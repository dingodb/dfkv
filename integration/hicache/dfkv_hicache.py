"""DingoFS HiCache storage backend for SGLang (loaded via --hicache-storage-backend
dynamic). Zero-copy v1 path: hands raw host-buffer pointers from
mem_pool_host.get_page_buffer_meta() straight to the DingoFS KV client (C ABI).

Key scheme:
- MLA (GLM-5.1): one packed-latent object per page. The latent is replicated
  across *TP*, so the key has NO tp_rank suffix and only tp_rank 0 writes
  (backup_skip). PP splits the model by *layer*, so the latent is NOT replicated
  across PP stages — when pp_size > 1 every key carries _pp{pp_rank} (including
  MLA) so stages holding different layer-slices do not collide.
- MHA: two objects (_k/_v) per page, suffixed by tp_size/tp_rank (+ _pp{pp_rank}
  when PP is on).

This mirrors SGLang's reference HiCacheFile suffix (hicache_storage.py), where
`enable_pp` appends `_{pp_size}_{pp_rank}` unconditionally — including MLA.

This file is the production plugin. On a GPU host it imports the real SGLang
HiCacheStorage; the test harness supplies a no-torch shim with the same surface.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from typing import List, Optional

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

from dfkv_access_log import (access_log, configure as _configure_access_log,
                            fmt_bytes as _fmt_bytes, fmt_pools as _fmt_pools,
                            fmt_pool_results as _fmt_pool_results)
from dfkv_metrics import Metrics as _Metrics, ClientStatsPoller as _ClientStatsPoller
from dfkv_telemetry import metrics as _push_metrics, config as _tcfg
from dfkv_telemetry import tracing as _tracing

_FLAG_IS_MLA = 0x1


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)


def resolve_node_dedup(cfg_value, env_value, is_mla: bool, tp_size: int):
    """Decide DFKV_CLIENT_NODE_DEDUP: (value_to_set | None, auto_enabled).

    Precedence: explicit extra-config beats the env, the env beats the auto
    default. The auto default enables the same-host rendezvous exactly for
    the topology it exists for — MLA (TP-replicated KV) with tp_size > 1;
    other topologies never rendezvous (per-rank keys differ) and are spared
    the /dev/shm arena. Pure function so the policy is unit-testable.
    """
    if cfg_value is not None:
        return ("1" if _truthy(cfg_value) else "0"), False
    if env_value is not None:
        return None, False  # operator's env stands as-is
    if is_mla and tp_size > 1:
        return "1", True
    return None, False


def _load_lib(path: Optional[str] = None) -> ctypes.CDLL:
    lib_path = (path or os.environ.get("DFKV_LIB")
                or os.path.join(os.environ.get("DFKV_BUILD", "/home/ketor/dfkv-dev/build"),
                                "libdfkv.so"))
    lib = ctypes.CDLL(lib_path)
    lib.dfkv_open.restype = ctypes.c_void_p
    lib.dfkv_open.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint32,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                              ctypes.c_uint32]
    lib.dfkv_put.restype = ctypes.c_int
    lib.dfkv_put.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_get.restype = ctypes.c_int
    lib.dfkv_get.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_exist.restype = ctypes.c_int
    lib.dfkv_exist.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.dfkv_register_memory.restype = ctypes.c_int
    lib.dfkv_register_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_batch_put.restype = ctypes.c_int
    lib.dfkv_batch_put.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
                                   ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int)]
    lib.dfkv_batch_get.restype = ctypes.c_int
    lib.dfkv_batch_get.argtypes = lib.dfkv_batch_put.argtypes
    lib.dfkv_batch_exist.restype = ctypes.c_int
    lib.dfkv_batch_exist.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.dfkv_set_members.restype = ctypes.c_int
    lib.dfkv_set_members.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.dfkv_refresh_members.restype = ctypes.c_int
    lib.dfkv_refresh_members.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.dfkv_start_mds_discovery.restype = ctypes.c_int
    lib.dfkv_start_mds_discovery.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    # Client registration (additive >= dfkv with the /clients/<id> lease). Guarded
    # at the call site for older libdfkv.so without the symbol (same pattern as the
    # vLLM/LMCache connectors — see integration/vllm + integration/lmcache).
    if hasattr(lib, "dfkv_start_client_registration"):
        lib.dfkv_start_client_registration.restype = ctypes.c_int
        lib.dfkv_start_client_registration.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                       ctypes.c_char_p, ctypes.c_char_p,
                                                       ctypes.c_char_p, ctypes.c_int]
    lib.dfkv_transport_mode.restype = ctypes.c_char_p
    lib.dfkv_transport_mode.argtypes = [ctypes.c_void_p]
    lib.dfkv_set_batch_concurrency.restype = ctypes.c_int
    lib.dfkv_set_batch_concurrency.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_stats_snapshot.restype = ctypes.c_uint64
    lib.dfkv_stats_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64]
    lib.dfkv_version.restype = ctypes.c_char_p
    lib.dfkv_version.argtypes = []
    lib.dfkv_close.restype = None
    lib.dfkv_close.argtypes = [ctypes.c_void_p]
    return lib


def _native_version(lib) -> str:
    """libdfkv.so version via the C dfkv_version(); "" if the symbol is missing
    (older lib) or the call fails. Never raises."""
    try:
        v = lib.dfkv_version()
        return v.decode("utf-8", "replace") if v else ""
    except Exception:
        return ""


def _read_snapshot(lib, h) -> str:
    """Read the C client's Prometheus metrics snapshot (size query, then fetch)."""
    need = int(lib.dfkv_stats_snapshot(h, None, 0))
    if need <= 0:
        return ""
    buf = ctypes.create_string_buffer(need + 1)
    lib.dfkv_stats_snapshot(h, buf, need + 1)
    return buf.value.decode("utf-8", "replace")


# A valid non-null pointer for zero-length "marker" puts (the logical-anchor
# "kv" pool of V4/DSA models). The payload is 0 bytes, so the pointer is never
# dereferenced, but it must be non-null so the batch slot isn't treated as a
# failed (null-key) entry by the C ABI.
_MARKER_BUF = ctypes.create_string_buffer(1)
_MARKER_PTR = ctypes.addressof(_MARKER_BUF)


def _arrays(subkeys, ptrs, sizes):
    """Build parallel C arrays (keys, ptrs, sizes) for a batch call."""
    n = len(subkeys)
    kbuf = [k.encode() for k in subkeys]
    karr = (ctypes.c_char_p * n)(*kbuf)
    parr = (ctypes.c_void_p * n)(*[ctypes.c_void_p(int(p)) for p in ptrs])
    sarr = (ctypes.c_uint64 * n)(*[int(s) for s in sizes])
    out = (ctypes.c_int * n)()
    return karr, parr, sarr, out, kbuf


class DfkvHiCache(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, kwargs: Optional[dict] = None):
        cfg = (kwargs or {}) or (getattr(storage_config, "extra_config", None) or {})
        self.cfg = cfg
        # Enforce the zero-copy deploy contract. SGLang's cache_controller only
        # selects the zero-copy v1 path (batch_set_v1/batch_get_v1) for a
        # 'dynamic' backend when extra_config.interface_v1 is truthy; otherwise it
        # uses the generic set/get copy path. The generic path is implemented and
        # correct, but slower (extra host copies) and for MLA every TP rank writes
        # the page redundantly. Fail fast so a launch-config omission can't quietly
        # degrade to the copy path.
        if not cfg.get("interface_v1"):
            raise ValueError(
                "dfkv requires extra_config interface_v1=1 to select the zero-copy "
                "v1 RDMA path. Omitting it falls back to the generic copy path "
                "(slower; MLA writes redundant per-rank copies)."
            )
        self.model = (storage_config.model_name or "").replace("/", "-")
        self.tp_rank = int(storage_config.tp_rank)
        self.tp_size = int(storage_config.tp_size)
        self.is_mla = bool(storage_config.is_mla_model)
        # Pipeline-parallel rank/size. PP splits the model across stages by
        # layer, so each pp_rank holds a *different* slice of KV (unlike TP,
        # where MLA latent is replicated). Storage keys MUST therefore carry
        # pp_rank when pp_size > 1, or PP stages overwrite each other's pages.
        # Mirrors SGLang's reference HiCacheFile suffix (hicache_storage.py:
        # `if enable_pp: config_suffix += f"_{pp_size}_{pp_rank}"`, applied
        # unconditionally — including MLA models).
        self.pp_rank = int(getattr(storage_config, "pp_rank", 0))
        self.pp_size = int(getattr(storage_config, "pp_size", 1))
        self.enable_pp = self.pp_size > 1
        # One-shot guard for the logical-anchor notice (V4/DSA models whose
        # primary "kv" pool holds no host buffer — see _note_logical_anchor_once).
        self._anchor_noop_warned = False
        # Access log: idempotent per process (first instance wins). Configured
        # here so tp_rank/model are available for the {rank} path placeholder.
        _configure_access_log(cfg, tp_rank=self.tp_rank, model=self.model)
        self._alog_tag = f"r{self.tp_rank}"
        # Client-side read/write counters (Prometheus when available). Lets ops
        # confirm SGLang->dfkv volume from /metrics instead of parsing access logs.
        self._metrics = _Metrics(self.tp_rank)
        # Load the native lib up front so its version can ride on the fleet-metrics
        # resource (dfkv_native_version). The actual client (dfkv_open) is created
        # later, inside the access-log "init" block. The plugin ships with the dfkv
        # repo, so its package version == the native lib version.
        self._lib = _load_lib(cfg.get("lib_path"))
        native_ver = _native_version(self._lib)
        # Unified fleet metrics pushed over OTLP to the central Collector (off by
        # default; zero cost when DFKV_METRICS_ENABLED is unset). Powers the
        # cross-connector global dashboard (per connector_id / type).
        _push_metrics.configure(cfg, connector_type=_tcfg.TYPE_HICACHE,
                                tp_rank=self.tp_rank, model=self.model,
                                version=native_ver, native_version=native_ver)
        # Connector-side request tracing (off by default; zero cost when off).
        # Emits a span per op for slow / sampled / failed requests over OTLP
        # /v1/traces — same identity + endpoint as the fleet metrics above.
        _tracing.configure(cfg, connector_type=_tcfg.TYPE_HICACHE,
                           tp_rank=self.tp_rank, model=self.model,
                           version=native_ver, native_version=native_ver)
        # When metrics are on, enable the C client's active per-peer latency probe
        # (read at dfkv_open below) so every cache node shows avg/max latency even
        # when idle. extra_config 'probe_interval_ms' overrides; 0 disables.
        if _push_metrics.is_enabled():
            probe_ms = int(float(_tcfg.resolve(
                cfg, "probe_interval_ms", _tcfg.ENV_PROBE_INTERVAL_MS, 5000)))
            os.environ["DFKV_PROBE_INTERVAL_MS"] = str(probe_ms)
        # Log the open/discovery setup (the access log is live from here on; the
        # earlier interface_v1 check raises before config is resolved).
        with access_log("init", lambda: f"{self._alog_tag} {self.model} "
                        f"tp={self.tp_rank}/{self.tp_size} mla={int(self.is_mla)}") as r:
            mds = cfg.get("mds_endpoints", "")
            members = cfg.get("members", "")
            if not mds and not members:
                raise ValueError("dingofs hicache: extra_config needs 'mds_endpoints' (MDS discovery) or 'members' (static)")
            # RDMA write pipelining: depth>1 keeps multiple PUTs in flight on one
            # connection, hiding per-op latency (single-rank MLA writes are
            # latency-bound). The C client reads DFKV_RDMA_DEPTH when it builds the
            # transport inside dfkv_open below, so set it from extra_config first.
            # NOTE: the dfkv_server must set the SAME (or larger) DFKV_RDMA_DEPTH in
            # its own env -- client depth must be <= server depth.
            if cfg.get("rdma_depth"):
                os.environ["DFKV_RDMA_DEPTH"] = str(int(cfg["rdma_depth"]))
            if _truthy(cfg.get("require_rdma")):
                os.environ["DFKV_REQUIRE_RDMA"] = "1"
                if not _truthy(os.environ.get("DFKV_RDMA")):
                    os.environ["DFKV_RDMA"] = "1"
            # rail_affinity (per-tp_rank narrowing) is DEPRECATED and now a no-op:
            # it keyed off tp_rank, which is always 0 under DP-attention (every rank
            # is its own attention TP group of size 1), so it collapsed all ranks to
            # one rail. NUMA-aware rail selection now lives in the C++ client: keep
            # the full multi-rail DFKV_RDMA_DEV and set DFKV_RDMA_NUMA=1, and the
            # client picks a NUMA-local rail per connection (works for TP and DP).
            if cfg.get("rail_affinity"):
                import sys as _sys
                print("[dfkv] WARNING: 'rail_affinity' is deprecated and ignored; "
                      "set DFKV_RDMA_NUMA=1 + multi-rail DFKV_RDMA_DEV for NUMA-aware "
                      "rail selection in the client.", file=_sys.stderr, flush=True)
            if cfg.get("rdma_numa"):
                os.environ.setdefault("DFKV_RDMA_NUMA", "1")
            # Same-host GET rendezvous (phase 5): dedups TP-replicated L3 loads
            # across the rank processes of one node (HiCache destinations are
            # HOST memory — the host flavor applies). Since phase 9 it is ON BY
            # DEFAULT exactly for the topology it exists for: MLA (replicated
            # KV) with tp_size > 1 — the measured 8x lockstep read case. Other
            # topologies gain nothing (per-rank keys never rendezvous) and are
            # spared the 512 MiB /dev/shm arena. Explicit settings always win:
            # extra-config `node_dedup` beats the env, the env beats the auto
            # default; "0" anywhere disables.
            _dedup, _auto = resolve_node_dedup(
                cfg.get("node_dedup"), os.environ.get("DFKV_CLIENT_NODE_DEDUP"),
                self.is_mla, self.tp_size)
            if _dedup is not None:
                os.environ["DFKV_CLIENT_NODE_DEDUP"] = _dedup
            if _auto:
                print(f"[dfkv] node-dedup auto-enabled (mla, tp={self.tp_size}): "
                      "same-host rendezvous collapses replicated L3 loads; "
                      "set node_dedup=0 or DFKV_CLIENT_NODE_DEDUP=0 to disable.",
                      file=sys.stderr, flush=True)
            # Phase 9: exist gate on the backup path. Recomputed pages whose
            # KV is ALREADY in L3 were re-backed wholesale (measured 485 GB
            # per hot round on the canary — write pressure that starved the
            # very reads the cache exists for). One batch_exist (collapsed by
            # the rendezvous) filters them out. backup_exist_gate=0 disables.
            self._backup_exist_gate = _truthy(
                cfg.get("backup_exist_gate",
                        os.environ.get("DFKV_BACKUP_EXIST_GATE", "1")))
            # self._lib was loaded above (before configure) so the native version
            # could be reported; dfkv_open uses that same handle here.
            flags = _FLAG_IS_MLA if self.is_mla else 0
            model_hash = int(cfg.get("model_hash", 0)) & 0xFFFFFFFFFFFFFFFF
            self._h = self._lib.dfkv_open(
                members.encode(), model_hash,
                int(cfg.get("page_size", 64)), int(cfg.get("dtype_tag", 0)), flags,
                self.tp_size, self.tp_rank,
                int(cfg.get("layer_num", 0)), int(cfg.get("head_num", 0)),
                int(cfg.get("head_dim", 0)))
            if not self._h:
                raise RuntimeError("dfkv_open failed")
            mode_b = self._lib.dfkv_transport_mode(self._h)
            self.transport_mode = (
                mode_b.decode("utf-8", errors="replace") if mode_b else "unknown"
            )
            if (_truthy(cfg.get("require_rdma")) or
                    _truthy(os.environ.get("DFKV_REQUIRE_RDMA"))):
                if self.transport_mode != "rdma":
                    self._lib.dfkv_close(self._h)
                    self._h = None
                    raise RuntimeError(
                        "dfkv requires RDMA zero-copy transport, "
                        f"got {self.transport_mode}"
                    )
            if cfg.get("batch_concurrency"):
                self._lib.dfkv_set_batch_concurrency(
                    self._h, ctypes.c_uint64(int(cfg["batch_concurrency"])))
            if mds:
                group = cfg.get("mds_group", "default")
                poll_ms = int(cfg.get("mds_poll_ms", 3000))
                rc = self._lib.dfkv_start_mds_discovery(self._h, mds.encode(), group.encode(), poll_ms)
                if rc != 0:
                    raise RuntimeError("dfkv_start_mds_discovery failed")
                # Register this SGLang HiCache connector as a cache consumer so
                # `dfkvctl clients` can answer "who is using dfkv" (parity with the
                # vLLM/LMCache connectors added in v1.15.0). Best-effort: a missing
                # symbol (older libdfkv.so) or a registration failure is logged,
                # never fatal — the data path is already up via discovery above.
                # Default on; opt out with extra_config client_register=0 or
                # DFKV_CLIENT_REGISTER=0. SGLang HiCache is a prefix L3 cache with
                # no producer/consumer split, so no 'role' field (the CLI shows '-'
                # for it, same as LMCache which has no role either).
                if _truthy(cfg.get("client_register",
                                    os.environ.get("DFKV_CLIENT_REGISTER", "1"))):
                    cid = _tcfg.resolve_connector_id(cfg, tp_rank=self.tp_rank)
                    info = (f"type={_tcfg.TYPE_HICACHE},model={self.model},"
                            f"tp_size={self.tp_size},tp_rank={self.tp_rank},"
                            f"ver={native_ver}")
                    try:
                        rc2 = self._lib.dfkv_start_client_registration(
                            self._h, mds.encode(), group.encode(), cid.encode(),
                            info.encode(), 10000)
                        if rc2 != 0:
                            raise RuntimeError(f"rc={rc2}")
                    except AttributeError:
                        pass  # older libdfkv.so without the symbol — skip silently
                    except Exception as e:  # noqa: BLE001 — never block startup
                        import warnings
                        warnings.warn(
                            f"dfkv client registration skipped (mds={mds!r}): {e}",
                            stacklevel=2)
            self.mem_pool_host = None
            mode = f" transport={self.transport_mode}"
            r.result = ("ok mds-discovery" if mds else "ok static") + mode

        # Background mirror of the C client's metrics snapshot onto Prometheus
        # (sleeping daemon thread; off the request path). client_stats_poll_s<=0
        # disables it. extra_config wins, then env, then 10s default.
        self._poller = None
        try:
            poll_s = cfg.get("client_stats_poll_s")
            if poll_s is None:
                poll_s = os.environ.get("DFKV_CLIENT_STATS_POLL_S", 10)
            poll_s = float(poll_s)
        except (TypeError, ValueError):
            poll_s = 10.0
        if poll_s > 0:
            self._poller = _ClientStatsPoller(
                lambda: _read_snapshot(self._lib, self._h), self.tp_rank, poll_s)
            self._poller.start()
        # Per-cache-node latency: parse the same snapshot and push avg/max per peer
        # over OTLP (no-op unless metrics enabled). Off the request path.
        peer_poll_s = float(_tcfg.resolve(
            cfg, "peer_latency_poll_s", _tcfg.ENV_PEER_POLL_S,
            poll_s if poll_s > 0 else 10.0))
        self._peer_lat_poller = _push_metrics.start_peer_latency_poller(
            lambda: _read_snapshot(self._lib, self._h), peer_poll_s)

    def __del__(self):
        try:
            if getattr(self, "_poller", None):
                self._poller.stop()
                self._poller = None
        except Exception:
            pass
        try:
            if getattr(self, "_peer_lat_poller", None):
                self._peer_lat_poller.stop()
                self._peer_lat_poller = None
        except Exception:
            pass
        try:
            if getattr(self, "_h", None):
                self._lib.dfkv_close(self._h)
                self._h = None
        except Exception:
            pass

    def register_memory(self, base: int, size: int) -> bool:
        """Register a host memory region for RDMA zero-copy (registered once per
        connection; buffers inside it then transfer with no per-op MR register).
        No-op on TCP. Returns True on success."""
        if not base or not size:
            return False
        return self._lib.dfkv_register_memory(
            self._h, ctypes.c_void_p(int(base)), ctypes.c_uint64(int(size))) == 0

    def _register_pool_buffers(self, pool) -> int:
        """Best-effort: register a host pool's backing buffer(s) so its pages
        transfer zero-copy without per-op MR registration. Probes the common
        SGLang HostKVCache backing-buffer attributes; failure is non-fatal (pages
        just fall back to ad-hoc per-buffer registration). Returns #regions done."""
        done = 0
        for attr in ("kv_buffer", "host_kv_buffer", "data_buffer", "buffer", "data"):
            buf = getattr(pool, attr, None)
            if buf is None:
                continue
            tensors = buf if isinstance(buf, (list, tuple)) else [buf]
            for t in tensors:
                if not (hasattr(t, "data_ptr") and hasattr(t, "numel")
                        and hasattr(t, "element_size")):
                    continue
                try:
                    if self.register_memory(t.data_ptr(), t.numel() * t.element_size()):
                        done += 1
                except Exception:
                    pass
            if done:
                break  # first attribute that yielded registrable tensors wins
        return done

    def register_mem_pool_host(self, mem_pool_host):
        self.mem_pool_host = mem_pool_host
        with access_log("register_mem_pool_host",
                        lambda: f"{self._alog_tag}") as r:
            n = 0
            try:
                n = self._register_pool_buffers(mem_pool_host)
            except Exception as e:  # never fail HiCache setup over an optimization
                r.result = f"skip ({type(e).__name__})"
                return
            r.result = f"registered {n} region(s)" if n else "no backing buffer found"

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        name = str(host_pool_name)
        self.registered_pools[name] = host_pool
        with access_log("register_mem_host_pool_v2",
                        lambda: f"{self._alog_tag} {name}") as r:
            n = 0
            try:
                n = self._register_pool_buffers(host_pool)
            except Exception as e:
                r.result = f"skip ({type(e).__name__})"
                return
            r.result = f"registered {n} region(s)" if n else "no backing buffer found"

    def set_members(self, members: str):
        """Hot-swap cluster membership, e.g. 'n1=ip:12000,n2=ip:12000'."""
        self._lib.dfkv_set_members(self._h, members.encode())

    def refresh_members(self, seed: str) -> bool:
        """Discover cluster membership from a seed node ('ip:port') and apply it.
        Lets the cluster grow/shrink without restarting clients. Returns True on
        success (seed reachable and returned a non-empty member list)."""
        return self._lib.dfkv_refresh_members(self._h, seed.encode()) == 0

    def start_mds_discovery(self, mds_endpoints: str, group: str = "default", poll_ms: int = 3000) -> bool:
        """Start background MDS-based discovery. mds_endpoints: comma-separated 'ip:port' list.
        Returns True on success."""
        return self._lib.dfkv_start_mds_discovery(self._h, mds_endpoints.encode(), group.encode(), poll_ms) == 0

    # --- key scheme: MLA single object (no tp_rank suffix); MHA two objects ---
    # PP note: when pp_size > 1, every path appends _pp{pp_rank} so stages
    # holding different layer-slices of KV do not collide on the same key.
    # This applies to MLA too — PP splits by layer, the latent is NOT
    # replicated across PP stages (only across TP). See __init__.
    def _pp_suffix(self) -> str:
        return f"_pp{self.pp_rank}" if self.enable_pp else ""

    def _keys(self, page_hash: str) -> List[str]:
        if self.is_mla:
            return [f"{self.model}/{page_hash}_k{self._pp_suffix()}"]
        base = f"{self.model}/{page_hash}_{self.tp_size}_{self.tp_rank}{self._pp_suffix()}"
        return [base + "_k", base + "_v"]

    def _sub(self) -> int:
        return 1 if self.is_mla else 2

    def _flatten(self, keys, ptrs, sizes):
        """Expand per-page keys into per-object (sub) flat arrays."""
        sub = self._sub()
        assert len(ptrs) == len(keys) * sub, (len(ptrs), len(keys), sub)
        sks, sp, ss = [], [], []
        for i, k in enumerate(keys):
            for j, sk in enumerate(self._keys(k)):
                sks.append(sk); sp.append(int(ptrs[i * sub + j])); ss.append(int(sizes[i * sub + j]))
        return sub, sks, sp, ss

    def _fold(self, flat_results, npages, sub):
        """A page succeeds iff all its sub-objects succeeded."""
        return [all(flat_results[i * sub + j] for j in range(sub)) for i in range(npages)]

    def _batch_exist_flat(self, subkeys) -> List[bool]:
        if not subkeys:
            return []
        kbuf = [s.encode() for s in subkeys]
        karr = (ctypes.c_char_p * len(kbuf))(*kbuf)
        out = (ctypes.c_int * len(kbuf))()
        rc = self._lib.dfkv_batch_exist(self._h, karr, len(kbuf), out)
        if rc != 0:
            return [False] * len(kbuf)
        return [out[i] == 1 for i in range(len(kbuf))]

    def _note_logical_anchor_once(self):
        """One-time notice that the primary KV pool is a logical anchor.

        SGLang builds a *logical anchor* (LogicalHostPool) as the primary "kv"
        pool for V4/DSA multi-pool models such as GLM-5.2 — it carries no KV
        tensor, so get_page_buffer_meta() returns None. The real KV rides the v2
        side-pool path (batch_set_v2/batch_get_v2); the v1 anchor only writes an
        empty "kv" marker so the v2 existence check can anchor the hit prefix
        (mirrors SGLang's reference backend). Informational, not an error — the
        path works, but it is newly enabled, so surface it once for ops to
        confirm the hit rate via metrics."""
        if self._anchor_noop_warned:
            return
        self._anchor_noop_warned = True
        import sys as _sys
        print("[dfkv] NOTE: primary KV pool is a logical anchor "
              "(get_page_buffer_meta -> None) — a V4/DSA multi-pool model "
              f"(model={self.model!r}, e.g. GLM-5.2). The v1 anchor writes an "
              "empty 'kv' marker; real KV rides the v2 side-pool path. Verify L3 "
              "hit rate via metrics; if low, the LMCache MP connector "
              "(docs/CONNECTORS.md #4.5) is the alternative path for GLM-5.x DSA.",
              file=_sys.stderr, flush=True)

    def _write_anchor_markers(self, keys) -> List[bool]:
        """Write an empty (0-byte) marker object per "kv" sub-key so a later
        batch_exists / batch_exists_v2 can find the primary-pool prefix.

        Used only for the logical-anchor case (V4/DSA, e.g. GLM-5.2): the anchor
        pool holds no KV buffer, so there is nothing to zero-copy — but SGLang's
        v2 existence check still gates the hit prefix on the primary "kv" keys,
        exactly as its reference backend does by writing an empty get_data_page().
        The marker carries the connector's geometry header (so a cross-geometry
        reader still misses); the matching read (batch_get_v1) is a no-op. Returns
        a per-page success list (a page succeeds iff all its sub-object markers
        were written)."""
        sub = self._sub()
        sks = [sk for k in keys for sk in self._keys(k)]
        if not sks:
            return []
        karr, parr, sarr, out, _ = _arrays(sks, [_MARKER_PTR] * len(sks),
                                           [0] * len(sks))
        self._lib.dfkv_batch_put(self._h, karr, parr, sarr, len(sks), out)
        return self._fold([out[i] == 1 for i in range(len(sks))], len(keys), sub)

    # --- zero-copy v1 batch path (the one the controller calls) ---
    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        n = len(keys)
        with _tracing.span("batch_set_v1", n) as _sp, \
                access_log("batch_set_v1", lambda: f"{self._alog_tag} {n} keys") as r:
            # MLA backup_skip: latent is replicated across TP, only rank 0 writes.
            if self.is_mla and self.tp_rank != 0:
                r.result = "backup_skip"
                if _sp:
                    _sp.attrs = {"dfkv.backup_skip": True}
                return [True] * n
            meta = self.mem_pool_host.get_page_buffer_meta(host_indices)
            if meta is None:
                # Primary "kv" pool is a *logical anchor* holding no KV buffer
                # (V4/DSA-compressed models, e.g. GLM-5.2: SGLang registers a
                # LogicalHostPool whose get_page_buffer_meta() returns None). The
                # real KV lives in compressed side-pools written via batch_set_v2;
                # there is nothing to zero-copy on the anchor, but we still write
                # an empty "kv" marker per page so the v2 existence check can
                # anchor the hit prefix (SGLang's reference backend does the same
                # with an empty get_data_page()). Single-pool models (MLA/MHA)
                # never return None, so this branch is inert for them.
                self._note_logical_anchor_once()
                res = self._write_anchor_markers(keys)
                r.result = f"anchor_marker {sum(res)}/{n}"
                if _sp:
                    _sp.hits = sum(res)
                    _sp.attrs = {"dfkv.anchor_marker": True}
                return res
            ptrs, sizes = meta
            sub, sks, sp, ss = self._flatten(keys, ptrs, sizes)
            t0 = time.perf_counter()
            flat = self._put_flat(sks, sp, ss)
            dur = time.perf_counter() - t0
            res = self._fold(flat, n, sub)
            r.result = f"ok {sum(res)}/{n}"
            if _sp:
                _sp.hits = sum(res); _sp.bytes = sum(ss)
            self._metrics.on_set(pages=n, ok_pages=sum(res), nbytes=sum(ss), seconds=dur)
            # Fleet op metrics (put/get/exist/...) are accumulated in the C++
            # KVClient (the chokepoint all connectors share) and forwarded over
            # OTLP by the snapshot poller — see dfkv_telemetry.parse_client_ops.
            return res

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        n = len(keys)
        with _tracing.span("batch_get_v1", n) as _sp, \
                access_log("batch_get_v1", lambda: f"{self._alog_tag} {n} keys") as r:
            meta = self.mem_pool_host.get_page_buffer_meta(host_indices)
            if meta is None:
                # Logical anchor pool, no buffer to scatter into (V4/DSA models,
                # e.g. GLM-5.2). The "kv" prefix was already confirmed present by
                # batch_exists_v2 (via the empty markers written on backup); there
                # is no anchor payload to load. Report all pages present so
                # _page_get_zero_copy counts the anchor prefix complete and the
                # hybrid controller then loads the real KV from side-pools via
                # batch_get_v2. Returning False here would make kv_completed_pages
                # < prefix and skip that side-pool load entirely. Inert for
                # single-pool models (non-None).
                self._note_logical_anchor_once()
                r.result = "anchor_noop"
                if _sp:
                    _sp.attrs = {"dfkv.anchor_noop": True}
                return [True] * n
            ptrs, sizes = meta
            sub, sks, sp, ss = self._flatten(keys, ptrs, sizes)
            karr, parr, sarr, out, _kb = _arrays(sks, sp, ss)
            t0 = time.perf_counter()
            self._lib.dfkv_batch_get(self._h, karr, parr, sarr, len(sks), out)
            dur = time.perf_counter() - t0
            res = self._fold([out[i] == 1 for i in range(len(sks))], n, sub)
            r.result = f"hits={sum(res)}/{n}"
            if _sp:
                _sp.hits = sum(res); _sp.bytes = sum(ss)
            self._metrics.on_get(pages=n, hit_pages=sum(res), nbytes=sum(ss), seconds=dur)
            # Fleet op metrics now come from the C++ KVClient snapshot (above).
            return res

    def batch_exists(self, keys, extra_info=None) -> int:
        total = len(keys)
        with _tracing.span("batch_exists", total) as _sp, \
                access_log("batch_exists", lambda: f"{self._alog_tag} {total} keys") as r:
            # longest contiguous prefix of pages whose every sub-object exists
            sub = self._sub()
            sks = [sk for k in keys for sk in self._keys(k)]
            page_ok = self._fold(self._batch_exist_flat(sks), total, sub)
            n = 0
            for ok in page_ok:
                if not ok:
                    break
                n += 1
            r.result = f"prefix={n}/{total}"
            if _sp:
                _sp.hits = n
            return n

    # --- v2 pool-aware interface (multi-pool models: Mamba/SWA/DeepSeek-V4) ---
    def _pool_keys(self, pool_name: str, page_hash: str) -> List[str]:
        # primary KV pool keeps the MLA/MHA split; auxiliary pools are single-object.
        # Both carry the PP suffix for the same layer-slice reason as _keys().
        pps = self._pp_suffix()
        if pool_name in ("kv", "__default__"):
            return self._keys(page_hash)
        base = f"{self.model}/{page_hash}_{pool_name}{pps}"
        return [base + "_k"] if self.is_mla else [base + "_k", base + "_v"]

    def _pool_sub(self, pool_name: str) -> int:
        if pool_name in ("kv", "__default__"):
            return self._sub()
        return 1 if self.is_mla else 2

    def _put_flat(self, sks, sp, ss) -> List[bool]:
        """batch_put with the phase-9 exist gate: sub-objects already in L3
        are reported stored without rewriting. Falls back to a straight put
        when the gate is off or everything is missing (cold path unchanged
        except one exist round-trip)."""
        if self._backup_exist_gate and sks:
            present = list(self._batch_exist_flat(sks))
            todo = [i for i, p in enumerate(present) if not p]
            if not todo:
                return [True] * len(sks)
            if len(todo) < len(sks):
                karr, parr, sarr, out, _kb = _arrays(
                    [sks[i] for i in todo], [sp[i] for i in todo],
                    [ss[i] for i in todo])
                self._lib.dfkv_batch_put(self._h, karr, parr, sarr,
                                         len(todo), out)
                for m, i in enumerate(todo):
                    present[i] = out[m] == 1
                return present
        karr, parr, sarr, out, _kb = _arrays(sks, sp, ss)
        self._lib.dfkv_batch_put(self._h, karr, parr, sarr, len(sks), out)
        return [out[i] == 1 for i in range(len(sks))]

    def _v2_io(self, transfers, putting):
        results = {}
        segments = []
        sks, sp, ss = [], [], []
        for tr in transfers:
            name = str(tr.name)
            keys = tr.keys or []
            # MLA backup_skip: only tp_rank 0 writes the replicated latent pools.
            if putting and self.is_mla and self.tp_rank != 0:
                results[name] = [True] * len(keys)
                continue
            pool = self.registered_pools[name]
            ptrs, sizes = pool.get_page_buffer_meta(tr.host_indices)
            sub = self._pool_sub(name)
            start = len(sks)
            for i, k in enumerate(keys):
                for j, sk in enumerate(self._pool_keys(name, k)):
                    sks.append(sk); sp.append(int(ptrs[i * sub + j])); ss.append(int(sizes[i * sub + j]))
            segments.append((name, len(keys), sub, start, len(sks)))
        if sks and putting:
            flat = self._put_flat(sks, sp, ss)
        elif sks:
            karr, parr, sarr, out, _ = _arrays(sks, sp, ss)
            self._lib.dfkv_batch_get(self._h, karr, parr, sarr, len(sks), out)
            flat = [out[i] == 1 for i in range(len(sks))]
        else:
            flat = []
        for name, nkeys, sub, start, end in segments:
            results[name] = self._fold(flat[start:end], nkeys, sub)
        return results

    def batch_set_v2(self, transfers, extra_info=None) -> dict:
        nkeys = sum(len(tr.keys or []) for tr in (transfers or []))
        with _tracing.span("batch_set_v2", nkeys) as _sp, \
                access_log("batch_set_v2",
                           lambda: f"{self._alog_tag} {_fmt_pools(transfers)}") as r:
            res = self._v2_io(transfers, putting=True)
            r.result = _fmt_pool_results(res)
            if _sp:
                _sp.hits = sum(sum(rs) for rs in res.values())
            return res

    def batch_get_v2(self, transfers, extra_info=None) -> dict:
        nkeys = sum(len(tr.keys or []) for tr in (transfers or []))
        with _tracing.span("batch_get_v2", nkeys) as _sp, \
                access_log("batch_get_v2",
                           lambda: f"{self._alog_tag} {_fmt_pools(transfers)}") as r:
            res = self._v2_io(transfers, putting=False)
            r.result = _fmt_pool_results(res)
            if _sp:
                _sp.hits = sum(sum(rs) for rs in res.values())
            return res

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        total = len(keys)
        with access_log("batch_exists_v2",
                        lambda: f"{self._alog_tag} {total} keys, "
                                f"{_fmt_pools(pool_transfers)}") as r:
            from sglang.srt.mem_cache.hicache_storage import PoolTransferResult, PoolHitPolicy
            # primary KV prefix
            kv_pages = self.batch_exists(keys)
            hit = {"kv": kv_pages} if kv_pages else {}
            final = kv_pages
            for tr in (pool_transfers or []):
                if final == 0:
                    break
                name = str(tr.name)
                sub = self._pool_sub(name)
                sks = [sk for k in keys[:kv_pages]
                       for sk in self._pool_keys(name, k)]
                present = self._fold(self._batch_exist_flat(sks), kv_pages, sub)
                if tr.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                    # Only the *last* `trailing` pages of a candidate prefix need
                    # this pool (e.g. SWA sliding window / Mamba state) — matches
                    # SGLang's reference batch_exists_v2. The previous `all(present)`
                    # wrongly required every prefix page, collapsing SWA hits to 0
                    # once early window pages had been evicted.
                    trailing = max(1, len(tr.keys) if tr.keys else 1)
                    boundary = 0
                    for prefix_len in range(kv_pages, 0, -1):
                        if all(present[i] for i in
                               range(max(0, prefix_len - trailing), prefix_len)):
                            boundary = prefix_len
                            break
                else:  # ALL_PAGES
                    boundary = 0
                    for ok in present:
                        if not ok:
                            break
                        boundary += 1
                if boundary:
                    hit[name] = boundary
                final = min(final, boundary)
            result = PoolTransferResult(final, hit)
            r.result = (f"kv={result.kv_hit_pages}/{total} "
                        + ",".join(f"{k}={v}" for k, v in hit.items()
                                   if k != "kv")).strip()
            return result

    # --- required abstract methods (non zero-copy / introspection) ---
    def exists(self, key) -> bool:
        with access_log("exists", lambda: f"{self._alog_tag} {key}") as r:
            found = all(self._lib.dfkv_exist(self._h, sk.encode()) == 1
                        for sk in self._keys(key))
            r.result = "found" if found else "not_found"
            return found

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        nbytes = 0
        with access_log("set",
                        lambda: f"{self._alog_tag} {key}, {_fmt_bytes(nbytes)}") as r:
            if value is None:
                r.result = "fail none"
                return False
            sk = self._keys(key)[0]
            # SGLang's L3 backup path (_generic_page_set -> batch_set) passes torch
            # Tensors, not bytes. Take the raw tensor bytes via data_ptr (dtype-
            # agnostic, works for fp8 which numpy can't represent). Tensor must stay
            # alive across the call (local `t`).
            if hasattr(value, "data_ptr"):
                t = value.detach().cpu().contiguous()
                nbytes = t.numel() * t.element_size()
                ok = self._lib.dfkv_put(self._h, sk.encode(),
                                        ctypes.c_void_p(t.data_ptr()),
                                        ctypes.c_uint64(nbytes)) == 0
            else:
                mv = memoryview(value).cast("B")
                nbytes = len(mv)
                buf = (ctypes.c_char * nbytes).from_buffer_copy(mv)
                ok = self._lib.dfkv_put(self._h, sk.encode(),
                                        ctypes.cast(buf, ctypes.c_void_p),
                                        ctypes.c_uint64(nbytes)) == 0
            r.result = "ok" if ok else "fail"
            return ok

    def get(self, key, target_location=None, target_sizes=None):
        # Generic (non zero-copy) read: dfkv_get reads the page bytes straight
        # into target_location's buffer (a host flat-page tensor). Symmetric with
        # set() (whole page under _keys[0]). Returns target_location on hit, None
        # on miss. SGLang's prod path uses batch_get_v1; this serves the generic
        # path + direct/test callers.
        nbytes = 0
        with access_log("get",
                        lambda: f"{self._alog_tag} {key}, {_fmt_bytes(nbytes)}") as r:
            if target_location is None:
                r.result = "miss no_target"
                return None
            sk = self._keys(key)[0]
            nbytes = target_location.numel() * target_location.element_size()
            rc = self._lib.dfkv_get(self._h, sk.encode(),
                                    ctypes.c_void_p(target_location.data_ptr()),
                                    ctypes.c_uint64(nbytes))
            r.result = "hit" if rc == 1 else "miss"
            return target_location if rc == 1 else None

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        n = len(keys)
        with access_log("batch_set", lambda: f"{self._alog_tag} {n} keys") as r:
            if values is None:
                r.result = "fail none"
                return False
            # list (not short-circuiting all()) so every key is attempted and the
            # logged count is accurate; controller treats the bool as all-or-nothing.
            oks = [self.set(k, v) for k, v in zip(keys, values)]
            r.result = f"ok {sum(oks)}/{n}"
            return all(oks)

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        n = len(keys)
        with access_log("batch_get", lambda: f"{self._alog_tag} {n} keys") as r:
            if target_locations is None:
                r.result = "miss no_targets"
                return [None] * n
            res = [self.get(k, t) for k, t in zip(keys, target_locations)]
            r.result = f"hits={sum(1 for x in res if x is not None)}/{n}"
            return res
