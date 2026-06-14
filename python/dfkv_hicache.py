"""DingoFS HiCache storage backend for SGLang (loaded via --hicache-storage-backend
dynamic). Zero-copy v1 path: hands raw host-buffer pointers from
mem_pool_host.get_page_buffer_meta() straight to the DingoFS KV client (C ABI).

MLA (GLM-5.1): one packed-latent object per page, key has NO tp_rank suffix, and
only tp_rank 0 writes (backup_skip) since the latent is replicated across TP.

This file is the production plugin. On a GPU host it imports the real SGLang
HiCacheStorage; the test harness supplies a no-torch shim with the same surface.
"""
from __future__ import annotations

import ctypes
import os
from typing import List, Optional

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

_FLAG_IS_MLA = 0x1


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
    lib.dfkv_close.restype = None
    lib.dfkv_close.argtypes = [ctypes.c_void_p]
    return lib


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
        mds = cfg.get("mds_endpoints", "")
        members = cfg.get("members", "")
        if not mds and not members:
            raise ValueError("dingofs hicache: extra_config needs 'mds_endpoints' (MDS discovery) or 'members' (static)")
        self._lib = _load_lib(cfg.get("lib_path"))
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
        if mds:
            group = cfg.get("mds_group", "default")
            poll_ms = int(cfg.get("mds_poll_ms", 3000))
            rc = self._lib.dfkv_start_mds_discovery(self._h, mds.encode(), group.encode(), poll_ms)
            if rc != 0:
                raise RuntimeError("dfkv_start_mds_discovery failed")
        self.mem_pool_host = None

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                self._lib.dfkv_close(self._h)
                self._h = None
        except Exception:
            pass

    def register_mem_pool_host(self, mem_pool_host):
        self.mem_pool_host = mem_pool_host

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

    # --- key scheme: MLA single object (no rank suffix); MHA two objects ---
    def _keys(self, page_hash: str) -> List[str]:
        if self.is_mla:
            return [f"{self.model}/{page_hash}_k"]
        base = f"{self.model}/{page_hash}_{self.tp_size}_{self.tp_rank}"
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

    # --- zero-copy v1 batch path (the one the controller calls) ---
    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        # MLA backup_skip: latent is replicated across TP, only rank 0 writes.
        if self.is_mla and self.tp_rank != 0:
            return [True] * len(keys)
        ptrs, sizes = self.mem_pool_host.get_page_buffer_meta(host_indices)
        sub, sks, sp, ss = self._flatten(keys, ptrs, sizes)
        karr, parr, sarr, out, _kb = _arrays(sks, sp, ss)
        self._lib.dfkv_batch_put(self._h, karr, parr, sarr, len(sks), out)
        return self._fold([out[i] == 1 for i in range(len(sks))], len(keys), sub)

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        ptrs, sizes = self.mem_pool_host.get_page_buffer_meta(host_indices)
        sub, sks, sp, ss = self._flatten(keys, ptrs, sizes)
        karr, parr, sarr, out, _kb = _arrays(sks, sp, ss)
        self._lib.dfkv_batch_get(self._h, karr, parr, sarr, len(sks), out)
        return self._fold([out[i] == 1 for i in range(len(sks))], len(keys), sub)

    def batch_exists(self, keys, extra_info=None) -> int:
        # longest contiguous prefix of pages whose every sub-object exists
        sub = self._sub()
        sks = [sk for k in keys for sk in self._keys(k)]
        karr = (ctypes.c_char_p * len(sks))(*[s.encode() for s in sks])
        out = (ctypes.c_int * len(sks))()
        self._lib.dfkv_batch_exist(self._h, karr, len(sks), out)
        page_ok = self._fold([out[i] == 1 for i in range(len(sks))], len(keys), sub)
        n = 0
        for ok in page_ok:
            if not ok:
                break
            n += 1
        return n

    # --- v2 pool-aware interface (multi-pool models: Mamba/SWA/DeepSeek-V4) ---
    def _pool_keys(self, pool_name: str, page_hash: str) -> List[str]:
        # primary KV pool keeps the MLA/MHA split; auxiliary pools are single-object.
        if pool_name in ("kv", "__default__"):
            return self._keys(page_hash)
        base = f"{self.model}/{page_hash}_{pool_name}"
        return [base + "_k"] if self.is_mla else [base + "_k", base + "_v"]

    def _pool_sub(self, pool_name: str) -> int:
        if pool_name in ("kv", "__default__"):
            return self._sub()
        return 1 if self.is_mla else 2

    def _v2_io(self, transfers, putting):
        results = {}
        for tr in transfers:
            name = str(tr.name)
            # MLA backup_skip: only tp_rank 0 writes the replicated latent pools.
            if putting and self.is_mla and self.tp_rank != 0:
                results[name] = [True] * len(tr.keys or [])
                continue
            pool = self.registered_pools[name]
            ptrs, sizes = pool.get_page_buffer_meta(tr.host_indices)
            sub = self._pool_sub(name)
            keys = tr.keys or []
            sks, sp, ss = [], [], []
            for i, k in enumerate(keys):
                for j, sk in enumerate(self._pool_keys(name, k)):
                    sks.append(sk); sp.append(int(ptrs[i * sub + j])); ss.append(int(sizes[i * sub + j]))
            karr, parr, sarr, out, _ = _arrays(sks, sp, ss)
            (self._lib.dfkv_batch_put if putting else self._lib.dfkv_batch_get)(
                self._h, karr, parr, sarr, len(sks), out)
            results[name] = self._fold([out[i] == 1 for i in range(len(sks))], len(keys), sub)
        return results

    def batch_set_v2(self, transfers, extra_info=None) -> dict:
        return self._v2_io(transfers, putting=True)

    def batch_get_v2(self, transfers, extra_info=None) -> dict:
        return self._v2_io(transfers, putting=False)

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        from sglang.srt.mem_cache.hicache_storage import PoolTransferResult, PoolHitPolicy
        # primary KV prefix
        kv_pages = self.batch_exists(keys)
        hit = {"kv": kv_pages} if kv_pages else {}
        final = kv_pages
        for tr in (pool_transfers or []):
            if final == 0:
                break
            name = str(tr.name)
            present = [all(self._lib.dfkv_exist(self._h, sk.encode()) == 1
                           for sk in self._pool_keys(name, k)) for k in keys[:kv_pages]]
            if tr.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                boundary = kv_pages if all(present) else 0
            else:  # ALL_PAGES
                boundary = 0
                for ok in present:
                    if not ok:
                        break
                    boundary += 1
            if boundary:
                hit[name] = boundary
            final = min(final, boundary)
        return PoolTransferResult(final, hit)

    # --- required abstract methods (non zero-copy / introspection) ---
    def exists(self, key) -> bool:
        return all(self._lib.dfkv_exist(self._h, sk.encode()) == 1 for sk in self._keys(key))

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        if value is None:
            return False
        sk = self._keys(key)[0]
        # SGLang's L3 backup path (_generic_page_set -> batch_set) passes torch
        # Tensors, not bytes. Take the raw tensor bytes via data_ptr (dtype-
        # agnostic, works for fp8 which numpy can't represent). Tensor must stay
        # alive across the call (local `t`).
        if hasattr(value, "data_ptr"):
            t = value.detach().cpu().contiguous()
            nbytes = t.numel() * t.element_size()
            return self._lib.dfkv_put(self._h, sk.encode(),
                                      ctypes.c_void_p(t.data_ptr()),
                                      ctypes.c_uint64(nbytes)) == 0
        mv = memoryview(value).cast("B")
        buf = (ctypes.c_char * len(mv)).from_buffer_copy(mv)
        return self._lib.dfkv_put(self._h, sk.encode(), ctypes.cast(buf, ctypes.c_void_p),
                                  ctypes.c_uint64(len(mv))) == 0

    def get(self, key, target_location=None, target_sizes=None):
        # Generic (non zero-copy) read: dfkv_get reads the page bytes straight
        # into target_location's buffer (a host flat-page tensor). Symmetric with
        # set() (whole page under _keys[0]). Returns target_location on hit, None
        # on miss. SGLang's prod path uses batch_get_v1; this serves the generic
        # path + direct/test callers.
        if target_location is None:
            return None
        sk = self._keys(key)[0]
        nbytes = target_location.numel() * target_location.element_size()
        rc = self._lib.dfkv_get(self._h, sk.encode(),
                                ctypes.c_void_p(target_location.data_ptr()),
                                ctypes.c_uint64(nbytes))
        return target_location if rc == 1 else None

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        if values is None:
            return False
        return all(self.set(k, v) for k, v in zip(keys, values))

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        if target_locations is None:
            return [None] * len(keys)
        return [self.get(k, t) for k, t in zip(keys, target_locations)]
