"""Mirror the C client's ring/MDS health snapshot onto Prometheus.

Why: the vLLM connector's per-op metrics (``vllm:dfkv_store_*`` in metrics.py)
flow through vLLM's KVConnectorStats reduction on the frontend, but they cannot
express the client's *membership* health -- an empty ring (writes routed
nowhere, ok=0) or an unreachable ``mds_endpoint``. Those live only in the C
client (per worker process) and, before this, surfaced on vLLM only in the
client log. This poller re-exports the ring size + MDS reachability the C client
already publishes so the failure is visible on the SCRAPE (ring_members==0 /
mds_reachable==0), matching the production incident class of a silent empty ring.

The snapshot text is the C client's Prometheus-format dump (``dfkv_stats_snapshot``
in _cabi.py). The parsing + poll loop mirror the SGLang HiCache poller in
integration/hicache/dfkv_metrics.py (ClientStatsPoller); it ships as a separate
package, so the minimal parser is copied here rather than imported across
integration packages.

Metrics (labels: tp_rank), ``vllm:`` prefix per the connector convention:
  vllm:dfkv_client_ring_members                 ring size the client sees (0 = empty ring)
  vllm:dfkv_client_mds_reachable                1 if the MDS answered the last poll, else 0
  vllm:dfkv_client_mds_unreachable_polls_total  MDS discovery polls that failed (mirrors the C counter)
  vllm:dfkv_client_transport_info               1, labeled with the live transport (rdma/tcp)
"""
import ctypes
import threading

try:
    from prometheus_client import Gauge as _PromGauge
    _HAVE_PROM = True
except Exception:  # prometheus_client absent -> in-process gauges only
    _HAVE_PROM = False

# Snapshot metric name (unprefixed, as the C client emits it) -> exported name.
_GAUGE_SPECS = [
    ("dfkv_client_ring_members", "vllm:dfkv_client_ring_members",
     "ring members the dfkv client sees (0 = empty ring)"),
    ("dfkv_client_mds_reachable", "vllm:dfkv_client_mds_reachable",
     "1 if the MDS answered the last discovery poll, else 0"),
    ("dfkv_client_mds_unreachable_polls_total",
     "vllm:dfkv_client_mds_unreachable_polls_total",
     "MDS discovery polls that could not reach the MDS (mirrors the C counter)"),
]
_SOURCE_NAMES = frozenset(src for src, _, _ in _GAUGE_SPECS)

# Prometheus Gauges are process-global singletons (re-creating a name raises), so
# build them once at import and reuse across the per-process client instances.
_PROM = {}
_TRANSPORT_PROM = None
if _HAVE_PROM:
    try:
        for _src, _name, _help in _GAUGE_SPECS:
            # 'liveall' keeps each live rank's current value under SGLang/vLLM
            # multiprocess metrics mode (PROMETHEUS_MULTIPROC_DIR); tp_rank labels them.
            _PROM[_src] = _PromGauge(_name, _help, ["tp_rank"],
                                     multiprocess_mode="liveall")
        _TRANSPORT_PROM = _PromGauge(
            "vllm:dfkv_client_transport_info",
            "dfkv client live transport (value is always 1)",
            ["tp_rank", "transport"], multiprocess_mode="liveall")
    except Exception:
        _HAVE_PROM = False
        _PROM = {}
        _TRANSPORT_PROM = None


def read_snapshot(lib, h) -> str:
    """Read the C client's Prometheus metrics snapshot (size query, then fetch).
    Mirrors integration/hicache/dfkv_hicache.py:_read_snapshot."""
    need = int(lib.dfkv_stats_snapshot(h, None, 0))
    if need <= 0:
        return ""
    buf = ctypes.create_string_buffer(need + 1)
    lib.dfkv_stats_snapshot(h, buf, need + 1)
    return buf.value.decode("utf-8", "replace")


class ClientStatsPoller:
    """Polls the C client's Prometheus-text snapshot and mirrors the ring/MDS
    gauges onto Prometheus. One sleeping daemon thread, off the request path."""

    def __init__(self, get_text, tp_rank, transport="", interval_s=15.0):
        self._get_text = get_text
        self._rank = str(int(tp_rank))
        self._transport = transport or ""
        self._interval = float(interval_s)
        self._gauges = {src: 0 for src in _SOURCE_NAMES}  # last-seen (tests/debug)
        self._stop = threading.Event()
        self._thread = None
        self._warned = False

    @staticmethod
    def _parse(text):
        # Copied from integration/hicache/dfkv_metrics.py:ClientStatsPoller._parse.
        out = {}
        for line in text.splitlines():
            if not line or line[0] == "#":
                continue
            sp = line.rfind(" ")
            if sp <= 0:
                continue
            name = line[:sp]
            brace = name.find("{")
            if brace != -1:
                name = name[:brace]
            if name in _SOURCE_NAMES:
                try:
                    out[name] = out.get(name, 0) + int(line[sp + 1:])
                except ValueError:
                    pass
        return out

    def poll_once(self):
        vals = self._parse(self._get_text() or "")
        for src in _SOURCE_NAMES:
            v = vals.get(src, 0)
            self._gauges[src] = v
            if _HAVE_PROM and src in _PROM:
                _PROM[src].labels(self._rank).set(v)

    def gauges(self):
        return dict(self._gauges)

    def _loop(self):
        while not self._stop.wait(self._interval):
            try:
                self.poll_once()
            except Exception:
                if not self._warned:  # log once, never let it kill the thread
                    self._warned = True
                    import warnings
                    warnings.warn("dfkv client stats poll failed (further errors "
                                  "suppressed)", stacklevel=2)

    def start(self):
        if self._interval <= 0:
            return  # disabled
        # transport is static per client; publish it once up front.
        if _HAVE_PROM and _TRANSPORT_PROM is not None and self._transport:
            _TRANSPORT_PROM.labels(self._rank, self._transport).set(1)
        try:
            self.poll_once()  # immediate first read
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop, name="dfkv-client-stats",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
