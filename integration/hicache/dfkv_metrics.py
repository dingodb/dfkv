"""Client-side read/write counters for the dfkv HiCache plugin.

Why: under MLA/DSA, SGLang's own `cached_tokens_total{storage}` is collapsed into
the device tier and does not move, so confirming "did SGLang write/read dfkv, and
how much" previously required parsing per-op access logs. These counters make the
volume directly observable.

Always keeps in-process integer counters (cheap, queryable via snapshot() for
tests/debug). If prometheus_client is importable they are mirrored to Prometheus
Counters; with SGLang's multiprocess metrics mode (PROMETHEUS_MULTIPROC_DIR set)
those automatically aggregate onto the server's /metrics endpoint. No-op cost is a
few attribute increments when Prometheus is absent.

Metric names (labels: tp_rank):
  dfkv_client_set_calls_total      batch_set_v1 calls that actually wrote (MLA: rank0 only)
  dfkv_client_set_pages_total      KV pages offered to set
  dfkv_client_set_ok_pages_total   pages the server acked OK
  dfkv_client_set_bytes_total      bytes sent (sum of sub-object sizes)
  dfkv_client_get_calls_total      batch_get_v1 calls
  dfkv_client_get_pages_total      KV pages requested
  dfkv_client_get_hit_pages_total  pages returned as hits
  dfkv_client_get_bytes_total      bytes requested (sum of sub-object sizes)

v2 (hybrid/DSA side-pool) path. For V4/DSA models (e.g. GLM-5.2) the real KV
rides these while the *_v1 counters see only empty anchor markers, so the L3
write/hit rate lives here. The DSA L3 hit rate =
exist_v2_hit_pages_total / exist_v2_probe_pages_total.
  dfkv_client_set_v2_calls_total       batch_set_v2 calls that wrote (MLA: rank0 only)
  dfkv_client_set_v2_pages_total       side-pool pages offered to set_v2
  dfkv_client_set_v2_ok_pages_total    side-pool pages acked OK
  dfkv_client_set_v2_bytes_total       bytes sent on set_v2
  dfkv_client_get_v2_calls_total       batch_get_v2 calls
  dfkv_client_get_v2_pages_total       side-pool pages requested
  dfkv_client_get_v2_hit_pages_total   side-pool pages returned as hits
  dfkv_client_get_v2_bytes_total       bytes requested on get_v2
  dfkv_client_exist_v2_calls_total     batch_exists_v2 calls
  dfkv_client_exist_v2_probe_pages_total  candidate KV prefix pages probed
  dfkv_client_exist_v2_hit_pages_total    usable hit-prefix pages (kv_hit_pages)
"""
import threading

try:
    from prometheus_client import (Counter as _PromCounter, Gauge as _PromGauge,
                                    Histogram as _PromHistogram)
    _HAVE_PROM = True
except Exception:  # prometheus_client absent -> in-process counters only
    _HAVE_PROM = False

# Latency buckets (seconds) for the set/get batch-call duration histograms.
_HIST_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
_HIST = {}
if _HAVE_PROM:
    try:
        _HIST["set"] = _PromHistogram("dfkv_client_set_seconds",
                                      "batch_set_v1 call duration seconds",
                                      ["tp_rank"], buckets=_HIST_BUCKETS)
        _HIST["get"] = _PromHistogram("dfkv_client_get_seconds",
                                      "batch_get_v1 call duration seconds",
                                      ["tp_rank"], buckets=_HIST_BUCKETS)
        _HIST["set_v2"] = _PromHistogram("dfkv_client_set_v2_seconds",
                                         "batch_set_v2 I/O duration seconds",
                                         ["tp_rank"], buckets=_HIST_BUCKETS)
        _HIST["get_v2"] = _PromHistogram("dfkv_client_get_v2_seconds",
                                         "batch_get_v2 I/O duration seconds",
                                         ["tp_rank"], buckets=_HIST_BUCKETS)
    except Exception:
        _HIST = {}

_SPECS = [
    ("set_calls", "dfkv_client_set_calls_total", "batch_set_v1 calls that wrote"),
    ("set_pages", "dfkv_client_set_pages_total", "KV pages offered to set"),
    ("set_ok_pages", "dfkv_client_set_ok_pages_total", "pages acked OK by server"),
    ("set_bytes", "dfkv_client_set_bytes_total", "bytes sent on set"),
    ("get_calls", "dfkv_client_get_calls_total", "batch_get_v1 calls"),
    ("get_pages", "dfkv_client_get_pages_total", "KV pages requested"),
    ("get_hit_pages", "dfkv_client_get_hit_pages_total", "pages returned as hits"),
    ("get_bytes", "dfkv_client_get_bytes_total", "bytes requested on get"),
    # v2 (hybrid/DSA side-pool) path. For V4/DSA models (e.g. GLM-5.2) the real KV
    # rides batch_set_v2/batch_get_v2 while the v1 "kv" path only writes/reads
    # empty anchor markers, so the *_v1 counters above cannot express the L3
    # hit/write rate for those models — these do. batch_exists_v2 is the prefix
    # existence probe that drives the scheduler's hit prefix, so its hit rate
    # (exist_v2_hit_pages / exist_v2_probe_pages) is the L3 hit-rate signal.
    ("set_v2_calls", "dfkv_client_set_v2_calls_total", "batch_set_v2 calls that wrote"),
    ("set_v2_pages", "dfkv_client_set_v2_pages_total", "side-pool pages offered to set_v2"),
    ("set_v2_ok_pages", "dfkv_client_set_v2_ok_pages_total", "side-pool pages acked OK on set_v2"),
    ("set_v2_bytes", "dfkv_client_set_v2_bytes_total", "bytes sent on set_v2"),
    ("get_v2_calls", "dfkv_client_get_v2_calls_total", "batch_get_v2 calls"),
    ("get_v2_pages", "dfkv_client_get_v2_pages_total", "side-pool pages requested on get_v2"),
    ("get_v2_hit_pages", "dfkv_client_get_v2_hit_pages_total", "side-pool pages returned as hits on get_v2"),
    ("get_v2_bytes", "dfkv_client_get_v2_bytes_total", "bytes requested on get_v2"),
    ("exist_v2_calls", "dfkv_client_exist_v2_calls_total", "batch_exists_v2 calls"),
    ("exist_v2_probe_pages", "dfkv_client_exist_v2_probe_pages_total", "candidate KV prefix pages probed by exists_v2"),
    ("exist_v2_hit_pages", "dfkv_client_exist_v2_hit_pages_total", "usable hit-prefix pages from exists_v2 (kv_hit_pages)"),
]

# Prometheus Counters are process-global singletons (re-creating the same name
# raises), so build them once at import and reuse across plugin instances/ranks.
_PROM = {}
if _HAVE_PROM:
    for _attr, _name, _help in _SPECS:
        try:
            _PROM[_attr] = _PromCounter(_name, _help, ["tp_rank"])
        except Exception:
            _HAVE_PROM = False
            _PROM = {}
            break


class Metrics:
    """Per-plugin-instance counters; mirrors to process-global Prometheus Counters."""

    def __init__(self, tp_rank):
        self._rank = str(int(tp_rank))
        self._lock = threading.Lock()
        self._c = {attr: 0 for attr, _, _ in _SPECS}
        # histogram observation counts (tests/debug)
        self._obs = {"set": 0, "get": 0, "set_v2": 0, "get_v2": 0}

    def _add(self, **deltas):
        with self._lock:
            for k, v in deltas.items():
                self._c[k] += v
        if _HAVE_PROM:
            for k, v in deltas.items():
                if v:
                    _PROM[k].labels(self._rank).inc(v)

    def _observe(self, op, seconds):
        with self._lock:
            self._obs[op] += 1
        if _HAVE_PROM and op in _HIST:
            _HIST[op].labels(self._rank).observe(seconds)

    def on_set(self, pages, ok_pages, nbytes, seconds=None):
        self._add(set_calls=1, set_pages=pages, set_ok_pages=ok_pages, set_bytes=nbytes)
        if seconds is not None:
            self._observe("set", seconds)

    def on_get(self, pages, hit_pages, nbytes, seconds=None):
        self._add(get_calls=1, get_pages=pages, get_hit_pages=hit_pages, get_bytes=nbytes)
        if seconds is not None:
            self._observe("get", seconds)

    def on_set_v2(self, pages, ok_pages, nbytes, seconds=None):
        self._add(set_v2_calls=1, set_v2_pages=pages, set_v2_ok_pages=ok_pages,
                  set_v2_bytes=nbytes)
        if seconds is not None:
            self._observe("set_v2", seconds)

    def on_get_v2(self, pages, hit_pages, nbytes, seconds=None):
        self._add(get_v2_calls=1, get_v2_pages=pages, get_v2_hit_pages=hit_pages,
                  get_v2_bytes=nbytes)
        if seconds is not None:
            self._observe("get_v2", seconds)

    def on_exists_v2(self, probe_pages, hit_pages):
        self._add(exist_v2_calls=1, exist_v2_probe_pages=probe_pages,
                  exist_v2_hit_pages=hit_pages)

    def snapshot(self):
        with self._lock:
            snap = dict(self._c)
            snap.update({f"{op}_observations": cnt for op, cnt in self._obs.items()})
            return snap


# --- client-side snapshot mirroring (from the C client via dfkv_stats_snapshot) ---
#
# The C++ KVClient accumulates client-observed counters (ops served, IO errors,
# peer-health transitions) that the Python layer can't see. A sleeping daemon
# thread polls the snapshot text and mirrors the aggregate counters onto
# Prometheus Counters by delta, so they aggregate across the per-TP-rank
# processes in SGLang's multiprocess metrics mode. Off the request hot path.

# Counters mirrored by positive delta (monotonic across polls).
_CLIENT_COUNTER_NAMES = [
    "dfkv_client_ops_served_total",
    "dfkv_client_io_errors_total",
    "dfkv_client_unhealthy_skips_total",
    "dfkv_client_peer_marked_bad_total",
    "dfkv_client_peer_recovered_total",
    "dfkv_client_mds_unreachable_polls_total",
]
# Gauges mirrored by set() to the current value each poll: ring size and MDS
# reachability, so a bad/unreachable mds_endpoint is visible on the SCRAPE
# (ring_members==0 / mds_reachable==0), not just in the client log.
_CLIENT_GAUGE_NAMES = [
    "dfkv_client_ring_members",
    "dfkv_client_mds_reachable",
]
# The union the snapshot parser captures (per-peer labeled series stay excluded).
_CLIENT_NAMES = _CLIENT_COUNTER_NAMES + _CLIENT_GAUGE_NAMES

_CLIENT_PROM = {}
_CLIENT_GAUGE_PROM = {}
if _HAVE_PROM:
    for _n in _CLIENT_COUNTER_NAMES:
        try:
            _CLIENT_PROM[_n] = _PromCounter(_n, _n, ["tp_rank"])
        except Exception:
            _CLIENT_PROM = {}
            break
    for _n in _CLIENT_GAUGE_NAMES:
        try:
            # multiprocess_mode is consulted only under PROMETHEUS_MULTIPROC_DIR;
            # 'liveall' keeps each live rank's current value (tp_rank labels them).
            _CLIENT_GAUGE_PROM[_n] = _PromGauge(_n, _n, ["tp_rank"],
                                                multiprocess_mode="liveall")
        except Exception:
            _CLIENT_GAUGE_PROM = {}
            break


class ClientStatsPoller:
    """Polls a Prometheus-text snapshot provider and mirrors the aggregate client
    counters onto Prometheus Counters by delta. One sleeping daemon thread."""

    def __init__(self, get_text, tp_rank, interval_s=10.0):
        self._get_text = get_text
        self._rank = str(int(tp_rank))
        self._interval = float(interval_s)
        self._last = {n: 0 for n in _CLIENT_COUNTER_NAMES}
        self._totals = {n: 0 for n in _CLIENT_COUNTER_NAMES}  # in-process mirror (tests/debug)
        self._gauges = {n: 0 for n in _CLIENT_GAUGE_NAMES}    # last-seen gauge values
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _parse(text):
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
            if name in _CLIENT_NAMES:
                try:
                    out[name] = out.get(name, 0) + int(line[sp + 1:])
                except ValueError:
                    pass
        return out

    def poll_once(self):
        vals = self._parse(self._get_text() or "")
        for n in _CLIENT_COUNTER_NAMES:  # counters: mirror by positive delta
            v = vals.get(n, 0)
            d = v - self._last[n]
            if d > 0:
                self._last[n] = v
                self._totals[n] += d
                if _HAVE_PROM and n in _CLIENT_PROM:
                    _CLIENT_PROM[n].labels(self._rank).inc(d)
        for n in _CLIENT_GAUGE_NAMES:  # gauges: mirror the current value each poll
            v = vals.get(n, 0)
            self._gauges[n] = v
            if _HAVE_PROM and n in _CLIENT_GAUGE_PROM:
                _CLIENT_GAUGE_PROM[n].labels(self._rank).set(v)

    def totals(self):
        return dict(self._totals)

    def gauges(self):
        return dict(self._gauges)

    def _loop(self):
        while not self._stop.wait(self._interval):
            try:
                self.poll_once()
            except Exception:
                pass  # never let a transient snapshot error kill the thread

    def start(self):
        if self._interval <= 0:
            return  # disabled
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
