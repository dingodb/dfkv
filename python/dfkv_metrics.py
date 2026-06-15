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
"""
import threading

try:
    from prometheus_client import Counter as _PromCounter
    _HAVE_PROM = True
except Exception:  # prometheus_client absent -> in-process counters only
    _HAVE_PROM = False

_SPECS = [
    ("set_calls", "dfkv_client_set_calls_total", "batch_set_v1 calls that wrote"),
    ("set_pages", "dfkv_client_set_pages_total", "KV pages offered to set"),
    ("set_ok_pages", "dfkv_client_set_ok_pages_total", "pages acked OK by server"),
    ("set_bytes", "dfkv_client_set_bytes_total", "bytes sent on set"),
    ("get_calls", "dfkv_client_get_calls_total", "batch_get_v1 calls"),
    ("get_pages", "dfkv_client_get_pages_total", "KV pages requested"),
    ("get_hit_pages", "dfkv_client_get_hit_pages_total", "pages returned as hits"),
    ("get_bytes", "dfkv_client_get_bytes_total", "bytes requested on get"),
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

    def _add(self, **deltas):
        with self._lock:
            for k, v in deltas.items():
                self._c[k] += v
        if _HAVE_PROM:
            for k, v in deltas.items():
                if v:
                    _PROM[k].labels(self._rank).inc(v)

    def on_set(self, pages, ok_pages, nbytes):
        self._add(set_calls=1, set_pages=pages, set_ok_pages=ok_pages, set_bytes=nbytes)

    def on_get(self, pages, hit_pages, nbytes):
        self._add(get_calls=1, get_pages=pages, get_hit_pages=hit_pages, get_bytes=nbytes)

    def snapshot(self):
        with self._lock:
            return dict(self._c)
