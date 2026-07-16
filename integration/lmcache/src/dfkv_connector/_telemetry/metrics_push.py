# SPDX-License-Identifier: Apache-2.0
"""Unified per-connector fleet metrics, pushed to a central OTel Collector.

Why push (not Prometheus pull): connectors live inside many short-lived,
dynamically-scheduled inference processes (vLLM / SGLang / LMCache), so a central
Prometheus would have to discover and reach every one of them. Instead each
connector *self-reports* over OTLP to a central Collector — matching the
"report to center" model — which fans out to Prometheus + Grafana.

What it powers (the global cluster dashboard):
  * how many connector instances, by type (hicache/lmcache/vllm), by connector_id
  * put/get request count, key count, bytes, avg latency (sum/count), p99, max
  * success vs failure counts and ratio
  * drill into one connector_id

Design (mirrors dfkv_access_log / dfkv_metrics):
  * Off by default => ``op()`` returns a frozen no-op singleton, ~tens of ns,
    OTel is never imported. The master switch is DFKV_METRICS_ENABLED (or the
    umbrella DFKV_TELEMETRY_ENABLED).
  * When on, cheap in-process counters are always kept (queryable via
    ``snapshot()`` for tests/debug) and mirrored to OTel instruments that a
    PeriodicExportingMetricReader pushes to ``OTEL_EXPORTER_OTLP_ENDPOINT``.
  * If the OTel SDK is not installed, the recorder logs once and degrades to
    in-process-only (never crashes the connector).

Identity (connector_id, connector_type, host, pid, tp_rank) is attached as OTel
*resource* attributes; the Collector's prometheus exporter is configured with
``resource_to_telemetry_conversion`` so they become queryable Prometheus labels.
``dfkv_connector_info`` (=1) is emitted every interval so an idle connector is
still visible / countable and ``count by(connector_type)`` works.

Metric model (op is an attribute, so put/get share one set of names):
  dfkv_connector_ops_total{op,status}   request count, status in {ok,fail}
  dfkv_connector_keys_total{op}         keys touched
  dfkv_connector_bytes_total{op}        bytes moved
  dfkv_connector_op_seconds{op}         duration histogram (avg=sum/count, p99)
  dfkv_connector_op_max_seconds{op}     max duration since last export (gauge)
  dfkv_connector_info                   =1 heartbeat (identity via resource attrs)
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from typing import Any, Optional

from . import config

# Latency histogram bucket boundaries (seconds); same family the SGLang plugin
# uses in dfkv_metrics so dashboards line up.
_HIST_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025,
                 0.05, 0.1, 0.25, 0.5, 1.0)

_OPS = ("put", "get", "exist")


# ---------------------------------------------------------------------------
# No-op (disabled) path — what op() returns when metrics are off. Zero cost.
# ---------------------------------------------------------------------------

class _NoopOp:
    __slots__ = ()
    status: Any = None

    def __enter__(self) -> "_NoopOp":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __bool__(self) -> bool:
        # Falsy so callers can `if _m: _m.bytes = sum(...)` and skip evaluating
        # the (potentially O(batch)) metric args entirely when metrics are off.
        return False

    def __setattr__(self, name: str, value: Any) -> None:
        pass  # allow `m.status = "fail"` / `m.keys = n` without raising


_NOOP_OP = _NoopOp()


class _NoopRecorder:
    enabled = False

    def op(self, op_name, num_keys=0, num_bytes=0):
        return _NOOP_OP

    def record(self, op_name, keys=0, nbytes=0, seconds=0.0, status="ok"):
        pass

    def update_peer_latency(self, peer_stats):
        pass

    def update_client_ops(self, client_ops):
        pass

    def update_client_dedup(self, dedup):
        pass

    def peer_latency_snapshot(self):
        return {}

    def info(self):
        pass

    def snapshot(self):
        return {}

    def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# Real (enabled) op context manager — times the op, records on exit.
# ---------------------------------------------------------------------------

class _Op:
    """Per-operation context manager. Set ``.status`` ('ok'/'fail') and/or adjust
    ``.keys``/``.bytes`` before exit; otherwise status is inferred from whether
    the block raised, and keys/bytes default to the constructor values."""

    __slots__ = ("_rec", "_name", "keys", "bytes", "_start", "status")

    def __init__(self, rec: "_Recorder", name: str, num_keys: int, num_bytes: int):
        self._rec = rec
        self._name = name
        self.keys = int(num_keys)
        self.bytes = int(num_bytes)
        self.status: Optional[str] = None
        self._start = 0.0

    def __enter__(self) -> "_Op":
        self._start = time.perf_counter()
        return self

    def __bool__(self) -> bool:
        return True  # truthy: the `if _m:` guard runs the metric-arg eval

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.perf_counter() - self._start
        status = self.status or ("fail" if exc_type is not None else "ok")
        self._rec._record(self._name, self.keys, self.bytes, duration, status)
        return False  # never swallow the caller's exception


# ---------------------------------------------------------------------------
# Real recorder — in-process counters always, OTel mirror when available.
# ---------------------------------------------------------------------------

class _Recorder:
    enabled = True

    def __init__(self, connector_type: str, connector_id: str, tp_rank: int,
                 model: str, endpoint: str, interval_ms: int, protocol: str = "",
                 readers=None, version: str = "", native_version: str = "",
                 exporter: str = "stdlib"):
        self.connector_type = connector_type
        self.connector_id = connector_id
        self.tp_rank = int(tp_rank)
        self.model = model
        # version: the connector package version (pip dist, e.g. dfkv-vllm).
        # native_version: the linked libdfkv.so version (C dfkv_version()).
        # Both ride on the resource so the dashboard can spot version skew.
        self.version = version or ""
        self.native_version = native_version or ""
        self._lock = threading.Lock()
        # In-process mirror; the stdlib exporter reads these to build OTLP-JSON.
        self._ops = {(o, s): 0 for o in _OPS for s in ("ok", "fail")}
        self._keys = {o: 0 for o in _OPS}
        self._bytes = {o: 0 for o in _OPS}
        self._dur_sum = {o: 0.0 for o in _OPS}
        self._dur_cnt = {o: 0 for o in _OPS}
        self._dur_max = {o: 0.0 for o in _OPS}  # since last export; reset on read
        # Per-op latency bucket counts (len = len(_HIST_BUCKETS)+1) for the OTLP
        # histogram (-> p99 in Grafana). One cheap branchy increment per op.
        self._nb = len(_HIST_BUCKETS)
        self._dur_buckets = {o: [0] * (self._nb + 1) for o in _OPS}
        # Per-cache-node latency (peer -> (avg_seconds, max_seconds)), fed by a
        # PeerLatencyPoller reading the C client's snapshot.
        self._peer = {}
        # Per-op (put/get/exist) metrics, sourced from the C KVClient snapshot
        # (dfkv_client_op_*) by the same poller. The C++ client is the convergent
        # chokepoint all connectors funnel through, so the op accounting lives
        # there once; here we just forward the absolute values over OTLP with this
        # connector's identity. {op: {requests,keys,hits,bytes,max,lat_sum,
        # lat_count, buckets:[(le_str, cum_count)]}}.
        self._client_ops = {}
        self._client_dedup = {}
        self._otel = None
        self._provider = None
        self._stdlib = None
        # Default exporter = the pure-stdlib OTLP/HTTP-JSON pusher (zero third-party
        # deps, nothing to pip-install in the container, no dependency shadowing).
        # `readers` (test seam) or exporter="otel" selects the OpenTelemetry SDK;
        # if the SDK is requested but absent, fall back to the stdlib pusher.
        if readers is not None or str(exporter).lower() == config.EXPORTER_OTEL:
            self._setup_otel(endpoint, interval_ms, protocol, readers)
            if self._otel is None and readers is None:
                self._setup_stdlib(endpoint, interval_ms)
        else:
            self._setup_stdlib(endpoint, interval_ms)

    def _setup_stdlib(self, endpoint: str, interval_ms: int) -> None:
        from . import otlp_json
        self._stdlib = otlp_json.StdlibExporter(
            self, endpoint, max(1000, int(interval_ms)) / 1000.0)
        self._stdlib.start()

    # -- OTel wiring (guarded; degrades to in-process-only if SDK absent) --
    # `readers` is a test seam: when given (e.g. an InMemoryMetricReader), it is
    # used as-is and the OTLP exporter is not constructed. Production passes None.
    def _setup_otel(self, endpoint: str, interval_ms: int, protocol: str,
                    readers=None) -> None:
        try:
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.metrics.view import (
                ExplicitBucketHistogramAggregation, View)
            from opentelemetry.sdk.resources import Resource

            res_attrs = {
                "service.name": "dfkv-{}-connector".format(self.connector_type or "unknown"),
                "service.namespace": "dfkv",
                "dfkv.connector_id": self.connector_id,
                "dfkv.connector_type": self.connector_type,
                "dfkv.host": _hostname(),
                "dfkv.pid": os.getpid(),
                "dfkv.tp_rank": self.tp_rank,
            }
            # Version labels (omitted when unknown so we never emit empty labels):
            # dfkv_version = connector package, dfkv_native_version = libdfkv.so.
            if self.version:
                res_attrs["dfkv.version"] = self.version
            if self.native_version:
                res_attrs["dfkv.native_version"] = self.native_version
            res = Resource.create(res_attrs)
            if readers is None:
                proto = (protocol or "grpc").lower()
                if "http" in proto:
                    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                        OTLPMetricExporter)
                else:
                    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                        OTLPMetricExporter)
                exporter = OTLPMetricExporter(endpoint=endpoint) if endpoint else OTLPMetricExporter()
                readers = [PeriodicExportingMetricReader(
                    exporter, export_interval_millis=max(1000, int(interval_ms)))]
            view = View(instrument_name="dfkv_connector_op_seconds",
                        aggregation=ExplicitBucketHistogramAggregation(_HIST_BUCKETS))
            self._provider = MeterProvider(metric_readers=list(readers), resource=res, views=[view])
            meter = self._provider.get_meter("dfkv.connector")
            self._otel = _OtelInstruments(meter, self)
        except Exception as exc:  # SDK missing or misconfigured -> in-process only
            sys.stderr.write(
                "[dfkv.telemetry] OTel metrics push disabled ({}); "
                "in-process counters still kept. Install the 'otel' extra to "
                "enable push.\n".format(exc))
            self._otel = None

    def op(self, op_name: str, num_keys: int = 0, num_bytes: int = 0) -> _Op:
        return _Op(self, op_name, num_keys, num_bytes)

    def record(self, op_name: str, keys: int = 0, nbytes: int = 0,
               seconds: float = 0.0, status: str = "ok") -> None:
        """Direct record for sites that already measured the duration (e.g. the
        SGLang plugin times the C call itself)."""
        self._record(op_name, keys, nbytes, seconds, status)

    def info(self) -> None:
        # Identity heartbeat is driven by the observable gauge callback (OTel) or
        # is a no-op for the in-process-only path. Kept for API symmetry.
        pass

    def _ensure_op(self, op_name: str) -> None:
        """Register the per-op accumulators the first time an op is seen. The
        seed set (_OPS) is created up front so common series are always present;
        any other op (e.g. put_indexer/get_indexer/exist_v2) is added on demand.
        Caller holds self._lock (or is __init__)."""
        if op_name not in self._dur_sum:
            self._keys[op_name] = 0
            self._bytes[op_name] = 0
            self._dur_sum[op_name] = 0.0
            self._dur_cnt[op_name] = 0
            self._dur_max[op_name] = 0.0
            self._dur_buckets[op_name] = [0] * (self._nb + 1)

    def op_names(self) -> list:
        with self._lock:
            return list(self._dur_sum.keys())

    def _record(self, op_name: str, keys: int, nbytes: int,
                seconds: float, status: str) -> None:
        if seconds < 0:
            seconds = 0.0
        with self._lock:
            self._ensure_op(op_name)
            self._ops[(op_name, status)] = self._ops.get((op_name, status), 0) + 1
            self._keys[op_name] += int(keys)
            self._bytes[op_name] += int(nbytes)
            self._dur_sum[op_name] += seconds
            self._dur_cnt[op_name] += 1
            if seconds > self._dur_max[op_name]:
                self._dur_max[op_name] = seconds
            bi = self._nb
            for i in range(self._nb):
                if seconds <= _HIST_BUCKETS[i]:
                    bi = i
                    break
            self._dur_buckets[op_name][bi] += 1
        if self._otel is not None:
            self._otel.record(op_name, keys, nbytes, seconds, status)

    def take_max(self, op_name: str) -> float:
        """Read-and-reset the max duration since the last export (for the gauge)."""
        with self._lock:
            v = self._dur_max.get(op_name, 0.0)
            self._dur_max[op_name] = 0.0
            return v

    def update_peer_latency(self, peer_stats: dict) -> None:
        """peer_stats: {peer: (avg_seconds, max_seconds)} from the snapshot poller."""
        with self._lock:
            self._peer = dict(peer_stats)

    def update_client_ops(self, client_ops: dict) -> None:
        """Absolute per-op metrics parsed from the C KVClient snapshot
        (dfkv_client_op_*). Forwarded as-is (cumulative) by the OTLP exporter."""
        with self._lock:
            self._client_ops = dict(client_ops)

    def update_client_dedup(self, dedup: dict) -> None:
        """Absolute same-host rendezvous counters (dfkv_client_[gpu_]dedup_*)."""
        with self._lock:
            self._client_dedup = dict(dedup)

    def peer_latency_snapshot(self) -> dict:
        with self._lock:
            return dict(self._peer)

    def export_snapshot(self) -> dict:
        """Consistent aggregate snapshot for the stdlib OTLP/JSON exporter. Reads-
        and-resets the per-op max (matching the OTel observable-gauge semantics)."""
        with self._lock:
            snap = {
                "ops": dict(self._ops),
                "keys": dict(self._keys),
                "bytes": dict(self._bytes),
                "dur_sum": dict(self._dur_sum),
                "dur_cnt": dict(self._dur_cnt),
                "dur_buckets": {o: list(b) for o, b in self._dur_buckets.items()},
                "dur_max": dict(self._dur_max),
                "bounds": list(_HIST_BUCKETS),
                "peer": dict(self._peer),
                "client_ops": {o: dict(d) for o, d in self._client_ops.items()},
                "client_dedup": dict(self._client_dedup),
            }
            for o in list(self._dur_max):
                self._dur_max[o] = 0.0  # read-and-reset
        return snap

    def snapshot(self) -> dict:
        with self._lock:
            snap = {
                "connector_id": self.connector_id,
                "connector_type": self.connector_type,
                "ops": {"{}/{}".format(o, s): c for (o, s), c in self._ops.items()},
                "keys": dict(self._keys),
                "bytes": dict(self._bytes),
                "dur_sum": dict(self._dur_sum),
                "dur_cnt": dict(self._dur_cnt),
                "dur_max": dict(self._dur_max),
            }
        return snap

    def shutdown(self) -> None:
        if self._stdlib is not None:
            try:
                self._stdlib.stop()
            except Exception:
                pass
            self._stdlib = None
        if self._provider is not None:
            try:
                self._provider.shutdown()
            except Exception:
                pass
            self._provider = None


class _OtelInstruments:
    """Holds the OTel instruments and mirrors a recorded op onto them. Only
    constructed when the OTel SDK imported successfully."""

    def __init__(self, meter, rec: "_Recorder"):
        self._rec = rec
        self.ops = meter.create_counter(
            "dfkv_connector_ops_total", unit="1",
            description="connector ops, by op and status")
        self.keys = meter.create_counter(
            "dfkv_connector_keys_total", unit="1",
            description="KV keys touched, by op")
        self.bytes = meter.create_counter(
            "dfkv_connector_bytes_total", unit="By",
            description="bytes moved, by op")
        self.seconds = meter.create_histogram(
            "dfkv_connector_op_seconds", unit="s",
            description="op duration seconds, by op")
        meter.create_observable_gauge(
            "dfkv_connector_op_max_seconds", callbacks=[self._observe_max],
            unit="s", description="max op duration since last export, by op")
        meter.create_observable_gauge(
            "dfkv_client_peer_latency_avg_seconds", callbacks=[self._observe_peer_avg],
            unit="s", description="avg latency to each cache node, by peer")
        meter.create_observable_gauge(
            "dfkv_client_peer_latency_max_seconds", callbacks=[self._observe_peer_max],
            unit="s", description="max latency to each cache node, by peer")
        # unit="" (not "1"): a unit of "1" makes the OTel->Prometheus convention
        # rename the gauge to dfkv_connector_info_ratio. Empty unit keeps the name.
        meter.create_observable_gauge(
            "dfkv_connector_info", callbacks=[self._observe_info], unit="",
            description="connector liveness/identity heartbeat (=1)")

    def record(self, op_name, keys, nbytes, seconds, status):
        attrs = {"op": op_name}
        self.ops.add(1, {"op": op_name, "status": status})
        if keys:
            self.keys.add(int(keys), attrs)
        if nbytes:
            self.bytes.add(int(nbytes), attrs)
        self.seconds.record(seconds, attrs)

    def _observe_max(self, options):
        from opentelemetry.metrics import Observation
        return [Observation(self._rec.take_max(o), {"op": o})
                for o in self._rec.op_names()]

    def _observe_peer_avg(self, options):
        from opentelemetry.metrics import Observation
        return [Observation(avg, {"peer": p})
                for p, (avg, _mx) in self._rec.peer_latency_snapshot().items()]

    def _observe_peer_max(self, options):
        from opentelemetry.metrics import Observation
        return [Observation(mx, {"peer": p})
                for p, (_avg, mx) in self._rec.peer_latency_snapshot().items()]

    def _observe_info(self, options):
        from opentelemetry.metrics import Observation
        return [Observation(1)]


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Module-level facade (mirrors dfkv_access_log's module API).
# ---------------------------------------------------------------------------

_recorder = _NoopRecorder()  # type: ignore[assignment]
_configured = False
_cfg_lock = threading.Lock()


def configure(cfg: Optional[dict] = None, connector_type: str = "",
              tp_rank: int = 0, model: str = "", version: str = "",
              native_version: str = "", _readers=None) -> None:
    """Initialize the push-metrics recorder (idempotent; first call wins).

    Call once from each connector's ``__init__``. When metrics are disabled this
    leaves a frozen no-op recorder in place (zero cost). ``version`` (the
    connector package version) and ``native_version`` (the linked libdfkv.so,
    from the C ``dfkv_version()``) become the ``dfkv_version`` /
    ``dfkv_native_version`` labels. ``_readers`` is a test-only seam to inject
    metric readers in place of the OTLP exporter."""
    global _recorder, _configured
    with _cfg_lock:
        if _configured:
            return
        _configured = True
        cfg = cfg or {}
        if not config.metrics_enabled(cfg):
            _recorder = _NoopRecorder()
            return
        connector_id = config.resolve_connector_id(cfg, tp_rank)
        endpoint = str(config.resolve(cfg, "otlp_endpoint", config.ENV_OTLP_ENDPOINT, "")).strip()
        protocol = str(config.resolve(cfg, "otlp_protocol", config.ENV_OTLP_PROTOCOL, "")).strip()
        try:
            interval_ms = int(config.resolve(
                cfg, "metrics_export_interval_ms", config.ENV_EXPORT_INTERVAL_MS, 10000))
        except (TypeError, ValueError):
            interval_ms = 10000
        exporter = str(config.resolve(
            cfg, "metrics_exporter", config.ENV_METRICS_EXPORTER,
            config.EXPORTER_STDLIB)).lower()
        _recorder = _Recorder(connector_type, connector_id, tp_rank, model,
                              endpoint, interval_ms, protocol, readers=_readers,
                              version=version, native_version=native_version,
                              exporter=exporter)


def op(op_name: str, num_keys: int = 0, num_bytes: int = 0):
    """Cheap when disabled (frozen singleton); times + records when enabled."""
    return _recorder.op(op_name, num_keys, num_bytes)


def record(op_name: str, keys: int = 0, nbytes: int = 0,
           seconds: float = 0.0, status: str = "ok") -> None:
    """Record a finished op whose duration was already measured by the caller."""
    _recorder.record(op_name, keys, nbytes, seconds, status)


# ---------------------------------------------------------------------------
# Per-cache-node latency: parse the C client snapshot, push avg/max per peer.
# The C++ KVClient active prober (DFKV_PROBE_INTERVAL_MS>0) fills
# dfkv_client_peer_latency_* in dfkv_stats_snapshot; this poller re-emits a
# windowed avg + lifetime max as OTLP gauges for the dashboard.
# ---------------------------------------------------------------------------

_PEER_NAMES = ("dfkv_client_peer_latency_seconds_sum",
               "dfkv_client_peer_latency_seconds_count",
               "dfkv_client_peer_latency_max_seconds")


def _extract_label(labels: str, key: str):
    needle = key + '="'
    i = labels.find(needle)
    if i == -1:
        return None
    j = labels.find('"', i + len(needle))
    return labels[i + len(needle):j] if j != -1 else None


def parse_peer_latency(text: str) -> dict:
    """Parse dfkv_client_peer_latency_* lines -> {peer: {'sum','count','max'}}."""
    out = {}
    for line in (text or "").splitlines():
        if not line or line[0] == "#":
            continue
        sp = line.rfind(" ")
        if sp <= 0:
            continue
        name_labels, val = line[:sp], line[sp + 1:]
        brace = name_labels.find("{")
        if brace == -1:
            continue
        name = name_labels[:brace]
        if name not in _PEER_NAMES:
            continue
        rb = name_labels.rfind("}")
        labels = name_labels[brace + 1:rb] if rb != -1 else ""
        peer = _extract_label(labels, "peer")
        if peer is None:
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        d = out.setdefault(peer, {})
        if name.endswith("_sum"):
            d["sum"] = v
        elif name.endswith("_count"):
            d["count"] = v
        else:
            d["max"] = v
    return out


# Scalar dfkv_client_op_* families (cumulative counters + a max gauge) keyed by
# the field name we store them under for the OTLP forwarder.
_OP_SCALARS = {
    "dfkv_client_op_requests_total": "requests",
    "dfkv_client_op_keys_total": "keys",
    "dfkv_client_op_hits_total": "hits",
    "dfkv_client_op_bytes_total": "bytes",
    "dfkv_client_op_max_seconds": "max",
    "dfkv_client_op_latency_seconds_sum": "lat_sum",
    "dfkv_client_op_latency_seconds_count": "lat_count",
}


def parse_client_ops(text: str) -> dict:
    """Parse dfkv_client_op_* lines from the C KVClient snapshot into
    {op: {requests,keys,hits,bytes,max,lat_sum,lat_count, buckets:[(le_str,cum)]}}.
    The op accounting is done once in C++ (the convergent chokepoint); this just
    forwards it. buckets keep the cumulative (le) counts for OTLP conversion."""
    out = {}
    for line in (text or "").splitlines():
        if not line or line[0] == "#":
            continue
        sp = line.rfind(" ")
        if sp <= 0:
            continue
        name_labels, val = line[:sp], line[sp + 1:]
        brace = name_labels.find("{")
        if brace == -1:
            continue
        name = name_labels[:brace]
        rb = name_labels.rfind("}")
        labels = name_labels[brace + 1:rb] if rb != -1 else ""
        op = _extract_label(labels, "op")
        if op is None:
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        d = out.setdefault(op, {})
        field = _OP_SCALARS.get(name)
        if field is not None:
            d[field] = v
        elif name == "dfkv_client_op_latency_seconds_bucket":
            le = _extract_label(labels, "le")
            if le is not None:
                d.setdefault("buckets", []).append((le, v))
    return out


_DEDUP_COUNTERS = (
    "dfkv_client_dedup_hits_total",
    "dfkv_client_dedup_fetches_total",
    "dfkv_client_dedup_wait_hits_total",
    "dfkv_client_dedup_wait_timeouts_total",
    "dfkv_client_gpu_dedup_hits_total",
    "dfkv_client_gpu_dedup_fetches_total",
    "dfkv_client_gpu_dedup_wait_hits_total",
    "dfkv_client_gpu_dedup_wait_timeouts_total",
)


def parse_client_dedup(text: str) -> dict:
    """Extract the same-host rendezvous counters (host + GPU flavors) from the C
    client snapshot. These have NO op= label, so parse_client_ops skips them —
    yet they ARE the "was the L3 read collapsed to 1x or fanned out 8x" signal,
    the top of the hit-rate funnel. {counter_name: value}, present ones only."""
    out = {}
    wanted = set(_DEDUP_COUNTERS)
    for line in (text or "").splitlines():
        if not line or line[0] == "#":
            continue
        sp = line.rfind(" ")
        if sp <= 0:
            continue
        name = line[:sp]
        if "{" in name:
            continue  # dedup counters are label-free
        if name in wanted:
            try:
                out[name] = float(line[sp + 1:])
            except ValueError:
                pass
    return out


class PeerLatencyPoller:
    """Sleeping daemon thread: polls the C client's metrics snapshot, computes a
    windowed avg (delta-sum / delta-count) + lifetime max per peer, AND forwards
    the per-op metrics (dfkv_client_op_*) so all connectors report their full
    request mix from the one C++ chokepoint. Off the request hot path."""

    def __init__(self, get_text, recorder, interval_s=10.0):
        self._get_text = get_text
        self._rec = recorder
        self._interval = float(interval_s)
        self._last = {}  # peer -> (sum, count) at the previous poll
        self._stop = threading.Event()
        self._thread = None

    def poll_once(self):
        text = self._get_text() or ""
        # Forward the per-op metrics (computed once in C++) over OTLP, with this
        # connector's identity. Absolute/cumulative — no windowing needed.
        self._rec.update_client_ops(parse_client_ops(text))
        self._rec.update_client_dedup(parse_client_dedup(text))
        stats = parse_peer_latency(text)
        result = {}
        for peer, d in stats.items():
            s = d.get("sum", 0.0)
            c = d.get("count", 0.0)
            mx = d.get("max", 0.0)
            ls, lc = self._last.get(peer, (0.0, 0.0))
            ds, dc = s - ls, c - lc
            avg = (ds / dc) if dc > 0 else (s / c if c > 0 else 0.0)
            self._last[peer] = (s, c)
            result[peer] = (avg, mx)
        self._rec.update_peer_latency(result)
        return result

    def _loop(self):
        while not self._stop.wait(self._interval):
            try:
                self.poll_once()
            except Exception:
                pass  # never let a transient snapshot error kill the thread

    def start(self):
        if self._interval <= 0:
            return
        try:
            self.poll_once()
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop, name="dfkv-peer-latency",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


_peer_poller = None


def start_peer_latency_poller(get_text, interval_s=10.0):
    """Start the per-peer latency poller bound to the active recorder. No-op when
    metrics are disabled. ``get_text()`` returns the C client snapshot text (e.g.
    via dfkv_stats_snapshot). Returns the poller (or None)."""
    global _peer_poller
    if not _recorder.enabled:
        return None
    _peer_poller = PeerLatencyPoller(get_text, _recorder, interval_s)
    _peer_poller.start()
    return _peer_poller


def is_enabled() -> bool:
    return _recorder.enabled


def snapshot() -> dict:
    return _recorder.snapshot()


def shutdown() -> None:
    global _peer_poller
    if _peer_poller is not None:
        _peer_poller.stop()
        _peer_poller = None
    _recorder.shutdown()


def _reset_for_test() -> None:
    """Test seam: drop config + recorder so a test can reconfigure from env."""
    global _recorder, _configured, _peer_poller
    with _cfg_lock:
        if _peer_poller is not None:
            _peer_poller.stop()
            _peer_poller = None
        try:
            _recorder.shutdown()
        except Exception:
            pass
        _recorder = _NoopRecorder()
        _configured = False
