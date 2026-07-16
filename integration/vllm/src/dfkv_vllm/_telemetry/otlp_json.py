# SPDX-License-Identifier: Apache-2.0
"""Pure-standard-library OTLP/HTTP-JSON metrics exporter — zero third-party deps.

The OTLP spec allows HTTP transport with JSON encoding; the OpenTelemetry
Collector accepts it on ``:4318/v1/metrics`` (``Content-Type: application/json``).
That lets a connector push its fleet metrics using only ``urllib``/``json``/
``threading`` — no ``opentelemetry`` SDK, no ``protobuf``/``grpcio``/``requests`` —
so there's nothing to ``pip install`` inside the inference container, and nothing
that can shadow the engine's own dependencies. The OTel SDK stays an opt-in
alternative (``DFKV_METRICS_EXPORTER=otel``).

It reads the recorder's in-process aggregates each interval and emits:
  dfkv_connector_ops_total{op,status}   Sum (cumulative, monotonic)
  dfkv_connector_keys_total{op}         Sum
  dfkv_connector_bytes_total{op}        Sum
  dfkv_connector_op_seconds{op}         Histogram (buckets -> p99 in Grafana)
  dfkv_connector_op_max_seconds{op}     Gauge (max since last export)
  dfkv_connector_info                   Gauge (=1; identity via resource attrs)
  dfkv_client_peer_latency_{avg,max}_seconds{peer}  Gauge
Identity rides on OTLP *resource* attributes (the Collector's
resource_to_telemetry_conversion turns them into Prometheus labels).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.request


def _attr(key, value):
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": str(value)}
    elif isinstance(value, float):
        v = {"doubleValue": value}
    else:
        v = {"stringValue": str(value)}
    return {"key": key, "value": v}


def metrics_url(endpoint):
    """Resolve OTEL_EXPORTER_OTLP_ENDPOINT to the metrics path. A base endpoint
    gets ``/v1/metrics`` appended; a full ``.../v1/metrics`` is used verbatim."""
    ep = (endpoint or "http://localhost:4318").strip()
    if "://" not in ep:
        ep = "http://" + ep
    ep = ep.rstrip("/")
    if ep.endswith("/v1/metrics"):
        return ep
    return ep + "/v1/metrics"


def _host():
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def build_payload(rec, snap, start_ns, now_ns):
    """Build the OTLP/HTTP-JSON ExportMetricsServiceRequest from a recorder + an
    aggregate snapshot. Pure data; unit-testable without a network."""
    res_attrs = [
        _attr("service.name", "dfkv-%s-connector" % (rec.connector_type or "unknown")),
        _attr("service.namespace", "dfkv"),
        _attr("dfkv.connector_id", rec.connector_id),
        _attr("dfkv.connector_type", rec.connector_type),
        _attr("dfkv.host", _host()),
        _attr("dfkv.pid", os.getpid()),
        _attr("dfkv.tp_rank", int(rec.tp_rank)),
    ]
    if getattr(rec, "version", ""):
        res_attrs.append(_attr("dfkv.version", rec.version))
    if getattr(rec, "native_version", ""):
        res_attrs.append(_attr("dfkv.native_version", rec.native_version))

    t = str(now_ns)
    st = str(start_ns)
    metrics = []

    # --- counters (cumulative, monotonic Sums) ---
    ops_dps = []
    for (op, status), c in snap["ops"].items():
        if c:
            ops_dps.append({"attributes": [_attr("op", op), _attr("status", status)],
                            "startTimeUnixNano": st, "timeUnixNano": t, "asInt": str(c)})
    if ops_dps:
        metrics.append({"name": "dfkv_connector_ops_total",
                        "sum": {"aggregationTemporality": 2, "isMonotonic": True,
                                "dataPoints": ops_dps}})

    for name, key in (("dfkv_connector_keys_total", "keys"),
                      ("dfkv_connector_bytes_total", "bytes")):
        dps = [{"attributes": [_attr("op", op)], "startTimeUnixNano": st,
                "timeUnixNano": t, "asInt": str(v)}
               for op, v in snap[key].items() if v]
        if dps:
            metrics.append({"name": name,
                            "sum": {"aggregationTemporality": 2, "isMonotonic": True,
                                    "dataPoints": dps}})

    # --- op latency histogram (buckets -> p99) ---
    bounds = list(snap["bounds"])
    hist_dps = []
    for op, cnt in snap["dur_cnt"].items():
        if cnt:
            hist_dps.append({
                "attributes": [_attr("op", op)],
                "startTimeUnixNano": st, "timeUnixNano": t,
                "count": str(cnt), "sum": snap["dur_sum"][op],
                "bucketCounts": [str(c) for c in snap["dur_buckets"][op]],
                "explicitBounds": bounds,
            })
    if hist_dps:
        metrics.append({"name": "dfkv_connector_op_seconds",
                        "histogram": {"aggregationTemporality": 2, "dataPoints": hist_dps}})

    # --- gauges: per-op max, info heartbeat, per-peer latency ---
    max_dps = [{"attributes": [_attr("op", op)], "timeUnixNano": t, "asDouble": v}
               for op, v in snap["dur_max"].items() if v]
    if max_dps:
        metrics.append({"name": "dfkv_connector_op_max_seconds",
                        "gauge": {"dataPoints": max_dps}})

    metrics.append({"name": "dfkv_connector_info",
                    "gauge": {"dataPoints": [{"timeUnixNano": t, "asInt": "1"}]}})

    peer = snap["peer"]
    if peer:
        metrics.append({"name": "dfkv_client_peer_latency_avg_seconds",
                        "gauge": {"dataPoints": [
                            {"attributes": [_attr("peer", p)], "timeUnixNano": t, "asDouble": avg}
                            for p, (avg, _mx) in peer.items()]}})
        metrics.append({"name": "dfkv_client_peer_latency_max_seconds",
                        "gauge": {"dataPoints": [
                            {"attributes": [_attr("peer", p)], "timeUnixNano": t, "asDouble": mx}
                            for p, (_avg, mx) in peer.items()]}})

    # --- per-op metrics forwarded from the C++ KVClient snapshot (dfkv_client_op_*),
    #     re-emitted with this connector's identity. One name set across all
    #     connectors since they share the C chokepoint. Values are absolute/
    #     cumulative (counters) so no per-window delta is needed.
    cops = snap.get("client_ops") or {}
    if cops:
        for fld, mname in (("requests", "dfkv_connector_op_requests_total"),
                           ("keys", "dfkv_connector_op_keys_total"),
                           ("hits", "dfkv_connector_op_hits_total"),
                           ("bytes", "dfkv_connector_op_bytes_total")):
            dps = [{"attributes": [_attr("op", op)], "startTimeUnixNano": st,
                    "timeUnixNano": t, "asInt": str(int(d.get(fld, 0)))}
                   for op, d in cops.items() if d.get(fld)]
            if dps:
                metrics.append({"name": mname,
                                "sum": {"aggregationTemporality": 2,
                                        "isMonotonic": True, "dataPoints": dps}})
        max_dps = [{"attributes": [_attr("op", op)], "timeUnixNano": t,
                    "asDouble": float(d.get("max", 0.0))}
                   for op, d in cops.items() if d.get("max")]
        if max_dps:
            metrics.append({"name": "dfkv_connector_op_max_seconds",
                            "gauge": {"dataPoints": max_dps}})
        hist_dps = []
        for op, d in cops.items():
            cnt = d.get("lat_count")
            buckets = d.get("buckets")
            if not cnt or not buckets:
                continue
            # Prometheus cumulative (le) buckets -> OTLP explicitBounds + per-bucket
            # counts. The +Inf bucket = count - last cumulative.
            fin = sorted(((float(le), cum) for le, cum in buckets
                          if le not in ("+Inf", "Inf")), key=lambda x: x[0])
            bounds = [b for b, _ in fin]
            total = float(cnt)
            counts, prev = [], 0.0
            for _, c in fin:
                counts.append(str(int(max(0.0, c - prev))))
                prev = c
            counts.append(str(int(max(0.0, total - prev))))  # +Inf bucket
            hist_dps.append({
                "attributes": [_attr("op", op)],
                "startTimeUnixNano": st, "timeUnixNano": t,
                "count": str(int(total)), "sum": float(d.get("lat_sum", 0.0)),
                "bucketCounts": counts, "explicitBounds": bounds,
            })
        if hist_dps:
            metrics.append({"name": "dfkv_connector_op_latency_seconds",
                            "histogram": {"aggregationTemporality": 2,
                                          "dataPoints": hist_dps}})

    # --- same-host rendezvous counters (dfkv_client_[gpu_]dedup_*), forwarded
    #     under this connector's identity. The top of the hit-rate funnel: how
    #     much of the TP-replicated L3 read load was collapsed to 1x vs fanned
    #     out per rank. Label-free cumulative counters -> per-name OTLP sums.
    dedup = snap.get("client_dedup") or {}
    for cname, val in sorted(dedup.items()):
        metrics.append({"name": cname.replace("dfkv_client_", "dfkv_connector_"),
                        "sum": {"aggregationTemporality": 2, "isMonotonic": True,
                                "dataPoints": [{"startTimeUnixNano": st,
                                                "timeUnixNano": t,
                                                "asInt": str(int(val))}]}})

    return {"resourceMetrics": [{
        "resource": {"attributes": res_attrs},
        "scopeMetrics": [{"scope": {"name": "dfkv.connector"}, "metrics": metrics}],
    }]}


class StdlibExporter:
    """Background thread that POSTs the recorder's aggregates as OTLP/HTTP-JSON
    every ``interval_s``. Off the request hot path. Never routes through a proxy
    (the Collector is an internal endpoint)."""

    def __init__(self, recorder, endpoint, interval_s=10.0):
        self._rec = recorder
        self._url = metrics_url(endpoint)
        self._interval = float(interval_s)
        self._start_ns = time.time_ns()
        self._stop = threading.Event()
        self._thread = None
        # An opener with an empty ProxyHandler ignores HTTP(S)_PROXY for this POST.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def build(self, now_ns=None):
        now_ns = now_ns or time.time_ns()
        snap = self._rec.export_snapshot()
        return build_payload(self._rec, snap, self._start_ns, now_ns)

    def push_once(self):
        body = json.dumps(self.build()).encode("utf-8")
        req = urllib.request.Request(self._url, data=body,
                                     headers={"Content-Type": "application/json"})
        resp = self._opener.open(req, timeout=5)
        try:
            resp.read()
            return resp.status
        finally:
            resp.close()

    def _loop(self):
        while not self._stop.wait(self._interval):
            try:
                self.push_once()
            except Exception:
                pass  # transient collector hiccup must not kill the thread

    def start(self):
        if self._interval <= 0:
            return
        self._thread = threading.Thread(target=self._loop, name="dfkv-otlp-json",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
