"""Unit tests for the shared dfkv_telemetry push-metrics layer.

GPU/OTel/Prometheus-free: exercises the in-process counter mirror and the
zero-cost-when-off path. (End-to-end OTLP push is validated in the M3 backend
stack, which needs the OTel SDK + a running Collector.)
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))  # <repo>/python

import dfkv_telemetry  # noqa: E402
from dfkv_telemetry import config, metrics  # noqa: E402

_ENV_KEYS = [
    config.ENV_METRICS_ENABLED, config.ENV_TELEMETRY_ENABLED,
    config.ENV_CONNECTOR_ID, config.ENV_OTLP_ENDPOINT,
    config.ENV_EXPORT_INTERVAL_MS,
]


class TelemetryTestBase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        metrics._reset_for_test()

    def tearDown(self):
        metrics._reset_for_test()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class DisabledByDefaultTest(TelemetryTestBase):
    def test_off_by_default_is_noop(self):
        metrics.configure({}, connector_type=config.TYPE_VLLM, tp_rank=0)
        self.assertFalse(metrics.is_enabled())
        # op() returns the frozen no-op; setting attrs / raising is harmless.
        with metrics.op("put", num_keys=3, num_bytes=4096) as m:
            m.status = "fail"
            m.keys = 99
        self.assertEqual(metrics.snapshot(), {})

    def test_noop_does_not_swallow_exceptions(self):
        metrics.configure({}, connector_type=config.TYPE_LMCACHE)
        with self.assertRaises(ValueError):
            with metrics.op("get", num_keys=1):
                raise ValueError("boom")


class ConnectorIdTest(TelemetryTestBase):
    def test_default_id_is_host_pid_rank(self):
        cid = config.resolve_connector_id({}, tp_rank=2)
        self.assertEqual(cid.rsplit(":", 1)[1], "2")
        self.assertEqual(cid, "{}:{}:2".format(__import__("socket").gethostname(), os.getpid()))

    def test_env_overrides_default(self):
        os.environ[config.ENV_CONNECTOR_ID] = "node-A-rank0"
        self.assertEqual(config.resolve_connector_id({}, tp_rank=7), "node-A-rank0")

    def test_extra_config_wins_over_env(self):
        os.environ[config.ENV_CONNECTOR_ID] = "from-env"
        self.assertEqual(
            config.resolve_connector_id({"connector_id": "from-cfg"}, tp_rank=0),
            "from-cfg")


class VersionLabelTest(TelemetryTestBase):
    """version / native_version ride on the recorder (and, with OTel, the
    resource) so the dashboard can spot connector vs libdfkv.so version skew."""

    def test_dist_version_unknown_is_empty_not_raises(self):
        # A distribution that surely isn't installed -> "" (never raises).
        self.assertEqual(config.dist_version("dfkv-nonexistent-xyz"), "")
        self.assertEqual(config.dist_version(""), "")

    def test_recorder_captures_versions_when_enabled(self):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM, tp_rank=0,
                          version="2.0.0", native_version="1.6.3")
        self.assertTrue(metrics.is_enabled())
        self.assertEqual(metrics._recorder.version, "2.0.0")
        self.assertEqual(metrics._recorder.native_version, "1.6.3")

    def test_versions_default_empty(self):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        self.assertEqual(metrics._recorder.version, "")
        self.assertEqual(metrics._recorder.native_version, "")


class EnabledInProcessTest(TelemetryTestBase):
    """Enabled via env, OTel absent in this env => in-process-only counters."""

    def _enable(self, **cfg):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure(cfg, connector_type=config.TYPE_HICACHE, tp_rank=1,
                          model="m")

    def test_records_requests_keys_bytes_status(self):
        self._enable()
        self.assertTrue(metrics.is_enabled())
        with metrics.op("put", num_keys=3, num_bytes=4096) as m:
            m.status = "ok"
        with metrics.op("get", num_keys=5, num_bytes=8192):  # inferred ok
            pass
        snap = metrics.snapshot()
        self.assertEqual(snap["ops"]["put/ok"], 1)
        self.assertEqual(snap["ops"]["get/ok"], 1)
        self.assertEqual(snap["keys"]["put"], 3)
        self.assertEqual(snap["keys"]["get"], 5)
        self.assertEqual(snap["bytes"]["put"], 4096)
        self.assertEqual(snap["bytes"]["get"], 8192)
        self.assertEqual(snap["connector_type"], config.TYPE_HICACHE)

    def test_exception_marks_fail(self):
        self._enable()
        try:
            with metrics.op("put", num_keys=2, num_bytes=10):
                raise RuntimeError("io error")
        except RuntimeError:
            pass
        snap = metrics.snapshot()
        self.assertEqual(snap["ops"]["put/fail"], 1)
        self.assertEqual(snap["ops"].get("put/ok", 0), 0)

    def test_duration_sum_count_and_max_reset(self):
        self._enable()
        rec = metrics._recorder  # the real recorder
        with metrics.op("get", num_keys=1, num_bytes=1):
            pass
        snap = metrics.snapshot()
        self.assertEqual(snap["dur_cnt"]["get"], 1)
        self.assertGreaterEqual(snap["dur_sum"]["get"], 0.0)
        # take_max reads-and-resets (the observable gauge relies on this).
        first = rec.take_max("get")
        self.assertGreaterEqual(first, 0.0)
        self.assertEqual(rec.take_max("get"), 0.0)

    def test_umbrella_switch_enables(self):
        os.environ[config.ENV_TELEMETRY_ENABLED] = "true"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        self.assertTrue(metrics.is_enabled())

    def test_configure_is_idempotent(self):
        self._enable()
        # second configure (even disabling) must not replace the live recorder
        metrics.configure({"metrics": False}, connector_type=config.TYPE_VLLM)
        self.assertTrue(metrics.is_enabled())


class LiteralZeroOffTest(TelemetryTestBase):
    """When off, `if _m:` must be falsy so the connector never evaluates the
    (potentially O(batch)) metric args — literal zero overhead."""

    def test_disabled_op_is_falsy_and_skips_arg_eval(self):
        metrics.configure({}, connector_type=config.TYPE_VLLM)  # off by default
        calls = []

        def expensive():
            calls.append(1)
            return 4096

        with metrics.op("put", num_keys=3) as m:
            self.assertFalse(bool(m))
            if m:
                m.bytes = expensive()
        self.assertEqual(calls, [])           # never evaluated while off
        self.assertEqual(metrics.snapshot(), {})

    def test_enabled_op_is_truthy_and_records_guarded_bytes(self):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        calls = []

        def expensive():
            calls.append(1)
            return 4096

        with metrics.op("put", num_keys=3) as m:
            self.assertTrue(bool(m))
            if m:
                m.bytes = expensive()
        self.assertEqual(calls, [1])
        self.assertEqual(metrics.snapshot()["bytes"]["put"], 4096)


_SAMPLE_SNAPSHOT = """# HELP dfkv_client_peer_latency_seconds ...
# TYPE dfkv_client_peer_latency_seconds histogram
dfkv_client_peer_latency_seconds_bucket{le="0.0005",peer="10.0.0.1:9000"} 2
dfkv_client_peer_latency_seconds_sum{peer="10.0.0.1:9000"} 0.0030
dfkv_client_peer_latency_seconds_count{peer="10.0.0.1:9000"} 3
dfkv_client_peer_latency_max_seconds{peer="10.0.0.1:9000"} 0.0015
dfkv_client_peer_latency_seconds_sum{peer="10.0.0.2:9000"} 0.0100
dfkv_client_peer_latency_seconds_count{peer="10.0.0.2:9000"} 2
dfkv_client_peer_latency_max_seconds{peer="10.0.0.2:9000"} 0.0060
"""


class PeerLatencyTest(TelemetryTestBase):
    def test_parse_peer_latency(self):
        from dfkv_telemetry.metrics_push import parse_peer_latency
        d = parse_peer_latency(_SAMPLE_SNAPSHOT)
        self.assertEqual(d["10.0.0.1:9000"], {"sum": 0.003, "count": 3.0, "max": 0.0015})
        self.assertAlmostEqual(d["10.0.0.2:9000"]["max"], 0.006)
        self.assertEqual(set(d), {"10.0.0.1:9000", "10.0.0.2:9000"})

    def test_poller_computes_avg_and_max(self):
        from dfkv_telemetry.metrics_push import PeerLatencyPoller
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        rec = metrics._recorder
        poller = PeerLatencyPoller(lambda: _SAMPLE_SNAPSHOT, rec, interval_s=0)
        res = poller.poll_once()
        self.assertAlmostEqual(res["10.0.0.1:9000"][0], 0.001)   # avg 0.003/3
        self.assertAlmostEqual(res["10.0.0.1:9000"][1], 0.0015)  # max
        self.assertAlmostEqual(res["10.0.0.2:9000"][0], 0.005)   # avg 0.01/2
        # recorder state is updated (observable gauges read it)
        self.assertIn("10.0.0.1:9000", rec.peer_latency_snapshot())

    def test_poller_windowed_avg_uses_deltas(self):
        from dfkv_telemetry.metrics_push import PeerLatencyPoller
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        texts = iter([
            "dfkv_client_peer_latency_seconds_sum{peer=\"p\"} 0.002\n"
            "dfkv_client_peer_latency_seconds_count{peer=\"p\"} 2\n"
            "dfkv_client_peer_latency_max_seconds{peer=\"p\"} 0.0015\n",
            "dfkv_client_peer_latency_seconds_sum{peer=\"p\"} 0.005\n"
            "dfkv_client_peer_latency_seconds_count{peer=\"p\"} 3\n"
            "dfkv_client_peer_latency_max_seconds{peer=\"p\"} 0.0030\n",
        ])
        poller = PeerLatencyPoller(lambda: next(texts), metrics._recorder, interval_s=0)
        poller.poll_once()                       # baseline
        res = poller.poll_once()                 # delta: (0.005-0.002)/(3-2) = 0.003
        self.assertAlmostEqual(res["p"][0], 0.003)
        self.assertAlmostEqual(res["p"][1], 0.003)  # lifetime max


class StdlibExporterTest(TelemetryTestBase):
    """The default exporter is the pure-stdlib OTLP/HTTP-JSON pusher (no OTel SDK)."""

    def _enabled(self):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_HICACHE, tp_rank=2)
        return metrics._recorder

    def test_default_exporter_is_stdlib_not_otel(self):
        rec = self._enabled()
        self.assertTrue(metrics.is_enabled())
        self.assertIsNotNone(rec._stdlib)   # stdlib pusher started
        self.assertIsNone(rec._otel)        # OTel SDK not used

    def test_build_payload_structure_and_values(self):
        rec = self._enabled()
        with metrics.op("put", num_keys=3) as m:
            if m:
                m.bytes = 4096
        with metrics.op("get", num_keys=5) as m:
            if m:
                m.bytes = 8192
        try:
            with metrics.op("put", num_keys=1) as m:
                raise RuntimeError("io")
        except RuntimeError:
            pass
        rm = rec._stdlib.build()["resourceMetrics"][0]
        ra = {a["key"]: a["value"] for a in rm["resource"]["attributes"]}
        self.assertEqual(ra["dfkv.connector_type"]["stringValue"], "hicache")
        self.assertIn("dfkv.connector_id", ra)
        by_name = {m["name"]: m for m in rm["scopeMetrics"][0]["metrics"]}
        for n in ("dfkv_connector_ops_total", "dfkv_connector_keys_total",
                  "dfkv_connector_bytes_total", "dfkv_connector_op_seconds",
                  "dfkv_connector_info"):
            self.assertIn(n, by_name)
        ops = {}
        for dp in by_name["dfkv_connector_ops_total"]["sum"]["dataPoints"]:
            a = {x["key"]: x["value"]["stringValue"] for x in dp["attributes"]}
            ops[(a["op"], a["status"])] = int(dp["asInt"])
        self.assertEqual(ops[("put", "ok")], 1)
        self.assertEqual(ops[("put", "fail")], 1)
        self.assertEqual(ops[("get", "ok")], 1)
        # histogram: bucketCounts length == bounds + 1 (overflow)
        hist = by_name["dfkv_connector_op_seconds"]["histogram"]["dataPoints"][0]
        self.assertEqual(len(hist["bucketCounts"]), len(hist["explicitBounds"]) + 1)

    def test_metrics_url_normalization(self):
        from dfkv_telemetry.otlp_json import metrics_url
        self.assertEqual(metrics_url("http://h:4318"), "http://h:4318/v1/metrics")
        self.assertEqual(metrics_url("http://h:4318/"), "http://h:4318/v1/metrics")
        self.assertEqual(metrics_url("h:4318"), "http://h:4318/v1/metrics")
        self.assertEqual(metrics_url("http://h:4318/v1/metrics"), "http://h:4318/v1/metrics")
        self.assertEqual(metrics_url("http://h:4318/v1/metrics/"), "http://h:4318/v1/metrics")
        self.assertEqual(metrics_url(""), "http://localhost:4318/v1/metrics")

    def test_push_once_to_local_http_server(self):
        import http.server
        import json as _json
        import socketserver
        import threading
        captured = {}

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                captured["body"] = self.rfile.read(n)
                captured["ctype"] = self.headers.get("Content-Type")
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        srv = socketserver.TCPServer(("127.0.0.1", 0), H)
        port = srv.server_address[1]
        threading.Thread(target=srv.handle_request, daemon=True).start()

        rec = self._enabled()
        with metrics.op("put", num_keys=2) as m:
            if m:
                m.bytes = 100
        from dfkv_telemetry.otlp_json import StdlibExporter
        exp = StdlibExporter(rec, "http://127.0.0.1:%d" % port, interval_s=0)
        status = exp.push_once()
        srv.server_close()
        self.assertEqual(status, 200)
        self.assertEqual(captured["ctype"], "application/json")
        body = _json.loads(captured["body"].decode())
        names = [m["name"] for m in body["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]]
        self.assertIn("dfkv_connector_ops_total", names)


try:
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: F401
    _HAVE_OTEL = True
except Exception:
    _HAVE_OTEL = False


@unittest.skipUnless(_HAVE_OTEL, "opentelemetry SDK not installed")
class OtelPushTest(TelemetryTestBase):
    """Validates the real OTel instrument output via an in-memory reader (no
    network). Skipped when the SDK is absent; exercised in CI images with the
    'otel' extra and in the M3 backend stack end-to-end."""

    def _collect(self, reader):
        out = {}
        res = {}
        data = reader.get_metrics_data()
        for rm in data.resource_metrics:
            res = dict(rm.resource.attributes)
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    pts = []
                    for dp in metric.data.data_points:
                        attrs = dict(dp.attributes)
                        if hasattr(dp, "value"):
                            pts.append((attrs, dp.value))
                        else:
                            pts.append((attrs, ("count", dp.count)))
                    out[metric.name] = pts
        return out, res

    def test_otlp_instruments_emit_expected_series(self):
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        os.environ[config.ENV_CONNECTOR_ID] = "node-A-rank0"
        reader = InMemoryMetricReader()
        metrics.configure({}, connector_type=config.TYPE_VLLM, tp_rank=0,
                          model="m", version="2.0.0", native_version="1.6.3",
                          _readers=[reader])
        self.assertTrue(metrics.is_enabled())
        with metrics.op("put", num_keys=3, num_bytes=4096) as m:
            m.status = "ok"
        with metrics.op("get", num_keys=5, num_bytes=8192):
            pass
        try:
            with metrics.op("put", num_keys=2, num_bytes=10):
                raise RuntimeError("io")
        except RuntimeError:
            pass

        found, res = self._collect(reader)
        # identity rides on resource attributes (-> Prometheus labels via the
        # Collector's resource_to_telemetry_conversion).
        self.assertEqual(res.get("dfkv.connector_id"), "node-A-rank0")
        self.assertEqual(res.get("dfkv.connector_type"), config.TYPE_VLLM)
        # version labels (-> dfkv_version / dfkv_native_version in Prometheus)
        self.assertEqual(res.get("dfkv.version"), "2.0.0")
        self.assertEqual(res.get("dfkv.native_version"), "1.6.3")
        ops = {tuple(sorted(a.items())): v for a, v in found["dfkv_connector_ops_total"]}
        self.assertEqual(ops[(("op", "put"), ("status", "ok"))], 1)
        self.assertEqual(ops[(("op", "put"), ("status", "fail"))], 1)
        self.assertEqual(ops[(("op", "get"), ("status", "ok"))], 1)
        keys = {a["op"]: v for a, v in found["dfkv_connector_keys_total"]}
        self.assertEqual(keys["put"], 5)  # 3 + 2
        self.assertEqual(keys["get"], 5)
        nbytes = {a["op"]: v for a, v in found["dfkv_connector_bytes_total"]}
        self.assertEqual(nbytes["put"], 4106)
        self.assertEqual(nbytes["get"], 8192)
        self.assertIn("dfkv_connector_op_seconds", found)
        self.assertIn("dfkv_connector_op_max_seconds", found)
        self.assertTrue(any(v == 1 for _, v in found["dfkv_connector_info"]))
        metrics.shutdown()


if __name__ == "__main__":
    unittest.main()
