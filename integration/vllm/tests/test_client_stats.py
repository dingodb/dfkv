"""ClientStatsPoller parses the C client snapshot into ring/MDS health gauges.

Pure Python: feeds a canned Prometheus-text snapshot to the poller (no libdfkv,
no GPU) and asserts the parsed values, then — when prometheus_client is present —
that the vllm:dfkv_client_* gauges carry them on the scrape."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfkv_vllm.client_stats import ClientStatsPoller

# A healthy snapshot: 5-member ring, MDS reachable, no failed polls. Includes
# HELP/TYPE comment lines, a labeled per-peer series, and an unrelated metric the
# parser must ignore.
_HEALTHY = """\
# HELP dfkv_client_ring_members ring size
# TYPE dfkv_client_ring_members gauge
dfkv_client_ring_members 5
dfkv_client_mds_reachable 1
dfkv_client_mds_unreachable_polls_total 0
dfkv_client_ops_served_total 123456
dfkv_client_peer_latency_seconds{peer="10.0.0.1"} 0.0012
"""

# The production incident: empty ring, MDS unreachable, failed polls piling up.
_EMPTY_RING = """\
dfkv_client_ring_members 0
dfkv_client_mds_reachable 0
dfkv_client_mds_unreachable_polls_total 7
"""


def test_parses_healthy_snapshot():
    p = ClientStatsPoller(lambda: _HEALTHY, tp_rank=0, interval_s=0)
    p.poll_once()
    g = p.gauges()
    assert g["dfkv_client_ring_members"] == 5
    assert g["dfkv_client_mds_reachable"] == 1
    assert g["dfkv_client_mds_unreachable_polls_total"] == 0


def test_parses_empty_ring_snapshot():
    p = ClientStatsPoller(lambda: _EMPTY_RING, tp_rank=3, interval_s=0)
    p.poll_once()
    g = p.gauges()
    assert g["dfkv_client_ring_members"] == 0
    assert g["dfkv_client_mds_reachable"] == 0
    assert g["dfkv_client_mds_unreachable_polls_total"] == 7


def test_missing_lines_default_to_zero():
    p = ClientStatsPoller(lambda: "dfkv_client_ring_members 2\n", tp_rank=0, interval_s=0)
    p.poll_once()
    g = p.gauges()
    assert g["dfkv_client_ring_members"] == 2
    assert g["dfkv_client_mds_reachable"] == 0


def test_empty_and_malformed_text_never_raises():
    for txt in ("", None, "garbage-with-no-space\n", "dfkv_client_ring_members notanint\n"):
        p = ClientStatsPoller(lambda t=txt: t, tp_rank=0, interval_s=0)
        p.poll_once()
        assert p.gauges()["dfkv_client_ring_members"] == 0


def test_zero_interval_disables_thread():
    p = ClientStatsPoller(lambda: _HEALTHY, tp_rank=0, interval_s=0)
    p.start()
    assert p._thread is None
    p.stop()  # must be safe even when never started


def test_prometheus_gauges_reflect_snapshot():
    prometheus_client = pytest.importorskip("prometheus_client")
    p = ClientStatsPoller(lambda: _HEALTHY, tp_rank=1,
                          transport="rdma", interval_s=60)
    p.start()  # publishes transport_info + an immediate poll; loop won't fire in 60s
    reg = prometheus_client.REGISTRY
    assert reg.get_sample_value(
        "vllm:dfkv_client_ring_members", {"tp_rank": "1"}) == 5
    assert reg.get_sample_value(
        "vllm:dfkv_client_mds_reachable", {"tp_rank": "1"}) == 1
    assert reg.get_sample_value(
        "vllm:dfkv_client_mds_unreachable_polls_total", {"tp_rank": "1"}) == 0
    assert reg.get_sample_value(
        "vllm:dfkv_client_transport_info",
        {"tp_rank": "1", "transport": "rdma"}) == 1
    p.stop()
