# SPDX-License-Identifier: Apache-2.0
"""Shared config resolution + env constants for dfkv connector telemetry.

The precedence rule (``extra_config`` key wins, then env var, then default)
mirrors ``dfkv_access_log._resolve`` so operators learn one mental model across
the access log, the push-metrics layer and (later) tracing.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Optional

# --- master switches (off by default => zero cost) -------------------------
# Metrics push is on when DFKV_METRICS_ENABLED is truthy, OR the umbrella
# DFKV_TELEMETRY_ENABLED is truthy (the umbrella also turns on tracing later).
ENV_METRICS_ENABLED = "DFKV_METRICS_ENABLED"
ENV_TELEMETRY_ENABLED = "DFKV_TELEMETRY_ENABLED"

# --- OTLP push target (standard OTel env, shared with the C++ side) ---------
ENV_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
ENV_OTLP_PROTOCOL = "OTEL_EXPORTER_OTLP_PROTOCOL"  # grpc | http/protobuf

# --- dfkv-namespaced knobs --------------------------------------------------
ENV_CONNECTOR_ID = "DFKV_CONNECTOR_ID"
ENV_EXPORT_INTERVAL_MS = "DFKV_METRICS_EXPORT_INTERVAL_MS"   # OTLP push cadence (default 10000)
ENV_PROBE_INTERVAL_MS = "DFKV_PROBE_INTERVAL_MS"             # C++ per-peer probe (default off; 5000 when on)
ENV_PEER_POLL_S = "DFKV_PEER_LATENCY_POLL_S"                 # snapshot->push cadence (default 10)

# connector_type values the dashboard groups by.
TYPE_HICACHE = "hicache"
TYPE_LMCACHE = "lmcache"
TYPE_VLLM = "vllm"


def truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def resolve(cfg: Optional[dict], key: str, env: str, default: Any) -> Any:
    """extra_config key wins; then env var; then default."""
    cfg = cfg or {}
    if cfg.get(key) is not None:
        return cfg[key]
    if env in os.environ:
        return os.environ[env]
    return default


def metrics_enabled(cfg: Optional[dict]) -> bool:
    """Whether the push-metrics layer should be active for this process."""
    v = resolve(cfg, "metrics", ENV_METRICS_ENABLED, None)
    if v is not None:
        return truthy(v)
    return truthy(resolve(cfg, "telemetry", ENV_TELEMETRY_ENABLED, False))


def dist_version(dist_name: str) -> str:
    """Installed version of a pip distribution (e.g. "dfkv-vllm"), or "" if it
    can't be determined. Used to label fleet metrics with the connector package
    version so a rolling connector upgrade is visible per instance. Never raises."""
    if not dist_name:
        return ""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # pragma: no cover (py<3.8; not supported but be safe)
        return ""
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return ""
    except Exception:  # pragma: no cover (defensive: never break the connector)
        return ""


def resolve_connector_id(cfg: Optional[dict], tp_rank: int = 0) -> str:
    """Stable identity for this connector instance, surfaced as a metric label
    so the dashboard can enumerate / drill into a single instance.

    Explicit ``connector_id`` (extra_config or DFKV_CONNECTOR_ID) wins; otherwise
    auto-derive ``<host>:<pid>:<tp_rank>`` which is unique per process/rank."""
    cid = resolve(cfg, "connector_id", ENV_CONNECTOR_ID, "")
    if cid:
        return str(cid)
    try:
        host = socket.gethostname()
    except Exception:
        host = "unknown"
    return "{}:{}:{}".format(host, os.getpid(), int(tp_rank))
