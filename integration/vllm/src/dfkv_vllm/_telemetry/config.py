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
# DFKV_TELEMETRY_ENABLED is truthy. Tracing is on when DFKV_TRACING_ENABLED is
# truthy, OR the same umbrella DFKV_TELEMETRY_ENABLED is truthy.
ENV_METRICS_ENABLED = "DFKV_METRICS_ENABLED"
ENV_TELEMETRY_ENABLED = "DFKV_TELEMETRY_ENABLED"
ENV_TRACING_ENABLED = "DFKV_TRACING_ENABLED"

# --- OTLP push target (standard OTel env, shared with the C++ side) ---------
ENV_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
ENV_OTLP_PROTOCOL = "OTEL_EXPORTER_OTLP_PROTOCOL"  # grpc | http/protobuf

# --- dfkv-namespaced knobs --------------------------------------------------
ENV_CONNECTOR_ID = "DFKV_CONNECTOR_ID"
ENV_EXPORT_INTERVAL_MS = "DFKV_METRICS_EXPORT_INTERVAL_MS"   # OTLP push cadence (default 10000)
# Which exporter pushes the metrics: "stdlib" (default; pure-stdlib OTLP/HTTP-JSON,
# zero third-party deps) or "otel" (the OpenTelemetry SDK, if installed).
ENV_METRICS_EXPORTER = "DFKV_METRICS_EXPORTER"
EXPORTER_STDLIB = "stdlib"
EXPORTER_OTEL = "otel"
ENV_PROBE_INTERVAL_MS = "DFKV_PROBE_INTERVAL_MS"             # C++ per-peer probe (default off; 5000 when on)
ENV_PEER_POLL_S = "DFKV_PEER_LATENCY_POLL_S"                 # snapshot->push cadence (default 10)

# --- tracing knobs (connector-side spans pushed over OTLP /v1/traces) -------
ENV_TRACE_SLOW_REQUEST_MS = "DFKV_TRACE_SLOW_REQUEST_MS"     # >= this latency => trace it (default 1000; 0=off)
ENV_TRACE_SAMPLE_PERCENT = "DFKV_TRACE_SAMPLE_PERCENT"       # extra 0..100% of requests traced (default 0)
ENV_TRACE_EXPORT_INTERVAL_MS = "DFKV_TRACE_EXPORT_INTERVAL_MS"  # span flush cadence (default 5000)
ENV_TRACE_MAX_BUFFERED_SPANS = "DFKV_TRACE_MAX_BUFFERED_SPANS"  # bounded buffer; oldest dropped (default 2048)

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


# --- required-config validation --------------------------------------------
# Important connector parameters (the per-model isolation identity, and the ring
# to connect to) are checked at startup so the inference service refuses to
# START rather than serve with a silently-wrong config. The isolation identity
# is the sharpest case: a forgotten model_hash used to default to 0 (the shared
# keyspace), which reads as valid but lets two models collide/share KV. The
# checks distinguish "not passed" from "passed 0" and abort on either, unless
# the operator explicitly opts into the shared keyspace.
ENV_MODEL_HASH = "DFKV_MODEL_HASH"
ENV_ALLOW_SHARED_KEYSPACE = "DFKV_ALLOW_SHARED_KEYSPACE"

_UINT64_MAX = 0xFFFFFFFFFFFFFFFF
_MISSING = object()


class DfkvConfigError(ValueError):
    """A required dfkv connector config value is missing or invalid.

    Raised at connector startup so the inference service refuses to start
    instead of serving with a broken isolation identity (silently sharing or
    colliding KV across models) or with no ring to connect to. Messages name the
    offending parameter and how to set it, including the explicit
    shared-keyspace opt-in.
    """


def allow_shared_keyspace(cfg: Optional[dict] = None, override: Optional[bool] = None) -> bool:
    """Whether the operator EXPLICITLY opted into the shared (no-isolation)
    keyspace via allow_shared_keyspace=1 / DFKV_ALLOW_SHARED_KEYSPACE=1.
    `override` (when not None) wins, for callers that resolved it themselves."""
    if override is not None:
        return bool(override)
    return truthy(resolve(cfg, "allow_shared_keyspace",
                          ENV_ALLOW_SHARED_KEYSPACE, False))


def require_model_hash(cfg: Optional[dict] = None, *, allow_shared: Optional[bool] = None) -> int:
    """Return a validated nonzero uint64 model_hash, or raise DfkvConfigError.

    model_hash is the per-model isolation identity of the dfkv keyspace and is
    REQUIRED: a missing / zero / non-integer / out-of-uint64 value aborts
    startup, UNLESS the operator opts into the shared keyspace (returns 0 then).
    The missing-vs-zero distinction is intentional — ``cfg.get('model_hash', 0)``
    used to mask a forgotten param as the shared keyspace."""
    shared = allow_shared_keyspace(cfg, allow_shared)
    raw = resolve(cfg, "model_hash", ENV_MODEL_HASH, _MISSING)
    if raw is _MISSING or (isinstance(raw, str) and not raw.strip()):
        if shared:
            return 0
        raise DfkvConfigError(
            "model_hash is required as the per-model isolation identity but was "
            "not set. Set a nonzero per-model value (extra_config 'model_hash' or "
            "env DFKV_MODEL_HASH), or allow_shared_keyspace=1 to opt into the "
            "shared keyspace (no cross-model isolation).")
    if isinstance(raw, bool):  # bool is an int subclass; reject it explicitly
        raise DfkvConfigError("model_hash must be an integer (uint64), not a bool.")
    try:
        v = raw if isinstance(raw, int) else int(str(raw), 0)
    except (TypeError, ValueError):
        raise DfkvConfigError(
            "model_hash must be an integer (uint64); got {!r}.".format(raw))
    if v < 0 or v > _UINT64_MAX:
        raise DfkvConfigError(
            "model_hash out of uint64 range [0, 2^64): {}.".format(v))
    if v == 0:
        if shared:
            return 0
        raise DfkvConfigError(
            "model_hash=0 is the shared keyspace (no cross-model isolation). Set "
            "a nonzero per-model value, or allow_shared_keyspace=1 to opt in.")
    return v


def require_isolation_name(name: Any, *, field: str = "model_name",
                           cfg: Optional[dict] = None,
                           allow_shared: Optional[bool] = None) -> str:
    """Validate an isolation-namespace string (the source a connector hashes into
    model_hash, e.g. LMCache's model_name). Empty / blank / non-string makes
    every deployment collide on one derived keyspace, so it aborts startup unless
    the shared keyspace is explicitly allowed."""
    if isinstance(name, str) and name.strip():
        return name
    if allow_shared_keyspace(cfg, allow_shared):
        return name if isinstance(name, str) else ""
    raise DfkvConfigError(
        "{0} is required as the isolation namespace (it derives the dfkv "
        "model_hash); an empty {0} makes every deployment share one keyspace. "
        "Set {0}, or allow_shared_keyspace=1 to opt in.".format(field))


def require_ring_endpoint(members: Any = None, mds_endpoints: Any = None) -> None:
    """At least one of static ``members`` or ``mds_endpoints`` must be set, else
    the client has no dfkv ring to connect to. Raises DfkvConfigError otherwise."""
    if (str(members).strip() if members else "") or \
            (str(mds_endpoints).strip() if mds_endpoints else ""):
        return
    raise DfkvConfigError(
        "no dfkv ring to connect to: set either 'members' (static member list) "
        "or 'mds_endpoints' (MDS discovery).")


def metrics_enabled(cfg: Optional[dict]) -> bool:
    """Whether the push-metrics layer should be active for this process."""
    v = resolve(cfg, "metrics", ENV_METRICS_ENABLED, None)
    if v is not None:
        return truthy(v)
    return truthy(resolve(cfg, "telemetry", ENV_TELEMETRY_ENABLED, False))


def tracing_enabled(cfg: Optional[dict]) -> bool:
    """Whether connector-side request tracing should be active for this process.

    Mirrors ``metrics_enabled``: an explicit ``tracing`` key / DFKV_TRACING_ENABLED
    wins; otherwise the umbrella DFKV_TELEMETRY_ENABLED turns it on too."""
    v = resolve(cfg, "tracing", ENV_TRACING_ENABLED, None)
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
    auto-derive ``<host>_<pid>_<tp_rank>`` which is unique per process/rank.

    The separator MUST be in the MDS ``IsValidGroupOrId`` alphabet
    ``[A-Za-z0-9._-]`` — a ``:`` (the previous default) makes the whole id fail
    validation, so ``MdsServer::UpsertClient`` returns ``kInvalid``, the
    registrar's ``SendOnce`` never sees ``kOk``, and ``MdsRegistrar::Loop`` is
    stuck retrying ``kClientRegister`` forever and never advances to the
    heartbeat loop — ``dfkvctl clients`` stays empty despite thousands of
    register requests. Underscore is path-safe and not part of any hostname
    convention (hostnames are ``[a-z0-9-]``), so it cannot collide with the
    host segment."""
    cid = resolve(cfg, "connector_id", ENV_CONNECTOR_ID, "")
    if cid:
        return str(cid)
    try:
        host = socket.gethostname()
    except Exception:
        host = "unknown"
    return "{}_{}_{}".format(host, os.getpid(), int(tp_rank))
