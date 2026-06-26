# dfkv local observability stack

A vendor-neutral, runnable-locally backend for the dfkv telemetry layer:

```
connectors (vllm/lmcache/hicache) --OTLP push--> otel-collector --+--> prometheus --> grafana
dfkv_server / dfkv_mds  <--Prometheus pull (/metrics)-------------/        ^
                                                  otel-collector --traces--> tempo --/
```

- **Connectors PUSH** their fleet metrics over OTLP to the Collector (they are
  many and dynamically scheduled, so pull-discovery is impractical).
- **dfkv C++ daemons are PULLED**: Prometheus scrapes `dfkv_server`/`dfkv_mds`
  `/metrics` directly (few, long-lived). Start them with `--metrics-port`.
- The Collector re-exposes the pushed metrics at `:8889` for Prometheus and
  forwards traces (later milestones) to Tempo.

## Bring it up

```bash
docker compose -f deploy/observability/docker-compose.yml up -d
```

| Service | URL | Notes |
|---|---|---|
| Grafana | http://localhost:3000 | anon admin; dashboards **"dfkv — cluster overview"** (connectors) + **"dfkv — backend (cache nodes + MDS)"** auto-provisioned |
| Prometheus | http://localhost:9090 | |
| Collector OTLP | grpc `localhost:4317`, http `localhost:4318` | connectors push here |
| Collector scrape | http://localhost:8889/metrics | what Prometheus reads |
| Tempo | http://localhost:3200 | traces (later milestones) |

## Point a connector at it

Telemetry is **off by default** (zero cost). Turn it on per connector process:

```bash
export DFKV_METRICS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317      # gRPC; use :4318 for http
# optional: a stable, human-readable id (else auto = host:pid:tp_rank)
export DFKV_CONNECTOR_ID=myhost-rank0
# optional: active per-cache-node latency probe (every 5s) so idle nodes still
# show avg/max latency. Off unless set. (The SGLang plugin auto-enables it.)
export DFKV_PROBE_INTERVAL_MS=5000
# install the optional OTel deps once: pip install 'dfkv-vllm[otel]'  (or dfkv-connector[otel])
```

Then run vLLM / LMCache / SGLang as usual. Metrics appear in Grafana within ~15s.

## Configuration reference

All telemetry is opt-in. Configure via env vars (and, for the SGLang plugin,
equivalent `extra_config` keys, precedence **extra_config > env > default**).

| Env var | extra_config key | Default | Effect |
|---|---|---|---|
| `DFKV_METRICS_ENABLED` | `metrics` | `0` (off) | master switch for push metrics |
| `DFKV_TELEMETRY_ENABLED` | `telemetry` | `0` | umbrella switch (metrics + future traces) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `otlp_endpoint` | SDK default | Collector endpoint (grpc `:4317` / http `:4318`) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `otlp_protocol` | `grpc` | `grpc` or `http/protobuf` |
| `DFKV_CONNECTOR_ID` | `connector_id` | `<host>:<pid>:<tp_rank>` | stable instance id label |
| `DFKV_METRICS_EXPORT_INTERVAL_MS` | `metrics_export_interval_ms` | `10000` | OTLP push cadence (min 1000) |
| `DFKV_PROBE_INTERVAL_MS` | `probe_interval_ms` | `0` off (`5000` when metrics on) | C++ active per-peer latency probe |
| `DFKV_PEER_LATENCY_POLL_S` | `peer_latency_poll_s` | `10` | snapshot→push cadence for per-peer latency |

**Cost model.** When metrics are **off**: the connector op path evaluates no
metric args (a falsy no-op guard), the C++ datapath is byte-for-byte unchanged,
and the OTel SDK is never imported — effectively zero. When **on**: each op does
an in-memory aggregate update (no I/O); a background thread pushes the aggregated
state over OTLP every `DFKV_METRICS_EXPORT_INTERVAL_MS` (time-triggered, not
size-triggered — metrics are aggregates, not an event buffer).

## Dashboards

Two dashboards split by audience: **connector / business-traffic view** vs
**backend / infrastructure-health view**. Both read the same Prometheus.

### 1. "dfkv — cluster overview" (connectors — pushed via OTLP)

1. **Connector instances / by type / inventory table** — how many connectors,
   which type (hicache/lmcache/vllm), by `connector_id`. The inventory table also
   shows each instance's `dfkv_version` (connector pkg) and `dfkv_native_version`
   (libdfkv.so), so a half-finished rolling upgrade / version skew is visible.
2. **PUT/GET req-rate, keys/s, bytes/s** — volume per op.
3. **Op latency avg / p99 / max** — per op.
4. **Ops by status + success rate** — success vs failure and the ratio.
5. **Per-cache-node latency** (avg / max per node) — from the client's active
   per-peer probe (`DFKV_PROBE_INTERVAL_MS`); visible even when idle.

Use the **Connector type** / **Connector id** template variables at the top to
drill into one type or one instance.

### 2. "dfkv — backend (cache nodes + MDS)" (C++ daemons — scraped via pull)

Fed by the `dfkv_server` / `dfkv_mds` Prometheus scrape jobs (start the daemons
with `--metrics-port`). Three sections:

- **Cache nodes** — hit ratio, cache/exist ops/s, op latency avg/p99 by op,
  read/write throughput, evictions, errors by op/status, capacity (used + per
  disk), connections, and a node inventory table (version / uptime).
- **RDMA** — active connections, completions/s & errors/s, idle reclaims/s.
- **MDS** — members, register/keepalive/list ops/s, etcd lease grants & errors,
  and a replica inventory table (version per instance).

Use the **Cache node** template variable to filter by `instance`.

> Note: the C++ client-side health counters (`dfkv_client_io_errors_total`,
> `unhealthy_skips`, `peer_marked_bad/recovered`, `peer_errors`) are **not** on
> this central Prometheus — they are mirrored only onto SGLang's own `/metrics`
> via `dfkv_metrics.py` (needs `prometheus_client` + multiproc mode). Only
> `dfkv_client_peer_latency_*` reaches the center (pushed via OTLP), and it lives
> on the connector dashboard. Wiring those counters to OTLP push is future work.

## Metric reference (what the connectors emit)

| Metric | Type | Labels |
|---|---|---|
| `dfkv_connector_info` | gauge (=1) | identity: `dfkv_connector_id/type/host/pid/tp_rank` + version: `dfkv_version` (connector pkg), `dfkv_native_version` (libdfkv.so) |
| `dfkv_connector_ops_total` | counter | `op`, `status` |
| `dfkv_connector_keys_total` | counter | `op` |
| `dfkv_connector_bytes_total` | counter | `op` |
| `dfkv_connector_op_seconds` | histogram | `op` |
| `dfkv_connector_op_max_seconds` | gauge | `op` |

Identity rides on OTLP **resource attributes**; the Collector's
`resource_to_telemetry_conversion` turns them into the `dfkv_connector_*` labels
above (with `.`→`_`). Metric-name suffixing is disabled so names are verbatim.

## Trace backend: Tempo vs Jaeger

Tempo is used for first-class Grafana trace↔metrics↔logs correlation. To use a
standalone Jaeger UI instead, swap the `tempo` service for
`jaegertracing/all-in-one` (`COLLECTOR_OTLP_ENABLED=true`, ports 16686/4317) and
point the Collector's `otlp/tempo` exporter at it.
