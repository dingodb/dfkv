# dfkv Client Configuration

The single reference for **client-side** knobs — the env vars and connector
config a KV client (SGLang HiCache / vLLM / LMCache) reads when it talks to a
dfkv cluster. Per-connector wiring lives in `docs/{hicache,lmcache,vllm}/DEPLOY.md`;
this page is the cross-connector catalogue.

> **First, the short answer to "what changed for the client in v1.7.0":**
> **Almost nothing.** The v1.7.0 server features — the **slab storage engine**
> (`--store-engine`) and the **RAM hot tier** (`DFKV_RAM_TIER`) — are **entirely
> server-side**. They change how a node stores bytes on disk / in RAM; the wire
> protocol, the returned bytes, and the client API are identical. A client
> **cannot tell** which engine a node runs and needs **no config** for them.
> The one genuinely new *client* knob is **`DFKV_WIRE_VERSION`** (opt-in wire v2,
> below), and even that defaults to the old behavior. An existing v1.6.x client
> talks to a v1.7.x server unchanged.

---

## 1. Connection & discovery

| env | default | notes |
|-----|---------|-------|
| `DFKV_LIB` (a.k.a. `DFKV_BUILD`) | — | absolute path to `libdfkv.so`. Connector config (`extra_config.lib`) overrides it. |
| `DFKV_MEMBERS` | — | **legacy static** member list `name=ip:port,...` for single-node / simple setups. Prefer MDS discovery. |

MDS dynamic discovery (production) is set via the connector's config, not env:
`mds_endpoints=ip:port,...` + `mds_group=<group>` (must match `dfkv_server --group`).
See the per-connector DEPLOY docs. The client polls the MDS and rebuilds the
weighted Ketama ring on each etcd epoch.

## 2. Transport (TCP / RDMA)

| env | default | recommended | notes |
|-----|---------|-------------|-------|
| `DFKV_RDMA` | unset = TCP | `1` | select native-verbs RDMA; unset ⇒ TCP fallback |
| `DFKV_RDMA_DEV` | — | all local 400G ports, comma list (`ib7s400p0,…,p7`) | RDMA rail(s); comma list = multi-rail. Required when `DFKV_RDMA=1`. |
| `DFKV_REQUIRE_RDMA` | `0` | `1` in prod | fail startup if no RDMA device instead of silently falling back to TCP |
| `DFKV_RDMA_DEPTH` | `1` | **keep 1** | in-flight requests per connection; a latency hider, **not** a throughput knob (GET/PUT are depth-flat). Scale via multi-connection fan-out. |
| `DFKV_RDMA_NUMA` | `0` | `1` on multi-NUMA hosts | pin buffers/threads to the rail's NUMA node + pick a NUMA-local rail |
| `DFKV_RDMA_MAX_PAYLOAD_BYTES` | 64 MiB | — | client-side max value size (must not exceed the server's) |

## 3. Wire protocol version — **new in v1.7.0**

| env | default | notes |
|-----|---------|-------|
| `DFKV_WIRE_VERSION` | `1` | `2` opts the client into **wire v2**: each request carries an 8-byte `seq` echoed in the reply, validated by the client (defense against a torn/misordered stream). |

Rules:
- **Servers dual-accept** v1 and v2 → **no flag-day**. You can flip clients to v2
  independently of the server rollout, and vice-versa.
- **TCP only.** The RDMA transport is v1-only (its zero-copy receive can't yet
  adapt to the longer v2 prefix), so `DFKV_WIRE_VERSION=2` is **silently ignored
  on the RDMA path** — an RDMA client always sends v1, which every server accepts.
  Safe to set globally; it only affects TCP connections.
- The env is read **once** per process (cached), so set it before the client
  starts.
- Default `1` = byte-identical to v1.6.x, so leaving it unset is always safe.

## 4. Block identity (no config, informational)

v1.7.0 widened the block key to **96 bits** (`id` = MD5[0..8), `index` =
MD5[8..12)) to make same-model hash collisions vanishingly unlikely. This is
automatic inside `libdfkv` — **no env, no config**. One consequence worth knowing:
a **v1.6.x client and a v1.7.x client compute different keys for the same logical
block**, so if you run *both* client versions against *one* ring they won't share
each other's cache entries (a clean cross-version miss → recompute, never
corruption). Within a single client version everything is consistent. Routing is
unaffected (it uses a separate hash), so mixing client versions never mis-routes.

## 5. Connector tuning (batching / behavior)

Common across connectors (some are also settable via `kv_connector_extra_config`;
see the per-connector DEPLOY doc for precedence):

| env | default | notes |
|-----|---------|-------|
| `DFKV_CONNECTOR_ID` | — | logical client id for metrics labels |
| `DFKV_CONNECTOR_BATCH_MAX_KEYS` | connector-specific | cap keys per batch RPC |
| `DFKV_CONNECTOR_GET_PARALLELISM` | connector-specific | concurrent GET fan-out |
| `DFKV_CONNECTOR_ASSUME_EXISTS` | `0` | skip the Exist probe before load (trade a probe for a possible miss) |
| `DFKV_TP_RANK` | — | tensor-parallel rank (MLA: only rank 0 writes) |

## 6. Observability (opt-in, off the datapath)

| area | env | see |
|------|-----|-----|
| Fleet metrics push (OTLP) | `DFKV_METRICS_ENABLED=1`, `DFKV_METRICS_EXPORTER`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `DFKV_PROBE_INTERVAL_MS`, `DFKV_CLIENT_STATS_POLL_S`, `DFKV_PEER_LATENCY_POLL_S` | [METRICS.md](METRICS.md) §3.4, [deploy/observability/CONNECTOR-USAGE.md](../deploy/observability/CONNECTOR-USAGE.md) |
| Distributed tracing push | `DFKV_TRACING_ENABLED=1`, `DFKV_TRACE_SLOW_REQUEST_MS`, `DFKV_TRACE_SAMPLE_PERCENT`, `DFKV_TRACE_EXPORT_INTERVAL_MS` | [tracing.md](tracing.md) |
| Per-op access log | `DFKV_ACCESS_LOG_ENABLED=1`, `DFKV_ACCESS_LOG_PATH`, `DFKV_ACCESS_LOG_THRESHOLD_US`, `DFKV_ACCESS_LOG_MAX_BYTES`, `DFKV_ACCESS_LOG_BACKUP_COUNT` | [access_log.md](access_log.md) |

## 7. NOT client config — server-side only (common confusion)

These are set on `dfkv_server` / its systemd unit, **never on the client**, and
have **zero client-visible effect** (see [ARCHITECTURE.md](ARCHITECTURE.md) §5–7):

| flag / env | side | what it does |
|------------|------|--------------|
| `--store-engine=file\|slab` / `DFKV_STORE_ENGINE` | **server** | on-disk storage backend (per node) |
| `DFKV_RAM_TIER` / `DFKV_RAM_TIER_BYTES` | **server** | write-through RAM hot tier |
| `DFKV_SERVER_URING` | **server** | io_uring async-GET serve path |

A mixed fleet (some nodes slab, some file; some with the RAM tier, some without)
serves every client identically — pick engines per node freely.

---

## Minimal client setup (vLLM example)

```bash
DFKV_RDMA=1 \
DFKV_RDMA_DEV=ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7 \
DFKV_LIB=/opt/dfkv/libdfkv.so \
python -m vllm.entrypoints.openai.api_server ... \
  --kv-transfer-config '{"kv_connector":"DfkvStoreConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"mds_endpoints":"192.168.0.8:28150,192.168.0.9:28150",
    "mds_group":"glm"}}'
# DFKV_WIRE_VERSION unset ⇒ v1 (default). No slab/RAM knobs on the client.
```

Full per-connector config: [hicache/DEPLOY.md](hicache/DEPLOY.md) ·
[vllm/DEPLOY.md](vllm/DEPLOY.md) · [lmcache/DEPLOY.md](lmcache/DEPLOY.md).
