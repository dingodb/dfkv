# dfkv — distributed KV cache for LLM inference (SGLang · LMCache · vLLM)

[![CI](https://github.com/dingodb/dfkv/actions/workflows/ci.yml/badge.svg)](https://github.com/dingodb/dfkv/actions/workflows/ci.yml)

A small, **self-contained** distributed key-value cache that pools GPU-node NVMe
SSDs into a shared, large-capacity KV pool for LLM inference (e.g. GLM-5.1 / MLA,
DeepSeek-V4), **without any DingoFS / brpc / S3-RADOS dependency** — it runs on
its own (only its built-in MDS + etcd for dynamic membership). It plugs into
three engines through thin adapters over one portable core:

- **SGLang HiCache** as an L3 external KV store (`--hicache-storage-backend dynamic`).
- **LMCache** as a `RemoteConnector`.
- **vLLM** directly as a `KVConnectorBase_V1` (GPUDirect RDMA, no LMCache).

> Origin: extracted from the DingoFS branch `feat/kvcache-sglang`
> (`src/cache/kvclient`). The portable core has zero coupling to DingoFS, so it
> lives here as an independent repo. To instead fuse these semantics into the
> production `dingo-cache` (brpc + MDS), see `docs/INTEGRATION.md`.

## What it is
- **`dfkv_server`** — a cache-node daemon. Disk + LRU, **cache-only** (a miss is
  a clean NotFound; no object-store fallback), synchronous durable-visible writes.
  Supports **multiple NVMe SSDs per node** (`--dir d1,d2,d3`, intra-node Ketama).
  With `--mds`, `--group`, `--id`, `--advertise`, `--weight` it registers into the
  MDS tier; the old static `--members` flag has been removed. Pluggable storage
  backend `--store-engine=file|slab` (default `file`) and an optional
  write-through **RAM hot tier** (`DFKV_RAM_TIER=1`) — see
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **`dfkv_mds`** — stateless Membership Directory Service daemon. Flags:
  `--listen <port>` and `--etcd <host:port>` (default `127.0.0.1:2379`). The only
  etcd client in the system; holds each node's etcd lease on its behalf. Deploy as
  N replicas — no load-balancer needed; nodes and clients each pick any reachable
  MDS and fail over automatically.
- **`libdfkv.so`** — C ABI client (key→consistent-hash routing, value header with
  CRC + model/page/dtype/layer geometry guard, Put/Get/Exist).
- **`integration/hicache/dfkv_hicache.py`** — SGLang `HiCacheStorage` plugin loaded via
  `--hicache-storage-backend dynamic` (no SGLang fork). MLA: one packed-latent
  object per page, no tp_rank suffix, `backup_skip` (only tp_rank 0 writes).

## Design in one breath
SGLang HiCache (zero-copy v1) → `dfkv_hicache.py` (ctypes) → `libdfkv` client
(Ketama route + header wrap/verify) → TCP/RDMA → `dfkv_server` → optional RAM hot
tier → DiskCacheGroup over N NVMe (per-disk `StoreEngine`: `file` or `slab`), LRU.
Distributed = client-side consistent hashing; no replication (regenerable KV →
node loss = miss → recompute). Full architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Membership** is managed by the MDS tier (`dfkv_mds` + etcd). Nodes register
with the MDS on startup and send periodic heartbeats; etcd leases (TTL 30 s)
are the liveness signal. Clients call `dfkv_start_mds_discovery(c, "ep1,ep2",
group, poll_ms)` to poll the MDS and rebuild the weighted consistent-hash ring
whenever the epoch (etcd revision) advances. Two-layer offline detection:
**layer-2** — etcd lease expiry → MDS view changes → client epoch → ring rebuild
(authoritative removal, ≤ 30 s); **layer-1** — `PeerHealth` fast avoidance: a
peer that fails transport IO is short-circuited to miss for a cooldown period
without any ring change. The legacy static path (`dfkv_open(members=...)` /
`dfkv_set_members`) still exists for simple or single-node setups.

## Build & test (no GPU / no RDMA needed)
```bash
cmake -S . -B build            # add -DDFKV_STATIC_LIBSTDCXX=ON for portable binaries
cmake --build build -j
ctest --test-dir build --output-on-failure   # C++ gtests + the Python plugin test
```
Artifacts: `build/dfkv_server`, `build/dfkv_mds`, `build/libdfkv.so`.

## Run a cluster
```bash
# 1. Start etcd (one or three nodes, external)

# 2. Start MDS replicas (stateless, any number)
dfkv_mds --listen 9400 --etcd 127.0.0.1:2379

# 3. On each cache node (--mds requires --id and --advertise)
dfkv_server --dir /mnt/disk1/dfkv,/mnt/disk2/dfkv,/mnt/disk3/dfkv \
            --port 12000 --cap 6597069766656 \
            --mds 10.0.0.1:9400,10.0.0.2:9400 \
            --group default --id n1 --advertise 10.0.0.10:12000

# 4. Client: MDS-based discovery (recommended)
#    dfkv_start_mds_discovery(c, "10.0.0.1:9400,10.0.0.2:9400", "default", 3000);
# OR legacy static path (single-node / simple setups)
#    dfkv_open("n1=10.0.0.10:12000,...", ...)
```
Full dfkv CLUSTER deploy runbook (etcd + MDS + systemd units): `docs/DEPLOY.md`.
Per-engine connect/config: `docs/hicache/DEPLOY.md` · `docs/vllm/DEPLOY.md` · `docs/lmcache/DEPLOY.md`.

## Layout
```
src/        portable C++ core: common/ (shared types) · utils/ (generic helpers) ·
            transport/ (TCP/RDMA + v1/v2 wire protocol) · cache/ (StoreEngine: file
            KVStore | slab SlabAllocator+DiskSlabStore · RamTier · dfkv_server) ·
            client/ (KV client + C ABI) · mds/ (membership service + dfkv_mds) · tools/ (CLIs)
integration/hicache/  dfkv_hicache.py (SGLang dynamic backend plugin) + dfkv_telemetry/
                      (canonical shared telemetry pkg, vendored by the other connectors)
integration/lmcache/  dfkv_connector  (LMCache RemoteConnector, ctypes over libdfkv.so)
integration/vllm/     dfkv_vllm       (vLLM KVConnectorBase_V1, GPUDirect RDMA, bypass LMCache)
test/       gtest suites + test/python (unittest + no-torch sglang shim)
docs/       ARCHITECTURE.md (layers · storage engines · RAM hot tier · wire protocol) ·
            DEPLOY.md (dfkv CLUSTER deploy: etcd + MDS + server + systemd) · INTEGRATION.md (fuse into dingo-cache)
docs/hicache/  SGLang HiCache connector docs (DEPLOY — connect/config/use)
docs/lmcache/  LMCache connector docs (DESIGN · IMPLEMENTATION · DEPLOY)
docs/vllm/     vLLM connector docs (DEPLOY — config reference + recommended settings)
```

## Engine integrations
- **SGLang HiCache**: `integration/hicache/dfkv_hicache.py` — see `docs/hicache/DEPLOY.md` (connect/config/use;
  cluster deploy is `docs/DEPLOY.md`).
- **LMCache**: `integration/lmcache/` (`dfkv_connector`) — see `docs/lmcache/DESIGN.md`,
  `docs/lmcache/IMPLEMENTATION.md`, `docs/lmcache/DEPLOY.md`.
- **vLLM (direct)**: `integration/vllm/` (`dfkv_vllm`) — a `KVConnectorBase_V1`
  connector occupying the same `--kv-transfer-config` slot as `MooncakeStoreConnector`,
  storing/loading KV **directly over GPUDirect RDMA** (no LMCache, no host bounce).
  Pure-Python ctypes over `libdfkv.so`; uses the scatter-gather batch API to coalesce
  per-chunk keys. Validated on H100 + IB with DeepSeek-V4 (multi kv_cache_group / MLA +
  SWA), full cross-restart and cross-DP prefix hit. See `docs/vllm/DEPLOY.md` (config
  reference + recommended settings) and `integration/vllm/README.md`.

## Operability & performance features
- **Pluggable storage engine** (`--store-engine=file|slab`, default `file`): the
  `file` engine is one file per block (battle-tested); the `slab` engine is a
  fixed pool of pre-allocated **extent files** carved into slots by a media-agnostic
  size-class allocator, with a compact `slots.tbl` so the index **rebuilds on
  restart** (cache warmth across a rolling upgrade) — removing the one-file-per-block
  hazards (tmp leak / ENOSPC dead-end / unbounded inodes / lock-held unlink /
  open-per-GET). Crash-safe (CRC32 per slot record). Off by default.
- **RAM hot tier** (`DFKV_RAM_TIER=1`, off by default): a pre-registered RAM arena
  fronting the disk — PUT is **write-through** (synchronously visible, async-flushed
  to disk) and a warm GET is served **straight from the arena over RDMA** (zero-copy
  scatter-send from the arena MR, no open/pread/disk), removing the disk-bound COLD
  load bottleneck. Send-in-flight slot pinning + flush backpressure keep it correct;
  `dfkv_ram_*` metrics expose hit-rate + backpressure. See
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §5–6.
- **Versioned wire protocol** (v1/v2): v2 adds a request `seq` echoed in the reply;
  servers **dual-accept** both versions (no flag-day rolling upgrade). Client opt-in
  via `DFKV_WIRE_VERSION=2` (default v1). Block identity is **96-bit** (id + index
  from MD5) to make same-model hash collisions — a silent cross-key read — vanishingly
  unlikely. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §3–4.
- **Connection pooling + keep-alive** (TCP_NODELAY): ~250× lower latency vs dial-per-call.
- **Batch APIs** with concurrent fan-out across nodes (`BatchPut/Get/Exist`, C ABI + plugin).
- **Connect/IO timeouts + stale-connection retry**: a hung node fails fast, never hangs.
- **Observability** ([docs/METRICS.md](docs/METRICS.md)): opt-in embedded Prometheus
  `/metrics` on `dfkv_server` and `dfkv_mds` (`--metrics-port`); sampled op-latency
  histogram, eviction/error/per-disk/RDMA counters server-side; client-side counters
  (peer health, IO errors) via `dfkv_stats_snapshot` + a plugin poller. **Opt-in and
  off the datapath** — no `--metrics-port` ⇒ no listener, behavior unchanged.
  The three connectors (vLLM / LMCache / SGLang HiCache) can also **push** fleet
  metrics (ops/keys/bytes, op latency, per-peer latency) over OTLP to a central
  Collector → Grafana — opt-in via `DFKV_METRICS_ENABLED=1`, **zero-dependency stdlib
  exporter by default**; see [deploy/observability/CONNECTOR-USAGE.md](deploy/observability/CONNECTOR-USAGE.md)
  and [docs/METRICS.md](docs/METRICS.md) §3.4.
- **Dynamic membership**: MDS discovery (`dfkv_start_mds_discovery`) polls the MDS
  tier and rebuilds the weighted Ketama ring on each etcd-epoch change. Legacy
  `SetMembers()` hot-swap and `dfkv_refresh_members` (single-seed query) are still
  supported.
- **CLI tools**: `dfkv_smoke` (roundtrip check), `dfkvctl` — per-node ops
  (`put/get/exist/stat`) plus cluster views: `dfkvctl ring` (membership + ring vnode
  share) and `dfkvctl stat --all` (per-node metrics + cluster aggregate) via MDS.
- **RDMA transport** (gated `-DDFKV_WITH_RDMA=ON`, native libibverbs RC): device
  selected **by name** (`DFKV_RDMA_DEV=ib7s400p0`, comma-list = multi-rail), QP
  bootstrapped over a tiny TCP channel so the 400G data fabric needs no IP and may
  be separate from the IP network. **Automatic TCP fallback** when no device or
  `DFKV_RDMA` unset. Validated on 400G InfiniBand.
- **Zero-copy GET both ends**: the server reads the block straight into the send
  buffer; the client scatters the payload directly into the caller's buffer (e.g.
  a SGLang HiCache registered host page) — no intermediate copies.
- **Optional pipelining** (`DFKV_RDMA_DEPTH=K`): K requests in flight per connection.
  A network-latency hider, **not a throughput knob** — GET and PUT are both
  depth-flat (the per-connection serve loop is in-order; benchmarked GET ~1.24 GB/s at
  depth 1 == 32). The throughput levers are **multi-connection fan-out**
  (`batch_concurrency`) and **fewer/larger keys**. See `docs/datapath-perf-notes.md`.
- **NUMA-aware rail selection** (`DFKV_RDMA_NUMA=1`): pins buffers/serve-threads to
  the rail's NUMA node AND, with a multi-rail `DFKV_RDMA_DEV`, picks a NUMA-local
  rail per connection (falls back to round-robin over all rails when no local rail
  exists). Off by default; vendor-neutral (sysfs + `sched_getcpu`, no libnuma/CUDA).
- **HiCache v2** (PoolTransfer) for multi-pool models (Mamba/SWA/DeepSeek-V4).
- **Packaging**: CPack (deb/rpm/tgz) + Dockerfile; **graceful shutdown**; leveled logging.

## Status
TDD; **264 C++ ctest entries (default) / 288 (RDMA+io_uring) + Python plugin &
connector tests green**, 0 warnings, **ThreadSanitizer-clean**.
CI: gcc/clang build+test, TSan, RDMA datapath (Soft-RoCE loopback), RDMA compile-check, static-artifact build. License: Apache-2.0.
Architecture & design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Rollout: `docs/DEPLOY.md`.
