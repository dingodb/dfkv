# dfkv — distributed KV cache for SGLang HiCache

[![CI](https://github.com/dingodb/dfkv/actions/workflows/ci.yml/badge.svg)](https://github.com/dingodb/dfkv/actions/workflows/ci.yml)

A small, **self-contained** distributed key-value cache that plugs into SGLang's
HiCache as its L3 external KV store. Built to pool GPU-node NVMe SSDs into a
shared, large-capacity KVCache pool for LLM inference (e.g. GLM-5.1 / MLA),
**without any DingoFS / brpc / MDS / S3-RADOS dependency** — it runs on its own.

> Origin: extracted from the DingoFS branch `feat/kvcache-sglang`
> (`src/cache/kvclient`). The portable core has zero coupling to DingoFS, so it
> lives here as an independent repo. To instead fuse these semantics into the
> production `dingo-cache` (brpc + MDS), see `docs/INTEGRATION.md`.

## What it is
- **`dfkv_server`** — a cache-node daemon. Disk + LRU, **cache-only** (a miss is
  a clean NotFound; no object-store fallback), synchronous durable-visible writes.
  Supports **multiple NVMe SSDs per node** (`--dir d1,d2,d3`, intra-node Ketama).
  With `--mds`, `--group`, `--id`, `--advertise`, `--weight` it registers into the
  MDS tier; the old static `--members` flag has been removed.
- **`dfkv_mds`** — stateless Membership Directory Service daemon. Flags:
  `--listen <port>` and `--etcd <host:port>` (default `127.0.0.1:2379`). The only
  etcd client in the system; holds each node's etcd lease on its behalf. Deploy as
  N replicas — no load-balancer needed; nodes and clients each pick any reachable
  MDS and fail over automatically.
- **`libdfkv.so`** — C ABI client (key→consistent-hash routing, value header with
  CRC + model/page/dtype/layer geometry guard, Put/Get/Exist).
- **`python/dfkv_hicache.py`** — SGLang `HiCacheStorage` plugin loaded via
  `--hicache-storage-backend dynamic` (no SGLang fork). MLA: one packed-latent
  object per page, no tp_rank suffix, `backup_skip` (only tp_rank 0 writes).

## Design in one breath
SGLang HiCache (zero-copy v1) → `dfkv_hicache.py` (ctypes) → `libdfkv` client
(Ketama route + header wrap/verify) → TCP/RDMA → `dfkv_server` (DiskCacheGroup
over N NVMe, LRU). Distributed = client-side consistent hashing; no replication
(regenerable KV → node loss = miss → recompute).

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
Full rollout runbook (etcd + MDS + systemd units): `docs/DEPLOY.md`.

## Layout
```
src/        portable C++ core (headers + .cc) + dfkv_server_main.cc + dfkv_mds_main.cc
python/     dfkv_hicache.py  (SGLang dynamic backend plugin)
integration/lmcache/  dfkv_connector  (LMCache RemoteConnector, ctypes over libdfkv.so)
tests/      gtest suites + tests/python (unittest + no-torch sglang shim)
docs/       DEPLOY.md (standalone rollout) · INTEGRATION.md (fuse into dingo-cache)
docs/hicache/  SGLang HiCache plugin docs (access_log, module README)
docs/lmcache/  LMCache connector docs (DESIGN · IMPLEMENTATION · DEPLOY)
```

## Engine integrations
- **SGLang HiCache**: `python/dfkv_hicache.py` — see `docs/hicache/` and `docs/DEPLOY.md`.
- **LMCache**: `integration/lmcache/` (`dfkv_connector`) — see `docs/lmcache/DESIGN.md`,
  `docs/lmcache/IMPLEMENTATION.md`, `docs/lmcache/DEPLOY.md`.

## Operability & performance features
- **Connection pooling + keep-alive** (TCP_NODELAY): ~250× lower latency vs dial-per-call.
- **Batch APIs** with concurrent fan-out across nodes (`BatchPut/Get/Exist`, C ABI + plugin).
- **Connect/IO timeouts + stale-connection retry**: a hung node fails fast, never hangs.
- **Metrics**: server counters + Prometheus text (`dfkvctl stat <node>` / `kStats` op).
- **Dynamic membership**: MDS discovery (`dfkv_start_mds_discovery`) polls the MDS
  tier and rebuilds the weighted Ketama ring on each etcd-epoch change. Legacy
  `SetMembers()` hot-swap and `dfkv_refresh_members` (single-seed query) are still
  supported.
- **CLI tools**: `dfkv_smoke` (roundtrip check), `dfkvctl` (put/get/exist/stat).
- **RDMA transport** (gated `-DDFKV_WITH_RDMA=ON`, native libibverbs RC): device
  selected **by name** (`DFKV_RDMA_DEV=ib7s400p0`, comma-list = multi-rail), QP
  bootstrapped over a tiny TCP channel so the 400G data fabric needs no IP and may
  be separate from the IP network. **Automatic TCP fallback** when no device or
  `DFKV_RDMA` unset. Validated on 400G InfiniBand.
- **Zero-copy GET both ends**: the server reads the block straight into the send
  buffer; the client scatters the payload directly into the caller's buffer (e.g.
  a SGLang HiCache registered host page) — no intermediate copies.
- **Optional pipelining** (`DFKV_RDMA_DEPTH=K`): K requests in flight per connection.
- **HiCache v2** (PoolTransfer) for multi-pool models (Mamba/SWA/DeepSeek-V4).
- **Packaging**: CPack (deb/rpm/tgz) + Dockerfile; **graceful shutdown**; leveled logging.

## Status
TDD; **53 C++ ctest entries + 7 Python tests green**, 0 warnings, **ThreadSanitizer-clean**.
CI: gcc/clang build+test, TSan, RDMA compile-check, static-artifact build. License: Apache-2.0.
See `docs/DEPLOY.md` (rollout) and the round report in the ai_david KB.
