# dfkv — distributed KV cache for SGLang HiCache

[![CI](https://github.com/ketor/dfkv/actions/workflows/ci.yml/badge.svg)](https://github.com/ketor/dfkv/actions/workflows/ci.yml)

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
- **`libdfkv.so`** — C ABI client (key→consistent-hash routing, value header with
  CRC + model/page/dtype/layer geometry guard, Put/Get/Exist).
- **`python/dfkv_hicache.py`** — SGLang `HiCacheStorage` plugin loaded via
  `--hicache-storage-backend dynamic` (no SGLang fork). MLA: one packed-latent
  object per page, no tp_rank suffix, `backup_skip` (only tp_rank 0 writes).

## Design in one breath
SGLang HiCache (zero-copy v1) → `dfkv_hicache.py` (ctypes) → `libdfkv` client
(Ketama route + header wrap/verify) → TCP → `dfkv_server` (DiskCacheGroup over
N NVMe, LRU). Membership is a **static list** of node addresses (no MDS).
Distributed = client-side consistent hashing; no replication (regenerable KV →
node loss = miss → recompute). RDMA is intentionally not included yet.

## Build & test (no GPU / no RDMA needed)
```bash
cmake -S . -B build            # add -DDFKV_STATIC_LIBSTDCXX=ON for portable binaries
cmake --build build -j
ctest --test-dir build --output-on-failure   # C++ gtests + the Python plugin test
```
Artifacts: `build/dfkv_server`, `build/libdfkv.so`.

## Run a cluster
```bash
# on each node (one or more NVMe dirs; --cap is total, split across disks)
dfkv_server --dir /mnt/disk1/dfkv,/mnt/disk2/dfkv,/mnt/disk3/dfkv --port 12000 --cap 6597069766656
```
Point the SGLang plugin at all nodes via `members="n1=ip:12000,n2=ip:12000,..."`.
Full rollout runbook: `docs/DEPLOY.md`.

## Layout
```
src/        portable C++ core (headers + .cc) + dfkv_server_main.cc
python/     dfkv_hicache.py  (SGLang dynamic backend plugin)
tests/      gtest suites + tests/python (unittest + no-torch sglang shim)
docs/       DEPLOY.md (standalone rollout) · INTEGRATION.md (fuse into dingo-cache)
```

## Operability & performance features
- **Connection pooling + keep-alive** (TCP_NODELAY): ~250× lower latency vs dial-per-call.
- **Batch APIs** with concurrent fan-out across nodes (`BatchPut/Get/Exist`, C ABI + plugin).
- **Connect/IO timeouts + stale-connection retry**: a hung node fails fast, never hangs.
- **Metrics**: server counters + Prometheus text (`dfkvctl stat <node>` / `kStats` op).
- **Dynamic membership**: `SetMembers()` hot-swaps the ring (no client restart).
- **CLI tools**: `dfkv_smoke` (roundtrip check), `dfkvctl` (put/get/exist/stat).
- **RDMA transport** (gated `-DDFKV_WITH_RDMA=ON`, native libibverbs RC): device
  selected **by name** (`DFKV_RDMA_DEV=ib7s400p0`, comma-list = multi-rail), QP
  bootstrapped over a tiny TCP channel so the 400G data fabric needs no IP and may
  be separate from the IP network. **Automatic TCP fallback** (`DFKV_RDMA=1`);
  `DFKV_RDMA_CM=1` selects the librdmacm/IP-routed fallback. Validated on hd03 400G.
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
