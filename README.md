# dfkv — distributed KV cache for SGLang HiCache

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
- **`python/dingofs_hicache.py`** — SGLang `HiCacheStorage` plugin loaded via
  `--hicache-storage-backend dynamic` (no SGLang fork). MLA: one packed-latent
  object per page, no tp_rank suffix, `backup_skip` (only tp_rank 0 writes).

## Design in one breath
SGLang HiCache (zero-copy v1) → `dingofs_hicache.py` (ctypes) → `libdfkv` client
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
python/     dingofs_hicache.py  (SGLang dynamic backend plugin)
tests/      gtest suites + tests/python (unittest + no-torch sglang shim)
docs/       DEPLOY.md (standalone rollout) · INTEGRATION.md (fuse into dingo-cache)
```

## Status
TDD, 7 rounds; **45 tests green** (39 C++ via ctest + 6 Python). License: Apache-2.0.
