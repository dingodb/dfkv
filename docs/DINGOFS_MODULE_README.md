# DingoFS KV cache for SGLang HiCache (`src/cache/kvclient`)

Branch `feat/kvcache-sglang`. A KV-by-hash cache path so SGLang's HiCache can use
DingoFS as its L3 external KV store (GLM-5.1 / MLA). Two layers:

1. **Portable semantic core (this dir)** — no brpc/MDS deps, builds & is fully
   unit/integration tested on a plain Linux box (gcc + cmake, no GPU/RDMA):
   - `value_header.h` — 48B header (model/page/dtype/layer geometry; no payload
     checksum as of v3 — integrity is left to RC RDMA/RoCE ICRC, keeping the
     datapath zero-touch); geometry mismatch ⇒ safe MISS (recompute), never wrong
     bytes. MLA latent is TP-layout invariant ⇒ tp_rank not part of the match.
   - `kv_types.h` / `key_map.h` — sglang page-hash → `BlockKey`. **F1 fix:**
     `BlockKey.size` is a fixed constant, so Put/Get/Exist build the identical
     identity key and route identically (payload length lives in the header).
   - `con_hash.{h,cc}` (+ `md5.h`) — Ketama ring (client-side routing).
   - `kv_store.{h,cc}` — single-disk local store: disk + LRU + **cache-only / no
     S3** (miss = clean NotFound); `Cache()` is synchronous & durable-visible.
   - `disk_cache_group.{h,cc}` — **multi-NVMe per node** (like dingo-cache
     `--cache_dir=d1,d2,d3`): one `KVStore` per disk, intra-node Ketama routes a
     block to one disk, total capacity split across disks. `dfkv_server --dir`
     accepts comma-separated paths.
   - `transport.h` + `tcp_transport.{h,cc}` — transport abstraction + a real TCP
     loopback impl used by the standalone harness.
   - `kv_node_server.{h,cc}` + `dfkv_server_main.cc` — a cache-node daemon
     (`dfkv_server`) over the wire protocol.
   - `kv_client.{h,cc}` — routes (conhash) → wraps value w/ header → Put
     (SyncCache) / Get (verify header geometry) / Exist.
   - `dfkv_c_api.{h,cc}` — C ABI (`libdfkv.so`) for the Python plugin (ctypes).
   - `python/dingofs_hicache.py` — the SGLang `HiCacheStorage` plugin
     (`--hicache-storage-backend dynamic`). MLA: single object/page, no rank
     suffix, `backup_skip` (only tp_rank 0 writes).

2. **Full-build brpc integration (see `INTEGRATION.md`)** — wires the same core
   into the production `dingo-cache` (brpc `BlockCacheService` Exist/SyncCache
   handlers, `--kv_cache_only`, `NullStorageClientPool`, a `DingofsTransport`
   that routes via the existing `RemoteBlockCache`/MDS PeerGroup, and a nanobind
   module). RDMA data-path is intentionally deferred (separate task).

## Build & test the portable core (no GPU)
```
cmake -S /home/ketor/dfkv-dev -B /home/ketor/dfkv-dev/build && cmake --build /home/ketor/dfkv-dev/build
ctest --test-dir /home/ketor/dfkv-dev/build --output-on-failure   # 33 C++ tests
cd test/unit/cache/kvclient/python && python3 -m unittest test_dingofs_hicache  # 6 plugin tests
```

## Deploy for pre-prod FUNCTIONAL validation (no GPU needed to test the path)
- Run one `dfkv_server` per node (`--dir <nvme path> --port <p> --cap <bytes>`).
- Launch SGLang with `--hicache-storage-backend dynamic` +
  `--hicache-storage-backend-extra-config '{"backend_name":"dingofs",
  "module_path":"dingofs_hicache","class_name":"DingoFSHiCache","interface_v1":1,
  "members":"n1=ip:port,...","model_hash":...,"page_size":64,"dtype_tag":...,
  "layer_num":78,"head_num":1,"head_dim":576}'` and `PYTHONPATH`/`DFKV_LIB` set.
