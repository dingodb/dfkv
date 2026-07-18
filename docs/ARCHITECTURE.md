# dfkv Architecture

A tour of how dfkv is put together: the layers a request flows through, the
pluggable storage engines, the optional RAM hot tier, and the wire protocol.
For deployment see [DEPLOY.md](DEPLOY.md); for metrics see
[METRICS.md](METRICS.md); for engine connectors and the **client-side**
env/config reference see [CONNECTORS.md](CONNECTORS.md).

> **Client vs server config:** the storage engine (§5) and RAM tier (§6) are
> **server-side only** — a client needs no config for them and can't tell which a
> node runs. No client-side config is required for any v1.7.x feature (the
> 96-bit key, §3, is automatic). See [CONNECTORS.md](CONNECTORS.md) §1.

---

## 1. Layers

```
        LLM inference engine (SGLang HiCache / LMCache / vLLM)
                              │  thin adapter (ctypes / KVConnector)
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  libdfkv client  — key→96-bit id, Ketama route, header guard  │
   │                    Put / Get / Exist / Remove + batch/SG APIs  │
   └───────────────┬──────────────────────────────┬───────────────┘
        MDS discovery (etcd epoch)          transport: TCP or RDMA (RC)
                    │                               │  versioned wire frame
                    ▼                               ▼
   ┌──────────────────────┐        ┌──────────────────────────────────────┐
   │ dfkv_mds (+ etcd)    │        │  dfkv_server (cache node)             │
   │ membership directory │        │  ┌────────────────────────────────┐  │
   └──────────────────────┘        │  │ RamTier (opt) — write-through   │  │
                                    │  │  RAM arena, RDMA zero-copy GET  │  │
                                    │  └───────────────┬────────────────┘  │
                                    │      miss ▼      │ async flush        │
                                    │  ┌──────────────────────────────────┐│
                                    │  │ DiskCacheGroup — Ketama over N    ││
                                    │  │  disks, each a StoreEngine:       ││
                                    │  │   file (KVStore)  |  slab         ││
                                    │  └──────────────────────────────────┘│
                                    └──────────────────────────────────────┘
```

Distribution is **client-side consistent hashing**; there is no replication
(regenerable KV → a node loss is a miss → recompute). Membership is dynamic via
the MDS tier — see the README for the two-layer offline detection.

---

## 2. Request data path

**PUT** `client.Put(key, value)`
1. Client computes the block identity and wraps the value with a 48-byte
   `ValueHeader` (model / page-size / dtype / layer geometry guard), then routes
   to one node by Ketama over the key.
2. Server `ProcessRequest(kCache)` / `CacheDirect` (RDMA aligned): if the RAM
   tier is on, the value is **written through** to a RAM slot (synchronously
   visible) and flushed to disk in the background; otherwise it goes straight to
   the disk `StoreEngine`.

**GET** `client.Get(key, dst)`
1. Route + request. Server checks the RAM tier first (served from the arena, no
   disk); on a miss it reads from the disk engine. On RDMA the payload is
   scatter-sent zero-copy (from the arena MR, or straight from the O_DIRECT read
   buffer) into the caller's registered buffer.
2. The client verifies the `ValueHeader` geometry — a mismatch (wrong model /
   page-size / dtype / layer, or a stale/corrupt block) is treated as a **miss**
   (recompute), never silent wrong bytes.

---

## 3. Block identity (96-bit)

A block's identity is `BlockKey{id, index, size}`:

- `id` = MD5[0..8) of the key string (little-endian u64) — the ring routing hash
  input and the primary identity.
- `index` = MD5[8..12) (little-endian u32) — extends identity to **96 bits**.
- `size` = a fixed identity constant (never the payload length, so Put/Get/Exist
  build the same `Filename()`).

Why 96 bits: at the ~1e9 lifetime-write scale of a 5 TiB × N-node ring, a 64-bit
id alone has a few-percent birthday-collision probability — and a collision is
**not a clean miss but a silent cross-key read** (the geometry header only checks
model/page/dtype/layer, so two same-model pages that collide pass validation).
Filling the previously-always-zero `index` with more hash bits cuts collision
probability by 2³² with no wire or storage change; `id` is unchanged so routing
is unaffected.

---

## 4. Wire protocol

Fixed-prefix binary frames (`src/transport/wire.h`), protocol version 1:

| | request | response |
|-|---------|----------|
| prefix bytes | 42 | 10 |

Both prefixes start with a 1-byte protocol version so a mixed-version deploy
**fails fast** (an unknown version is rejected and the connection dropped)
instead of mis-parsing. Replies are paired to requests by strict FIFO order —
on TCP that order is the stream's; on RDMA the RC transport guarantees reliable
in-order delivery in hardware.

> History: v1.7.0/v1.7.1 shipped an experimental opt-in "wire v2" (a request
> `seq` echoed in the reply, `DFKV_WIRE_VERSION=2`). It was **removed in
> v1.7.2** as speculative: the RDMA data path — where production traffic runs —
> already gets ordering + ICRC from RC hardware and never supported v2 (its
> zero-copy receive posts the scatter split before the version is known), and no
> deployment used it on TCP. If out-of-order replies are ever needed, a
> request-id echo will be designed for that purpose then.

---

## 5. Storage engine layer

`DiskCacheGroup` routes a block to one disk (Ketama over `BlockKey.Filename()`)
and holds one `StoreEngine` per disk. The backend is selected by
`--store-engine` / `DFKV_STORE_ENGINE` (default `file`):

```
StoreEngine (interface): Cache / CacheDirect / Range / RangeInto /
                         RangeDirect / RangeDirectPrep / IsCached / Remove + stats
   ├── file  →  KVStore        (one file per block; the original engine)
   └── slab  →  DiskSlabStore  (extent files + slots.tbl; the rework)
```

### 5a. `file` engine — KVStore (default)

One file per block under `blocks/<bucket>/…`, O_DIRECT read/write, per-shard
`shared_mutex` index + CLOCK-second-chance LRU. Battle-tested; the default. Its
"one file per block" geometry is the root of several operational hazards (tmp
leak, ENOSPC dead-end, unbounded inode growth, lock-held unlink, open-per-GET),
which the slab engine removes.

### 5b. `slab` engine — DiskSlabStore (opt-in)

A fixed pool of pre-allocated **extent files**, each bound on demand to one
**size class** and carved into uniform slots by the media-agnostic
`SlabAllocator`:

- **SlabAllocator** — owns slot *layout* (which extent, which offset), not bytes.
  Size-class slab + per-class CLOCK eviction + pin refcount. `slot_granularity`
  bounds the per-extent slot count. Reused by the RAM tier (§6).
- **DiskSlabStore** — maps a key's slot to an extent-file offset (buffered I/O;
  extent fds stay resident, no open-per-GET) and records `{key → slot}` in a
  compact 64-byte-per-slot **`slots.tbl`**. On restart the index **rebuilds from
  `slots.tbl`**, keeping cache warmth across a rolling upgrade without per-block
  file churn.

**Crash safety**: every `slots.tbl` record is CRC32-checked, so a torn record
reads as free (its key becomes a clean miss = recompute, never corruption). A
meta magic mismatch is refused; an extent/granularity config mismatch re-inits
fresh. Eviction leaves a freed slot's record as-is (a slot's record always
reflects its last occupant), so an unreused evicted slot merely "resurrects" its
still-valid content-addressed key on restart.

**When to use slab**: it is a strategic replacement for the file engine on nodes
where the per-block-file hazards bite. It is **off by default**; switching a node
needs a clean-disk cold start (a separate ops migration).

---

## 6. RAM hot tier (P3, opt-in)

A COLD dfkv load is disk-bound (~480 MB/s O_DIRECT) and dominates PD-decode TTFT.
The RAM tier fronts the disk with a pre-registered RAM arena so a PD-warm GET is
served straight from RAM over RDMA — no open, no pread, no disk. Enabled with
`DFKV_RAM_TIER=1` (`DFKV_RAM_TIER_BYTES` sizes the arena; default off).

**Write-through**: `Put` copies the value into a RAM slot (via a `SlabAllocator`
over the arena), makes it **synchronously visible** (read-after-write), and
enqueues a background flush to the disk engine.

**State machine = allocator pin refcount.** The slot lifecycle maps directly onto
the allocator's pin count — this is why the allocator was built media-agnostic:

| state | pin holders | evictable? |
|-------|-------------|------------|
| RAM_ONLY (flush pending) | flush-pin | no |
| in-flight (RDMA send reading arena) | send-pin | no |
| DURABLE & idle | none (refcount 0) | **yes** |

- **flush-pin** — taken on `Put`, released when the async flush reaches disk.
- **send-pin** — taken on `GetPrep` (an RDMA send reads the shared arena in
  place), released on the send completion (`IBV_WC_SEND`).

**RDMA zero-copy serve**: the arena is registered once as a pool MR on each
connection's PD (`RegisterMemory`). On a RAM hit the serve loop scatter-sends
`[response | arena bytes]` straight from the arena MR and records the send-pin's
release token in `release_on_send[]` (mirrors the existing `rearm_on_send[]`);
the pin is released when that send's `IBV_WC_SEND` fires. Connection teardown
releases any outstanding pins, so a torn-down connection never leaks a pinned
slot. When the RAM tier is off, none of this path is wired and the serve loop is
byte-identical to before.

**Backpressure & durability**: if the arena fills with non-evictable
(flush-pending / in-flight) slots, `Put` declines and the caller takes the normal
synchronous disk write — never blocks, never breaks read-after-write. Only a
DURABLE slot may be evicted, so an eviction never loses the sole copy. RAM is
volatile but dfkv is a cache (a miss = recompute), so a crash costs only a
recompute of the unflushed tail.

Observability: `dfkv_ram_hit/miss/put/put_bypass/flushed/flush_dropped/
evictions_total` + `ram_objects` / `ram_flush_backlog` (emitted only when
enabled) — see [METRICS.md](METRICS.md).

---

## 7. Configuration matrix (feature defaults)

Every subsystem added by the storage/protocol rework is **off by default** —
a stock `v1.7.0` node behaves exactly like `v1.6.x`.

| Feature | Enable with | Default | Notes |
|---------|-------------|---------|-------|
| slab storage engine | `--store-engine=slab` / `DFKV_STORE_ENGINE=slab` | `file` | needs clean-disk cold start |
| RAM hot tier | `DFKV_RAM_TIER=1` (+ `DFKV_RAM_TIER_BYTES`) | off | write-through + RDMA zero-copy GET |
| RAM tier lock shards | `--ram-tier-shards` / `DFKV_RAM_TIER_SHARDS` | 8 | 1-64; per-shard lock is the >8-connection concurrency ceiling; auto-halved while a shard would hold <32 extents |
| RDMA transport | build `-DDFKV_WITH_RDMA=ON`, `DFKV_RDMA=1` | TCP | device by name `DFKV_RDMA_DEV` |
| io_uring async GET | build `-DDFKV_WITH_URING`, `DFKV_SERVER_URING=1` | off | disk-read path only |

---

## 8. Source map

```
src/common/     ValueHeader, BlockKey, Status — portable shared types
src/transport/  wire.h (wire frames) · tcp_transport · rdma_transport / rdma_verbs
src/cache/      store_engine.h (interface) · kv_store (file) ·
                slab_allocator + disk_slab_store (slab) · ram_tier (RAM hot tier) ·
                disk_cache_group (per-disk routing) · kv_node_server · rdma_server ·
                dfkv_server_main
src/client/     kv_client + C ABI · key_map (96-bit identity)
src/mds/        membership service + dfkv_mds
src/tools/      dfkvctl / dfkv_smoke / dfkv_bench
```
