# dfkv RDMA datapath — perf notes & investigated-but-deferred items

## Current datapath (as of 187d241)

- RC **two-sided SEND/RECV**, two-end **zero-copy** (server reads disk straight into
  the registered send buffer; client SEND/RECV scatters straight into the caller's
  registered HiCache page). One round-trip per key.
- Batch ops (`CacheFrom` for PUT, `RangeInto` for GET) take N keys per call and
  **pipeline** them in windows of `ep.depth()` (`DFKV_RDMA_DEPTH`): up to `depth`
  requests in flight on one connection, hiding per-op latency. Each key is still its
  own wire SEND/RECV.

## #2 — coalescing contiguous pages into larger RDMA transfers: investigated, deferred

**Question:** should we merge multiple KV pages into fewer, larger RDMA transfers to
beat a small-object penalty?

**Finding — the premise mostly doesn't apply to MLA:**
- GLM-5.1 is MLA → **one object per page ≈ 2.6 MiB**. That is already in the bench's
  large-transfer sweet spot (3-node bench: 2 MiB→15.5, 4 MiB→23.5 GB/s GET). The
  per-page wire transfer is NOT small, so there is no small-object penalty to fix.
- Coalescing N keys into one wire message is **not free**: each key is a separate
  server-side object/file with its own value header, so it needs a new multi-key
  wire op (server reads N files into N regions of one buffer → one big RDMA) plus
  larger registered direct buffers (`DFKV_RDMA_MAX_PAYLOAD_BYTES`, default 64 MiB,
  while ordinary control buffers stay small). Cost is real; gain on already-large, bandwidth-bound transfers is
  marginal.
- The genuine latency lever for multi-key batches is **pipelining (#1, `DFKV_RDMA_DEPTH`)**,
  which keeps `depth` pages in flight without any protocol change. That is implemented.
- (MHA models split a page into `_k`+`_v` = 2 sub-objects/page → there coalescing the
  pair could halve ops. Not relevant to MLA/GLM-5.1.)

**Decision:** do not add key-coalescing now. Use depth pipelining (#1). Revisit only
if a future model is MHA with small pages (small-object, message-rate-bound).

## #1 — write pipelining: measured on an 8x400G IB testbed (depth flat; multi-connection is the lever)

Empirical (8x400G IB testbed, 3-node, 1 MiB PUT, single writer thread, batch 64):

| knob | PUT GB/s |
|------|----------|
| `DFKV_RDMA_DEPTH` 1 / 8 / 16 (1 connection) | 2.50 / 2.55 / 2.50 — **flat** |
| `batch_concurrency` 1 / 8 / 16 | 1.34 / **3.35** / 3.28 — **2.5x at 8, saturates** |

- **Depth pipelining does NOT raise PUT.** The server's per-connection serve loop
  processes requests serially (each does the O_DIRECT disk write inline), so a
  client that pipelines `depth` PUTs just queues them at the server. Depth is kept
  as an opt-in knob (`rdma_depth` extra_config / `DFKV_RDMA_DEPTH`) — it can help on
  a network-latency-bound link — but on a disk-bound path it is flat.
- **Multi-connection fan-out is the real write lever**: splitting a batch across N
  connections hits N parallel server serve threads. `batch_concurrency` **defaults
  to 8** in the client, so the SGLang plugin's single-writer (MLA rank0) path
  **already parallelizes writes 8-way** with no extra config. It saturates at ~8 on
  this hardware, so there is no large untapped win here; the write ceiling is the
  single-rank r0 path + server serial-per-connection writes, already mitigated by
  the 8-way fan-out.

## One-sided RDMA WRITE/READ vs two-sided SEND/RECV

A colleague noted one-sided (READ/WRITE) often beats two-sided (SEND/RECV). True in
general (no remote CPU in the data path, no RECV-WQE matching → lower latency, higher
small-message IOPS). For dfkv specifically:

- Values live on **NVMe**, not pre-registered RAM, so a pure client-initiated one-sided
  READ can't read the value directly — it needs a round-trip to stage the block first
  (more trips than today's single SEND/RECV for cold data).
- The realistic form is **server-initiated RDMA WRITE (or WRITE_WITH_IMM) of the GET
  response into the client's registered HiCache buffer** (request carries the client
  buffer addr+rkey), and client WRITE for PUT payload + small SEND to commit.
- dfkv is **already two-end zero-copy** with SEND/RECV, so the big copy→zero-copy win is
  banked. The marginal one-sided gain is **latency/jitter + freed remote CPU**, which is
  modest on 2.6 MiB **bandwidth-bound** transfers (large for IOPS-bound small objects).

**Decision:** candidate for a future A/B (WRITE_WITH_IMM GET response prototype, measured
on real RDMA hardware), but **not a blind rewrite** — the data we have (large, bandwidth-bound,
already zero-copy) does not justify rewriting the datapath on "should be faster". Measure
first.
