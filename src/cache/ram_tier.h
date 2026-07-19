/* RamTier — a write-through RAM hot tier that fronts the disk cache.
 *
 * Motivation (P3): a COLD dfkv load is disk-bound (~480 MB/s O_DIRECT), which
 * dominates PD decode TTFT. For PD-warm GETs, serving the KV straight from a
 * pre-registered RAM arena over RDMA -- no open, no pread, no disk -- removes
 * that bottleneck. PUT is write-through: the value is copied into a RAM slot and
 * made visible synchronously (read-after-write), then flushed to disk in the
 * background for durability and capacity overflow.
 *
 * Slot lifecycle reuses SlabAllocator (size-class + CLOCK + pin); the design's
 * state machine maps directly onto the allocator's pin refcount:
 *   - flush-pin: taken on Put, released when the async flush reaches disk. While
 *     held the slot is RAM_ONLY and must not be evicted (no disk copy yet).
 *   - send-pin: taken on GetPrep (an RDMA send reads the shared arena in place),
 *     released on send completion.
 *   refcount == 0  <=>  DURABLE && no send in flight  <=>  evictable.
 * SlabAllocator only ever evicts unpinned slots, so this is exactly the design's
 * "only a flushed, not-in-flight slot may be reclaimed" rule (gaps 10.1 + 4.3).
 *
 * Backpressure (gap 10.3): if the arena is full of non-evictable slots (flush
 * fell behind), Put returns false and the caller falls back to the normal
 * synchronous disk write -- never blocks, never fails read-after-write.
 * Alignment (gap 10.4): the arena base is posix_memalign(4096) and slot sizes
 * are 4096-multiples, so a slot address is O_DIRECT-aligned for the flusher.
 *
 * SHARDING: every op used to funnel through ONE mutex (index/alloc/flushq/pins)
 * — with 16 serve threads + 16 flush workers the single lock was the measured
 * write-path serialization point (each RAM op also took the allocator's lock
 * UNDER it: two locks per op). The arena is still one allocation (one reg_mr),
 * but it is partitioned into DFKV_RAM_TIER_SHARDS (default 8) independent
 * shards — own SlabAllocator, mutex, index, flush queue and flush workers —
 * routed by hash(key) so a key's whole lifecycle stays on one shard. Hot paths
 * only contend within a shard. Backpressure is per shard (a full shard
 * declines while others admit); the hash keeps that rare and unbiased.
 *
 * Off by default (DFKV_RAM_TIER=0); this class is the standalone core, wired in
 * by later PRs. RDMA MR registration is deferred (SetArenaMr) so this stays
 * hermetically testable without libibverbs. */
#ifndef DFKV_RAM_TIER_H_
#define DFKV_RAM_TIER_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/slab_allocator.h"
#include "common/kv_types.h"

namespace dfkv {

class RamTier {
 public:
  struct Options {
    uint64_t bytes = (4ull << 30);       // arena size (DFKV_RAM_TIER_BYTES)
    uint32_t slot_granularity = 4096;    // slot quantum (>= O_DIRECT align)
    uint32_t flush_retries = 3;          // per-item flush attempts before drop
    // Flush worker threads draining the shard queues (distributed round-robin
    // over the shards, at least one per shard). One DIO stream sustains only
    // ~1.6-3 GB/s -- far below the arena's ingest -- so the drain rate, not
    // the arena size, decides when Put backpressure kicks in. The server
    // defaults this to 4x the disk count, capped at 16.
    uint32_t flush_threads = 1;
    // Background free-slot reclaimer cadence (ms; 0 = off). Keeps demand-driven
    // free-slot headroom per class so Put's allocator call stays pop-free-slot;
    // without it a full arena runs the CLOCK eviction sweep inline under the
    // shard lock on every admission.
    uint32_t reclaim_interval_ms = 10;
  };

  // Persists a slot to disk. Returns true on success (slot -> DURABLE). Called
  // off the hot path by the background flusher; `data` points into the arena
  // (4 KiB-aligned) and `cap` is the slot's full size, so the sink may take the
  // O_DIRECT CacheDirect path -- zeroing padding bytes inside [len, cap) is fine
  // (the slot is this key's; readers only ever see [0, len)). A buffered sink
  // can ignore cap.
  using FlushFn =
      std::function<bool(const BlockKey& key, char* data, size_t len, size_t cap)>;
  // Batched flush sink (optional). One arena slot per item, same contract as
  // FlushFn; the server wires this to the disk group's CacheDirectBatch so a
  // worker's whole dequeue rides one lock-amortized (and, with uring, one-
  // submit) store visit -- small-object flushing is IOPS-bound at one
  // synchronous DIO write per key otherwise. Unset = per-item FlushFn.
  struct FlushItem { BlockKey key; char* data; size_t len; size_t cap; };
  using FlushBatchFn = std::function<std::vector<bool>(const std::vector<FlushItem>&)>;
  void set_flush_batch(FlushBatchFn fn) { flush_batch_ = std::move(fn); }

  // A pinned arena location handed to the RDMA send path. `token` must be passed
  // back to Release() once the send completes (releases the send-pin).
  struct Hit {
    const char* ptr = nullptr;
    size_t len = 0;
    void* mr = nullptr;      // arena MR (nullptr until SetArenaMr); for RDMA lkey
    uint64_t token = 0;      // opaque; pass to Release()
  };

  RamTier(Options opt, FlushFn flush);
  ~RamTier();                // stops the flushers, joins

  RamTier(const RamTier&) = delete;
  RamTier& operator=(const RamTier&) = delete;

  bool ok() const { return arena_ != nullptr; }

  // Write-through: copy into a RAM slot + make visible + enqueue flush. Returns
  // true if accepted into RAM (caller may skip the sync disk write; the flusher
  // will persist). Returns false on backpressure (shard full of non-evictable
  // slots) or if len exceeds the shard -- caller then does the normal disk write.
  bool Put(const BlockKey& key, const void* data, size_t len);

  // Read-promotion entry: install a value that is ALREADY durable on disk
  // (a coalesced cold read with fan-in evidence). Same allocation/index logic
  // as Put, but the slot is born durable — never enqueued for flush, never
  // holds a flush-pin, so it costs zero DIO write bandwidth and is evictable
  // by CLOCK the moment no send is in flight. Best-effort: a full shard
  // declines (false) and the promotion is silently skipped; promoted slots can
  // never crowd out the write path's flush resources.
  bool PutDurable(const BlockKey& key, const void* data, size_t len);

  // On hit, pins the slot (send-pin) and returns its arena location. The caller
  // MUST call Release(hit.token) when the RDMA send completes. Miss => false.
  bool GetPrep(const BlockKey& key, uint64_t offset, uint64_t length, Hit* out);
  void Release(uint64_t token);

  bool Contains(const BlockKey& key) const;
  bool Remove(const BlockKey& key);  // drop from RAM (cache); true if present

  // Explicit miss accounting for GET paths that consult the tier via Contains()
  // instead of GetPrep(). The uring prep path partitions keys up front
  // (RAM-resident -> sync arena serve, absent -> async disk read); its absent
  // branch never reaches GetPrep, so without this the dominant read path
  // under-reports misses to zero. Contains() itself must NOT count: the exist
  // path probes it too, and exist probes are not GETs.
  void CountMiss() { misses_.fetch_add(1, std::memory_order_relaxed); }

  // Registers the arena's RDMA MR (set once by the RDMA server after reg_mr).
  void SetArenaMr(void* mr);
  char* arena() const { return arena_; }
  uint64_t arena_bytes() const { return opt_.bytes; }
  size_t shards() const { return shards_.size(); }  // test/diagnostic

  // metrics (relaxed)
  uint64_t Hits() const { return hits_.load(std::memory_order_relaxed); }
  uint64_t Misses() const { return misses_.load(std::memory_order_relaxed); }
  uint64_t Puts() const { return puts_.load(std::memory_order_relaxed); }
  uint64_t Promoted() const { return promotes_.load(std::memory_order_relaxed); }
  uint64_t PutBypass() const { return put_bypass_.load(std::memory_order_relaxed); }
  uint64_t Flushed() const { return flushed_.load(std::memory_order_relaxed); }
  uint64_t FlushDropped() const { return flush_dropped_.load(std::memory_order_relaxed); }
  uint64_t Reclaimed() const { return reclaimed_.load(std::memory_order_relaxed); }
  uint64_t Rebalanced() const { return rebalanced_.load(std::memory_order_relaxed); }
  uint64_t Evictions() const;
  uint64_t UsedBytes() const;   // resident slot bytes
  size_t Count() const;
  size_t FlushBacklog() const;  // queued, not-yet-durable items

 private:
  struct Entry {
    uint64_t offset = 0;   // byte offset into arena_ (GLOBAL, not shard-local)
    uint32_t len = 0;      // payload length
    uint32_t cap = 0;      // slot size (>= len, 4 KiB multiple) -- flusher's DIO cap
    bool durable = false;  // flushed to disk (metrics; eviction uses alloc pin)
  };
  struct QItem { std::string fn; BlockKey key; uint32_t tries = 0; };

  // One independent slice of the tier. All state a key's lifecycle touches
  // (allocator, index, writing set, send-pins, flush queue) lives here, under
  // this shard's mutex only. base_off is the shard's byte offset into arena_.
  struct Shard {
    uint64_t base_off = 0;
    std::unique_ptr<SlabAllocator> alloc;
    mutable std::mutex mu;
    std::unordered_map<std::string, Entry> index;
    std::unordered_set<std::string> writing;
    std::unordered_map<uint64_t, std::string> pinned;  // token -> fn (send-pins)
    uint64_t next_token = 1;
    std::deque<QItem> flushq;
    std::condition_variable cv;
    bool stop = false;
    std::vector<uint64_t> reclaim_last_puts;  // reclaim-thread-local snapshot
  };

  Shard& ShardFor(const std::string& fn) {
    return *shards_[std::hash<std::string>{}(fn) % shards_.size()];
  }
  const Shard& ShardFor(const std::string& fn) const {
    return *shards_[std::hash<std::string>{}(fn) % shards_.size()];
  }

  void FlushLoop(Shard& s);
  // Max items one flush worker drains per store visit. Big enough to amortize
  // the two store-lock hops + reach a useful uring submit width, small enough
  // that a batch's slots stay flush-pinned only briefly.
  static constexpr size_t kFlushBatchMax = 16;
  // Send-pin tokens encode their shard in the low bits so Release() can route
  // without a global map. 6 bits = up to 64 shards (raised from 4/16 in phase
  // 10: a clean GET scaling sweep peaked at 8 threads with the default 8 shards
  // then DEGRADED — the per-shard lock is the >8-connection concurrency ceiling
  // for both read and write. Raising the cap lets read/write-heavy multi-client
  // deployments set DFKV_RAM_TIER_SHARDS past 16; the default stays 8).
  static constexpr int kTokenShardBits = 6;
  static constexpr size_t kMaxShards = 1u << kTokenShardBits;
  void ReclaimTick(Shard& s, size_t shard_idx);
  // Rebalance rate cap: extents moved per tick per hot class (32 MiB default
  // extents; converges in well under a second at the 10 ms tick).
  static constexpr size_t kGrowExtentsPerTick = 8;
  static void DropLocked(Shard& s, const std::string& fn);  // s.mu held

  Options opt_;
  FlushFn flush_;
  FlushBatchFn flush_batch_;
  char* arena_ = nullptr;
  std::atomic<void*> arena_mr_{nullptr};
  uint64_t extent_bytes_ = 0;   // SlabAllocator extent size; global arena offset
                                // of a slot = shard.base_off + ref.extent *
                                // extent_bytes_ + ref.offset
  std::vector<std::unique_ptr<Shard>> shards_;
  std::vector<std::thread> flushers_;
  // Background free-slot reclaimer (own cv/mutex so shutdown never races the
  // flushers' cv protocol). One thread sweeps all shards.
  std::thread reclaim_thread_;
  std::condition_variable reclaim_cv_;
  std::mutex reclaim_mu_;
  bool reclaim_stop_ = false;

  std::atomic<uint64_t> hits_{0}, misses_{0}, puts_{0}, put_bypass_{0}, promotes_{0};
  std::atomic<uint64_t> flushed_{0}, flush_dropped_{0}, reclaimed_{0}, rebalanced_{0};
};

}  // namespace dfkv

#endif  // DFKV_RAM_TIER_H_
