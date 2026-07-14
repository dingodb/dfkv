/* DiskSlabStore — a disk-backed KV cache node built on SlabAllocator.
 *
 * Replaces the "one file per block" KVStore geometry (whose tmp-leak / ENOSPC-
 * dead-end / unbounded-inode / lock-held-unlink pathologies all stem from that
 * base) with a fixed pool of pre-allocated EXTENT files carved into slots by the
 * media-agnostic SlabAllocator. A compact slots.tbl records which key occupies
 * each slot so the index rebuilds on restart -- keeping cache warmth across a
 * rolling upgrade without the per-block file churn.
 *
 * I/O is buffered (page cache) for simplicity and correctness; extent fds are
 * kept resident (no open()-per-GET). O_DIRECT is a future flag. Cache semantics:
 * a miss is a clean NotFound (upper layer recomputes), so a crash that loses the
 * last few unsynced table records only costs those keys a recompute -- never
 * corruption (each record is CRC-checked; a torn record reads as free). */
#ifndef DFKV_DISK_SLAB_STORE_H_
#define DFKV_DISK_SLAB_STORE_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/slab_allocator.h"
#include "cache/store_engine.h"
#include "common/kv_types.h"
#include "common/status.h"

namespace dfkv {

class DiskSlabStore : public StoreEngine {
 public:
  struct Options {
    std::string dir;
    uint64_t capacity_bytes = (1ull << 30);
    uint64_t extent_bytes = (1ull << 30);    // 1 GiB per extent file
    // Slot-size quantum: every class slot_size is a multiple of this, so the
    // per-extent slot count (and thus slots.tbl size) is bounded. Real KV blocks
    // are MiB-scale, so 1 MiB is a fine default; tests use small values.
    uint64_t slot_granularity = (1ull << 20);
    // O_DIRECT on the aligned data paths (CacheDirect writes, RangeDirect reads,
    // async prep): zero page-cache footprint -- on GPU nodes the cache/dirty
    // growth of buffered I/O competes with training/inference memory, so the
    // DEPLOYMENT default is direct (DiskCacheGroup: DFKV_SLAB_WRITE=buffered
    // opts out); burst absorption comes from the explicit RAM tier instead.
    // Extents are fallocate'd in this mode: DIO OVERWRITES parallelize on XFS,
    // allocating writes into holes would serialize on the exclusive iolock.
    // If the filesystem rejects O_DIRECT (tmpfs in tests/CI), the store falls
    // back to buffered -- DirectWritesActive() reports the resolved truth.
    bool direct_writes = false;
    // Background free-slot reclaimer cadence (ms; 0 = off). Keeps a demand-
    // driven headroom of free slots per active class so Put's fast path stays
    // pop-free-slot; without it every Put under capacity pressure runs the
    // CLOCK eviction sweep inline under the store lock (the measured write-
    // path serialization on a full store).
    uint32_t reclaim_interval_ms = 50;
    // slots.tbl fdatasync cadence (ms; 0 = off). Bounds the crash window in
    // which a REUSED slot's new payload is on disk while the reused record is
    // still page-cache-only -- rebuild would then resurrect the PREVIOUS
    // occupant's key pointing at the new occupant's bytes. Payload writes give
    // no such window in direct mode (DIO is durable at return); this closes
    // the record side to <= the cadence instead of the ~30s dirty expiry.
    uint32_t table_sync_ms = 100;
  };

  // One store's runtime counters (see KvNodeServer's dfkv_slab_* metrics).
  struct Stats {
    uint64_t dio_write_fallbacks = 0;  // direct mode: writes that fell back buffered
    uint64_t dio_read_fallbacks = 0;   // direct mode: aligned reads that fell back
    uint64_t table_syncs = 0;          // fdatasync cycles actually performed
    uint64_t bind_wipes = 0;           // extent table regions wiped on (re)bind
    uint64_t steals = 0;               // allocator cross-class extent steals
    uint64_t extent_returns = 0;       // fully-free extents returned to the pool
    uint64_t deferred_removes = 0;     // Removes deferred behind in-flight I/O
    uint64_t inflight = 0;             // keys with an unlocked read/write in flight
    uint64_t prep_holds = 0;           // outstanding async-prep slot holds
    uint64_t reclaimed_slots = 0;      // slots freed by the background reclaimer
    uint64_t rebalanced_extents = 0;   // extents moved hot<-cold by the reclaimer
    uint64_t batched_writes = 0;       // payload writes that rode a batch submit
    uint64_t uring_write_batches = 0;  // io_uring one-submit rounds (0 = loop path)
  };

  // Opens (or creates) the store under Options::dir, pre-allocating extents and
  // rebuilding the index from slots.tbl. `*ok` (nullable) reports success; on a
  // fatal open error the store is left empty and every op returns kIOError.
  explicit DiskSlabStore(Options opt, bool* ok = nullptr);
  ~DiskSlabStore();

  DiskSlabStore(const DiskSlabStore&) = delete;
  DiskSlabStore& operator=(const DiskSlabStore&) = delete;

  Status Cache(const BlockKey& key, const void* data, size_t len) override;
  // Buffered engine: the aligned direct-PUT buffer is just written as bytes.
  Status CacheDirect(const BlockKey& key, char* data, size_t len, size_t cap) override;
  // Batched CacheDirect: ONE lock hold allocates every slot, payloads go out
  // unlocked (io_uring one-submit when built+enabled, else a sequential loop),
  // ONE lock hold commits. Per-item semantics identical to CacheDirect.
  std::vector<Status> CacheDirectBatch(const std::vector<CacheBatchItem>& items) override;
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out) override;
  Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                   char* dst, size_t dst_cap, size_t* out_len) override;
  // Buffered engine: read straight into io_buf and point out_data at it (no
  // O_DIRECT alignment head); the RDMA server scatter-sends from io_buf.
  Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                     char* io_buf, size_t io_cap, const char** out_data,
                     size_t* out_len) override;
  // Async prep: read-acquire the slot (pin + inflight, held via RangePrep::token
  // until RangeRelease) and hand back a dup'd extent fd + slot-absolute aligned
  // window. The dup is BUFFERED, so a hot page-cache slot completes the uring
  // read without touching the device.
  Status RangeDirectPrep(const BlockKey& key, uint64_t offset, uint64_t length,
                         size_t io_cap, RangePrep* out) override;
  void RangeRelease(uint64_t token) override;
  bool IsCached(const BlockKey& key) const override;
  Status Remove(const BlockKey& key) override;

  size_t Count() const override;
  uint64_t UsedBytes() const override;
  uint64_t Capacity() const;
  uint64_t Evictions() const override;
  uint64_t EvictedBytes() const override;
  const std::string& Dir() const override { return opt_.dir; }
  uint64_t TableRebuilt() const { return table_rebuilt_; }  // records restored at open
  // Resolved I/O mode: direct_writes requested AND the filesystem took O_DIRECT.
  bool DirectWritesActive() const { return !extent_dio_fds_.empty(); }
  Stats GetStats() const;

 private:
  static constexpr size_t kRecBytes = 64;

  bool OpenOrInit();                 // create/verify meta, extents, table
  void Rebuild();                    // scan slots.tbl -> allocator + payload map
  bool WriteRecord(const SlabAllocator::SlotRef& r, const BlockKey& key,
                   uint32_t payload_len, bool valid);
  bool WritePayload(const SlabAllocator::SlotRef& r, const void* data, size_t len);
  // O_DIRECT payload write from the caller's ALIGNED buffer (CacheDirect path):
  // zeroes the padding bytes in place (cap allows) and pwrites the 4 KiB-rounded
  // length via the extent's DIO fd. Caller pre-checked alignment/cap fit.
  bool WritePayloadDirect(const SlabAllocator::SlotRef& r, char* data, size_t len);
  // Shared allocate/pin -> unlocked payload write -> commit skeleton for
  // Cache/CacheDirect; `write_payload` does just the payload I/O for the slot.
  template <typename WriteFn>
  Status CacheImpl(const BlockKey& key, size_t len, const WriteFn& write_payload);
  // Resolve fn's slot + committed payload length and acquire it for an unlocked
  // read (pin + inflight count, under mu_). False = miss (absent, or a write
  // still in flight). Balanced by a Release{...}Locked call.
  bool AcquireForRead(const std::string& fn, SlabAllocator::SlotRef* ref, uint32_t* plen);
  // Drop one in-flight hold on fn (unpin + count). The LAST releaser executes a
  // Remove that arrived while the key was in flight (deferred_remove_): freeing
  // the slot mid-I/O would let the allocator hand its bytes to a new writer while
  // our unlocked pread/pwrite still touches them (SlabAllocator::Remove frees a
  // slot immediately even when pinned -- pins only block EVICTION).
  // Call with mu_ held.
  void ReleaseInflightLocked(const BlockKey& key, const std::string& fn);
  uint64_t TableOffset(uint32_t extent, uint32_t slot) const {
    return static_cast<uint64_t>(extent) * max_slots_per_extent_ * kRecBytes +
           static_cast<uint64_t>(slot) * kRecBytes;
  }

  Options opt_;
  uint32_t num_extents_ = 0;
  uint32_t max_slots_per_extent_ = 0;
  std::unique_ptr<SlabAllocator> alloc_;
  std::vector<int> extent_fds_;      // resident, one per extent (buffered)
  std::vector<int> extent_dio_fds_;  // O_DIRECT twins (only when direct_writes)
  int table_fd_ = -1;                // slots.tbl
  // Outstanding async-prep holds: token -> the key to release (see RangeRelease).
  struct PrepHold { BlockKey key; std::string fn; };
  std::unordered_map<uint64_t, PrepHold> prep_holds_;
  uint64_t prep_token_seq_ = 0;
  // key.Filename() -> its true payload length (slot_size >= this). Rebuilt on
  // open from the table; the allocator only tracks slot size.
  std::unordered_map<std::string, uint32_t> payload_len_;
  // Keys with an unlocked pread/pwrite in flight (value = op count), and keys
  // whose Remove arrived during that window (executed by the last releaser).
  std::unordered_map<std::string, uint32_t> inflight_;
  std::unordered_set<std::string> deferred_remove_;
  // Guards the in-memory maps and the allocate/lookup+pin step ONLY -- never
  // held across payload I/O (a pin protects the slot from eviction in the
  // unlocked window, inflight_/deferred_remove_ protect it from Remove;
  // payload_len_ install after the write is the reader-visible commit).
  mutable std::mutex mu_;
  bool ok_ = false;
  uint64_t table_rebuilt_ = 0;
  uint64_t evicted_bytes_ = 0;
  uint64_t deferred_remove_total_ = 0;  // under mu_
  std::atomic<uint64_t> dio_write_fallbacks_{0};
  std::atomic<uint64_t> dio_read_fallbacks_{0};
  std::atomic<uint64_t> table_syncs_{0};
  std::atomic<uint64_t> bind_wipes_{0};
  // slots.tbl sync thread (see Options::table_sync_ms): fdatasync only when
  // records were written since the last cycle.
  std::atomic<uint64_t> record_writes_{0};
  uint64_t synced_marker_ = 0;  // sync-thread-local snapshot of record_writes_
  std::thread sync_thread_;
  std::condition_variable sync_cv_;
  std::mutex sync_mu_;
  bool sync_stop_ = false;
  // Background free-slot reclaimer (see Options::reclaim_interval_ms): runs
  // ReclaimTick every interval, evicting ahead of demand in bounded batches
  // and rebalancing extents from cold classes to hot ones on a full store.
  void ReclaimTick();
  // Rebalance rate cap: extents moved per tick per hot class. Each move evicts
  // a donor extent's residents + wipes its table region under the lock, so it
  // is deliberately slow-drip (converges in seconds at the 50 ms tick).
  static constexpr size_t kGrowExtentsPerTick = 2;
  std::atomic<uint64_t> reclaimed_{0};
  std::atomic<uint64_t> rebalanced_{0};
  std::atomic<uint64_t> batched_writes_{0};
  std::atomic<uint64_t> uring_write_batches_{0};
  bool uring_write_enabled_ = false;   // DFKV_SLAB_URING_WRITE (needs DFKV_WITH_URING build)
  void* uring_w_ = nullptr;            // io_uring* (lazily created, guarded by uring_w_mu_)
  std::mutex uring_w_mu_;
  std::vector<uint64_t> reclaim_last_puts_;  // reclaim-thread-local puts snapshot
  std::thread reclaim_thread_;
  std::condition_variable reclaim_cv_;
  std::mutex reclaim_mu_;
  bool reclaim_stop_ = false;
};

}  // namespace dfkv

#endif  // DFKV_DISK_SLAB_STORE_H_
