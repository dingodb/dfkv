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

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache/slab_allocator.h"
#include "common/kv_types.h"
#include "common/status.h"

namespace dfkv {

class DiskSlabStore {
 public:
  struct Options {
    std::string dir;
    uint64_t capacity_bytes = (1ull << 30);
    uint64_t extent_bytes = (1ull << 30);    // 1 GiB per extent file
    // Slot-size quantum: every class slot_size is a multiple of this, so the
    // per-extent slot count (and thus slots.tbl size) is bounded. Real KV blocks
    // are MiB-scale, so 1 MiB is a fine default; tests use small values.
    uint64_t slot_granularity = (1ull << 20);
  };

  // Opens (or creates) the store under Options::dir, pre-allocating extents and
  // rebuilding the index from slots.tbl. `*ok` (nullable) reports success; on a
  // fatal open error the store is left empty and every op returns kIOError.
  explicit DiskSlabStore(Options opt, bool* ok = nullptr);
  ~DiskSlabStore();

  DiskSlabStore(const DiskSlabStore&) = delete;
  DiskSlabStore& operator=(const DiskSlabStore&) = delete;

  Status Cache(const BlockKey& key, const void* data, size_t len);
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out);
  Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                   char* dst, size_t dst_cap, size_t* out_len);
  bool IsCached(const BlockKey& key) const;
  Status Remove(const BlockKey& key);

  size_t Count() const;
  uint64_t UsedBytes() const;
  uint64_t Capacity() const;
  uint64_t Evictions() const;
  uint64_t TableRebuilt() const { return table_rebuilt_; }  // records restored at open

 private:
  static constexpr size_t kRecBytes = 64;

  bool OpenOrInit();                 // create/verify meta, extents, table
  void Rebuild();                    // scan slots.tbl -> allocator + payload map
  bool WriteRecord(const SlabAllocator::SlotRef& r, const BlockKey& key,
                   uint32_t payload_len, bool valid);
  bool WritePayload(const SlabAllocator::SlotRef& r, const void* data, size_t len);
  uint64_t TableOffset(uint32_t extent, uint32_t slot) const {
    return static_cast<uint64_t>(extent) * max_slots_per_extent_ * kRecBytes +
           static_cast<uint64_t>(slot) * kRecBytes;
  }

  Options opt_;
  uint32_t num_extents_ = 0;
  uint32_t max_slots_per_extent_ = 0;
  std::unique_ptr<SlabAllocator> alloc_;
  std::vector<int> extent_fds_;      // resident, one per extent
  int table_fd_ = -1;                // slots.tbl
  // key.Filename() -> its true payload length (slot_size >= this). Rebuilt on
  // open from the table; the allocator only tracks slot size.
  std::unordered_map<std::string, uint32_t> payload_len_;
  mutable std::mutex mu_;
  bool ok_ = false;
  uint64_t table_rebuilt_ = 0;
};

}  // namespace dfkv

#endif  // DFKV_DISK_SLAB_STORE_H_
