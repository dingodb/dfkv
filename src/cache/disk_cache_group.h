/* DiskCacheGroup — one cache node spanning multiple NVMe SSDs, mirroring
 * dingo-cache's intra-node layout (--cache_dir=d1,d2,d3). Each disk is an
 * independent KVStore (own LRU); a block is routed to one disk by a second-level
 * Ketama hash on BlockKey.Filename(). Total capacity is split evenly across
 * disks. A single-disk group degenerates to one KVStore. */
#ifndef DFKV_DISK_CACHE_GROUP_H_
#define DFKV_DISK_CACHE_GROUP_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/con_hash.h"
#include "cache/kv_store.h"
#include "common/kv_types.h"

namespace dfkv {

class DiskCacheGroup {
 public:
  struct Options {
    std::vector<std::string> cache_dirs;   // one per NVMe SSD
    uint64_t capacity_bytes = (1ull << 30);  // TOTAL, split evenly across disks
  };

  explicit DiskCacheGroup(Options opt);

  Status Cache(const BlockKey& key, const void* data, size_t len);
  Status CacheDirect(const BlockKey& key, char* data, size_t len, size_t cap);
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out);
  Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                   char* dst, size_t dst_cap, size_t* out_len);
  Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                     char* io_buf, size_t io_cap, const char** out_data,
                     size_t* out_len);
  // Cheap prep half of RangeDirect (no disk read); see KVStore::RangeDirectPrep.
  Status RangeDirectPrep(const BlockKey& key, uint64_t offset, uint64_t length,
                         size_t io_cap, KVStore::RangePrep* out);
  bool IsCached(const BlockKey& key) const;

  // Drop a cached block from its owning disk (routes like IsCached). kOk if
  // removed, kNotFound if absent. Backs the LMCache L2 eviction path.
  Status Remove(const BlockKey& key);

  uint64_t UsedBytes() const;   // summed across disks
  size_t Count() const;         // summed across disks
  size_t DiskCount() const { return disks_.size(); }
  uint64_t Evictions() const;     // summed across disks
  uint64_t EvictedBytes() const;  // summed across disks
  // Per-disk views for fine-grained metrics (i in [0, DiskCount)).
  const std::string& DiskPath(size_t i) const { return disks_[i]->Dir(); }
  uint64_t DiskUsedBytes(size_t i) const { return disks_[i]->UsedBytes(); }
  size_t DiskObjects(size_t i) const { return disks_[i]->Count(); }

 private:
  KVStore* Route(const BlockKey& key) const;

  std::vector<std::unique_ptr<KVStore>> disks_;
  std::unordered_map<std::string, KVStore*> by_id_;  // disk id -> store
  ConHash ring_;
};

}  // namespace dfkv

#endif  // DFKV_DISK_CACHE_GROUP_H_
