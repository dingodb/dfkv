/* DiskCacheGroup — one cache node spanning multiple NVMe SSDs, mirroring
 * dingo-cache's intra-node layout (--cache_dir=d1,d2,d3). Each disk is an
 * independent KVStore (own LRU); a block is routed to one disk by a second-level
 * Ketama hash on BlockKey.Filename(). Total capacity is split evenly across
 * disks. A single-disk group degenerates to one KVStore. */
#ifndef DINGOFS_SRC_CACHE_KVCLIENT_DISK_CACHE_GROUP_H_
#define DINGOFS_SRC_CACHE_KVCLIENT_DISK_CACHE_GROUP_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "con_hash.h"
#include "kv_store.h"
#include "kv_types.h"

namespace dingofs {
namespace cache {
namespace kv {

class DiskCacheGroup {
 public:
  struct Options {
    std::vector<std::string> cache_dirs;   // one per NVMe SSD
    uint64_t capacity_bytes = (1ull << 30);  // TOTAL, split evenly across disks
  };

  explicit DiskCacheGroup(Options opt);

  Status Cache(const BlockKey& key, const void* data, size_t len);
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out);
  bool IsCached(const BlockKey& key) const;

  uint64_t UsedBytes() const;   // summed across disks
  size_t Count() const;         // summed across disks
  size_t DiskCount() const { return disks_.size(); }

 private:
  KVStore* Route(const BlockKey& key) const;

  std::vector<std::unique_ptr<KVStore>> disks_;
  std::unordered_map<std::string, KVStore*> by_id_;  // disk id -> store
  ConHash ring_;
};

}  // namespace kv
}  // namespace cache
}  // namespace dingofs

#endif  // DINGOFS_SRC_CACHE_KVCLIENT_DISK_CACHE_GROUP_H_
