/* KVStore — cache-node local KV store: disk-backed, LRU-evicted, cache-only
 * (no S3 fallback; a miss is a clean NotFound). Cache() is synchronous: after
 * it returns the block is on disk and indexed (IsCached==true), giving the
 * cross-node read-after-write visibility the design requires. Portable (no
 * brpc/io_uring); the real cache node uses the dingofs DiskCache engine. */
#ifndef DINGOFS_SRC_CACHE_KVCLIENT_KV_STORE_H_
#define DINGOFS_SRC_CACHE_KVCLIENT_KV_STORE_H_

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

#include "kv_types.h"

namespace dingofs {
namespace cache {
namespace kv {

enum class Status { kOk, kNotFound, kCacheFull, kIOError, kInvalid };

const char* StatusName(Status s);

class KVStore {
 public:
  struct Options {
    std::string cache_dir;
    uint64_t capacity_bytes = (1ull << 30);
  };

  explicit KVStore(Options opt);

  // Synchronous, idempotent (skips if already present). No S3 upload.
  Status Cache(const BlockKey& key, const void* data, size_t len);
  // [offset, offset+length) from the local block; NotFound if absent.
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out);
  bool IsCached(const BlockKey& key) const;

  uint64_t UsedBytes() const;
  size_t Count() const;

 private:
  struct Entry {
    std::string path;
    uint64_t size;
    std::list<std::string>::iterator lru_it;  // position in lru_ (front = MRU)
  };
  void TouchLocked(const std::string& fname);
  void EvictIfNeededLocked();
  void RebuildIndex();

  Options opt_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, Entry> index_;  // filename -> entry
  std::list<std::string> lru_;                    // filenames, front = MRU
  uint64_t used_bytes_ = 0;
};

}  // namespace kv
}  // namespace cache
}  // namespace dingofs

#endif  // DINGOFS_SRC_CACHE_KVCLIENT_KV_STORE_H_
