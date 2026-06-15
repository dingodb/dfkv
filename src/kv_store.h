/* KVStore — cache-node local KV store: disk-backed, LRU-evicted, cache-only
 * (no S3 fallback; a miss is a clean NotFound). Cache() is synchronous: after
 * it returns the block is on disk and indexed (IsCached==true), giving the
 * cross-node read-after-write visibility the design requires. Portable (no
 * brpc/io_uring); the real cache node uses the dingofs DiskCache engine.
 *
 * Concurrency: the in-memory index + recency are SHARDED into `shards` stripes,
 * each guarded by its own std::shared_mutex (the dingofs `Shards` pattern). The
 * GET hot path (Range/RangeDirect/IsCached) takes a SHARED lock and only flips a
 * per-entry atomic CLOCK bit (no list mutation), so reads to a shard run
 * concurrently; Cache/eviction take the EXCLUSIVE lock. Eviction is per-shard
 * second-chance (CLOCK) — an LRU approximation; each shard owns capacity/shards
 * bytes. The bulk disk I/O always happens OUTSIDE the lock (the open fd pins the
 * inode, so a concurrent eviction can't pull bytes out from under a reader). */
#ifndef DFKV_KV_STORE_H_
#define DFKV_KV_STORE_H_

#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_types.h"

namespace dfkv {

enum class Status { kOk, kNotFound, kCacheFull, kIOError, kInvalid };

const char* StatusName(Status s);

class KVStore {
 public:
  struct Options {
    std::string cache_dir;
    uint64_t capacity_bytes = (1ull << 30);
    // Index/LRU stripes. Each shard owns capacity_bytes/shards and its own lock,
    // so concurrent ops to distinct shards never contend. Default 16; the strict
    // cross-key LRU unit test pins it to 1.
    size_t shards = 16;
  };

  explicit KVStore(Options opt);

  // Synchronous, idempotent (skips if already present). No S3 upload.
  Status Cache(const BlockKey& key, const void* data, size_t len);
  // [offset, offset+length) from the local block; NotFound if absent.
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out);
  // Like Range but reads straight into a caller buffer (no std::string), saving a
  // copy on the server GET path. Reads up to min(length, file-offset, dst_cap)
  // bytes into dst; *out_len = bytes read. NotFound if absent.
  Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                   char* dst, size_t dst_cap, size_t* out_len);
  // O_DIRECT range read into a caller-provided aligned buffer. The disk read may
  // cover an aligned superset of the requested range; *out_data points inside
  // io_buf at the exact requested bytes and can be scatter-sent directly.
  Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                     char* io_buf, size_t io_cap, const char** out_data,
                     size_t* out_len);
  bool IsCached(const BlockKey& key) const;

  uint64_t UsedBytes() const;
  size_t Count() const;

 private:
  struct Entry {
    std::string path;
    uint64_t size = 0;
    std::list<std::string>::iterator ring_it;  // position in the shard's CLOCK ring
    std::atomic<bool> referenced{false};        // CLOCK bit: set on access (read lock)
    Entry(std::string p, uint64_t s, std::list<std::string>::iterator it)
        : path(std::move(p)), size(s), ring_it(it) {}
  };
  struct Shard {
    mutable std::shared_mutex mu;
    std::unordered_map<std::string, Entry> index;  // filename -> entry
    std::list<std::string> ring;                    // CLOCK ring, front = newest
    uint64_t used_bytes = 0;
    uint64_t capacity = 0;
  };

  Shard& ShardFor(const std::string& fname) const;
  void EvictLocked(Shard& sh);  // CLOCK second-chance; exclusive lock held
  void RebuildIndex();

  Options opt_;
  std::vector<std::unique_ptr<Shard>> shards_;  // fixed after construction
  std::atomic<uint64_t> tmp_seq_{0};  // unique suffix for concurrent lock-free writes
};

}  // namespace dfkv

#endif  // DFKV_KV_STORE_H_
