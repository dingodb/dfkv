/* StoreEngine — the cache-node local storage backend interface.
 *
 * DiskCacheGroup routes a block to one backend per disk; both the original
 * file-per-block KVStore and the extent-slab DiskSlabStore implement this, so
 * the engine is selectable (--store-engine / DFKV_STORE_ENGINE) with the file
 * engine as the default. The RDMA zero-copy fast path (RangeDirect returning a
 * pointer into the caller's registered buffer) and its io_uring split
 * (RangeDirectPrep) are part of the interface; a buffered engine (slab) fills
 * io_buf with a plain read and declines the prep (returns non-kOk) so the RDMA
 * server falls back to the synchronous RangeDirect. */
#ifndef DFKV_STORE_ENGINE_H_
#define DFKV_STORE_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "common/kv_types.h"
#include "common/status.h"

namespace dfkv {

// Output of the cheap, lock-protected prep half of RangeDirect (see
// KVStore::RangeDirectPrep). A buffered engine leaves fd < 0 and returns a
// non-kOk status to signal "no async prep; use the synchronous RangeDirect".
struct RangePrep {
  int fd = -1;              // owned by caller on kOk (caller closes); -1 if no read
  uint64_t aligned_off = 0; // O_DIRECT-aligned read start
  size_t aligned_len = 0;   // O_DIRECT-aligned read length (multiple of 4096)
  size_t head = 0;          // offset of requested bytes within the aligned read
  size_t payload_len = 0;   // exact requested bytes (after clamp to file size)
};

class StoreEngine {
 public:
  virtual ~StoreEngine() = default;

  virtual Status Cache(const BlockKey& key, const void* data, size_t len) = 0;
  virtual Status CacheDirect(const BlockKey& key, char* data, size_t len,
                             size_t cap) = 0;
  virtual Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
                       std::string* out) = 0;
  virtual Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                           char* dst, size_t dst_cap, size_t* out_len) = 0;
  virtual Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                             char* io_buf, size_t io_cap, const char** out_data,
                             size_t* out_len) = 0;
  virtual Status RangeDirectPrep(const BlockKey& key, uint64_t offset,
                                 uint64_t length, size_t io_cap, RangePrep* out) = 0;
  virtual bool IsCached(const BlockKey& key) const = 0;
  virtual Status Remove(const BlockKey& key) = 0;

  virtual uint64_t UsedBytes() const = 0;
  virtual size_t Count() const = 0;
  virtual uint64_t Evictions() const = 0;
  virtual uint64_t EvictedBytes() const = 0;
  virtual const std::string& Dir() const = 0;
};

}  // namespace dfkv

#endif  // DFKV_STORE_ENGINE_H_
