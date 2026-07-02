/* Portable KV types mirroring dingofs BlockKey identity/layout (no brpc deps).
 * In the real build the SDK adapts these to the production DingoFS BlockKey; the Filename()/
 * StoreKey() formats match src/common/block/block_key.h. */
#ifndef DFKV_KV_TYPES_H_
#define DFKV_KV_TYPES_H_

#include <cstdint>
#include <string>

namespace dfkv {

struct BlockKey {
  uint64_t id = 0;
  uint32_t index = 0;
  uint32_t size = 0;

  std::string Filename() const {
    return std::to_string(id) + "_" + std::to_string(index) + "_" +
           std::to_string(size);
  }
  // blocks/{id/1e6}/{id/1e3}/{filename} — matches dingofs StoreKey buckets.
  std::string StoreKey() const {
    return "blocks/" + std::to_string(id / 1000000) + "/" +
           std::to_string(id / 1000) + "/" + Filename();
  }
  bool operator==(const BlockKey& o) const {
    return id == o.id && index == o.index && size == o.size;
  }
};

}  // namespace dfkv

#endif  // DFKV_KV_TYPES_H_
