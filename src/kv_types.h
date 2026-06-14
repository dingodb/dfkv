/* Portable KV types mirroring dingofs BlockKey identity/layout (no brpc deps).
 * In the real build the SDK adapts these to dingofs::BlockKey; the Filename()/
 * StoreKey() formats match src/common/block/block_key.h. */
#ifndef DINGOFS_SRC_CACHE_KVCLIENT_KV_TYPES_H_
#define DINGOFS_SRC_CACHE_KVCLIENT_KV_TYPES_H_

#include <cstdint>
#include <string>

namespace dingofs {
namespace cache {
namespace kv {

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

}  // namespace kv
}  // namespace cache
}  // namespace dingofs

#endif  // DINGOFS_SRC_CACHE_KVCLIENT_KV_TYPES_H_
