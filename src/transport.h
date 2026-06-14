/* Transport abstraction between the KV client and cache nodes.
 * Real build: a brpc-backed impl over dingofs RemoteBlockCache.
 * Test/harness build: TcpTransport (POSIX sockets). */
#ifndef DFKV_TRANSPORT_H_
#define DFKV_TRANSPORT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "kv_store.h"   // Status
#include "kv_types.h"

namespace dfkv {

// Wire op codes (shared by TcpTransport and KvNodeServer).
enum class WireOp : uint8_t { kCache = 1, kRange = 2, kExist = 3, kStats = 4 };
// Fixed request prefix: op(1) id(8) index(4) size(4) offset(8) length(8) payload_len(8)
constexpr size_t kReqPrefix = 1 + 8 + 4 + 4 + 8 + 8 + 8;  // = 41
// Fixed response prefix: status(1) data_len(8)
constexpr size_t kRespPrefix = 1 + 8;  // = 9

// One write in a batch (data must outlive the CacheMany call).
struct CacheItem { BlockKey key; const void* data; size_t len; };

class Transport {
 public:
  virtual ~Transport() = default;
  // Synchronous, durable-visible write (server uses KVStore::Cache).
  virtual Status Cache(const std::string& node, const BlockKey& key,
                       const void* data, size_t len) = 0;
  virtual Status Range(const std::string& node, const BlockKey& key,
                       uint64_t offset, uint64_t length, std::string* out) = 0;
  virtual Status Exist(const std::string& node, const BlockKey& key,
                       bool* exist) = 0;

  // True if CacheMany/RangeMany pipeline requests on one connection (RDMA). When
  // false (TCP), the client parallelizes batches across items with its own
  // threads instead, since the per-node loop here would be sequential.
  virtual bool pipelined() const { return false; }

  // Batch variants for one node. Default = sequential loop; RDMA overrides these
  // to pipeline multiple requests in flight on a single connection. All keys in
  // a RangeMany share (offset, length).
  virtual std::vector<Status> CacheMany(const std::string& node,
                                        const std::vector<CacheItem>& items) {
    std::vector<Status> r;
    r.reserve(items.size());
    for (const auto& it : items) r.push_back(Cache(node, it.key, it.data, it.len));
    return r;
  }
  virtual std::vector<Status> RangeMany(const std::string& node,
                                        const std::vector<BlockKey>& keys,
                                        uint64_t offset, uint64_t length,
                                        std::vector<std::string>* outs) {
    outs->assign(keys.size(), std::string());
    std::vector<Status> r;
    r.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
      r.push_back(Range(node, keys[i], offset, length, &(*outs)[i]));
    return r;
  }
};

}  // namespace dfkv

#endif  // DFKV_TRANSPORT_H_
