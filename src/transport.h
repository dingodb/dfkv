/* Transport abstraction between the KV client and cache nodes.
 * Real build: a brpc-backed impl over dingofs RemoteBlockCache.
 * Test/harness build: TcpTransport (POSIX sockets). */
#ifndef DFKV_TRANSPORT_H_
#define DFKV_TRANSPORT_H_

#include <cstdint>
#include <string>

#include "kv_store.h"   // Status
#include "kv_types.h"

namespace dfkv {

// Wire op codes (shared by TcpTransport and KvNodeServer).
enum class WireOp : uint8_t { kCache = 1, kRange = 2, kExist = 3, kStats = 4 };
// Fixed request prefix: op(1) id(8) index(4) size(4) offset(8) length(8) payload_len(8)
constexpr size_t kReqPrefix = 1 + 8 + 4 + 4 + 8 + 8 + 8;  // = 41
// Fixed response prefix: status(1) data_len(8)
constexpr size_t kRespPrefix = 1 + 8;  // = 9

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
};

}  // namespace dfkv

#endif  // DFKV_TRANSPORT_H_
