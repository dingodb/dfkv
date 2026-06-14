#ifndef DINGOFS_SRC_CACHE_KVCLIENT_TCP_TRANSPORT_H_
#define DINGOFS_SRC_CACHE_KVCLIENT_TCP_TRANSPORT_H_

#include "transport.h"

namespace dingofs {
namespace cache {
namespace kv {

// One short-lived TCP connection per call (simple, correct; the real build
// uses brpc connection pooling).
class TcpTransport : public Transport {
 public:
  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override;
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out) override;
  Status Exist(const std::string& node, const BlockKey& key,
               bool* exist) override;
};

}  // namespace kv
}  // namespace cache
}  // namespace dingofs

#endif  // DINGOFS_SRC_CACHE_KVCLIENT_TCP_TRANSPORT_H_
