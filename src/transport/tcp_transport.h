#ifndef DFKV_TCP_TRANSPORT_H_
#define DFKV_TCP_TRANSPORT_H_

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "transport/transport.h"

namespace dfkv {

// TCP transport with a per-node connection pool (keep-alive). Idle connections
// are reused; a stale pooled connection is transparently re-dialed once.
// Thread-safe.
class TcpTransport : public Transport {
 public:
  TcpTransport() = default;
  ~TcpTransport() override;

  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override;
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out) override;
  Status Exist(const std::string& node, const BlockKey& key,
               bool* exist) override;
  Status Remove(const std::string& node, const BlockKey& key) override;

  // Fetch a node's Prometheus-format metrics text (not part of Transport iface).
  Status Stats(const std::string& node, std::string* out);
  Status Members(const std::string& node, std::string* out) override;

  // connect/IO timeouts (ms); 0 = block indefinitely. Defaults bound hangs.
  void set_timeouts(int connect_ms, int io_ms) {
    connect_ms_ = connect_ms; io_ms_ = io_ms;
  }

 private:
  Status RoundTrip(const std::string& node, WireOp op, const BlockKey& key,
                   uint64_t offset, uint64_t length, const void* payload,
                   uint64_t payload_len, std::string* out);
  int Acquire(const std::string& node, bool* from_pool);
  void Release(const std::string& node, int fd);

  std::mutex mu_;
  std::unordered_map<std::string, std::vector<int>> pool_;  // node -> idle fds
  int connect_ms_ = 3000;
  int io_ms_ = 10000;
};

}  // namespace dfkv

#endif  // DFKV_TCP_TRANSPORT_H_
