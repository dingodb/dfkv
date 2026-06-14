/* RDMA client transport — native libibverbs RC, two-sided SEND/RECV.
 * Built only when DFKV_WITH_RDMA is defined. Mirrors the TCP wire frames so the
 * server request logic is shared. Connection-pooled per node.
 *
 * Device is selected BY NAME (env DFKV_RDMA_DEV, e.g. "ib7s400p0"), not by IP,
 * so the data plane can ride a 400G IB fabric that has no IP and is separate
 * from the IP network. The QP is bootstrapped over a small TCP channel to the
 * node's member address (an IP on the shared 200G/bond0 network). See
 * rdma_verbs.h for the control-plane/data-plane split rationale. */
#ifndef DFKV_RDMA_TRANSPORT_H_
#define DFKV_RDMA_TRANSPORT_H_

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "transport.h"

namespace dfkv {

class RdmaTransport : public Transport {
 public:
  static bool Available();  // true if at least one RDMA device is present

  // dev_name empty => env DFKV_RDMA_DEV, else first device.
  explicit RdmaTransport(size_t max_msg = (8u << 20), const std::string& dev_name = "");
  ~RdmaTransport() override;

  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override;
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out) override;
  Status Exist(const std::string& node, const BlockKey& key,
               bool* exist) override;

  bool pipelined() const override { return true; }
  // Pipelined: up to `depth_` requests in flight on a single connection.
  std::vector<Status> CacheMany(const std::string& node,
                                const std::vector<CacheItem>& items) override;
  std::vector<Status> RangeMany(const std::string& node,
                                const std::vector<BlockKey>& keys,
                                uint64_t offset, uint64_t length,
                                std::vector<std::string>* outs) override;

 private:
  struct Conn;
  Conn* Acquire(const std::string& node, bool* from_pool);
  void Release(const std::string& node, Conn* c);
  void Destroy(Conn* c);
  Status RoundTrip(const std::string& node, WireOp op, const BlockKey& key,
                   uint64_t offset, uint64_t length, const void* payload,
                   uint64_t payload_len, std::string* out);

  std::mutex mu_;
  std::unordered_map<std::string, std::vector<Conn*>> pool_;
  size_t max_msg_;
  size_t depth_;
  std::string dev_name_;
};

}  // namespace dfkv

#endif  // DFKV_RDMA_TRANSPORT_H_
