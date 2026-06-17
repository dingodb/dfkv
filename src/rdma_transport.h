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

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "transport.h"

namespace dfkv {

class RdmaTransport : public Transport {
 public:
  static bool Available();  // true if at least one RDMA device is present

  // dev_name empty => env DFKV_RDMA_DEV (comma-separated list = multi-rail; each
  // new connection round-robins across the devices), else first device.
  explicit RdmaTransport(size_t max_msg = (64u << 20), const std::string& dev_name = "");
  ~RdmaTransport() override;

  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override;
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out) override;
  Status Exist(const std::string& node, const BlockKey& key,
               bool* exist) override;
  Status Members(const std::string& node, std::string* out) override;

  void RegisterMemory(void* base, size_t size) override;
  std::string MetricsText() const override;  // dfkv_rdma_client_* (conns, per-rail)

  bool pipelined() const override { return true; }
  // Pipelined: up to `depth_` requests in flight on a single connection.
  std::vector<Status> CacheMany(const std::string& node,
                                const std::vector<CacheItem>& items) override;
  std::vector<Status> CacheFrom(const std::string& node,
                                const std::vector<CacheSrc>& srcs) override;
  std::vector<Status> RangeMany(const std::string& node,
                                const std::vector<BlockKey>& keys,
                                uint64_t offset, uint64_t length,
                                std::vector<std::string>* outs) override;
  std::vector<Status> RangeInto(const std::string& node,
                                const std::vector<BlockKey>& keys,
                                const std::vector<RangeDst>& dsts,
                                size_t header_size,
                                std::vector<std::string>* hdrs) override;

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
  // Caller memory regions to register on every connection (the host KV pool).
  // Guarded by mu_; snapshotted in Acquire and registered on the connection.
  std::vector<std::pair<void*, size_t>> pools_;
  size_t max_payload_;
  size_t control_cap_;
  size_t depth_;
  int connect_ms_ = 3000;             // bootstrap TCP connect timeout (DFKV_RDMA_CONNECT_MS)
  int io_ms_ = 10000;                 // bootstrap TCP IO timeout (DFKV_RDMA_IO_MS)
  size_t pool_max_ = 256;             // idle conns kept per node (DFKV_RDMA_POOL_MAX)
  std::vector<std::string> devs_;     // RDMA devices (multi-rail); "" = first
  std::vector<int> dev_node_;         // NUMA node per devs_ entry (-1 unknown)
  std::atomic<size_t> rr_{0};         // round-robin selector across devs_
  // observability (relaxed): connections opened total + per-rail breakdown +
  // declared MR regions. Connection opens are infrequent (pooled), off the op path.
  std::atomic<uint64_t> conns_opened_{0};
  std::atomic<uint64_t> mr_regions_{0};
  std::unique_ptr<std::atomic<uint64_t>[]> rail_conns_;  // sized to devs_.size()
};

}  // namespace dfkv

#endif  // DFKV_RDMA_TRANSPORT_H_
