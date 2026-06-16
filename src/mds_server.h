#ifndef DFKV_MDS_SERVER_H_
#define DFKV_MDS_SERVER_H_

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "etcd_client.h"
#include "http_client.h"
#include "kv_store.h"
#include "mds_metrics.h"
#include "membership.h"

namespace dfkv {

// Stateless MDS: nodes/clients connect over TCP (same wire framing as the cache
// node). The MDS is the only etcd client; it holds each member's lease on the
// node's behalf. The {key->leaseID} map is reconstructable from heartbeats, so
// the MDS keeps no durable state. Liveness = the lease (TTL kTtlSeconds).
class MdsServer {
 public:
  explicit MdsServer(const std::string& etcd_addr, int etcd_timeout_ms = 2000)
      : http_(etcd_addr, etcd_timeout_ms), etcd_(&http_) {}
  ~MdsServer();

  Status Start(int port);
  void Stop();
  int port() const { return port_; }
  std::string MetricsText() const { return metrics_.Render(); }

  static constexpr int kTtlSeconds = 30;

 private:
  void AcceptLoop();
  void Handle(int fd);
  Status Upsert(const std::string& group, const MemberInfo& m);
  Status ListMembers(const std::string& group, std::string* out);
  static std::string MemberKey(const std::string& group, const std::string& id);

  TcpHttpTransport http_;
  EtcdClient etcd_;
  std::atomic<bool> running_{false};
  int listen_fd_ = -1;
  int port_ = 0;
  std::thread accept_thread_;
  std::mutex conn_mu_;
  std::vector<int> conn_fds_;
  std::vector<std::thread> conn_threads_;
  std::mutex lease_mu_;
  std::map<std::string, int64_t> leases_;
  MdsMetrics metrics_;
};

}  // namespace dfkv

#endif  // DFKV_MDS_SERVER_H_
