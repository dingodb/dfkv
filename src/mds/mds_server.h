#ifndef DFKV_MDS_SERVER_H_
#define DFKV_MDS_SERVER_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mds/etcd_client.h"
#include "utils/http_client.h"
#include "common/status.h"
#include "mds/mds_metrics.h"
#include "common/membership.h"

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
  // Static counters + per-group ring aggregates. The aggregate half does ONE
  // etcd prefix range over /dfkv/v1/groups/ at scrape time (MDS stays
  // stateless; ~30s Prometheus cadence makes this negligible), decodes each
  // member's STA1 stats and sums them per group -- ring capacity / usage /
  // hit-rate / alarm counters become one MDS scrape instead of a fleet sweep.
  std::string MetricsText() { return metrics_.Render() + GroupMetricsText(); }
  std::string GroupMetricsText();
  // kListGroups backend: distinct group names under /dfkv/v1/groups/ (newline-
  // joined). Feeds `dfkvctl stats --all`.
  Status ListGroups(std::string* out);
  size_t live_conn_count();  // handler threads not yet reaped (test/diagnostic)
  // One read against etcd (a bounded RangePrefix on a probe key). Returns true
  // iff etcd answered. Used at startup to fail loud on a misconfigured --etcd
  // (a wrong endpoint/scheme otherwise runs "healthy" while every registration
  // silently fails), and by /healthz to reflect etcd reachability.
  bool ProbeEtcd() { return etcd_.RangePrefix("/dfkv/v1/_healthz_probe/").has_value(); }

  static constexpr int kTtlSeconds = 30;

 private:
  void AcceptLoop();
  void Handle(int fd);
  void ReapDoneLocked();  // join+erase finished handler threads; conn_mu_ held
  Status Upsert(const std::string& group, const MemberInfo& m);
  Status ListMembers(const std::string& group, std::string* out);
  // Client (consumer) registration — same lease/keepalive contract as Upsert/
  // ListMembers but under a disjoint etcd prefix (/clients/<id> vs /members/<id>)
  // so consumers never enter the placement ring. A consumer carries no data-path
  // port or stats; it is pure identity ("who is using dfkv").
  Status UpsertClient(const std::string& group, const MemberInfo& m);
  Status ListClients(const std::string& group, std::string* out);
  // Shared by Upsert/UpsertClient: lease keepalive + re-Put under the lock-free
  // pattern. Callers pass the right etcd key + the matching {key->lease} map+mu.
  Status UpsertLeased(const std::string& key, const MemberInfo& m,
                      std::map<std::string, int64_t>& leases, std::mutex& mu);
  static std::string MemberKey(const std::string& group, const std::string& id);
  static std::string ClientKey(const std::string& group, const std::string& id);

  TcpHttpTransport http_;
  EtcdClient etcd_;
  std::atomic<bool> running_{false};
  int listen_fd_ = -1;
  int port_ = 0;
  std::thread accept_thread_;
  struct Conn {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::mutex conn_mu_;
  std::vector<int> conn_fds_;
  std::vector<Conn> conns_;
  std::mutex lease_mu_;
  std::map<std::string, int64_t> leases_;
  std::mutex client_lease_mu_;
  std::map<std::string, int64_t> client_leases_;  // mirrors leases_ for /clients/
  MdsMetrics metrics_;
};

}  // namespace dfkv

#endif  // DFKV_MDS_SERVER_H_
