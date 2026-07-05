#ifndef DFKV_MDS_REGISTRAR_H_
#define DFKV_MDS_REGISTRAR_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <functional>

#include "mds/mds_endpoints.h"
#include "common/membership.h"

namespace dfkv {

// Node-side agent: registers this cache node with the MDS and keeps its lease
// alive via periodic heartbeats. Uses MdsEndpoints for no-LB multi-endpoint
// selection + failover. One background thread; steady_clock drives both the
// endpoint backoff clock and the heartbeat cadence.
//
// With is_client=true the SAME agent registers a cache *consumer* (an inference
// connector instance) under the /clients/<id> prefix instead of a cache node.
// Only the wire op differs (kClientRegister/kClientHeartbeat vs kRegister/
// kHeartbeat); lease/keepalive/failover logic is identical, so consumers get the
// same lease-TTL auto-cleanup as nodes — a dead client's key expires out of etcd
// within kTtlSeconds with no explicit deregister.
class MdsRegistrar {
 public:
  MdsRegistrar(std::vector<std::string> mds_eps, std::string group, MemberInfo self,
               int heartbeat_ms = 10000, int io_timeout_ms = 2000,
               bool is_client = false);
  ~MdsRegistrar();

  void Start();
  void Stop();

  // Optional dynamic-stats provider: called right before EVERY register/
  // heartbeat encode, so the STA1 payload carries values as of that beat
  // (~heartbeat_ms freshness at the MDS). Unset = no STA1 (legacy behavior).
  using StatsFn = std::function<MemberStats()>;
  void set_stats_provider(StatsFn fn) { stats_fn_ = std::move(fn); }

  bool RegisterOnce();
  bool HeartbeatOnce();

 private:
  bool SendOnce(uint8_t op);
  void Loop();
  bool WaitMs(int ms);
  uint64_t NowMs() const;

  MdsEndpoints eps_;
  std::string group_;
  MemberInfo self_;
  StatsFn stats_fn_;
  int hb_ms_;
  int io_ms_;
  bool is_client_;
  std::atomic<bool> running_{false};
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace dfkv

#endif  // DFKV_MDS_REGISTRAR_H_
