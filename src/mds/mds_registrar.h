#ifndef DFKV_MDS_REGISTRAR_H_
#define DFKV_MDS_REGISTRAR_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mds/mds_endpoints.h"
#include "common/membership.h"

namespace dfkv {

// Node-side agent: registers this cache node with the MDS and keeps its lease
// alive via periodic heartbeats. Uses MdsEndpoints for no-LB multi-endpoint
// selection + failover. One background thread; steady_clock drives both the
// endpoint backoff clock and the heartbeat cadence.
class MdsRegistrar {
 public:
  MdsRegistrar(std::vector<std::string> mds_eps, std::string group, MemberInfo self,
               int heartbeat_ms = 10000, int io_timeout_ms = 2000);
  ~MdsRegistrar();

  void Start();
  void Stop();

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
  int hb_ms_;
  int io_ms_;
  std::atomic<bool> running_{false};
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace dfkv

#endif  // DFKV_MDS_REGISTRAR_H_
