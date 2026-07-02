#ifndef DFKV_MDS_MEMBER_POLLER_H_
#define DFKV_MDS_MEMBER_POLLER_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mds/mds_endpoints.h"
#include "common/membership.h"

namespace dfkv {

// Client-side discovery: periodically polls the MDS for a group's member view and
// invokes on_change(members) whenever the epoch (etcd revision) changes. Endpoint
// selection + failover via MdsEndpoints. One background thread.
class MdsMemberPoller {
 public:
  using OnChange = std::function<void(const std::vector<MemberInfo>&)>;
  MdsMemberPoller(std::vector<std::string> mds_eps, std::string group, OnChange cb,
                  int poll_ms = 3000, int io_timeout_ms = 2000);
  ~MdsMemberPoller();

  void Start();
  void Stop();
  bool PollOnce();

 private:
  bool Query(std::vector<MemberInfo>* out, uint64_t* epoch);
  void Loop();
  bool WaitMs(int ms);
  uint64_t NowMs() const;

  MdsEndpoints eps_;
  std::string group_;
  OnChange cb_;
  int poll_ms_;
  int io_ms_;
  uint64_t last_epoch_ = 0;
  bool have_epoch_ = false;
  std::atomic<bool> running_{false};
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace dfkv

#endif  // DFKV_MDS_MEMBER_POLLER_H_
