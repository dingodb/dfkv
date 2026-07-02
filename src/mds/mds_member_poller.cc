#include "mds/mds_member_poller.h"

#include <unistd.h>

#include <chrono>
#include <utility>

#include "common/status.h"
#include "utils/net_util.h"
#include "utils/wire_limits.h"
#include "transport/wire.h"

namespace dfkv {

MdsMemberPoller::MdsMemberPoller(std::vector<std::string> mds_eps, std::string group,
                                 OnChange cb, int poll_ms, int io_timeout_ms)
    : eps_(std::move(mds_eps)),
      group_(std::move(group)),
      cb_(std::move(cb)),
      poll_ms_(poll_ms),
      io_ms_(io_timeout_ms) {}

MdsMemberPoller::~MdsMemberPoller() { Stop(); }

uint64_t MdsMemberPoller::NowMs() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool MdsMemberPoller::Query(std::vector<MemberInfo>* out, uint64_t* epoch) {
  uint64_t now = NowMs();
  std::string ep = eps_.Pick(now);
  if (ep.empty()) return false;
  int fd = net::Dial(ep, io_ms_, io_ms_);
  if (fd < 0) { eps_.MarkFailed(ep, now); return false; }
  char pre[kReqPrefix];
  EncodeReq(pre, WireOp::kListMembers, BlockKey{}, 0, 0, group_.size());
  bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
            (group_.empty() || net::WriteAll(fd, group_.data(), group_.size()));
  std::string data;
  if (ok) {
    char rp[kRespPrefix];
    Status st = Status::kInvalid;
    uint64_t dlen = 0;
    ok = net::ReadAll(fd, rp, kRespPrefix) && DecodeResp(rp, &st, &dlen, wire_limits::kMdsMaxRespData) &&
         st == Status::kOk;
    if (ok) { data.resize(dlen); ok = (dlen == 0) || net::ReadAll(fd, &data[0], dlen); }
  }
  ::close(fd);
  if (!ok) { eps_.MarkFailed(ep, now); return false; }
  eps_.MarkOk(ep);
  return DecodeMembers(data.data(), data.size(), out, epoch);
}

bool MdsMemberPoller::PollOnce() {
  std::vector<MemberInfo> ms;
  uint64_t epoch = 0;
  if (!Query(&ms, &epoch)) return false;  // failed RPC never clears the ring

  // Empty-view guard: a *successful* empty response after we have seen real
  // members is almost always etcd-outage recovery (all leases expired), not a
  // genuine teardown. Don't swap in an empty ring until it persists for
  // kEmptyViewsToAccept consecutive polls; adopting late costs nothing (an
  // empty ring can serve no requests anyway), while adopting early causes a
  // cluster-wide miss storm and a double remap when members re-register.
  if (ms.empty() && seen_nonempty_ &&
      ++consecutive_empty_ < kEmptyViewsToAccept) {
    ++empty_view_rejected_;
    return true;  // keep the last ring; do NOT invoke on_change
  }
  if (!ms.empty()) { seen_nonempty_ = true; consecutive_empty_ = 0; }

  if (!have_epoch_ || epoch != last_epoch_) {
    last_epoch_ = epoch;
    have_epoch_ = true;
    cb_(ms);
  }
  return true;
}

bool MdsMemberPoller::WaitMs(int ms) {
  std::unique_lock<std::mutex> lk(mu_);
  return cv_.wait_for(lk, std::chrono::milliseconds(ms),
                      [this] { return !running_.load(); });
}

void MdsMemberPoller::Loop() {
  while (running_.load()) {
    PollOnce();
    if (WaitMs(poll_ms_)) return;
  }
}

void MdsMemberPoller::Start() {
  if (running_.exchange(true)) return;
  th_ = std::thread([this] { Loop(); });
}

void MdsMemberPoller::Stop() {
  if (!running_.exchange(false)) return;
  cv_.notify_all();
  if (th_.joinable()) th_.join();
}

}  // namespace dfkv
