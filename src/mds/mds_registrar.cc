#include "mds/mds_registrar.h"

#include <unistd.h>

#include <chrono>
#include <utility>
#include <vector>

#include "common/status.h"
#include "mds/mds_proto.h"
#include "utils/net_util.h"
#include "utils/wire_limits.h"
#include "transport/wire.h"

namespace dfkv {

MdsRegistrar::MdsRegistrar(std::vector<std::string> mds_eps, std::string group,
                           MemberInfo self, int heartbeat_ms, int io_timeout_ms,
                           bool is_client)
    : eps_(std::move(mds_eps)),
      group_(std::move(group)),
      self_(std::move(self)),
      hb_ms_(heartbeat_ms),
      io_ms_(io_timeout_ms),
      is_client_(is_client) {}

MdsRegistrar::~MdsRegistrar() { Stop(); }

uint64_t MdsRegistrar::NowMs() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool MdsRegistrar::SendOnce(uint8_t op) {
  uint64_t now = NowMs();
  std::string ep = eps_.Pick(now);
  if (ep.empty()) return false;
  int fd = net::Dial(ep, io_ms_, io_ms_);
  if (fd < 0) { eps_.MarkFailed(ep, now); return false; }

  if (stats_fn_) {  // refresh dynamic stats for this beat (STA1)
    self_.stats = stats_fn_();
    self_.has_stats = true;
  }
  std::string payload = EncodeMemberReq(group_, self_);
  char pre[kReqPrefix];
  EncodeReq(pre, static_cast<WireOp>(op), BlockKey{}, 0, 0, payload.size());
  bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
            net::WriteAll(fd, payload.data(), payload.size());
  if (ok) {
    char rp[kRespPrefix];
    Status st = Status::kInvalid;
    uint64_t dlen = 0;
    ok = net::ReadAll(fd, rp, kRespPrefix) && DecodeResp(rp, &st, &dlen, wire_limits::kMdsMaxRespData) &&
         st == Status::kOk;
    if (ok && dlen) {
      std::vector<char> d(dlen);
      ok = net::ReadAll(fd, d.data(), dlen);
    }
  }
  ::close(fd);
  if (ok) eps_.MarkOk(ep); else eps_.MarkFailed(ep, now);
  return ok;
}

bool MdsRegistrar::RegisterOnce() {
  return SendOnce(static_cast<uint8_t>(is_client_ ? WireOp::kClientRegister
                                                   : WireOp::kRegister));
}
bool MdsRegistrar::HeartbeatOnce() {
  return SendOnce(static_cast<uint8_t>(is_client_ ? WireOp::kClientHeartbeat
                                                   : WireOp::kHeartbeat));
}

bool MdsRegistrar::WaitMs(int ms) {
  std::unique_lock<std::mutex> lk(mu_);
  return cv_.wait_for(lk, std::chrono::milliseconds(ms),
                      [this] { return !running_.load(); });
}

void MdsRegistrar::Loop() {
  const uint8_t reg_op = static_cast<uint8_t>(is_client_ ? WireOp::kClientRegister
                                                          : WireOp::kRegister);
  const uint8_t hb_op = static_cast<uint8_t>(is_client_ ? WireOp::kClientHeartbeat
                                                         : WireOp::kHeartbeat);
  while (running_.load()) {
    if (SendOnce(reg_op)) break;
    if (WaitMs(1000)) return;
  }
  while (running_.load()) {
    if (WaitMs(hb_ms_)) return;
    SendOnce(hb_op);
  }
}

void MdsRegistrar::Start() {
  if (running_.exchange(true)) return;
  th_ = std::thread([this] { Loop(); });
}

void MdsRegistrar::Stop() {
  if (!running_.exchange(false)) return;
  cv_.notify_all();
  if (th_.joinable()) th_.join();
}

}  // namespace dfkv
