#include "mds/mds_registrar.h"

#include <unistd.h>

#include <chrono>
#include <utility>
#include <vector>

#include "common/status.h"
#include "mds/mds_proto.h"
#include "utils/net_util.h"
#include "transport/wire.h"

namespace dfkv {

MdsRegistrar::MdsRegistrar(std::vector<std::string> mds_eps, std::string group,
                           MemberInfo self, int heartbeat_ms, int io_timeout_ms)
    : eps_(std::move(mds_eps)),
      group_(std::move(group)),
      self_(std::move(self)),
      hb_ms_(heartbeat_ms),
      io_ms_(io_timeout_ms) {}

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

  std::string payload = EncodeMemberReq(group_, self_);
  char pre[kReqPrefix];
  EncodeReq(pre, static_cast<WireOp>(op), BlockKey{}, 0, 0, payload.size());
  bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
            net::WriteAll(fd, payload.data(), payload.size());
  if (ok) {
    char rp[kRespPrefix];
    Status st = Status::kInvalid;
    uint64_t dlen = 0;
    ok = net::ReadAll(fd, rp, kRespPrefix) && DecodeResp(rp, &st, &dlen) &&
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

bool MdsRegistrar::RegisterOnce() { return SendOnce(static_cast<uint8_t>(WireOp::kRegister)); }
bool MdsRegistrar::HeartbeatOnce() { return SendOnce(static_cast<uint8_t>(WireOp::kHeartbeat)); }

bool MdsRegistrar::WaitMs(int ms) {
  std::unique_lock<std::mutex> lk(mu_);
  return cv_.wait_for(lk, std::chrono::milliseconds(ms),
                      [this] { return !running_.load(); });
}

void MdsRegistrar::Loop() {
  while (running_.load()) {
    if (SendOnce(static_cast<uint8_t>(WireOp::kRegister))) break;
    if (WaitMs(1000)) return;
  }
  while (running_.load()) {
    if (WaitMs(hb_ms_)) return;
    SendOnce(static_cast<uint8_t>(WireOp::kHeartbeat));
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
