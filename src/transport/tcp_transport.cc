#include "transport/tcp_transport.h"

#include <unistd.h>

#include <string>

#include "utils/net_util.h"

namespace dfkv {

namespace {
constexpr size_t kMaxIdlePerNode = 16;  // cap pooled idle conns per node

// Do one request/response on an existing fd. Returns false on any TRANSPORT
// failure (caller should drop the connection); on success sets *st to the
// server's response status (which may itself be NotFound etc.).
bool OneShot(int fd, WireOp op, const BlockKey& k, uint64_t offset,
             uint64_t length, const void* payload, uint64_t payload_len,
             Status* st, std::string* out) {
  char prefix[kReqPrefix];
  EncodeReq(prefix, op, k, offset, length, payload_len);
  if (!net::WriteAll(fd, prefix, kReqPrefix)) return false;
  if (payload_len && !net::WriteAll(fd, payload, payload_len)) return false;

  char rp[kRespPrefix];
  if (!net::ReadAll(fd, rp, kRespPrefix)) return false;
  uint64_t dlen = 0;
  if (!DecodeResp(rp, st, &dlen)) return false;  // bad protocol version
  if (dlen) {
    std::string data(dlen, '\0');
    if (!net::ReadAll(fd, &data[0], dlen)) return false;
    if (out) *out = std::move(data);
  } else if (out) {
    out->clear();
  }
  return true;
}
}  // namespace

TcpTransport::~TcpTransport() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [node, fds] : pool_)
    for (int fd : fds) ::close(fd);
}

int TcpTransport::Acquire(const std::string& node, bool* from_pool) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pool_.find(node);
    if (it != pool_.end() && !it->second.empty()) {
      int fd = it->second.back();
      it->second.pop_back();
      *from_pool = true;
      return fd;
    }
  }
  *from_pool = false;
  return net::Dial(node, connect_ms_, io_ms_);  // dial outside the lock
}

void TcpTransport::Release(const std::string& node, int fd) {
  std::lock_guard<std::mutex> lk(mu_);
  auto& fds = pool_[node];
  if (fds.size() >= kMaxIdlePerNode) {
    ::close(fd);
  } else {
    fds.push_back(fd);
  }
}

Status TcpTransport::RoundTrip(const std::string& node, WireOp op,
                              const BlockKey& k, uint64_t offset,
                              uint64_t length, const void* payload,
                              uint64_t payload_len, std::string* out) {
  // Up to 2 attempts: a stale pooled connection is dropped and re-dialed once.
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool from_pool = false;
    int fd = Acquire(node, &from_pool);
    if (fd < 0) return Status::kIOError;  // dial failed

    Status st = Status::kIOError;
    if (OneShot(fd, op, k, offset, length, payload, payload_len, &st, out)) {
      Release(node, fd);  // healthy -> back to pool
      return st;
    }
    ::close(fd);  // broken connection
    if (!from_pool) return Status::kIOError;  // a fresh conn failed -> real error
    // else: pooled conn was stale -> retry once with a fresh dial
  }
  return Status::kIOError;
}

Status TcpTransport::Cache(const std::string& node, const BlockKey& key,
                           const void* data, size_t len) {
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr);
}

Status TcpTransport::Range(const std::string& node, const BlockKey& key,
                           uint64_t offset, uint64_t length, std::string* out) {
  return RoundTrip(node, WireOp::kRange, key, offset, length, nullptr, 0, out);
}

Status TcpTransport::Stats(const std::string& node, std::string* out) {
  return RoundTrip(node, WireOp::kStats, BlockKey{}, 0, 0, nullptr, 0, out);
}

Status TcpTransport::Members(const std::string& node, std::string* out) {
  return RoundTrip(node, WireOp::kMembers, BlockKey{}, 0, 0, nullptr, 0, out);
}

Status TcpTransport::Exist(const std::string& node, const BlockKey& key,
                           bool* exist) {
  Status st = RoundTrip(node, WireOp::kExist, key, 0, 0, nullptr, 0, nullptr);
  if (st == Status::kOk) { *exist = true; return Status::kOk; }
  if (st == Status::kNotFound) { *exist = false; return Status::kOk; }
  return st;
}

Status TcpTransport::Remove(const std::string& node, const BlockKey& key) {
  // kRemove is a key-only request (no payload), Status-only response: kOk when a
  // block was dropped, kNotFound when it was already absent (both fine for the
  // caller), kIOError on a transport failure.
  return RoundTrip(node, WireOp::kRemove, key, 0, 0, nullptr, 0, nullptr);
}

}  // namespace dfkv
