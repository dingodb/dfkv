#include "transport/tcp_transport.h"

#include "utils/wire_limits.h"

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
             Status* st, std::string* out, uint64_t max_data,
             uint8_t wire_ver, uint64_t seq) {
  char prefix[kReqPrefixV2];
  size_t plen;
  if (wire_ver == kProtoVersionV2) {
    EncodeReqV2(prefix, op, k, offset, length, payload_len, seq);
    plen = kReqPrefixV2;
  } else {
    EncodeReq(prefix, op, k, offset, length, payload_len);
    plen = kReqPrefix;
  }
  if (!net::WriteAll(fd, prefix, plen)) return false;
  if (payload_len && !net::WriteAll(fd, payload, payload_len)) return false;

  char rp[kRespPrefixV2];
  if (!net::ReadAll(fd, rp, kRespPrefix)) return false;
  // A dual-accept server echoes our version; read the extra 8 bytes only when
  // it actually replied v2 (peek the version byte, not our request version, so
  // an old v1-only server that answers v1 is handled gracefully below).
  if (static_cast<uint8_t>(rp[0]) == kProtoVersionV2 &&
      !net::ReadAll(fd, rp + kRespPrefix, kRespPrefixV2 - kRespPrefix))
    return false;
  uint64_t dlen = 0, resp_seq = 0;
  // max_data caps the server-declared response length BEFORE the allocation
  // below (the caller knows what it asked for; a corrupt/hostile peer must
  // not be able to declare a 16 GiB body).
  uint8_t rver = DecodeResp(rp, st, &dlen, max_data, &resp_seq);
  if (!rver) return false;  // bad version / oversize
  // v2 request: the reply MUST be v2 and echo our seq. A seq mismatch (or an
  // old server that replied v1) is a protocol violation -- fail the round trip
  // so the caller drops the connection rather than accept a misattributed reply.
  if (wire_ver == kProtoVersionV2 && (rver != kProtoVersionV2 || resp_seq != seq))
    return false;
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
                              uint64_t payload_len, std::string* out,
                              uint64_t max_data) {
  // Up to 2 attempts: a stale pooled connection is dropped and re-dialed once.
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool from_pool = false;
    int fd = Acquire(node, &from_pool);
    if (fd < 0) return Status::kIOError;  // dial failed

    // Fresh per-request seq (v2 only): correlates the reply to this request.
    const uint64_t seq = seq_.fetch_add(1, std::memory_order_relaxed);
    Status st = Status::kIOError;
    if (OneShot(fd, op, k, offset, length, payload, payload_len, &st, out,
                max_data, wire_ver_, seq)) {
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
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr,
                   wire_limits::kStatusMaxRespData);
}

Status TcpTransport::Range(const std::string& node, const BlockKey& key,
                           uint64_t offset, uint64_t length, std::string* out) {
  // A Range reply carries at most the requested byte count.
  return RoundTrip(node, WireOp::kRange, key, offset, length, nullptr, 0, out,
                   length);
}

Status TcpTransport::Stats(const std::string& node, std::string* out) {
  return RoundTrip(node, WireOp::kStats, BlockKey{}, 0, 0, nullptr, 0, out,
                   wire_limits::kTextMaxRespData);
}

Status TcpTransport::Members(const std::string& node, std::string* out) {
  return RoundTrip(node, WireOp::kMembers, BlockKey{}, 0, 0, nullptr, 0, out,
                   wire_limits::kTextMaxRespData);
}

Status TcpTransport::Exist(const std::string& node, const BlockKey& key,
                           bool* exist) {
  Status st = RoundTrip(node, WireOp::kExist, key, 0, 0, nullptr, 0, nullptr,
                        wire_limits::kStatusMaxRespData);
  if (st == Status::kOk) { *exist = true; return Status::kOk; }
  if (st == Status::kNotFound) { *exist = false; return Status::kOk; }
  return st;
}

Status TcpTransport::Remove(const std::string& node, const BlockKey& key) {
  // kRemove is a key-only request (no payload), Status-only response: kOk when a
  // block was dropped, kNotFound when it was already absent (both fine for the
  // caller), kIOError on a transport failure.
  return RoundTrip(node, WireOp::kRemove, key, 0, 0, nullptr, 0, nullptr,
                   wire_limits::kStatusMaxRespData);
}

}  // namespace dfkv
