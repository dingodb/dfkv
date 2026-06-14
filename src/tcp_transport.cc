#include "tcp_transport.h"

#include <unistd.h>

#include <vector>

#include "net_util.h"

namespace dingofs {
namespace cache {
namespace kv {

namespace {

// Build the fixed 41B request prefix.
void EncodePrefix(char* p, WireOp op, const BlockKey& k, uint64_t offset,
                  uint64_t length, uint64_t payload_len) {
  p[0] = static_cast<char>(op);
  net::PutU64(p + 1, k.id);
  net::PutU32(p + 9, k.index);
  net::PutU32(p + 13, k.size);
  net::PutU64(p + 17, offset);
  net::PutU64(p + 25, length);
  net::PutU64(p + 33, payload_len);
}

// Send one request, read one response. Returns status; fills out (data) if any.
Status RoundTrip(const std::string& node, WireOp op, const BlockKey& k,
                 uint64_t offset, uint64_t length, const void* payload,
                 uint64_t payload_len, std::string* out) {
  int fd = net::Dial(node);
  if (fd < 0) return Status::kIOError;
  struct Closer { int fd; ~Closer() { ::close(fd); } } closer{fd};

  // Send the fixed prefix then the payload (no combined buffer => no extra
  // copy of the user payload; TCP_NODELAY keeps it to one segment for small ones).
  char prefix[kReqPrefix];
  EncodePrefix(prefix, op, k, offset, length, payload_len);
  if (!net::WriteAll(fd, prefix, kReqPrefix)) return Status::kIOError;
  if (payload_len && !net::WriteAll(fd, payload, payload_len)) return Status::kIOError;

  char rp[kRespPrefix];
  if (!net::ReadAll(fd, rp, kRespPrefix)) return Status::kIOError;
  Status st = static_cast<Status>(static_cast<uint8_t>(rp[0]));
  uint64_t dlen = net::GetU64(rp + 1);
  if (dlen) {
    std::string data(dlen, '\0');
    if (!net::ReadAll(fd, &data[0], dlen)) return Status::kIOError;
    if (out) *out = std::move(data);
  } else if (out) {
    out->clear();
  }
  return st;
}

}  // namespace

Status TcpTransport::Cache(const std::string& node, const BlockKey& key,
                           const void* data, size_t len) {
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr);
}

Status TcpTransport::Range(const std::string& node, const BlockKey& key,
                           uint64_t offset, uint64_t length, std::string* out) {
  return RoundTrip(node, WireOp::kRange, key, offset, length, nullptr, 0, out);
}

Status TcpTransport::Exist(const std::string& node, const BlockKey& key,
                           bool* exist) {
  Status st = RoundTrip(node, WireOp::kExist, key, 0, 0, nullptr, 0, nullptr);
  if (st == Status::kOk) { *exist = true; return Status::kOk; }
  if (st == Status::kNotFound) { *exist = false; return Status::kOk; }
  return st;
}

}  // namespace kv
}  // namespace cache
}  // namespace dingofs
