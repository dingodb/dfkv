#include "kv_node_server.h"

#include <vector>

#include "net_util.h"
#include "transport.h"

namespace dingofs {
namespace cache {
namespace kv {

KvNodeServer::KvNodeServer(const std::string& cache_dir, uint64_t capacity_bytes)
    : group_(DiskCacheGroup::Options{{cache_dir}, capacity_bytes}) {}

KvNodeServer::KvNodeServer(const std::vector<std::string>& cache_dirs,
                           uint64_t capacity_bytes)
    : group_(DiskCacheGroup::Options{cache_dirs, capacity_bytes}) {}

KvNodeServer::~KvNodeServer() { Stop(); }

Status KvNodeServer::Start(int port) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return Status::kIOError;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  if (::listen(listen_fd_, 128) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  socklen_t sl = sizeof(sa);
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&sa), &sl);
  port_ = ntohs(sa.sin_port);
  running_ = true;
  accept_thread_ = std::thread([this] { AcceptLoop(); });
  return Status::kOk;
}

void KvNodeServer::Stop() {
  bool was = running_.exchange(false);
  if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); ::close(listen_fd_); listen_fd_ = -1; }
  if (was && accept_thread_.joinable()) accept_thread_.join();
}

void KvNodeServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    std::thread([this, fd] { Handle(fd); ::close(fd); }).detach();
  }
}

void KvNodeServer::Handle(int fd) {
  char prefix[kReqPrefix];
  if (!net::ReadAll(fd, prefix, kReqPrefix)) return;
  WireOp op = static_cast<WireOp>(static_cast<uint8_t>(prefix[0]));
  BlockKey key;
  key.id = net::GetU64(prefix + 1);
  key.index = net::GetU32(prefix + 9);
  key.size = net::GetU32(prefix + 13);
  uint64_t offset = net::GetU64(prefix + 17);
  uint64_t length = net::GetU64(prefix + 25);
  uint64_t payload_len = net::GetU64(prefix + 33);

  std::vector<char> payload(payload_len);
  if (payload_len && !net::ReadAll(fd, payload.data(), payload_len)) return;

  Status st = Status::kInvalid;
  std::string data;
  switch (op) {
    case WireOp::kCache:
      st = group_.Cache(key, payload.data(), payload_len);
      break;
    case WireOp::kRange:
      st = group_.Range(key, offset, length, &data);
      break;
    case WireOp::kExist:
      st = group_.IsCached(key) ? Status::kOk : Status::kNotFound;
      break;
  }

  char rp[kRespPrefix];
  rp[0] = static_cast<char>(st);
  net::PutU64(rp + 1, data.size());
  if (!net::WriteAll(fd, rp, kRespPrefix)) return;
  if (!data.empty()) net::WriteAll(fd, data.data(), data.size());
}

}  // namespace kv
}  // namespace cache
}  // namespace dingofs
