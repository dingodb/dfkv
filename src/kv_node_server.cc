#include "kv_node_server.h"

#include <string>

#include <vector>

#include "net_util.h"
#include "transport.h"

namespace dfkv {

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
  // accept loop is done; drain in-flight connections: unblock their ReadAll,
  // then join handler threads so group_ outlives them.
  std::vector<int> fds;
  std::vector<std::thread> threads;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    fds.assign(conn_fds_.begin(), conn_fds_.end());
    threads.swap(conn_threads_);
  }
  for (int fd : fds) ::shutdown(fd, SHUT_RDWR);  // unblock blocked recv()
  for (auto& t : threads) if (t.joinable()) t.join();
}

std::string KvNodeServer::MetricsText() const {
  auto line = [](const char* k, size_t v) {
    return std::string(k) + " " + std::to_string(v) + "\n";
  };
  std::string s;
  s += line("dfkv_cache_put_total", m_cache_put());
  s += line("dfkv_cache_hit_total", m_cache_hit());
  s += line("dfkv_cache_miss_total", m_cache_miss());
  s += line("dfkv_exist_hit_total", m_exist_hit());
  s += line("dfkv_exist_miss_total", m_exist_miss());
  s += line("dfkv_bytes_written_total", bytes_written_.load(std::memory_order_relaxed));
  s += line("dfkv_bytes_read_total", bytes_read_.load(std::memory_order_relaxed));
  s += line("dfkv_accepts_total", AcceptCount());
  s += line("dfkv_objects", group_.Count());
  s += line("dfkv_used_bytes", group_.UsedBytes());
  s += line("dfkv_disks", group_.DiskCount());
  return s;
}

void KvNodeServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // avoid Nagle stalls
    accept_count_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(conn_mu_);
    conn_fds_.insert(fd);
    conn_threads_.emplace_back([this, fd] {
      Handle(fd);
      { std::lock_guard<std::mutex> lk(conn_mu_); conn_fds_.erase(fd); }
      ::close(fd);
    });
  }
}

// Keep-alive: serve requests on this connection until the peer closes it.
void KvNodeServer::Handle(int fd) {
  while (running_) {
    char prefix[kReqPrefix];
    if (!net::ReadAll(fd, prefix, kReqPrefix)) return;  // peer closed / error
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
        if (st == Status::kOk) {
          cache_put_.fetch_add(1, std::memory_order_relaxed);
          bytes_written_.fetch_add(payload_len, std::memory_order_relaxed);
        }
        break;
      case WireOp::kRange:
        st = group_.Range(key, offset, length, &data);
        if (st == Status::kOk) {
          cache_hit_.fetch_add(1, std::memory_order_relaxed);
          bytes_read_.fetch_add(data.size(), std::memory_order_relaxed);
        } else if (st == Status::kNotFound) {
          cache_miss_.fetch_add(1, std::memory_order_relaxed);
        }
        break;
      case WireOp::kExist:
        if (group_.IsCached(key)) { st = Status::kOk; exist_hit_.fetch_add(1, std::memory_order_relaxed); }
        else { st = Status::kNotFound; exist_miss_.fetch_add(1, std::memory_order_relaxed); }
        break;
      case WireOp::kStats:
        data = MetricsText();
        st = Status::kOk;
        break;
    }

    char rp[kRespPrefix];
    rp[0] = static_cast<char>(st);
    net::PutU64(rp + 1, data.size());
    if (!net::WriteAll(fd, rp, kRespPrefix)) return;
    if (!data.empty() && !net::WriteAll(fd, data.data(), data.size())) return;
  }
}

}  // namespace dfkv
