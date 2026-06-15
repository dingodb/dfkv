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
  // Listen on all interfaces so peer nodes can reach this cache node (a
  // distributed cluster needs a routable bind, not loopback). Access control is
  // by network isolation / firewall, per docs/DEPLOY.md.
  sa.sin_addr.s_addr = htonl(INADDR_ANY);
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
  if (!running_.exchange(false)) return;  // idempotent
  // shutdown() (a read of listen_fd_) unblocks accept(); join the accept loop so
  // there is no concurrent reader before we mutate/close listen_fd_.
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);
  if (accept_thread_.joinable()) accept_thread_.join();
  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  // accept loop is done; drain in-flight connections: unblock their recv(), then
  // join handler threads so group_ outlives them.
  std::vector<int> fds;
  std::vector<std::thread> threads;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    fds.assign(conn_fds_.begin(), conn_fds_.end());
    threads.swap(conn_threads_);
  }
  for (int fd : fds) ::shutdown(fd, SHUT_RDWR);
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

// Transport-agnostic request processing + metrics (shared by TCP and RDMA).
Status KvNodeServer::ProcessRequest(uint8_t op_raw, uint64_t id, uint32_t index,
                                    uint32_t ksize, uint64_t offset,
                                    uint64_t length, const char* payload,
                                    uint64_t payload_len, std::string* out_data) {
  WireOp op = static_cast<WireOp>(op_raw);
  BlockKey key{id, index, ksize};
  Status st = Status::kInvalid;
  switch (op) {
    case WireOp::kCache:
      st = group_.Cache(key, payload, payload_len);
      if (st == Status::kOk) {
        cache_put_.fetch_add(1, std::memory_order_relaxed);
        bytes_written_.fetch_add(payload_len, std::memory_order_relaxed);
      }
      break;
    case WireOp::kRange:
      st = group_.Range(key, offset, length, out_data);
      if (st == Status::kOk) {
        cache_hit_.fetch_add(1, std::memory_order_relaxed);
        bytes_read_.fetch_add(out_data->size(), std::memory_order_relaxed);
      } else if (st == Status::kNotFound) {
        cache_miss_.fetch_add(1, std::memory_order_relaxed);
      }
      break;
    case WireOp::kExist:
      if (group_.IsCached(key)) { st = Status::kOk; exist_hit_.fetch_add(1, std::memory_order_relaxed); }
      else { st = Status::kNotFound; exist_miss_.fetch_add(1, std::memory_order_relaxed); }
      break;
    case WireOp::kStats:
      *out_data = MetricsText();
      st = Status::kOk;
      break;
    case WireOp::kMembers:
      *out_data = members_;  // advertised cluster membership (for client discovery)
      st = Status::kOk;
      break;
    case WireOp::kRegister:
    case WireOp::kHeartbeat:
    case WireOp::kListMembers:
      st = Status::kInvalid;  // MDS ops are not served by a cache node
      break;
  }
  return st;
}

Status KvNodeServer::RangeInto(uint64_t id, uint32_t index, uint32_t ksize,
                               uint64_t offset, uint64_t length, char* dst,
                               size_t dst_cap, size_t* out_len) {
  BlockKey key{id, index, ksize};
  Status st = group_.RangeInto(key, offset, length, dst, dst_cap, out_len);
  if (st == Status::kOk) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(*out_len, std::memory_order_relaxed);
  } else if (st == Status::kNotFound) {
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
  }
  return st;
}

Status KvNodeServer::RangeDirect(uint64_t id, uint32_t index, uint32_t ksize,
                                 uint64_t offset, uint64_t length, char* io_buf,
                                 size_t io_cap, const char** out_data,
                                 size_t* out_len) {
  BlockKey key{id, index, ksize};
  Status st = group_.RangeDirect(key, offset, length, io_buf, io_cap, out_data, out_len);
  if (st == Status::kOk) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(*out_len, std::memory_order_relaxed);
  } else if (st == Status::kNotFound) {
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
  }
  return st;
}

// Keep-alive: serve requests on this connection until the peer closes it.
void KvNodeServer::Handle(int fd) {
  while (running_) {
    char prefix[kReqPrefix];
    if (!net::ReadAll(fd, prefix, kReqPrefix)) return;  // peer closed / error
    ReqFields rq;
    if (!DecodeReq(prefix, &rq)) return;  // bad protocol version => drop
    std::vector<char> payload(rq.payload_len);
    if (rq.payload_len && !net::ReadAll(fd, payload.data(), rq.payload_len)) return;

    std::string data;
    Status st = ProcessRequest(rq.op, rq.id, rq.index, rq.size, rq.offset,
                               rq.length, payload.data(), rq.payload_len, &data);

    char rp[kRespPrefix];
    EncodeResp(rp, st, data.size());
    if (!net::WriteAll(fd, rp, kRespPrefix)) return;
    if (!data.empty() && !net::WriteAll(fd, data.data(), data.size())) return;
  }
}

}  // namespace dfkv
