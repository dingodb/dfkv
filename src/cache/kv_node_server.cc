#include "cache/kv_node_server.h"

#include <chrono>
#include <string>

#include <vector>

#include "utils/net_util.h"
#include "utils/prom_escape.h"
#include "transport/transport.h"

namespace dfkv {

namespace {
// Monotonic seconds; only read on the ~1/64 sampled ops, so the cost is amortized
// away on the hot path.
inline double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

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
  std::vector<Conn> conns;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    fds.assign(conn_fds_.begin(), conn_fds_.end());
    conns.swap(conns_);
  }
  for (int fd : fds) ::shutdown(fd, SHUT_RDWR);
  for (auto& c : conns) if (c.th.joinable()) c.th.join();
}

void KvNodeServer::ReapDoneLocked() {
  for (auto it = conns_.begin(); it != conns_.end();) {
    if (it->done->load(std::memory_order_acquire)) {
      if (it->th.joinable()) it->th.join();
      it = conns_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t KvNodeServer::live_conn_count() {
  std::lock_guard<std::mutex> lk(conn_mu_);
  return conns_.size();
}

#ifndef DFKV_VERSION
#define DFKV_VERSION "dev"
#endif
#ifdef DFKV_WITH_RDMA
#define DFKV_BUILD_TRANSPORT "rdma"
#else
#define DFKV_BUILD_TRANSPORT "tcp"
#endif

std::string KvNodeServer::MetricsText() const {
  // Node identity as an inner label set (no braces), empty when unset so a
  // single-node scrape and older tooling keep the bare `metric N` form.
  std::string idlabels;
  if (!node_id_.empty() || !node_group_.empty())
    idlabels = "node=\"" + PromLabelEscape(node_id_) + "\",group=\"" +
               PromLabelEscape(node_group_) + "\"";
  // braces(extra): merge `extra` label set with the identity labels.
  auto braces = [&](const std::string& extra) {
    std::string in = extra;
    if (!idlabels.empty()) in += (in.empty() ? "" : ",") + idlabels;
    return in.empty() ? std::string() : "{" + in + "}";
  };
  const std::string lbl = braces("");
  std::string s;
  auto metric = [&](const char* name, const char* type, const char* help,
                    uint64_t v) {
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
    s += name; s += lbl; s += " "; s += std::to_string(v); s += "\n";
  };
  auto rd = [](const std::atomic<size_t>& a) { return a.load(std::memory_order_relaxed); };
  // build/version + uptime
  s += "# HELP dfkv_build_info Build and version info (constant 1)\n";
  s += "# TYPE dfkv_build_info gauge\n";
  s += std::string("dfkv_build_info{version=\"") + DFKV_VERSION +
       "\",transport=\"" DFKV_BUILD_TRANSPORT "\"";
  if (!node_id_.empty() || !node_group_.empty())
    s += ",node=\"" + PromLabelEscape(node_id_) + "\",group=\"" +
         PromLabelEscape(node_group_) + "\"";
  s += "} 1\n";
  uint64_t up = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start_time_).count();
  metric("dfkv_uptime_seconds", "gauge", "Seconds since node start", up);
  // counters
  metric("dfkv_cache_put_total", "counter", "Cache (PUT) ops accepted", m_cache_put());
  metric("dfkv_cache_hit_total", "counter", "Range (GET) ops that hit", m_cache_hit());
  metric("dfkv_cache_miss_total", "counter", "Range (GET) ops that missed", m_cache_miss());
  metric("dfkv_exist_hit_total", "counter", "Exist checks that hit", m_exist_hit());
  metric("dfkv_exist_miss_total", "counter", "Exist checks that missed", m_exist_miss());
  metric("dfkv_remove_ok_total", "counter", "Remove ops that dropped a block", m_remove_ok());
  metric("dfkv_remove_miss_total", "counter", "Remove ops for an absent block", m_remove_miss());
  metric("dfkv_bytes_written_total", "counter", "Payload bytes written",
         bytes_written_.load(std::memory_order_relaxed));
  metric("dfkv_bytes_read_total", "counter", "Payload bytes read",
         bytes_read_.load(std::memory_order_relaxed));
  metric("dfkv_accepts_total", "counter", "TCP connections accepted", AcceptCount());
  metric("dfkv_evictions_total", "counter", "Objects evicted (capacity pressure)",
         group_.Evictions());
  metric("dfkv_evicted_bytes_total", "counter", "Bytes evicted", group_.EvictedBytes());
  // gauges
  metric("dfkv_objects", "gauge", "Cached objects on this node", group_.Count());
  metric("dfkv_used_bytes", "gauge", "Bytes used on disk", group_.UsedBytes());
  metric("dfkv_disks", "gauge", "Backing NVMe disks", group_.DiskCount());
  metric("dfkv_open_connections", "gauge", "Currently open client connections",
         rd(open_connections_));
  // errors by op+status (one HELP/TYPE, multiple labeled series)
  s += "# HELP dfkv_errors_total Failed ops by op and status\n";
  s += "# TYPE dfkv_errors_total counter\n";
  s += "dfkv_errors_total" + braces("op=\"put\",status=\"io\"") + " " +
       std::to_string(rd(put_io_err_)) + "\n";
  s += "dfkv_errors_total" + braces("op=\"get\",status=\"io\"") + " " +
       std::to_string(rd(get_io_err_)) + "\n";
  s += "dfkv_errors_total" + braces("op=\"any\",status=\"invalid\"") + " " +
       std::to_string(rd(invalid_ops_)) + "\n";
  // per-disk gauges (one HELP/TYPE, one series per disk)
  s += "# HELP dfkv_disk_used_bytes Bytes used per backing disk\n";
  s += "# TYPE dfkv_disk_used_bytes gauge\n";
  for (size_t i = 0; i < group_.DiskCount(); ++i)
    s += "dfkv_disk_used_bytes" + braces("disk=\"" + PromLabelEscape(group_.DiskPath(i)) + "\"") + " " +
         std::to_string(group_.DiskUsedBytes(i)) + "\n";
  s += "# HELP dfkv_disk_objects Objects per backing disk\n";
  s += "# TYPE dfkv_disk_objects gauge\n";
  for (size_t i = 0; i < group_.DiskCount(); ++i)
    s += "dfkv_disk_objects" + braces("disk=\"" + PromLabelEscape(group_.DiskPath(i)) + "\"") + " " +
         std::to_string(group_.DiskObjects(i)) + "\n";
  // sampled op latency histograms (op label merged with identity); one HELP/TYPE,
  // two label sets (get/put).
  {
    std::string get_lbl = std::string("op=\"get\"") + (idlabels.empty() ? "" : "," + idlabels);
    std::string put_lbl = std::string("op=\"put\"") + (idlabels.empty() ? "" : "," + idlabels);
    s += "# HELP dfkv_op_latency_seconds Sampled server op latency seconds\n";
    s += "# TYPE dfkv_op_latency_seconds histogram\n";
    s += get_lat_.RenderBody("dfkv_op_latency_seconds", get_lbl);
    s += put_lat_.RenderBody("dfkv_op_latency_seconds", put_lbl);
  }
  return s;
}

void KvNodeServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // avoid Nagle stalls
    accept_count_.fetch_add(1, std::memory_order_relaxed);
    open_connections_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { open_connections_.fetch_sub(1, std::memory_order_relaxed); ::close(fd); break; }
    ReapDoneLocked();  // join handlers that finished since the last accept
    conn_fds_.insert(fd);
    auto done = std::make_shared<std::atomic<bool>>(false);
    conns_.push_back({std::thread([this, fd, done] {
                        Handle(fd);
                        open_connections_.fetch_sub(1, std::memory_order_relaxed);
                        { std::lock_guard<std::mutex> lk(conn_mu_); conn_fds_.erase(fd); }
                        ::close(fd);
                        done->store(true, std::memory_order_release);  // last act
                      }),
                      done});
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
    case WireOp::kCache: {
      bool samp = lat_sampler_.ShouldSample();
      double t0 = samp ? NowSec() : 0.0;
      st = group_.Cache(key, payload, payload_len);
      if (st == Status::kOk) {
        cache_put_.fetch_add(1, std::memory_order_relaxed);
        bytes_written_.fetch_add(payload_len, std::memory_order_relaxed);
      } else if (st == Status::kIOError) {
        put_io_err_.fetch_add(1, std::memory_order_relaxed);
      }
      if (samp) put_lat_.Observe(NowSec() - t0);
      break;
    }
    case WireOp::kRange: {
      bool samp = lat_sampler_.ShouldSample();
      double t0 = samp ? NowSec() : 0.0;
      st = group_.Range(key, offset, length, out_data);
      if (st == Status::kOk) {
        cache_hit_.fetch_add(1, std::memory_order_relaxed);
        bytes_read_.fetch_add(out_data->size(), std::memory_order_relaxed);
      } else if (st == Status::kNotFound) {
        cache_miss_.fetch_add(1, std::memory_order_relaxed);
      } else if (st == Status::kIOError) {
        get_io_err_.fetch_add(1, std::memory_order_relaxed);
      }
      if (samp) get_lat_.Observe(NowSec() - t0);
      break;
    }
    case WireOp::kExist:
      if (group_.IsCached(key)) { st = Status::kOk; exist_hit_.fetch_add(1, std::memory_order_relaxed); }
      else { st = Status::kNotFound; exist_miss_.fetch_add(1, std::memory_order_relaxed); }
      break;
    case WireOp::kRemove:
      st = group_.Remove(key);
      if (st == Status::kOk) remove_ok_.fetch_add(1, std::memory_order_relaxed);
      else if (st == Status::kNotFound) remove_miss_.fetch_add(1, std::memory_order_relaxed);
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
  if (st == Status::kInvalid) invalid_ops_.fetch_add(1, std::memory_order_relaxed);
  return st;
}

Status KvNodeServer::RangeInto(uint64_t id, uint32_t index, uint32_t ksize,
                               uint64_t offset, uint64_t length, char* dst,
                               size_t dst_cap, size_t* out_len) {
  BlockKey key{id, index, ksize};
  bool samp = lat_sampler_.ShouldSample();
  double t0 = samp ? NowSec() : 0.0;
  Status st = group_.RangeInto(key, offset, length, dst, dst_cap, out_len);
  if (st == Status::kOk) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(*out_len, std::memory_order_relaxed);
  } else if (st == Status::kNotFound) {
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
  } else if (st == Status::kIOError) {
    get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  if (samp) get_lat_.Observe(NowSec() - t0);
  return st;
}

Status KvNodeServer::CacheDirect(uint64_t id, uint32_t index, uint32_t ksize,
                                 char* data, size_t len, size_t cap) {
  BlockKey key{id, index, ksize};
  bool samp = lat_sampler_.ShouldSample();
  double t0 = samp ? NowSec() : 0.0;
  Status st = group_.CacheDirect(key, data, len, cap);
  if (st == Status::kOk) {
    cache_put_.fetch_add(1, std::memory_order_relaxed);
    bytes_written_.fetch_add(len, std::memory_order_relaxed);
  } else if (st == Status::kIOError) {
    put_io_err_.fetch_add(1, std::memory_order_relaxed);
  } else if (st == Status::kInvalid) {
    invalid_ops_.fetch_add(1, std::memory_order_relaxed);
  }
  if (samp) put_lat_.Observe(NowSec() - t0);
  return st;
}

Status KvNodeServer::RangeDirect(uint64_t id, uint32_t index, uint32_t ksize,
                                 uint64_t offset, uint64_t length, char* io_buf,
                                 size_t io_cap, const char** out_data,
                                 size_t* out_len) {
  BlockKey key{id, index, ksize};
  bool samp = lat_sampler_.ShouldSample();
  double t0 = samp ? NowSec() : 0.0;
  Status st = group_.RangeDirect(key, offset, length, io_buf, io_cap, out_data, out_len);
  if (st == Status::kOk) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(*out_len, std::memory_order_relaxed);
  } else if (st == Status::kNotFound) {
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
  } else if (st == Status::kIOError) {
    get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  if (samp) get_lat_.Observe(NowSec() - t0);
  return st;
}

Status KvNodeServer::RangeDirectPrep(uint64_t id, uint32_t index, uint32_t ksize,
                                     uint64_t offset, uint64_t length,
                                     size_t io_cap, KVStore::RangePrep* out) {
  BlockKey key{id, index, ksize};
  Status st = group_.RangeDirectPrep(key, offset, length, io_cap, out);
  // Only miss/io-error are final here; a kOk prep is accounted on read completion
  // (RangeDirectComplete) because the async read can still fail.
  if (st == Status::kNotFound) {
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
  } else if (st == Status::kIOError) {
    get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  return st;
}

void KvNodeServer::RangeDirectComplete(bool ok, size_t bytes_read) {
  if (ok) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(bytes_read, std::memory_order_relaxed);
  } else {
    get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
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
