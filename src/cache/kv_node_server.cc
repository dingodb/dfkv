#include "cache/kv_node_server.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

#include <vector>

#include "utils/net_util.h"
#include "utils/wire_limits.h"
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
    : group_(DiskCacheGroup::Options{{cache_dir}, capacity_bytes}) { InitRamTier(); }

KvNodeServer::KvNodeServer(const std::vector<std::string>& cache_dirs,
                           uint64_t capacity_bytes)
    : group_(DiskCacheGroup::Options{cache_dirs, capacity_bytes}) { InitRamTier(); }

KvNodeServer::~KvNodeServer() {
  // Order matters: Stop() first joins the accept + handler threads so no request
  // is still reading ram_; THEN reset ram_ (stops+joins its flusher, whose
  // callback calls group_.Cache -- group_ is a member, still alive here); group_
  // is destroyed last, after this body.
  Stop();
  ram_.reset();
}

void KvNodeServer::InitRamTier() {
  // Off by default. DFKV_RAM_TIER in {1,on,true,yes} enables the RAM hot tier;
  // DFKV_RAM_TIER_BYTES sizes the pre-registered arena (default 4 GiB).
  const char* e = std::getenv("DFKV_RAM_TIER");
  if (!e || !*e) return;
  const std::string v(e);
  if (v != "1" && v != "on" && v != "true" && v != "yes") return;
  RamTier::Options o;
  if (const char* b = std::getenv("DFKV_RAM_TIER_BYTES")) {
    unsigned long long n = std::strtoull(b, nullptr, 10);
    if (n > 0) o.bytes = n;
  }
  // Flusher persists a RAM slot to the disk group (same bytes GET returns).
  auto tier = std::make_unique<RamTier>(o, [this](const BlockKey& k, const void* d, size_t l) {
    return group_.Cache(k, d, l) == Status::kOk;
  });
  if (tier->ok()) ram_ = std::move(tier);  // arena alloc failed => stay disk-only
}

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
  metric("dfkv_wire_v1_requests_total", "counter", "TCP requests decoded as wire v1",
         wire_v1_requests_.load(std::memory_order_relaxed));
  metric("dfkv_wire_v2_requests_total", "counter", "TCP requests decoded as wire v2",
         wire_v2_requests_.load(std::memory_order_relaxed));
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
  // RAM hot tier (P3), emitted only when enabled so a disk-only node's scrape is
  // unchanged. ram_hit/ram_miss let you compute the RAM hit rate; ram_put_bypass
  // is the flush-backpressure signal from the design's gap 10.3.
  if (ram_) {
    metric("dfkv_ram_hit_total", "counter", "GETs served from the RAM hot tier", ram_->Hits());
    metric("dfkv_ram_miss_total", "counter", "GETs that missed RAM (fell to disk)", ram_->Misses());
    metric("dfkv_ram_put_total", "counter", "PUTs accepted into the RAM hot tier", ram_->Puts());
    metric("dfkv_ram_put_bypass_total", "counter", "PUTs bypassing RAM (flush backpressure)", ram_->PutBypass());
    metric("dfkv_ram_flushed_total", "counter", "RAM slots flushed to disk (now durable)", ram_->Flushed());
    metric("dfkv_ram_flush_dropped_total", "counter", "RAM slots dropped after flush failure", ram_->FlushDropped());
    metric("dfkv_ram_evictions_total", "counter", "RAM slots evicted under capacity pressure", ram_->Evictions());
    metric("dfkv_ram_objects", "gauge", "Blocks currently resident in the RAM hot tier", ram_->Count());
    metric("dfkv_ram_flush_backlog", "gauge", "RAM slots queued for flush (not yet durable)", ram_->FlushBacklog());
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
      // Write-through RAM tier: on acceptance the value is synchronously visible
      // in RAM (read-after-write) and flushed to disk in the background. On
      // backpressure (arena full of un-flushed slots) Put declines and we take
      // the normal disk path.
      if (ram_ && ram_->Put(key, payload, payload_len)) {
        st = Status::kOk;
        cache_put_.fetch_add(1, std::memory_order_relaxed);
        bytes_written_.fetch_add(payload_len, std::memory_order_relaxed);
      } else {
        st = group_.Cache(key, payload, payload_len);
        if (st == Status::kOk) {
          cache_put_.fetch_add(1, std::memory_order_relaxed);
          bytes_written_.fetch_add(payload_len, std::memory_order_relaxed);
        } else if (st == Status::kIOError) {
          put_io_err_.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (samp) put_lat_.Observe(NowSec() - t0);
      break;
    }
    case WireOp::kRange: {
      bool samp = lat_sampler_.ShouldSample();
      double t0 = samp ? NowSec() : 0.0;
      bool ram_served = false;
      if (ram_) {  // RAM hot tier: serve straight from the arena (copy out), no disk
        RamTier::Hit h;
        if (ram_->GetPrep(key, offset, length, &h)) {
          out_data->assign(h.ptr, h.len);
          ram_->Release(h.token);
          st = Status::kOk;
          cache_hit_.fetch_add(1, std::memory_order_relaxed);
          bytes_read_.fetch_add(out_data->size(), std::memory_order_relaxed);
          ram_served = true;
        }
      }
      if (!ram_served) {
        st = group_.Range(key, offset, length, out_data);
        if (st == Status::kOk) {
          cache_hit_.fetch_add(1, std::memory_order_relaxed);
          bytes_read_.fetch_add(out_data->size(), std::memory_order_relaxed);
        } else if (st == Status::kNotFound) {
          cache_miss_.fetch_add(1, std::memory_order_relaxed);
        } else if (st == Status::kIOError) {
          get_io_err_.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (samp) get_lat_.Observe(NowSec() - t0);
      break;
    }
    case WireOp::kExist:
      // A RAM-resident block (possibly RAM-only, not yet flushed) must count as
      // present, else Exist would report absent for a just-written key.
      if ((ram_ && ram_->Contains(key)) || group_.IsCached(key)) {
        st = Status::kOk; exist_hit_.fetch_add(1, std::memory_order_relaxed);
      } else {
        st = Status::kNotFound; exist_miss_.fetch_add(1, std::memory_order_relaxed);
      }
      break;
    case WireOp::kRemove: {
      // Drop from both tiers. RamTier::Remove is best-effort (a still-flushing or
      // in-flight slot declines and is reclaimed later under pressure); the L2
      // eviction this backs targets durable blocks, which drop cleanly.
      bool ram_had = ram_ && ram_->Remove(key);
      Status ds = group_.Remove(key);
      if (ram_had || ds == Status::kOk) {
        st = Status::kOk; remove_ok_.fetch_add(1, std::memory_order_relaxed);
      } else if (ds == Status::kNotFound) {
        st = Status::kNotFound; remove_miss_.fetch_add(1, std::memory_order_relaxed);
      } else {
        st = ds;
      }
      break;
    }
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
  if (ram_) {
    RamTier::Hit h;
    if (ram_->GetPrep(key, offset, length, &h)) {  // RAM hit: copy out, no disk
      size_t n = h.len < dst_cap ? h.len : dst_cap;
      std::memcpy(dst, h.ptr, n);
      ram_->Release(h.token);
      if (out_len) *out_len = n;
      cache_hit_.fetch_add(1, std::memory_order_relaxed);
      bytes_read_.fetch_add(n, std::memory_order_relaxed);
      if (samp) get_lat_.Observe(NowSec() - t0);
      return Status::kOk;
    }
  }
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
  // RAM write-through (same [ValueHeader|payload] blob GET returns); backpressure
  // or disabled falls through to the O_DIRECT disk write.
  Status st;
  if (ram_ && ram_->Put(key, data, len)) {
    st = Status::kOk;
    cache_put_.fetch_add(1, std::memory_order_relaxed);
    bytes_written_.fetch_add(len, std::memory_order_relaxed);
  } else {
    st = group_.CacheDirect(key, data, len, cap);
    if (st == Status::kOk) {
      cache_put_.fetch_add(1, std::memory_order_relaxed);
      bytes_written_.fetch_add(len, std::memory_order_relaxed);
    } else if (st == Status::kIOError) {
      put_io_err_.fetch_add(1, std::memory_order_relaxed);
    } else if (st == Status::kInvalid) {
      invalid_ops_.fetch_add(1, std::memory_order_relaxed);
    }
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
  if (ram_) {
    RamTier::Hit h;
    if (ram_->GetPrep(key, offset, length, &h)) {
      // RAM hit: copy the value into the caller's registered io_buf so the RDMA
      // layer scatter-sends it from there (no disk, no open, no pread). True
      // zero-copy (send straight from the arena MR) is B5-3.
      size_t n = h.len < io_cap ? h.len : io_cap;
      std::memcpy(io_buf, h.ptr, n);
      ram_->Release(h.token);
      if (out_data) *out_data = io_buf;
      if (out_len) *out_len = n;
      cache_hit_.fetch_add(1, std::memory_order_relaxed);
      bytes_read_.fetch_add(n, std::memory_order_relaxed);
      if (samp) get_lat_.Observe(NowSec() - t0);
      return Status::kOk;
    }
  }
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
  // A RAM-resident key has no fd to hand io_uring (and may be RAM-only, not yet
  // on disk). Decline the async prep (kInvalid) WITHOUT counting a miss so the
  // RDMA serve loop falls back to the synchronous RangeDirect, which serves it
  // from the arena. (Only reached when the RAM tier is enabled.)
  if (ram_ && ram_->Contains(key)) {
    if (out) *out = KVStore::RangePrep{};
    return Status::kInvalid;
  }
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

bool KvNodeServer::RamRangePrep(uint64_t id, uint32_t index, uint32_t ksize,
                                uint64_t offset, uint64_t length,
                                const char** out_ptr, size_t* out_len,
                                uint64_t* out_token) {
  if (!ram_) return false;
  RamTier::Hit h;
  if (!ram_->GetPrep(BlockKey{id, index, ksize}, offset, length, &h)) return false;
  if (out_ptr) *out_ptr = h.ptr;
  if (out_len) *out_len = h.len;
  if (out_token) *out_token = h.token;
  cache_hit_.fetch_add(1, std::memory_order_relaxed);
  bytes_read_.fetch_add(h.len, std::memory_order_relaxed);
  return true;
}

void KvNodeServer::RamRelease(uint64_t token) {
  if (ram_) ram_->Release(token);
}

// Keep-alive: serve requests on this connection until the peer closes it.
void KvNodeServer::Handle(int fd) {
  // Bound the declared payload_len BEFORE the allocation below: without this
  // a forged 42-byte prefix is a 16 GiB allocation. Same bound as the RDMA
  // path (utils/wire_limits.h).
  const uint64_t max_payload = max_request_payload_
                                   ? max_request_payload_
                                   : wire_limits::MaxRequestPayload();
  while (running_) {
    // Dual-accept v1/v2: read the common v1 prefix, then 8 more bytes only when
    // the version byte marks a v2 frame; reply in the SAME version. A v1 and a
    // v2 client thus share this server with no flag-day upgrade.
    char prefix[kReqPrefixV2];
    if (!net::ReadAll(fd, prefix, kReqPrefix)) return;  // peer closed / error
    if (static_cast<uint8_t>(prefix[0]) == kProtoVersionV2 &&
        !net::ReadAll(fd, prefix + kReqPrefix, kReqPrefixV2 - kReqPrefix))
      return;
    ReqFields rq;
    uint8_t ver = DecodeReq(prefix, &rq, max_payload);
    if (!ver) return;  // bad version / oversize => drop
    (ver == kProtoVersionV2 ? wire_v2_requests_ : wire_v1_requests_)
        .fetch_add(1, std::memory_order_relaxed);
    std::vector<char> payload(rq.payload_len);
    if (rq.payload_len && !net::ReadAll(fd, payload.data(), rq.payload_len)) return;

    std::string data;
    Status st = ProcessRequest(rq.op, rq.id, rq.index, rq.size, rq.offset,
                               rq.length, payload.data(), rq.payload_len, &data);

    char rp[kRespPrefixV2];
    size_t rlen;
    if (ver == kProtoVersionV2) { EncodeRespV2(rp, st, data.size(), rq.seq); rlen = kRespPrefixV2; }
    else { EncodeResp(rp, st, data.size()); rlen = kRespPrefix; }
    if (!net::WriteAll(fd, rp, rlen)) return;
    if (!data.empty() && !net::WriteAll(fd, data.data(), data.size())) return;
  }
}

}  // namespace dfkv
