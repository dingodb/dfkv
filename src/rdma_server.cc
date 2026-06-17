#include "rdma_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "net_util.h"     // ReadAll / WriteAll / Get*/Put*
#include "numa_util.h"    // pin serve thread to the device's NUMA node
#include "rdma_verbs.h"   // RcEndpoint, QpInfo
#include "transport.h"    // kReqPrefix, kRespPrefix
#include "value_header.h"

namespace dfkv {

namespace {
size_t EnvBytes(const char* name, size_t dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  errno = 0;
  char* end = nullptr;
  unsigned long long x = std::strtoull(v, &end, 10);
  if (errno != 0 || end == v || x == 0) return dflt;
  constexpr unsigned long long kMaxSge =
      static_cast<unsigned long long>(std::numeric_limits<uint32_t>::max() -
                                      ValueHeader::kSize - 2 * rdma::kDirectIoAlign);
  if (x > kMaxSge) x = kMaxSge;
  return static_cast<size_t>(x);
}

size_t ResolveMaxPayload(size_t configured) {
  size_t n = configured ? configured : (64u << 20);
  n = EnvBytes("DFKV_RDMA_MAX_PAYLOAD_BYTES", n);
  n = EnvBytes("DFKV_RDMA_MAX_MSG_BYTES", n);
  return n;
}

size_t ControlCapFor(size_t max_payload) {
  constexpr size_t kDefaultControlCap = 8u << 20;
  constexpr size_t kMinControlCap = kReqPrefix + ValueHeader::kSize;
  size_t cap = std::min(kDefaultControlCap, max_payload);
  return cap < kMinControlCap ? kMinControlCap : cap;
}
}  // namespace

RdmaServer::RdmaServer(Handler handler, size_t max_msg, const std::string& dev_name)
    : handler_(std::move(handler)),
      max_msg_(ResolveMaxPayload(max_msg)),
      control_cap_(ControlCapFor(max_msg_)),
      dev_name_(dev_name) {
  if (dev_name_.empty()) {
    const char* e = std::getenv("DFKV_RDMA_DEV");
    if (e && *e) dev_name_ = e;
  }
}

RdmaServer::~RdmaServer() { Stop(); }

Status RdmaServer::Start(int port) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return Status::kIOError;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_ANY);  // bootstrap reachable on any IP net
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

void RdmaServer::Stop() {
  if (!running_.exchange(false)) return;
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);  // wake accept()
  if (accept_thread_.joinable()) accept_thread_.join();
  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  // Wake every in-flight Serve thread out of WaitComp, then join them all so no
  // handler call can race the owner's destruction after Stop() returns.
  std::vector<Conn> conns;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    for (rdma::RcEndpoint* ep : live_eps_) ep->Wake();
    conns.swap(conns_);
  }
  for (auto& c : conns) if (c.th.joinable()) c.th.join();
}

// Join and drop any Serve threads that have already finished. Called from
// AcceptLoop under conn_mu_; only threads whose `done` is set are touched, and a
// thread sets `done` only after its final conn_mu_ release, so join() never
// blocks here. This keeps conns_ bounded by the live (not lifetime) conn count.
void RdmaServer::ReapDoneLocked() {
  for (auto it = conns_.begin(); it != conns_.end();) {
    if (it->done->load(std::memory_order_acquire)) {
      if (it->th.joinable()) it->th.join();
      it = conns_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t RdmaServer::live_conn_count() {
  std::lock_guard<std::mutex> lk(conn_mu_);
  return conns_.size();
}

void RdmaServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval tv{10, 0};  // bound the bootstrap handshake so a stalled client
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));  // can't hang Stop()
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { ::close(fd); break; }
    ReapDoneLocked();  // reap connections that finished since the last accept
    auto done = std::make_shared<std::atomic<bool>>(false);
    conns_.push_back({std::thread([this, fd, done] {
                        Serve(fd);
                        done->store(true, std::memory_order_release);  // last act
                      }),
                      done});
  }
}

namespace {
size_t ServerDepth() {
  // Pipeline depth (requests in flight per connection). Default 1 keeps per-conn
  // pinned memory low; set DFKV_RDMA_DEPTH>1 to enable pipelining (helps the
  // latency-bound PUT path; GET scales via more conns).
  const char* e = std::getenv("DFKV_RDMA_DEPTH");
  if (e && *e) { long v = std::strtol(e, nullptr, 10); if (v >= 1 && v <= 256) return (size_t)v; }
  return 1;
}

int ServerIdleMs() {
  // Per-connection idle timeout. A connection with no completions for this long
  // is reclaimed: its Serve thread returns (freeing the QP, pinned buffers, and
  // the thread itself, which ReapDoneLocked then joins). Without this, a Serve
  // thread blocks in WaitComp forever after a silent client disconnect (a torn-
  // down RC peer yields no completion), so a long-running server accumulates one
  // live thread per lifetime connection. Reclaiming idle connections is safe:
  // the client re-dials a stale pooled connection via RdmaTransport's 2-attempt
  // retry. Default 10 min keeps active/recently-used pooled conns alive; set
  // DFKV_RDMA_IDLE_MS=0 to disable (block forever, the legacy behavior).
  const char* e = std::getenv("DFKV_RDMA_IDLE_MS");
  if (e && *e) {
    long v = std::strtol(e, nullptr, 10);
    if (v <= 0) return -1;               // disabled => block forever
    if (v > 86400000) v = 86400000;      // clamp to 24h
    return static_cast<int>(v);
  }
  return 600000;  // 10 minutes
}

}  // namespace

void RdmaServer::Serve(int boot_fd) {
  // Bootstrap: client first names the device it wants us to use (same rail for
  // multi-rail); fall back to our configured default if it sends an empty name.
  char devbuf[rdma::kDevNameBytes];
  if (!net::ReadAll(boot_fd, devbuf, rdma::kDevNameBytes)) { ::close(boot_fd); return; }
  devbuf[rdma::kDevNameBytes - 1] = '\0';
  std::string dev = devbuf[0] ? std::string(devbuf) : dev_name_;

  rdma::RcEndpoint ep;
  const size_t K = ServerDepth();
  const size_t direct_cap = ValueHeader::kSize + max_msg_;
  if (!ep.Open(dev.empty() ? nullptr : dev.c_str(), control_cap_, K,
               /*ib_port=*/1, /*direct_io_buffers=*/true, direct_cap)) {
    ::close(boot_fd); return;
  }
  numa::PinThreadToNode(ep.numa_node());  // keep this conn's serve thread NUMA-local to its NIC
  // QP bootstrap: read client's info, send ours (symmetric to the client).
  char peer[rdma::kQpInfoBytes], mine[rdma::kQpInfoBytes];
  rdma::SerializeQpInfo(ep.Local(), mine);
  if (!net::ReadAll(boot_fd, peer, rdma::kQpInfoBytes) ||
      !net::WriteAll(boot_fd, mine, rdma::kQpInfoBytes)) {
    ::close(boot_fd); return;
  }
  if (!ep.Connect(rdma::ParseQpInfo(peer))) { ::close(boot_fd); return; }
  auto post_request_recv = [&](size_t slot) {
    if (!ep.dbuf(slot) || !ep.dmr(slot) || ep.dbuf_cap() <= ValueHeader::kSize)
      return ep.PostRecv(slot);
    return ep.PostRecvScatter(slot, ep.dbuf(slot) + ValueHeader::kSize,
                              ep.dbuf_cap() - ValueHeader::kSize, ep.dmr(slot),
                              kReqPrefix + ValueHeader::kSize);
  };

  // Post all K receives so the client may keep up to K requests in flight
  // (pipelining). recv buffers and send buffers are independent slot pools.
  bool armed = true;
  for (size_t i = 0; i < K; ++i) armed = armed && post_request_recv(i);
  if (!armed) { ::close(boot_fd); return; }
  // Tell the client we are ready (recvs posted) so its first SENDs won't hit RNR.
  char ready = 1;
  bool ok = net::WriteAll(boot_fd, &ready, 1);
  ::close(boot_fd);  // bootstrap done
  if (!ok) return;

  // Register this endpoint so Stop() can Wake() us out of WaitComp and join. The
  // running_ check under conn_mu_ closes the race with a concurrent Stop(): either
  // Stop sees us in live_eps_ (and wakes us) or we see running_==false here.
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) return;
    live_eps_.insert(&ep);
  }

  // Send-slot free list (a reply uses one send slot until its SEND completes).
  std::vector<size_t> free_send;
  free_send.reserve(K);
  for (size_t i = 0; i < K; ++i) free_send.push_back(i);

  struct Reply {
    bool scatter = false;
    bool defer_recv_rearm = false;
    size_t recv_slot = 0;
    size_t first_len = 0;  // plain: full sbuf length; scatter: sbuf header length
    const void* payload = nullptr;
    size_t payload_len = 0;
    ibv_mr* payload_mr = nullptr;
  };

  // Build the reply for one request. Generic ops are materialized in sbuf. For
  // kRange with a direct range handler set, O_DIRECT reads into ep.dbuf[recv_slot]
  // (4096-aligned + registered) and the wire reply is scatter-sent as
  // [resp-prefix from sbuf | value bytes from dbuf], with no payload memcpy.
  // For kCache with a direct cache handler set, the RDMA RECV has already placed
  // the user payload into ep.dbuf[recv_slot]+ValueHeader::kSize; only the 48B
  // value header is copied over so disk writes see one contiguous aligned blob.
  auto copy_payload = [&](size_t recv_slot, const ReqFields& rq, uint32_t recv_bytes,
                          std::string* payload) -> bool {
    payload->clear();
    if (rq.payload_len == 0) return true;
    if (recv_bytes < kReqPrefix + rq.payload_len) return false;
    if (rq.payload_len > static_cast<uint64_t>(ValueHeader::kSize + max_msg_))
      return false;
    payload->resize(static_cast<size_t>(rq.payload_len));
    const size_t head = static_cast<size_t>(
        std::min<uint64_t>(rq.payload_len, ValueHeader::kSize));
    if (head) std::memcpy(payload->data(), ep.rbuf(recv_slot) + kReqPrefix, head);
    const size_t rest = static_cast<size_t>(rq.payload_len) - head;
    if (rest != 0) {
      if (!ep.dbuf(recv_slot) || ep.dbuf_cap() < ValueHeader::kSize + rest) return false;
      std::memcpy(payload->data() + head, ep.dbuf(recv_slot) + ValueHeader::kSize, rest);
    }
    return true;
  };

  auto build_reply = [&](size_t send_slot, size_t recv_slot, const ReqFields& rq,
                         uint32_t recv_bytes,
                         Reply* reply) -> bool {
    char* sb = ep.sbuf(send_slot);
    auto invalid_reply = [&] {
      EncodeResp(sb, Status::kInvalid, 0);
      reply->first_len = kRespPrefix;
      return true;
    };
    if (rq.op == static_cast<uint8_t>(WireOp::kRange) && range_handler_) {
      if (!ep.dbuf(recv_slot) || !ep.dmr(recv_slot)) return false;
      if (rq.length > static_cast<uint64_t>(ValueHeader::kSize + max_msg_))
        return invalid_reply();
      const char* out_data = nullptr;
      size_t out_len = 0;
      Status st = range_handler_(rq.id, rq.index, rq.size, rq.offset, rq.length,
                                 ep.dbuf(recv_slot), ep.dbuf_cap(), &out_data, &out_len);
      uint64_t dlen = (st == Status::kOk) ? out_len : 0;
      EncodeResp(sb, st, dlen);
      reply->first_len = kRespPrefix;
      if (st == Status::kOk && dlen != 0) {
        if (!out_data) return false;
        reply->scatter = true;
        reply->defer_recv_rearm = true;
        reply->recv_slot = recv_slot;
        reply->payload = out_data;
        reply->payload_len = out_len;
        reply->payload_mr = ep.dmr(recv_slot);
      }
      return true;
    }

    if (rq.op == static_cast<uint8_t>(WireOp::kCache) && cache_direct_handler_) {
      if (!ep.dbuf(recv_slot) || !ep.dmr(recv_slot)) return false;
      if (rq.payload_len < ValueHeader::kSize ||
          rq.payload_len > static_cast<uint64_t>(ValueHeader::kSize + max_msg_) ||
          recv_bytes < kReqPrefix + rq.payload_len) {
        return invalid_reply();
      }
      std::memcpy(ep.dbuf(recv_slot), ep.rbuf(recv_slot) + kReqPrefix,
                  ValueHeader::kSize);
      Status st = cache_direct_handler_(
          rq.id, rq.index, rq.size, ep.dbuf(recv_slot),
          static_cast<size_t>(rq.payload_len), ep.dbuf_cap());
      EncodeResp(sb, st, 0);
      reply->first_len = kRespPrefix;
      return true;
    }

    std::string payload_buf;
    const char* payload = nullptr;
    if (rq.payload_len != 0) {
      if (!copy_payload(recv_slot, rq, recv_bytes, &payload_buf)) return invalid_reply();
      payload = payload_buf.data();
    }
    std::string data;
    Status st = handler_(rq.op, rq.id, rq.index, rq.size, rq.offset, rq.length,
                         payload, rq.payload_len, &data);
    if (kRespPrefix + data.size() > ep.cap()) return false;
    EncodeResp(sb, st, data.size());
    if (!data.empty()) std::memcpy(sb + kRespPrefix, data.data(), data.size());
    reply->first_len = kRespPrefix + data.size();
    return true;
  };

  // Single-threaded serve loop: reap completions and process each RECV inline, in
  // arrival (= request) order, replying on a free send slot. Replies MUST go out
  // in request order: the pipelined client binds each reply's destination buffer
  // at recv-post time (zero-copy scatter), so an out-of-order reply would land in
  // the wrong buffer. Depth K still gives K-in-flight pipelining; we just don't
  // reorder. (An earlier parallel GET worker pool was removed for this reason —
  // it broke zero-copy correctness for marginal gain; GET scales via connections.)
  std::vector<ibv_wc> wcs(K);
  constexpr size_t kNoSlot = static_cast<size_t>(-1);
  std::vector<size_t> rearm_on_send(K, kNoSlot);
  bool fail = false;
  const int idle_ms = ServerIdleMs();
  active_conns_.fetch_add(1, std::memory_order_relaxed);
  while (running_ && !fail) {
    int g = ep.WaitComp(wcs.data(), static_cast<int>(K), idle_ms);
    if (g == 0) { idle_reclaims_.fetch_add(1, std::memory_order_relaxed); break; }  // idle -> reclaim
    if (g < 0) break;  // error / Stop()'s Wake()
    for (int w = 0; w < g && !fail; ++w) {
      const ibv_wc& wc = wcs[w];
      if (wc.status != IBV_WC_SUCCESS) {
        completion_errors_.fetch_add(1, std::memory_order_relaxed);
        fail = true; break;
      }
      if (wc.opcode == IBV_WC_SEND) {
        size_t sid = static_cast<size_t>(wc.wr_id);
        if (sid < rearm_on_send.size() && rearm_on_send[sid] != kNoSlot) {
          if (!post_request_recv(rearm_on_send[sid])) { fail = true; break; }
          rearm_on_send[sid] = kNoSlot;
        }
        free_send.push_back(sid);
        continue;
      }
      if (wc.opcode != IBV_WC_RECV || wc.byte_len < kReqPrefix) { fail = true; break; }
      completions_.fetch_add(1, std::memory_order_relaxed);  // a request RECV
      size_t r = static_cast<size_t>(wc.wr_id);
      ReqFields rq;
      if (!DecodeReq(ep.rbuf(r), &rq)) { fail = true; break; }  // bad protocol version
      if (free_send.empty()) { fail = true; break; }
      size_t s = free_send.back(); free_send.pop_back();
      Reply reply;
      bool built = build_reply(s, r, rq, wc.byte_len, &reply);  // consumes rbuf/dbuf[r]
      if (!built) { fail = true; break; }
      if (reply.defer_recv_rearm) {
        rearm_on_send[s] = reply.recv_slot;
      } else if (!post_request_recv(r)) {
        fail = true; break;  // re-arm (request consumed)
      }
      bool sent = reply.scatter
                      ? ep.PostSendScatter(s, reply.first_len, reply.payload,
                                           reply.payload_len, reply.payload_mr)
                      : ep.PostSend(s, reply.first_len);
      if (!sent) { fail = true; break; }
    }
  }
  active_conns_.fetch_sub(1, std::memory_order_relaxed);
  { std::lock_guard<std::mutex> lk(conn_mu_); live_eps_.erase(&ep); }
  // ep dtor tears down the QP; the peer observes the drop as an error completion.
}

std::string RdmaServer::MetricsText() const {
  auto m = [](std::string& s, const char* name, const char* type, const char* help,
              uint64_t v) {
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
    s += name; s += " "; s += std::to_string(v); s += "\n";
  };
  std::string s;
  m(s, "dfkv_rdma_completions_total", "counter", "RDMA request completions served",
    Completions());
  m(s, "dfkv_rdma_completion_errors_total", "counter", "RDMA error completions",
    CompletionErrors());
  m(s, "dfkv_rdma_active_conns", "gauge", "RDMA connections currently serving",
    ActiveConns());
  m(s, "dfkv_rdma_idle_reclaims_total", "counter", "RDMA connections reclaimed on idle timeout",
    IdleReclaims());
  return s;
}

}  // namespace dfkv
