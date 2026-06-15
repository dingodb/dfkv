#include "rdma_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "net_util.h"     // ReadAll / WriteAll / Get*/Put*
#include "numa_util.h"    // pin serve thread to the device's NUMA node
#include "rdma_verbs.h"   // RcEndpoint, QpInfo
#include "transport.h"    // kReqPrefix, kRespPrefix

namespace dfkv {

RdmaServer::RdmaServer(Handler handler, size_t max_msg, const std::string& dev_name)
    : handler_(std::move(handler)), max_msg_(max_msg), dev_name_(dev_name) {
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
  std::vector<std::thread> threads;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    for (rdma::RcEndpoint* ep : live_eps_) ep->Wake();
    threads.swap(conn_threads_);
  }
  for (auto& t : threads) if (t.joinable()) t.join();
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
    conn_threads_.emplace_back([this, fd] { Serve(fd); });
  }
}

namespace {
size_t ServerDepth() {
  // Pipeline depth (requests in flight per connection). Default 1 keeps per-conn
  // buffer memory low (depth * max_msg * 2); set DFKV_RDMA_DEPTH>1 to enable
  // pipelining (helps the latency-bound PUT path; GET scales via more conns).
  const char* e = std::getenv("DFKV_RDMA_DEPTH");
  if (e && *e) { long v = std::strtol(e, nullptr, 10); if (v >= 1 && v <= 256) return (size_t)v; }
  return 1;
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
  if (!ep.Open(dev.empty() ? nullptr : dev.c_str(), max_msg_, K,
               /*ib_port=*/1, /*direct_io_buffers=*/true)) {
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
  // Post all K receives so the client may keep up to K requests in flight
  // (pipelining). recv buffers and send buffers are independent slot pools.
  bool armed = true;
  for (size_t i = 0; i < K; ++i) armed = armed && ep.PostRecv(i);
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
    size_t first_len = 0;  // plain: full sbuf length; scatter: sbuf header length
    const void* payload = nullptr;
    size_t payload_len = 0;
    ibv_mr* payload_mr = nullptr;
  };

  // Build the reply for one request. Generic ops are materialized in sbuf. For
  // kRange with a direct range handler set, O_DIRECT reads into ep.dbuf[slot]
  // (4096-aligned + registered) and the wire reply is scatter-sent as
  // [resp-prefix from sbuf | value bytes from dbuf], with no payload memcpy.
  auto build_reply = [&](size_t slot, const ReqFields& rq, const char* payload,
                         Reply* reply) -> bool {
    char* sb = ep.sbuf(slot);
    if (rq.op == static_cast<uint8_t>(WireOp::kRange) && range_handler_) {
      if (!ep.dbuf(slot) || !ep.dmr(slot)) return false;
      const uint64_t max_payload = ep.cap() > kRespPrefix ? ep.cap() - kRespPrefix : 0;
      if (rq.length > max_payload) {
        EncodeResp(sb, Status::kInvalid, 0);
        reply->first_len = kRespPrefix;
        return true;
      }
      const char* out_data = nullptr;
      size_t out_len = 0;
      Status st = range_handler_(rq.id, rq.index, rq.size, rq.offset, rq.length,
                                 ep.dbuf(slot), ep.dbuf_cap(), &out_data, &out_len);
      uint64_t dlen = (st == Status::kOk) ? out_len : 0;
      EncodeResp(sb, st, dlen);
      reply->first_len = kRespPrefix;
      if (st == Status::kOk && dlen != 0) {
        if (!out_data) return false;
        reply->scatter = true;
        reply->payload = out_data;
        reply->payload_len = out_len;
        reply->payload_mr = ep.dmr(slot);
      }
      return true;
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
  bool fail = false;
  while (running_ && !fail) {
    int g = ep.WaitComp(wcs.data(), static_cast<int>(K));
    if (g <= 0) break;
    for (int w = 0; w < g && !fail; ++w) {
      const ibv_wc& wc = wcs[w];
      if (wc.status != IBV_WC_SUCCESS) { fail = true; break; }
      if (wc.opcode == IBV_WC_SEND) { free_send.push_back(static_cast<size_t>(wc.wr_id)); continue; }
      if (wc.opcode != IBV_WC_RECV || wc.byte_len < kReqPrefix) { fail = true; break; }
      size_t r = static_cast<size_t>(wc.wr_id);
      ReqFields rq;
      if (!DecodeReq(ep.rbuf(r), &rq)) { fail = true; break; }  // bad protocol version
      if (free_send.empty()) { fail = true; break; }
      size_t s = free_send.back(); free_send.pop_back();
      const char* payload = (rq.payload_len && wc.byte_len >= kReqPrefix + rq.payload_len)
                                ? ep.rbuf(r) + kReqPrefix : nullptr;
      Reply reply;
      bool built = build_reply(s, rq, payload, &reply);   // consumes rbuf[r] for a PUT payload
      if (!ep.PostRecv(r)) { fail = true; break; }  // re-arm (request consumed)
      if (!built) { fail = true; break; }
      bool sent = reply.scatter
                      ? ep.PostSendScatter(s, reply.first_len, reply.payload,
                                           reply.payload_len, reply.payload_mr)
                      : ep.PostSend(s, reply.first_len);
      if (!sent) { fail = true; break; }
    }
  }
  { std::lock_guard<std::mutex> lk(conn_mu_); live_eps_.erase(&ep); }
  // ep dtor tears down the QP; the peer observes the drop as an error completion.
}

}  // namespace dfkv
