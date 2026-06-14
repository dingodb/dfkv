#include "rdma_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "net_util.h"     // ReadAll / WriteAll / Get*/Put*
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
  // Per-connection serve threads are detached: they block in RDMA completion
  // waits and are reclaimed on process exit. We intentionally do not join them
  // (no clean interrupt for a blocked CQ wait), which keeps Stop() bounded.
}

void RdmaServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    std::thread([this, fd] { Serve(fd); }).detach();
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
  if (!ep.Open(dev.empty() ? nullptr : dev.c_str(), max_msg_, K)) {
    ::close(boot_fd); return;
  }
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

  // Send-slot free list (a reply uses one send slot until its SEND completes).
  // Owned by the serve thread only (alloc on RECV, free on SEND comp) — no lock.
  std::vector<size_t> free_send;
  free_send.reserve(K);
  for (size_t i = 0; i < K; ++i) free_send.push_back(i);

  // Worker pool: when pipelining (K>1), the slow part of a GET (reading the
  // 2.74 MiB object) is dispatched to workers so a single connection's in-flight
  // reads run in parallel; the serve thread keeps reaping completions. post_send
  // on the shared QP is serialized by send_mu. PUT (large inline payload that
  // lives in rbuf) and the K==1 path stay on the serve thread. Workers are joined
  // before ep is destroyed, so ep always outlives them (no use-after-free).
  struct WorkItem { uint8_t op; uint64_t id; uint32_t index, ksize; uint64_t offset, length; size_t slot; };
  std::mutex qmu, qp_mu;  // qp_mu serializes all post_send/post_recv on the shared QP
  std::condition_variable qcv;
  std::deque<WorkItem> queue;
  std::atomic<bool> failed{false};
  bool done = false;
  const size_t W = (K > 1) ? std::min<size_t>(K, 8) : 0;
  auto worker = [&] {
    for (;;) {
      WorkItem it;
      {
        std::unique_lock<std::mutex> lk(qmu);
        qcv.wait(lk, [&] { return done || !queue.empty(); });
        if (queue.empty()) return;  // done
        it = queue.front(); queue.pop_front();
      }
      std::string data;
      Status st = handler_(it.op, it.id, it.index, it.ksize, it.offset, it.length,
                           nullptr, 0, &data);
      if (kRespPrefix + data.size() > ep.cap()) { failed = true; continue; }
      char* sb = ep.sbuf(it.slot);
      sb[0] = static_cast<char>(st);
      net::PutU64(sb + 1, data.size());
      if (!data.empty()) std::memcpy(sb + kRespPrefix, data.data(), data.size());
      std::lock_guard<std::mutex> sl(qp_mu);
      if (!ep.PostSend(it.slot, kRespPrefix + data.size())) failed = true;
    }
  };
  std::vector<std::thread> workers;
  for (size_t i = 0; i < W; ++i) workers.emplace_back(worker);

  std::vector<ibv_wc> wcs(K);
  bool fail = false;
  while (running_ && !fail && !failed.load(std::memory_order_relaxed)) {
    int g = ep.WaitComp(wcs.data(), static_cast<int>(K));
    if (g <= 0) break;
    for (int w = 0; w < g && !fail; ++w) {
      const ibv_wc& wc = wcs[w];
      if (wc.status != IBV_WC_SUCCESS) { fail = true; break; }
      if (wc.opcode == IBV_WC_SEND) {
        free_send.push_back(static_cast<size_t>(wc.wr_id));
        continue;
      }
      if (wc.opcode != IBV_WC_RECV || wc.byte_len < kReqPrefix) { fail = true; break; }
      size_t r = static_cast<size_t>(wc.wr_id);
      const char* req = ep.rbuf(r);
      const uint8_t op = static_cast<uint8_t>(req[0]);
      const uint64_t payload_len = net::GetU64(req + 33);
      if (free_send.empty()) { fail = true; break; }
      size_t s = free_send.back(); free_send.pop_back();

      // GET/Exist/Stats (no payload needed past parse) -> pool when pipelining.
      if (W > 0 && op != static_cast<uint8_t>(WireOp::kCache)) {
        WorkItem it{op, net::GetU64(req + 1), net::GetU32(req + 9), net::GetU32(req + 13),
                    net::GetU64(req + 17), net::GetU64(req + 25), s};
        { std::lock_guard<std::mutex> ql(qp_mu); if (!ep.PostRecv(r)) { fail = true; break; } }  // fields copied; safe to re-arm
        { std::lock_guard<std::mutex> lk(qmu); queue.push_back(it); }
        qcv.notify_one();
        continue;
      }
      // Inline: PUT (payload in rbuf) or non-pipelined path.
      const char* payload = (payload_len && wc.byte_len >= kReqPrefix + payload_len)
                                ? req + kReqPrefix : nullptr;
      std::string data;
      Status st = handler_(op, net::GetU64(req + 1), net::GetU32(req + 9),
                           net::GetU32(req + 13), net::GetU64(req + 17),
                           net::GetU64(req + 25), payload, payload_len, &data);
      { std::lock_guard<std::mutex> ql(qp_mu); if (!ep.PostRecv(r)) { fail = true; break; } }
      if (kRespPrefix + data.size() > ep.cap()) { fail = true; break; }
      char* sb = ep.sbuf(s);
      sb[0] = static_cast<char>(st);
      net::PutU64(sb + 1, data.size());
      if (!data.empty()) std::memcpy(sb + kRespPrefix, data.data(), data.size());
      std::lock_guard<std::mutex> sl(qp_mu);
      if (!ep.PostSend(s, kRespPrefix + data.size())) { fail = true; break; }
    }
  }
  // Drain workers before ep is destroyed (ep must outlive every worker).
  { std::lock_guard<std::mutex> lk(qmu); done = true; }
  qcv.notify_all();
  for (auto& t : workers) if (t.joinable()) t.join();
  // ep dtor tears down the QP; the peer observes the drop as an error completion.
}

}  // namespace dfkv
