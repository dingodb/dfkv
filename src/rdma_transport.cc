#include "rdma_transport.h"

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "net_util.h"     // Dial / WriteAll / ReadAll / Put*/Get*
#include "rdma_verbs.h"   // RcEndpoint, QpInfo

namespace dfkv {

namespace {
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

// Post w requests (already built in ep.sbuf[j], send length slen[j]) with one
// recv per slot, then reap all 2*w completions. recv wr_id = slot, so the reply
// for request j lands in ep.rbuf[j] (RC in-order delivery). rbytes[j] = reply
// length. Returns false (connection unusable) on any error completion.
bool RunWindow(rdma::RcEndpoint& ep, const std::vector<size_t>& slen,
               std::vector<uint32_t>* rbytes) {
  const size_t w = slen.size();
  rbytes->assign(w, 0);
  for (size_t j = 0; j < w; ++j)
    if (!ep.PostRecv(j) || !ep.PostSend(j, slen[j])) return false;
  std::vector<ibv_wc> wcs(2 * w);
  int need = static_cast<int>(2 * w);
  while (need > 0) {
    int g = ep.WaitComp(wcs.data(), static_cast<int>(2 * w));
    if (g <= 0) return false;
    for (int i = 0; i < g; ++i) {
      if (wcs[i].status != IBV_WC_SUCCESS) return false;
      if (wcs[i].opcode == IBV_WC_RECV)
        (*rbytes)[static_cast<size_t>(wcs[i].wr_id)] = wcs[i].byte_len;
      --need;
    }
  }
  return true;
}
}  // namespace

struct RdmaTransport::Conn {
  rdma::RcEndpoint ep;
};

bool RdmaTransport::Available() {
  int n = 0;
  ibv_device** devs = ibv_get_device_list(&n);
  if (devs) ibv_free_device_list(devs);
  return n > 0;
}

RdmaTransport::RdmaTransport(size_t max_msg, const std::string& dev_name)
    : max_msg_(max_msg), depth_(1), dev_name_(dev_name) {
  if (dev_name_.empty()) {
    const char* e = std::getenv("DFKV_RDMA_DEV");
    if (e && *e) dev_name_ = e;
  }
  const char* d = std::getenv("DFKV_RDMA_DEPTH");  // pipeline depth (must be <= server's)
  if (d && *d) { long v = std::strtol(d, nullptr, 10); if (v >= 1 && v <= 256) depth_ = (size_t)v; }
}

RdmaTransport::~RdmaTransport() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [node, cs] : pool_)
    for (Conn* c : cs) Destroy(c);
}

void RdmaTransport::Destroy(Conn* c) { delete c; }  // RcEndpoint dtor tears down QP/MRs

RdmaTransport::Conn* RdmaTransport::Acquire(const std::string& node, bool* from_pool) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pool_.find(node);
    if (it != pool_.end() && !it->second.empty()) {
      Conn* c = it->second.back(); it->second.pop_back();
      *from_pool = true; return c;
    }
  }
  *from_pool = false;
  // Bootstrap the QP over a short-lived TCP connection to the node's member
  // address (control plane). The data plane then rides the named RDMA device.
  int fd = net::Dial(node, /*connect_ms=*/3000, /*io_ms=*/10000);
  if (fd < 0) return nullptr;

  auto* c = new Conn();
  if (!c->ep.Open(dev_name_.empty() ? nullptr : dev_name_.c_str(), max_msg_, depth_)) {
    ::close(fd); delete c; return nullptr;
  }
  char mine[rdma::kQpInfoBytes], peer[rdma::kQpInfoBytes];
  rdma::SerializeQpInfo(c->ep.Local(), mine);
  // client sends its Qp info, then reads the server's; symmetric to the server.
  if (!net::WriteAll(fd, mine, rdma::kQpInfoBytes) ||
      !net::ReadAll(fd, peer, rdma::kQpInfoBytes)) {
    ::close(fd); delete c; return nullptr;
  }
  if (!c->ep.Connect(rdma::ParseQpInfo(peer))) { ::close(fd); delete c; return nullptr; }
  // Wait for the server's "ready" byte: it has posted its recvs, so our first
  // SEND won't hit RNR.
  char ready = 0;
  if (!net::ReadAll(fd, &ready, 1) || ready != 1) { ::close(fd); delete c; return nullptr; }
  ::close(fd);  // bootstrap done; QP is RTS
  return c;
}

void RdmaTransport::Release(const std::string& node, Conn* c) {
  std::lock_guard<std::mutex> lk(mu_);
  pool_[node].push_back(c);
}

Status RdmaTransport::RoundTrip(const std::string& node, WireOp op,
                                const BlockKey& k, uint64_t offset,
                                uint64_t length, const void* payload,
                                uint64_t payload_len, std::string* out) {
  if (kReqPrefix + payload_len > max_msg_) return Status::kInvalid;
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool from_pool = false;
    Conn* c = Acquire(node, &from_pool);
    if (!c) return Status::kIOError;
    rdma::RcEndpoint& ep = c->ep;

    EncodePrefix(ep.sbuf(0), op, k, offset, length, payload_len);
    if (payload_len) std::memcpy(ep.sbuf(0) + kReqPrefix, payload, payload_len);

    bool ok = ep.PostRecv(0) && ep.PostSend(0, kReqPrefix + payload_len);
    // Reap both the send and the recv completion (shared CQ; either order).
    uint32_t recv_bytes = 0;
    bool need_send = ok, need_recv = ok;
    while (ok && (need_send || need_recv)) {
      ibv_wc wc{};
      int g = ep.WaitComp(&wc, 1);
      if (g <= 0 || wc.status != IBV_WC_SUCCESS) { ok = false; break; }
      if (wc.opcode == IBV_WC_SEND) need_send = false;
      else if (wc.opcode == IBV_WC_RECV) { need_recv = false; recv_bytes = wc.byte_len; }
    }

    if (!ok) {
      Destroy(c);
      if (!from_pool) return Status::kIOError;
      continue;  // stale pooled conn -> retry fresh
    }
    if (recv_bytes < kRespPrefix) { Destroy(c); return Status::kIOError; }
    Status st = static_cast<Status>(static_cast<uint8_t>(ep.rbuf(0)[0]));
    uint64_t dlen = net::GetU64(ep.rbuf(0) + 1);
    if (out) {
      if (kRespPrefix + dlen > recv_bytes) { Destroy(c); return Status::kIOError; }
      out->assign(ep.rbuf(0) + kRespPrefix, dlen);
    }
    Release(node, c);
    return st;
  }
  return Status::kIOError;
}

Status RdmaTransport::Cache(const std::string& node, const BlockKey& key,
                            const void* data, size_t len) {
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr);
}
Status RdmaTransport::Range(const std::string& node, const BlockKey& key,
                            uint64_t offset, uint64_t length, std::string* out) {
  return RoundTrip(node, WireOp::kRange, key, offset, length, nullptr, 0, out);
}
Status RdmaTransport::Exist(const std::string& node, const BlockKey& key, bool* exist) {
  Status st = RoundTrip(node, WireOp::kExist, key, 0, 0, nullptr, 0, nullptr);
  if (st == Status::kOk) { *exist = true; return Status::kOk; }
  if (st == Status::kNotFound) { *exist = false; return Status::kOk; }
  return st;
}

std::vector<Status> RdmaTransport::CacheMany(const std::string& node,
                                             const std::vector<CacheItem>& items) {
  const size_t n = items.size();
  std::vector<Status> res(n, Status::kIOError);
  if (n == 0) return res;
  for (const auto& it : items)
    if (kReqPrefix + it.len > max_msg_) return Transport::CacheMany(node, items);  // oversized: fall back

  bool from_pool = false;
  Conn* c = Acquire(node, &from_pool);
  if (!c) return res;
  rdma::RcEndpoint& ep = c->ep;
  const size_t W = ep.depth();
  bool conn_ok = true;
  for (size_t base = 0; base < n && conn_ok; base += W) {
    const size_t w = std::min(W, n - base);
    std::vector<size_t> slen(w);
    for (size_t j = 0; j < w; ++j) {
      const CacheItem& it = items[base + j];
      EncodePrefix(ep.sbuf(j), WireOp::kCache, it.key, 0, 0, it.len);
      if (it.len) std::memcpy(ep.sbuf(j) + kReqPrefix, it.data, it.len);
      slen[j] = kReqPrefix + it.len;
    }
    std::vector<uint32_t> rbytes;
    if (!RunWindow(ep, slen, &rbytes)) { conn_ok = false; break; }
    for (size_t j = 0; j < w; ++j)
      if (rbytes[j] >= kRespPrefix)
        res[base + j] = static_cast<Status>(static_cast<uint8_t>(ep.rbuf(j)[0]));
  }
  if (conn_ok) Release(node, c); else Destroy(c);
  return res;
}

std::vector<Status> RdmaTransport::RangeMany(const std::string& node,
                                             const std::vector<BlockKey>& keys,
                                             uint64_t offset, uint64_t length,
                                             std::vector<std::string>* outs) {
  const size_t n = keys.size();
  outs->assign(n, std::string());
  std::vector<Status> res(n, Status::kIOError);
  if (n == 0) return res;
  if (kRespPrefix + length > max_msg_)  // reply wouldn't fit the recv buffer
    return Transport::RangeMany(node, keys, offset, length, outs);

  bool from_pool = false;
  Conn* c = Acquire(node, &from_pool);
  if (!c) return res;
  rdma::RcEndpoint& ep = c->ep;
  const size_t W = ep.depth();
  bool conn_ok = true;
  for (size_t base = 0; base < n && conn_ok; base += W) {
    const size_t w = std::min(W, n - base);
    std::vector<size_t> slen(w);
    for (size_t j = 0; j < w; ++j) {
      EncodePrefix(ep.sbuf(j), WireOp::kRange, keys[base + j], offset, length, 0);
      slen[j] = kReqPrefix;
    }
    std::vector<uint32_t> rbytes;
    if (!RunWindow(ep, slen, &rbytes)) { conn_ok = false; break; }
    for (size_t j = 0; j < w; ++j) {
      uint32_t rb = rbytes[j];
      if (rb < kRespPrefix) continue;
      Status st = static_cast<Status>(static_cast<uint8_t>(ep.rbuf(j)[0]));
      res[base + j] = st;
      if (st == Status::kOk) {
        uint64_t dlen = net::GetU64(ep.rbuf(j) + 1);
        if (kRespPrefix + dlen <= rb) (*outs)[base + j].assign(ep.rbuf(j) + kRespPrefix, dlen);
        else res[base + j] = Status::kIOError;
      }
    }
  }
  if (conn_ok) Release(node, c); else Destroy(c);
  return res;
}

}  // namespace dfkv
