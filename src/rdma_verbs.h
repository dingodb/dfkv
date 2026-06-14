/* Native libibverbs RC endpoint shared by the dfkv RDMA client and server.
 * Built only when DFKV_WITH_RDMA is defined.
 *
 * Why native verbs (not librdmacm)? rdma_cm resolves the RDMA device from an IP
 * address (IPoIB), so it can only use a device that has an IP. On these clusters
 * the IP lives on the 200G port while the 8x 400G ports have NO IP and sit on a
 * SEPARATE IB fabric. Native verbs opens a device BY NAME (e.g. ib7s400p0) and
 * bootstraps the QP over a tiny out-of-band TCP channel (which rides whatever IP
 * network the nodes share — 200G or bond0). Control plane and data plane are
 * thus decoupled: the QP handshake (LID/GID/QPN/PSN, ~32 B) goes over TCP, the
 * data rides the 400G fabric by LID. This is exactly perftest's non-`-R` mode.
 *
 * A connection uses RC two-sided SEND/RECV with a ring of `depth` slots so the
 * caller can keep multiple requests in flight (pipelining). Completions are
 * reaped via a completion channel (blocking, no busy-spin). */
#ifndef DFKV_RDMA_VERBS_H_
#define DFKV_RDMA_VERBS_H_

#include <infiniband/verbs.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dfkv {
namespace rdma {

// QP connection info exchanged over the TCP bootstrap channel (fixed 32 bytes,
// little-endian). lid==0 => use GRH/GID routing (RoCE); else IB LID routing.
struct QpInfo {
  uint32_t qpn = 0;
  uint32_t psn = 0;
  uint16_t lid = 0;
  uint8_t gid[16] = {0};
  uint8_t pad[6] = {0};
};
constexpr size_t kQpInfoBytes = 32;
void SerializeQpInfo(const QpInfo& in, char out[kQpInfoBytes]);
QpInfo ParseQpInfo(const char in[kQpInfoBytes]);

// Fixed-size device-name field the client sends first in the bootstrap so the
// server opens its QP on the matching device (same rail) for multi-rail setups.
constexpr size_t kDevNameBytes = 32;

// One RC connection: device context + PD + CQ(+channel) + QP + a ring of
// `depth` send and recv buffers (each `cap` bytes, registered once).
class RcEndpoint {
 public:
  RcEndpoint() = default;
  ~RcEndpoint();
  RcEndpoint(const RcEndpoint&) = delete;
  RcEndpoint& operator=(const RcEndpoint&) = delete;

  // Open device `dev_name` (nullptr/"" = first device), create RC QP in INIT,
  // allocate+register `depth` send & recv buffers of `cap` bytes. False on error.
  bool Open(const char* dev_name, size_t cap, size_t depth, uint8_t ib_port = 1);

  QpInfo Local() const { return local_; }              // my QP info (after Open)
  bool Connect(const QpInfo& remote);                  // INIT -> RTR -> RTS

  size_t depth() const { return depth_; }
  size_t cap() const { return cap_; }
  char* sbuf(size_t slot) { return sbuf_[slot]; }
  char* rbuf(size_t slot) { return rbuf_[slot]; }

  bool PostRecv(size_t slot);                          // arm recv into rbuf_[slot]
  bool PostSend(size_t slot, size_t len);              // SEND sbuf_[slot][0,len)

  // Block until at least one completion, drain up to `max` into out[]; returns
  // the count (>0) or <0 on error. wr_id of each wc = the slot. Used for both
  // single round-trips (max=1) and pipelined batches.
  int WaitComp(ibv_wc* out, int max);

 private:
  void Close();

  ibv_context* ctx_ = nullptr;
  ibv_pd* pd_ = nullptr;
  ibv_cq* cq_ = nullptr;
  ibv_comp_channel* chan_ = nullptr;
  ibv_qp* qp_ = nullptr;
  uint8_t ib_port_ = 1;
  ibv_mtu mtu_ = IBV_MTU_4096;
  unsigned cq_armed_unacked_ = 0;

  size_t cap_ = 0, depth_ = 0;
  std::vector<char*> sbuf_, rbuf_;
  std::vector<ibv_mr*> smr_, rmr_;
  QpInfo local_;
};

}  // namespace rdma
}  // namespace dfkv

#endif  // DFKV_RDMA_VERBS_H_
