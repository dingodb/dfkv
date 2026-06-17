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
#include <list>
#include <unordered_map>
#include <utility>
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
// Alignment used for server-side O_DIRECT read buffers. NVMe/xfs are happy with
// 4096, and it is a safe superset for 512-byte logical-sector devices.
constexpr size_t kDirectIoAlign = 4096;

// One RC connection: device context + PD + CQ(+channel) + QP + a ring of
// `depth` send and recv buffers (each `cap` bytes, registered once).
class RcEndpoint {
 public:
  RcEndpoint() = default;
  ~RcEndpoint();
  RcEndpoint(const RcEndpoint&) = delete;
  RcEndpoint& operator=(const RcEndpoint&) = delete;

  // Open device `dev_name` (nullptr/"" = first device), create RC QP in INIT,
  // allocate+register `depth` send & recv buffers of `cap` bytes. When
  // direct_io_buffers is true, also allocate one 4096-aligned registered buffer
  // per slot for O_DIRECT reads/writes that are scatter-transferred without a
  // payload copy. direct_io_cap lets that buffer be larger than the ordinary
  // control send/recv buffers.
  bool Open(const char* dev_name, size_t cap, size_t depth, uint8_t ib_port = 1,
            bool direct_io_buffers = false, size_t direct_io_cap = 0);

  QpInfo Local() const { return local_; }              // my QP info (after Open)
  bool Connect(const QpInfo& remote);                  // INIT -> RTR -> RTS

  size_t depth() const { return depth_; }
  size_t cap() const { return cap_; }
  int numa_node() const { return numa_node_; }  // device's NUMA node, or -1
  char* sbuf(size_t slot) { return sbuf_[slot]; }
  char* rbuf(size_t slot) { return rbuf_[slot]; }
  char* dbuf(size_t slot) { return dbuf_.empty() ? nullptr : dbuf_[slot]; }
  size_t dbuf_cap() const { return dbuf_cap_; }
  ibv_mr* dmr(size_t slot) { return dmr_.empty() ? nullptr : dmr_[slot]; }

  bool PostRecv(size_t slot);                          // arm recv into rbuf_[slot]
  bool PostSend(size_t slot, size_t len);              // SEND sbuf_[slot][0,len)

  // Register a large caller memory region (e.g. the WHOLE SGLang host KV pool)
  // once on this PD. After this, RegisterUser() for any buffer inside the region
  // returns this MR by range lookup with NO per-op ibv_reg_mr — registration is a
  // one-time connection-setup cost, not a per-transfer cost. Idempotent per base.
  bool AddPoolMr(void* base, size_t size);
  // Register every region not yet present as a pool MR. Called when a connection
  // is acquired so regions declared at any time land on every connection's PD.
  void EnsurePoolMrs(const std::vector<std::pair<void*, size_t>>& regions);

  // Resolve a caller buffer to an MR for zero-copy transfer. Fast path: a buffer
  // inside a registered pool region (AddPoolMr) returns that MR with no syscall.
  // Otherwise it is registered ad-hoc and kept in a small LRU cache (stable repeat
  // buffers stay cached). Valid as a recv target AND a send source (LOCAL_WRITE is
  // a superset of the no-flag local read a SEND does). nullptr on failure.
  ibv_mr* RegisterUser(void* addr, size_t len);
  // Scatter recv: first `hdr_bytes` of the message land in rbuf_[slot] (resp
  // prefix + value header), the remaining payload lands directly in `payload`
  // (must belong to `payload_mr` from RegisterUser) — zero copy into the caller.
  bool PostRecvScatter(size_t slot, void* payload, size_t payload_len,
                       ibv_mr* payload_mr, size_t hdr_bytes);
  // Scatter send: SGE0 = sbuf_[slot][0,hdr_len), SGE1 = `payload` (must belong to
  // `payload_mr`). Used by client PUT and server GET to gather a tiny header plus
  // a registered payload into one wire SEND with zero copy of the payload. Degrades
  // to a 1-SGE send when payload_len==0, so payload_mr may be null.
  bool PostSendScatter(size_t slot, size_t hdr_len, const void* payload,
                       size_t payload_len, ibv_mr* payload_mr);

  // Block until at least one completion, drain up to `max` into out[]; returns
  // the count (>0), or <0 on error / when Wake() is called. wr_id of each wc =
  // the slot. Used for single round-trips (max=1) and pipelined batches.
  // timeout_ms < 0 blocks forever (the default, used by the client). A finite
  // timeout returns 0 if no completion arrives in that window — the server uses
  // it to reclaim an idle/abandoned connection instead of blocking indefinitely.
  int WaitComp(ibv_wc* out, int max, int timeout_ms = -1);
  // Unblock a thread sitting in WaitComp (so the server can join its Serve
  // threads at shutdown). Thread-safe vs the waiter.
  void Wake();

 private:
  void Close();

  ibv_context* ctx_ = nullptr;
  ibv_pd* pd_ = nullptr;
  ibv_cq* cq_ = nullptr;
  ibv_comp_channel* chan_ = nullptr;
  ibv_qp* qp_ = nullptr;
  uint8_t ib_port_ = 1;
  int gid_index_ = 0;       // local source GID index (RoCEv2 on Ethernet link layer)
  int numa_node_ = -1;      // NUMA node of the device (for buffer/thread placement)
  int wake_rfd_ = -1, wake_wfd_ = -1;  // self-pipe to interrupt WaitComp on stop
  ibv_mtu mtu_ = IBV_MTU_4096;
  unsigned cq_armed_unacked_ = 0;

  size_t cap_ = 0, depth_ = 0, dbuf_cap_ = 0;
  std::vector<char*> sbuf_, rbuf_, dbuf_;
  std::vector<ibv_mr*> smr_, rmr_, dmr_;
  // Big pre-registered caller regions (the host KV pool). Registered once at
  // connection setup, never evicted; RegisterUser range-looks-up into these.
  struct PoolMr { uintptr_t base; size_t size; ibv_mr* mr; };
  std::vector<PoolMr> pool_mr_;
  // LRU-capped cache of caller-buffer MRs (addr -> {mr, lru-iterator}); front of
  // user_lru_ is MRU. Bounded so a workload with many distinct buffers can't leak
  // registrations (a stable HiCache pool stays fully cached and always hits).
  static constexpr size_t kMaxUserMr = 64;
  std::list<uintptr_t> user_lru_;
  std::unordered_map<uintptr_t, std::pair<ibv_mr*, std::list<uintptr_t>::iterator>> user_mr_;
  QpInfo local_;
};

}  // namespace rdma
}  // namespace dfkv

#endif  // DFKV_RDMA_VERBS_H_
