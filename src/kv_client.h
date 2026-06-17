/* KVClient — the standalone DingoFS KV cache client used by the SGLang HiCache
 * plugin. Routes keys via consistent hash, wraps values with a ValueHeader
 * (model/page/dtype/layer geometry), and speaks to cache nodes via a Transport.
 *  - Put  -> SyncCache (durable-visible write, header-wrapped)
 *  - Get  -> Range + header geometry verify (mismatch => miss)
 *  - Exist-> local existence check on the owning node
 * NOTE: the header carries geometry, not a payload checksum (CRC was dropped in
 * v3). A geometry mismatch is caught, but silent payload bit-rot on the wire/disk
 * is not detected here — we rely on the underlying transport/filesystem. */
#ifndef DFKV_KV_CLIENT_H_
#define DFKV_KV_CLIENT_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "con_hash.h"
#include "membership.h"
#include "mds_member_poller.h"
#include "peer_health.h"
#include "transport.h"
#include "value_header.h"

namespace dfkv {

struct KvPutItem { std::string key; const void* value; size_t n; };
struct KvGetItem { std::string key; void* out; size_t n; };

class KVClient {
 public:
  // members: (node_name, "ip:port"). self_hdr: this engine's geometry identity.
  // transport defaults to TcpTransport (owned) when nullptr.
  KVClient(std::vector<std::pair<std::string, std::string>> members,
           ValueHeader self_hdr, Transport* transport = nullptr);
  ~KVClient();

  bool Put(const std::string& key, const void* value, size_t n);
  bool Get(const std::string& key, void* out, size_t n);  // true = hit (exact n)
  // Variable-size get: learns payload length from the stored header. For CLI/
  // tooling where the caller doesn't know the page size up front.
  bool GetAuto(const std::string& key, std::string* out, size_t max_bytes = (64u << 20));
  // Variable-size get into a caller buffer of capacity `cap`. On hit, writes the
  // stored payload (whatever length it was Put with) and reports its true length
  // via *out_len; returns false (miss) on geometry mismatch, route/health failure,
  // or if the stored payload would not fit in `cap`. Unlike Get(), it does NOT
  // require the caller to know the exact stored size — the LMCache connector uses
  // it to read back variable-size (unfull) chunks.
  bool GetAuto(const std::string& key, void* out, size_t cap, size_t* out_len);
  bool Exist(const std::string& key);

  // Batched, concurrently fanned out across owning nodes. Per-item results.
  std::vector<bool> BatchPut(const std::vector<KvPutItem>& items);
  std::vector<bool> BatchGet(const std::vector<KvGetItem>& items);  // hit/miss
  // Variable-size batched get. items[i].n is the buffer CAPACITY (not the exact
  // payload size); out_lens (if non-null) receives the actual stored payload
  // length per item. Per-item hit/miss like BatchGet, but it accepts any stored
  // size that fits in the buffer instead of requiring payload_len == n — so it
  // reads back unfull/variable chunks. Keeps the RDMA zero-copy datapath for
  // full-size items (same RangeInto path as BatchGet).
  std::vector<bool> BatchGetAuto(const std::vector<KvGetItem>& items,
                                 std::vector<size_t>* out_lens);
  std::vector<bool> BatchExist(const std::vector<std::string>& keys);

  void set_batch_concurrency(size_t n) { batch_concurrency_ = n ? n : 1; }

  // Register a large caller memory region (e.g. the whole SGLang host KV pool) for
  // zero-copy transfer, so Put/Get never do a per-op RDMA MR registration — every
  // buffer inside the region resolves to the pre-registered pool MR. No-op on TCP.
  // Call once at startup (after the pool is allocated) before traffic.
  void RegisterMemory(void* base, size_t size) { t_->RegisterMemory(base, size); }

  // Hot-swap the cluster membership (rebuilds the consistent-hash ring).
  // Thread-safe vs concurrent Put/Get/Exist.
  void SetMembers(std::vector<std::pair<std::string, std::string>> members);

  // Apply a weighted member view (from MDS discovery); rebuilds the ring with
  // per-node vnode weight and resolves addr = "ip:port".
  void SetMembers(const std::vector<MemberInfo>& members);

  // Start background MDS discovery: poll the MDS endpoints for `group` and rebuild
  // the ring on each epoch change. Replaces static membership.
  void StartMdsDiscovery(std::vector<std::string> mds_eps, const std::string& group,
                         int poll_ms = 3000);
  void StopMdsDiscovery();

  // Discovery: query a seed node ("ip:port") for the current cluster membership
  // and SetMembers() it. Lets clients learn add/remove without a static list or
  // MDS — point at any live node. Returns true if a non-empty list was applied.
  bool RefreshMembers(const std::string& seed_addr);

  // Client-side Prometheus metrics text (ops served, IO errors, peer health
  // transitions, per-peer errors) plus transport-level counters (RDMA per-rail
  // connections, MR regions). Surfaced to the plugin via the C ABI.
  std::string MetricsSnapshot() const;
  const std::string& TransportMode() const { return transport_reason_; }

 private:
  std::string Route(const std::string& key) const;
  uint64_t NowMs() const;

  mutable std::mutex ring_mu_;  // guards ring_ + addr_
  ConHash ring_;
  std::map<std::string, std::string> addr_;  // name -> ip:port
  ValueHeader self_hdr_;
  std::unique_ptr<Transport> owned_;
  Transport* t_;
  std::string transport_reason_ = "unknown";
  size_t batch_concurrency_ = 8;
  std::unique_ptr<MdsMemberPoller> poller_;
  PeerHealth health_;
};

}  // namespace dfkv

#endif  // DFKV_KV_CLIENT_H_
