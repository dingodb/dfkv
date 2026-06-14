/* KVClient — the standalone DingoFS KV cache client used by the SGLang HiCache
 * plugin. Routes keys via consistent hash, wraps values with a ValueHeader
 * (model/page/dtype/layer + CRC), and speaks to cache nodes via a Transport.
 *  - Put  -> SyncCache (durable-visible write, header-wrapped)
 *  - Get  -> Range + header/CRC verify (mismatch or corruption => miss)
 *  - Exist-> local existence check on the owning node */
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
  bool Exist(const std::string& key);

  // Batched, concurrently fanned out across owning nodes. Per-item results.
  std::vector<bool> BatchPut(const std::vector<KvPutItem>& items);
  std::vector<bool> BatchGet(const std::vector<KvGetItem>& items);  // hit/miss
  std::vector<bool> BatchExist(const std::vector<std::string>& keys);

  void set_batch_concurrency(size_t n) { batch_concurrency_ = n ? n : 1; }

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

 private:
  std::string Route(const std::string& key) const;
  uint64_t NowMs() const;

  mutable std::mutex ring_mu_;  // guards ring_ + addr_
  ConHash ring_;
  std::map<std::string, std::string> addr_;  // name -> ip:port
  ValueHeader self_hdr_;
  std::unique_ptr<Transport> owned_;
  Transport* t_;
  size_t batch_concurrency_ = 8;
  std::unique_ptr<MdsMemberPoller> poller_;
  PeerHealth health_;
};

}  // namespace dfkv

#endif  // DFKV_KV_CLIENT_H_
