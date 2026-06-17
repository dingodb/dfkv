/* Transport abstraction between the KV client and cache nodes.
 * Real build: a brpc-backed impl over dingofs RemoteBlockCache.
 * Test/harness build: TcpTransport (POSIX sockets). */
#ifndef DFKV_TRANSPORT_H_
#define DFKV_TRANSPORT_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "kv_store.h"   // Status
#include "kv_types.h"
#include "wire.h"       // WireOp, kReqPrefix, kRespPrefix, Encode/DecodeReq/Resp

namespace dfkv {

// One write in a batch (data must outlive the CacheMany call).
struct CacheItem { BlockKey key; const void* data; size_t len; };
// One zero-copy read target: `n` payload bytes are written straight into `payload`.
struct RangeDst { void* payload; size_t n; };
// One zero-copy write source: `header` (e.g. the 48B ValueHeader) and `payload`
// live in separate buffers; the transport gathers [header|payload] on the wire
// without a client-side concat copy (RDMA scatter-send). Both must outlive the
// CacheFrom call and `payload` must stay unmodified until it returns.
struct CacheSrc {
  BlockKey key;
  const void* header; size_t header_len;
  const void* payload; size_t payload_len;
};

class Transport {
 public:
  virtual ~Transport() = default;
  // Synchronous, durable-visible write (server uses KVStore::Cache).
  virtual Status Cache(const std::string& node, const BlockKey& key,
                       const void* data, size_t len) = 0;
  virtual Status Range(const std::string& node, const BlockKey& key,
                       uint64_t offset, uint64_t length, std::string* out) = 0;
  virtual Status Exist(const std::string& node, const BlockKey& key,
                       bool* exist) = 0;

  // Query a node's advertised cluster member list (for discovery). Default: not
  // supported; TCP/RDMA override it. *out = "name=ip:port,name=ip:port,...".
  virtual Status Members(const std::string& node, std::string* out) {
    (void)node; (void)out; return Status::kInvalid;
  }

  // Transport-level Prometheus metrics (e.g. RDMA per-rail connections, MR
  // registrations). Default empty; the RDMA transport overrides it. Folded into
  // the client snapshot. Off the datapath (rendered on demand).
  virtual std::string MetricsText() const { return ""; }

  // Register a large caller memory region (e.g. the whole SGLang host KV pool) for
  // zero-copy transfer. A pipelined (RDMA) transport registers it once per
  // connection so the Get/Put datapath never does a per-op MR registration —
  // every buffer inside the region is resolved by range lookup. Default: no-op
  // (TCP copies, so it needs no registration). Call before traffic for best effect.
  virtual void RegisterMemory(void* base, size_t size) { (void)base; (void)size; }

  // True if CacheMany/RangeMany pipeline requests on one connection (RDMA). When
  // false (TCP), the client parallelizes batches across items with its own
  // threads instead, since the per-node loop here would be sequential.
  virtual bool pipelined() const { return false; }

  // Batch variants for one node. Default = sequential loop; RDMA overrides these
  // to pipeline multiple requests in flight on a single connection. All keys in
  // a RangeMany share (offset, length).
  virtual std::vector<Status> CacheMany(const std::string& node,
                                        const std::vector<CacheItem>& items) {
    std::vector<Status> r;
    r.reserve(items.size());
    for (const auto& it : items) r.push_back(Cache(node, it.key, it.data, it.len));
    return r;
  }
  virtual std::vector<Status> RangeMany(const std::string& node,
                                        const std::vector<BlockKey>& keys,
                                        uint64_t offset, uint64_t length,
                                        std::vector<std::string>* outs) {
    outs->assign(keys.size(), std::string());
    std::vector<Status> r;
    r.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
      r.push_back(Range(node, keys[i], offset, length, &(*outs)[i]));
    return r;
  }

  // Batch existence probe for one node. Default = sequential loop; RDMA overrides
  // it to pipeline kExist requests on a single connection (one Acquire/node)
  // instead of one round trip per key. (*exists)[i] is 1 iff keys[i] is present;
  // the returned Status[i] is the raw per-key outcome for caller-side health
  // accounting (kIOError => transport failure; anything else => node responded).
  virtual std::vector<Status> ExistMany(const std::string& node,
                                        const std::vector<BlockKey>& keys,
                                        std::vector<char>* exists) {
    exists->assign(keys.size(), 0);
    std::vector<Status> r;
    r.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      bool e = false;
      Status st = Exist(node, keys[i], &e);
      (*exists)[i] = (st == Status::kOk && e) ? 1 : 0;
      r.push_back(st);
    }
    return r;
  }

  // Zero-copy read: the payload for keys[i] lands directly in dsts[i].payload
  // (RDMA scatters [resp-prefix | first header_size bytes] into a scratch and the
  // rest into the caller buffer). hdrs[i] gets the first header_size value-header
  // bytes for the caller to verify. Default: a Range + split-copy (no zero-copy).
  virtual std::vector<Status> RangeInto(const std::string& node,
                                        const std::vector<BlockKey>& keys,
                                        const std::vector<RangeDst>& dsts,
                                        size_t header_size,
                                        std::vector<std::string>* hdrs) {
    hdrs->assign(keys.size(), std::string());
    std::vector<Status> r;
    r.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      std::string raw;
      Status st = Range(node, keys[i], 0, header_size + dsts[i].n, &raw);
      r.push_back(st);
      if (st != Status::kOk || raw.size() < header_size) continue;
      (*hdrs)[i].assign(raw.data(), header_size);
      size_t pn = raw.size() - header_size;
      if (pn > dsts[i].n) pn = dsts[i].n;
      std::memcpy(dsts[i].payload, raw.data() + header_size, pn);
    }
    return r;
  }

  // Zero-copy batched write: gathers [srcs[i].header | srcs[i].payload] for
  // keys[i] without a client concat copy (RDMA scatter-send). Default: concat
  // each into a temp buffer and route through CacheMany (no zero-copy) so TCP
  // and any non-pipelined transport stay correct. RDMA's override also falls
  // back here (which re-materializes the contiguous buffer) on oversize/reg failure.
  virtual std::vector<Status> CacheFrom(const std::string& node,
                                        const std::vector<CacheSrc>& srcs) {
    std::vector<std::vector<char>> bufs(srcs.size());
    std::vector<CacheItem> items;
    items.reserve(srcs.size());
    for (size_t i = 0; i < srcs.size(); ++i) {
      bufs[i].resize(srcs[i].header_len + srcs[i].payload_len);
      if (srcs[i].header_len) std::memcpy(bufs[i].data(), srcs[i].header, srcs[i].header_len);
      if (srcs[i].payload_len)
        std::memcpy(bufs[i].data() + srcs[i].header_len, srcs[i].payload, srcs[i].payload_len);
      items.push_back(CacheItem{srcs[i].key, bufs[i].data(), bufs[i].size()});
    }
    return CacheMany(node, items);
  }
};

}  // namespace dfkv

#endif  // DFKV_TRANSPORT_H_
