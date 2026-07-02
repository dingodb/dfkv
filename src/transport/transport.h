/* Transport abstraction between the KV client and cache nodes.
 * Real build: a brpc-backed impl over dingofs RemoteBlockCache.
 * Test/harness build: TcpTransport (POSIX sockets). */
#ifndef DFKV_TRANSPORT_H_
#define DFKV_TRANSPORT_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "common/status.h"
#include "common/kv_types.h"
#include "common/value_header.h"  // ValueHeader (for true stored-length in RangeIntoMulti)
#include "transport/wire.h"       // WireOp, kReqPrefix, kRespPrefix, Encode/DecodeReq/Resp

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

// Scatter-gather write source: one dfkv key whose stored payload is the in-order
// concatenation of N non-contiguous caller buffers. The transport gathers
// [header | payload[0] | payload[1] | ...] on the wire via a multi-SGE RDMA SEND
// without a client-side concat copy. All buffers must outlive the CacheFromMulti
// call and stay unmodified until it returns. payload_len = sum of segment sizes.
struct CacheSrcMulti {
  BlockKey key;
  const void* header; size_t header_len;
  std::vector<std::pair<const void*, size_t>> payloads;  // (ptr, size) in order
};
// Scatter-gather read target: the stored payload (one concatenated blob) is
// scattered across N caller buffers in order, each receiving its own size worth
// of bytes (RDMA multi-SGE RECV). The destination capacity = sum of segment sizes.
struct RangeDstMulti {
  std::vector<std::pair<void*, size_t>> payloads;  // (ptr, size) in order
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

  // Explicitly drop a block on its owning node (LMCache L2 eviction). Default:
  // not supported; TcpTransport/RdmaTransport override it with a kRemove round
  // trip. kOk = removed, kNotFound = absent, kIOError = transport failure.
  virtual Status Remove(const std::string& node, const BlockKey& key) {
    (void)node; (void)key; return Status::kInvalid;
  }

  // Query a node's advertised cluster member list (for discovery). Default: not
  // supported; TCP/RDMA override it. *out = "name=ip:port,name=ip:port,...".
  virtual Status Members(const std::string& node, std::string* out) {
    (void)node; (void)out; return Status::kInvalid;
  }

  // Batch remove for one node. Default = sequential loop over Remove (eviction
  // is off the hot path, so neither transport bothers to pipeline it). Returns
  // the per-key Status for caller-side health accounting.
  virtual std::vector<Status> RemoveMany(const std::string& node,
                                         const std::vector<BlockKey>& keys) {
    std::vector<Status> r;
    r.reserve(keys.size());
    for (const auto& k : keys) r.push_back(Remove(node, k));
    return r;
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

  // Scatter-gather write: each key's payload is the concatenation of N caller
  // segments. Default: concatenate the segments into a single CacheSrc and route
  // through CacheFrom (no payload zero-copy, but correct on any transport). The
  // RDMA transport overrides this to gather the segments via a multi-SGE SEND.
  virtual std::vector<Status> CacheFromMulti(
      const std::string& node, const std::vector<CacheSrcMulti>& srcs) {
    std::vector<std::vector<char>> bufs(srcs.size());
    std::vector<CacheSrc> flat;
    flat.reserve(srcs.size());
    for (size_t i = 0; i < srcs.size(); ++i) {
      size_t total = 0;
      for (const auto& p : srcs[i].payloads) total += p.second;
      bufs[i].resize(total);
      size_t off = 0;
      for (const auto& p : srcs[i].payloads) {
        if (p.second) std::memcpy(bufs[i].data() + off, p.first, p.second);
        off += p.second;
      }
      flat.push_back(CacheSrc{srcs[i].key, srcs[i].header, srcs[i].header_len,
                              bufs[i].data(), total});
    }
    return CacheFrom(node, flat);
  }

  // Scatter-gather read: the stored payload is split across each key's N caller
  // segments in order. Default: RangeInto a single contiguous scratch buffer, then
  // split-copy it into the caller segments by their sizes (no zero-copy, but
  // correct on any transport). out_lens[i] (if non-null) = total payload bytes for
  // key[i] (0 on miss). The RDMA transport overrides this to scatter via multi-SGE
  // RECV directly into the caller segments. hdrs[i] gets the value-header bytes.
  virtual std::vector<Status> RangeIntoMulti(
      const std::string& node, const std::vector<BlockKey>& keys,
      const std::vector<RangeDstMulti>& dsts, size_t header_size,
      std::vector<std::string>* hdrs, std::vector<size_t>* out_lens) {
    const size_t n = keys.size();
    hdrs->assign(n, std::string());
    if (out_lens) out_lens->assign(n, 0);
    // Concatenate each key's segments into one contiguous scratch RangeDst, run
    // the existing RangeInto, then split the scratch back into the segments.
    std::vector<std::vector<char>> scratch(n);
    std::vector<RangeDst> flat(n);
    for (size_t i = 0; i < n; ++i) {
      size_t cap = 0;
      for (const auto& p : dsts[i].payloads) cap += p.second;
      scratch[i].resize(cap);
      flat[i] = RangeDst{cap ? scratch[i].data() : nullptr, cap};
    }
    std::vector<Status> r = RangeInto(node, keys, flat, header_size, hdrs);
    for (size_t i = 0; i < n; ++i) {
      if (r[i] != Status::kOk || (*hdrs)[i].size() < header_size) continue;
      // The stored payload length is carried by the value header; the caller
      // (KVClient) verifies geometry. Here we split exactly the bytes we received
      // across the segments. The RangeInto fallback copied min(stored, cap) bytes
      // into scratch; replay that across the segments honoring each segment size.
      size_t off = 0;
      for (const auto& p : dsts[i].payloads) {
        if (p.second) std::memcpy(p.first, scratch[i].data() + off, p.second);
        off += p.second;
      }
      // out_lens = the TRUE received payload length (matching the RDMA override,
      // which reports received_bytes - header), not the sum-of-caps `off`. The
      // stored length lives in the value header; RangeInto delivered min(stored,cap).
      if (out_lens) {
        size_t stored = off;  // fall back to caps if the header can't be parsed
        ValueHeader vh;
        if (header_size >= ValueHeader::kSize &&
            ValueHeader::Parse((*hdrs)[i].data(), (*hdrs)[i].size(), &vh))
          stored = std::min<size_t>(vh.payload_len, off);
        (*out_lens)[i] = stored;
      }
    }
    return r;
  }
};

}  // namespace dfkv

#endif  // DFKV_TRANSPORT_H_
