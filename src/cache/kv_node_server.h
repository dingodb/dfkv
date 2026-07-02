/* KvNodeServer — a cache-node daemon for the harness: wraps a DiskCacheGroup
 * (one or more NVMe SSDs) and serves Cache/Range/Exist over the TCP wire
 * protocol. The real cache node is dingo-cache (brpc + DiskCache); this proves
 * the semantics end-to-end, including multi-disk intra-node sharding. */
#ifndef DFKV_KV_NODE_SERVER_H_
#define DFKV_KV_NODE_SERVER_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "cache/disk_cache_group.h"
#include "cache/ram_tier.h"
#include "utils/latency_hist.h"

namespace dfkv {

class KvNodeServer {
 public:
  // single disk (back-compat)
  KvNodeServer(const std::string& cache_dir, uint64_t capacity_bytes);
  // multiple NVMe SSDs on this node (total capacity split across disks)
  KvNodeServer(const std::vector<std::string>& cache_dirs,
               uint64_t capacity_bytes);
  ~KvNodeServer();

  Status Start(int port);  // port 0 => ephemeral; query with port()
  void Stop();
  // Cluster member list this node advertises for discovery (kMembers wire op):
  // "name=ip:port,name=ip:port,...". Clients query any node to learn the cluster.
  void set_members(const std::string& members) { members_ = members; }
  // Node identity for Prometheus labels. When unset, series are emitted unlabeled
  // (back-compat with single-node scrapes / older tooling).
  void set_identity(const std::string& id, const std::string& group) {
    node_id_ = id; node_group_ = group;
  }
  // Upper bound for a request frame's declared payload_len. The TCP handler
  // allocates payload_len BEFORE reading the payload bytes, so without a real
  // bound a 42-byte forged prefix triggers a 16 GiB allocation (one-frame OOM
  // DoS). Defaults to value-header + the env-resolved max value size — the
  // SAME bound the RDMA server enforces (utils/wire_limits.h).
  void set_max_request_payload(uint64_t n) { max_request_payload_ = n; }
  int port() const { return port_; }
  size_t Count() const { return group_.Count(); }
  uint64_t UsedBytes() const { return group_.UsedBytes(); }
  size_t DiskCount() const { return group_.DiskCount(); }
  size_t AcceptCount() const { return accept_count_.load(std::memory_order_relaxed); }
  size_t live_conn_count();  // handler threads not yet reaped (test/diagnostic)

  // metrics (relaxed atomics)
  size_t m_cache_put() const { return cache_put_.load(std::memory_order_relaxed); }
  size_t m_cache_hit() const { return cache_hit_.load(std::memory_order_relaxed); }
  size_t m_cache_miss() const { return cache_miss_.load(std::memory_order_relaxed); }
  size_t m_exist_hit() const { return exist_hit_.load(std::memory_order_relaxed); }
  size_t m_exist_miss() const { return exist_miss_.load(std::memory_order_relaxed); }
  size_t m_remove_ok() const { return remove_ok_.load(std::memory_order_relaxed); }
  size_t m_remove_miss() const { return remove_miss_.load(std::memory_order_relaxed); }
  size_t m_wire_v1() const { return wire_v1_requests_.load(std::memory_order_relaxed); }
  size_t m_wire_v2() const { return wire_v2_requests_.load(std::memory_order_relaxed); }
  std::string MetricsText() const;  // Prometheus text format

  // Transport-agnostic request processing (shared by the TCP handler and, when
  // built with DFKV_WITH_RDMA, the RDMA handler). Returns status; fills out_data.
  Status ProcessRequest(uint8_t op, uint64_t id, uint32_t index, uint32_t ksize,
                        uint64_t offset, uint64_t length, const char* payload,
                        uint64_t payload_len, std::string* out_data);

  // Zero-copy server-side GET: read the block straight into `dst` (e.g. the RDMA
  // send buffer), no intermediate std::string. *out_len = bytes read. Updates
  // hit/miss + bytes_read metrics like a Range.
  Status RangeInto(uint64_t id, uint32_t index, uint32_t ksize, uint64_t offset,
                   uint64_t length, char* dst, size_t dst_cap, size_t* out_len);

  // RDMA direct PUT: `data` is an O_DIRECT-aligned [ValueHeader|payload] buffer
  // owned by the RDMA receive slot. Writes it to disk without a payload-sized
  // CPU copy and updates PUT metrics like ProcessRequest(kCache).
  Status CacheDirect(uint64_t id, uint32_t index, uint32_t ksize, char* data,
                     size_t len, size_t cap);

  // RDMA direct GET: read an O_DIRECT-aligned superset into `io_buf`; *out_data
  // points inside that same buffer at the exact requested range so the RDMA layer
  // can scatter-send it without copying into sbuf.
  Status RangeDirect(uint64_t id, uint32_t index, uint32_t ksize, uint64_t offset,
                     uint64_t length, char* io_buf, size_t io_cap,
                     const char** out_data, size_t* out_len);

  // Async-friendly prep half of RangeDirect: index lookup + O_DIRECT open +
  // alignment math, NO disk read (see KVStore::RangeDirectPrep). The caller
  // issues the pread (io_uring) into its io_buf and closes out->fd afterward.
  // Updates miss/io-error counters; the hit/bytes-read counters are bumped by
  // RangeDirectComplete once the read result is known.
  Status RangeDirectPrep(uint64_t id, uint32_t index, uint32_t ksize,
                         uint64_t offset, uint64_t length, size_t io_cap,
                         KVStore::RangePrep* out);
  // Account a completed prep-based GET (called after the async read finishes).
  void RangeDirectComplete(bool ok, size_t bytes_read);

  // --- RAM hot tier zero-copy serve hooks (P3 B5-3) ----------------------------
  // These let the RDMA server send a RAM hit STRAIGHT from the arena MR (no copy
  // into the connection buffer). All no-ops / false when the RAM tier is off.
  bool ram_enabled() const { return ram_ != nullptr; }
  char* ram_arena() const { return ram_ ? ram_->arena() : nullptr; }
  uint64_t ram_arena_bytes() const { return ram_ ? ram_->arena_bytes() : 0; }
  // On a RAM hit, pins the slot and returns its arena pointer + length + a token;
  // the caller MUST call RamRelease(token) once the RDMA send completes (the NIC
  // reads the shared arena in place -- gap 10.1). Bumps hit/miss + bytes-read.
  bool RamRangePrep(uint64_t id, uint32_t index, uint32_t ksize, uint64_t offset,
                    uint64_t length, const char** out_ptr, size_t* out_len,
                    uint64_t* out_token);
  void RamRelease(uint64_t token);

 private:
  void AcceptLoop();
  void Handle(int fd);
  void InitRamTier();     // construct ram_ if DFKV_RAM_TIER is enabled (env)
  void ReapDoneLocked();  // join+erase finished handler threads; conn_mu_ held
  std::atomic<size_t> cache_put_{0}, cache_hit_{0}, cache_miss_{0};
  std::atomic<size_t> exist_hit_{0}, exist_miss_{0};
  std::atomic<size_t> remove_ok_{0}, remove_miss_{0};
  std::atomic<size_t> bytes_written_{0}, bytes_read_{0};
  // depth metrics: errors by op, live connections, sampled op latency
  std::atomic<size_t> put_io_err_{0}, get_io_err_{0}, invalid_ops_{0};
  std::atomic<size_t> open_connections_{0};
  Sampler lat_sampler_{64};        // 1-in-64 latency sampling (near-zero hot-path cost)
  LatencyHist get_lat_, put_lat_;  // server-side op latency (sampled)
  std::atomic<uint64_t> wire_v1_requests_{0}, wire_v2_requests_{0};  // by frame version
  std::string members_;  // advertised cluster membership (kMembers)
  uint64_t max_request_payload_ = 0;  // 0 = resolve default lazily (see .cc)
  std::string node_id_, node_group_;       // identity for Prometheus labels (optional)
  std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();

  DiskCacheGroup group_;
  // Optional RAM hot tier (P3). Declared AFTER group_ so it is destroyed FIRST:
  // ~RamTier stops+joins the flusher, whose callback calls group_.Cache, before
  // group_ is torn down. nullptr unless DFKV_RAM_TIER is enabled. Write-through:
  // PUT lands in RAM (sync-visible) + async-flushes to group_; GET checks RAM
  // first (served from the arena) then falls back to group_.
  std::unique_ptr<RamTier> ram_;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::atomic<size_t> accept_count_{0};
  std::thread accept_thread_;
  // connection drain: track per-connection handler threads + their fds so Stop()
  // can unblock (shutdown) and join them before group_ is destroyed. Finished
  // handlers are reaped incrementally at accept time (done flag), so a long-lived
  // node with churning clients doesn't accumulate dead thread handles.
  struct Conn {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::mutex conn_mu_;
  std::set<int> conn_fds_;
  std::vector<Conn> conns_;
};

}  // namespace dfkv

#endif  // DFKV_KV_NODE_SERVER_H_
