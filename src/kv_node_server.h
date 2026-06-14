/* KvNodeServer — a cache-node daemon for the harness: wraps a DiskCacheGroup
 * (one or more NVMe SSDs) and serves Cache/Range/Exist over the TCP wire
 * protocol. The real cache node is dingo-cache (brpc + DiskCache); this proves
 * the semantics end-to-end, including multi-disk intra-node sharding. */
#ifndef DFKV_KV_NODE_SERVER_H_
#define DFKV_KV_NODE_SERVER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "disk_cache_group.h"

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
  int port() const { return port_; }
  size_t Count() const { return group_.Count(); }
  uint64_t UsedBytes() const { return group_.UsedBytes(); }
  size_t DiskCount() const { return group_.DiskCount(); }
  size_t AcceptCount() const { return accept_count_.load(std::memory_order_relaxed); }

  // metrics (relaxed atomics)
  size_t m_cache_put() const { return cache_put_.load(std::memory_order_relaxed); }
  size_t m_cache_hit() const { return cache_hit_.load(std::memory_order_relaxed); }
  size_t m_cache_miss() const { return cache_miss_.load(std::memory_order_relaxed); }
  size_t m_exist_hit() const { return exist_hit_.load(std::memory_order_relaxed); }
  size_t m_exist_miss() const { return exist_miss_.load(std::memory_order_relaxed); }
  std::string MetricsText() const;  // Prometheus text format

 private:
  void AcceptLoop();
  void Handle(int fd);
  std::atomic<size_t> cache_put_{0}, cache_hit_{0}, cache_miss_{0};
  std::atomic<size_t> exist_hit_{0}, exist_miss_{0};
  std::atomic<size_t> bytes_written_{0}, bytes_read_{0};

  DiskCacheGroup group_;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::atomic<size_t> accept_count_{0};
  std::thread accept_thread_;
  // connection drain: track per-connection handler threads + their fds so Stop()
  // can unblock (shutdown) and join them before group_ is destroyed.
  std::mutex conn_mu_;
  std::set<int> conn_fds_;
  std::vector<std::thread> conn_threads_;
};

}  // namespace dfkv

#endif  // DFKV_KV_NODE_SERVER_H_
