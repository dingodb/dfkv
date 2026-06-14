/* KvNodeServer — a cache-node daemon for the harness: wraps a DiskCacheGroup
 * (one or more NVMe SSDs) and serves Cache/Range/Exist over the TCP wire
 * protocol. The real cache node is dingo-cache (brpc + DiskCache); this proves
 * the semantics end-to-end, including multi-disk intra-node sharding. */
#ifndef DINGOFS_SRC_CACHE_KVCLIENT_KV_NODE_SERVER_H_
#define DINGOFS_SRC_CACHE_KVCLIENT_KV_NODE_SERVER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "disk_cache_group.h"

namespace dingofs {
namespace cache {
namespace kv {

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

 private:
  void AcceptLoop();
  void Handle(int fd);

  DiskCacheGroup group_;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
};

}  // namespace kv
}  // namespace cache
}  // namespace dingofs

#endif  // DINGOFS_SRC_CACHE_KVCLIENT_KV_NODE_SERVER_H_
