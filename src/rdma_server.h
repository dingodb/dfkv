/* RDMA cache-node listener — native libibverbs RC. Built only when
 * DFKV_WITH_RDMA is defined. Reuses the wire frames + a request handler so the
 * cache logic (DiskCacheGroup + metrics) is shared with the TCP server.
 *
 * The "listener" is a plain TCP accept socket used only to bootstrap QPs
 * (exchange LID/GID/QPN over a few bytes). Each accepted connection opens an RC
 * QP on the named RDMA device (--rdma-dev / DFKV_RDMA_DEV) and serves requests
 * over RDMA. Using TCP accept (not rdma_get_request) makes Stop() interruptible:
 * shutdown(listen_fd) wakes accept() instantly — no shutdown hang. */
#ifndef DFKV_RDMA_SERVER_H_
#define DFKV_RDMA_SERVER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "kv_store.h"   // Status
#include "rdma_verbs.h"  // rdma::RcEndpoint

namespace dfkv {

class RdmaServer {
 public:
  using Handler = std::function<Status(
      uint8_t op, uint64_t id, uint32_t index, uint32_t ksize, uint64_t offset,
      uint64_t length, const char* payload, uint64_t payload_len,
      std::string* out_data)>;
  // Optional direct GET handler: read an O_DIRECT-aligned superset into `io_buf`
  // and return *out_data pointing inside that same registered buffer at the exact
  // requested range. When set, kRange replies scatter-send [resp | *out_data]
  // without copying the payload into sbuf.
  using RangeHandler = std::function<Status(
      uint64_t id, uint32_t index, uint32_t ksize, uint64_t offset,
      uint64_t length, char* io_buf, size_t io_cap, const char** out_data,
      size_t* out_len)>;
  // Optional direct PUT handler: `data` points at a 4096-aligned registered
  // buffer containing the full stored blob [ValueHeader|payload]. `cap` is the
  // usable direct-buffer capacity. The handler may zero O_DIRECT padding bytes
  // after len and write data directly to disk.
  using CacheDirectHandler = std::function<Status(
      uint64_t id, uint32_t index, uint32_t ksize, char* data, size_t len,
      size_t cap)>;

  // dev_name empty => env DFKV_RDMA_DEV, else first device.
  explicit RdmaServer(Handler handler, size_t max_msg = (64u << 20),
                      const std::string& dev_name = "");
  void set_range_handler(RangeHandler h) { range_handler_ = std::move(h); }
  void set_cache_direct_handler(CacheDirectHandler h) {
    cache_direct_handler_ = std::move(h);
  }
  ~RdmaServer();

  Status Start(int port);  // TCP bootstrap port
  void Stop();
  int port() const { return port_; }

  // Observability (used by tests): number of Serve threads not yet reaped.
  // Bounded under churn now that finished connections are joined as new ones
  // arrive — previously this grew without bound until Stop().
  size_t live_conn_count();

  // RDMA completion counters (relaxed atomics, incremented in the serve loop).
  uint64_t Completions() const { return completions_.load(std::memory_order_relaxed); }
  uint64_t CompletionErrors() const { return completion_errors_.load(std::memory_order_relaxed); }
  uint64_t ActiveConns() const { return active_conns_.load(std::memory_order_relaxed); }
  uint64_t IdleReclaims() const { return idle_reclaims_.load(std::memory_order_relaxed); }
  std::string MetricsText() const;  // Prometheus text (dfkv_rdma_*)

 private:
  void AcceptLoop();
  void Serve(int boot_fd);
  void ReapDoneLocked();  // join+erase finished Serve threads; conn_mu_ held

  // A live connection: its Serve thread plus a flag the thread sets (last thing
  // it does) so AcceptLoop can tell it has finished and join it without blocking.
  // The flag is a shared_ptr because conns_ reallocates on push_back — the thread
  // captures its own copy, so the atomic outlives any vector move/erase.
  struct Conn {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
  };

  Handler handler_;
  RangeHandler range_handler_;
  CacheDirectHandler cache_direct_handler_;
  size_t max_msg_;
  size_t control_cap_;
  std::string dev_name_;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  // Track per-connection Serve threads + their endpoints so Stop() can wake them
  // out of WaitComp and join them before the handler's owner is destroyed, and
  // so finished threads are reaped incrementally (see ReapDoneLocked).
  std::mutex conn_mu_;
  std::vector<Conn> conns_;
  std::unordered_set<rdma::RcEndpoint*> live_eps_;
  std::atomic<uint64_t> completions_{0}, completion_errors_{0}, active_conns_{0},
      idle_reclaims_{0};
};

}  // namespace dfkv

#endif  // DFKV_RDMA_SERVER_H_
