/* MetricsHttpServer — a tiny, self-contained HTTP/1.0 responder that exposes a
 * single render callback at GET /metrics (plus GET /healthz). No third-party HTTP
 * dependency: it accepts on its own port and thread and is OFF the datapath, so a
 * Prometheus scrape never touches the cache/RDMA hot path. Only the render
 * callback's relaxed-atomic reads run per scrape. Opt-in (a daemon enables it via
 * --metrics-port); when no port is given no listener exists and behavior is
 * unchanged. */
#ifndef DFKV_METRICS_HTTP_H_
#define DFKV_METRICS_HTTP_H_

#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "kv_store.h"  // Status

namespace dfkv {

class MetricsHttpServer {
 public:
  // render() returns the Prometheus exposition text served at /metrics. It is
  // called once per scrape on the HTTP thread (never on the datapath).
  explicit MetricsHttpServer(std::function<std::string()> render)
      : render_(std::move(render)) {}
  ~MetricsHttpServer();

  Status Start(int port);  // port 0 => ephemeral; query with port()
  void Stop();             // idempotent
  int port() const { return port_; }

 private:
  void AcceptLoop();
  void Handle(int fd);

  std::function<std::string()> render_;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  std::mutex conn_mu_;
  std::set<int> conn_fds_;
  std::vector<std::thread> conn_threads_;
};

}  // namespace dfkv

#endif  // DFKV_METRICS_HTTP_H_
