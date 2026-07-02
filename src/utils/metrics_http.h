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
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/status.h"

namespace dfkv {

class MetricsHttpServer {
 public:
  // render() returns the Prometheus exposition text served at /metrics. It is
  // called once per scrape on the HTTP thread (never on the datapath).
  explicit MetricsHttpServer(std::function<std::string()> render)
      : render_(std::move(render)) {}
  ~MetricsHttpServer();

  // port 0 => ephemeral (query with port()). bind_addr empty => all interfaces
  // (back-compat); pass e.g. "127.0.0.1" to restrict /metrics to loopback.
  Status Start(int port, const std::string& bind_addr = "");
  void Stop();             // idempotent
  int port() const { return port_; }
  // Handler threads not yet reaped (test/diagnostic). Bounded under scrape churn
  // now that finished connections are joined as new ones arrive — Prometheus
  // scrapes are Connection: close, so without reaping this grew per scrape.
  size_t live_conn_count();

 private:
  void AcceptLoop();
  void Handle(int fd);
  void ReapDoneLocked();  // join+erase finished handler threads; conn_mu_ held

  // A live connection: its handler thread + a flag the thread sets last, so
  // AcceptLoop can join it without blocking. shared_ptr because conns_ may
  // reallocate; the thread captures its own copy.
  struct Conn {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
  };

  std::function<std::string()> render_;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  std::mutex conn_mu_;
  std::set<int> conn_fds_;
  std::vector<Conn> conns_;
};

}  // namespace dfkv

#endif  // DFKV_METRICS_HTTP_H_
