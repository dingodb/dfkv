/* PeerLatency — per-cache-node client-observed latency, keyed by "ip:port".
 * Fed by KVClient's active prober (a low-frequency background round trip to each
 * known member) so every node shows avg/max latency even when idle. Each peer
 * gets a lock-free LatencyHist (avg=sum/count, p99) plus a running max gauge.
 * The per-peer map is mutex-guarded and cardinality-capped (a churning client
 * can't grow it without bound); the prober is off the datapath so the mutex is
 * never on the hot path. Rendered into the client metrics snapshot. */
#ifndef DFKV_PEER_LATENCY_H_
#define DFKV_PEER_LATENCY_H_

#include <cstdio>
#include <map>
#include <mutex>
#include <string>

#include "utils/latency_hist.h"
#include "utils/prom_escape.h"

namespace dfkv {

class PeerLatency {
 public:
  // Record one observed round-trip latency (seconds) for a peer ("ip:port").
  void Observe(const std::string& peer, double seconds) {
    if (seconds < 0) seconds = 0;
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(peer);
    if (it == peers_.end()) {
      if (peers_.size() >= kMaxPeers) return;  // bound cardinality / series
      it = peers_.try_emplace(peer).first;     // in-place (LatencyHist isn't movable)
    }
    it->second.hist.Observe(seconds);
    if (seconds > it->second.max_seconds) it->second.max_seconds = seconds;
  }

  // Prometheus text: a per-peer latency histogram + a per-peer max gauge.
  std::string Render() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::string s;
    s += "# HELP dfkv_client_peer_latency_seconds Client-observed per-peer op latency seconds\n";
    s += "# TYPE dfkv_client_peer_latency_seconds histogram\n";
    for (const auto& [peer, e] : peers_)
      s += e.hist.RenderBody("dfkv_client_peer_latency_seconds",
                             "peer=\"" + PromLabelEscape(peer) + "\"");
    s += "# HELP dfkv_client_peer_latency_max_seconds Max client-observed per-peer latency seconds\n";
    s += "# TYPE dfkv_client_peer_latency_max_seconds gauge\n";
    char v[64];
    for (const auto& [peer, e] : peers_) {
      std::snprintf(v, sizeof(v), "%.6f", e.max_seconds);
      s += "dfkv_client_peer_latency_max_seconds{peer=\"" + PromLabelEscape(peer) + "\"} " + v + "\n";
    }
    return s;
  }

  // test accessor
  size_t peers() const {
    std::lock_guard<std::mutex> lk(mu_);
    return peers_.size();
  }

 private:
  struct Entry {
    LatencyHist hist;
    double max_seconds = 0;
  };
  static constexpr size_t kMaxPeers = 4096;
  mutable std::mutex mu_;
  std::map<std::string, Entry> peers_;
};

}  // namespace dfkv

#endif  // DFKV_PEER_LATENCY_H_
