#ifndef DFKV_PEER_HEALTH_H_
#define DFKV_PEER_HEALTH_H_

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "utils/prom_escape.h"

namespace dfkv {

// Layer-1 fast avoidance: a peer (keyed by "ip:port") that fails a transport IO
// is marked unhealthy for cooldown_ms; while unhealthy the client short-circuits
// requests to it (miss) WITHOUT touching the ring. A served response (hit or
// logical miss) clears it. Thread-safe: BatchGet/Put route from many threads.
//
// It is also the client's uniform metrics chokepoint: every op calls Healthy()
// then exactly one of MarkBad()/MarkGood(), so the client-observed counters live
// here at no extra hot-path cost (the mutex is already taken on these calls).
class PeerHealth {
 public:
  // base_cooldown_ms doubles per consecutive failure up to max_cooldown_ms. A
  // FIXED cooldown shorter than the connect timeout (2s < 3s) meant a dead peer
  // was re-dialed on the data path every base period, stalling each batch on a
  // full connect timeout indefinitely; exponential backoff caps that cost at
  // one connect per max_cooldown once a peer is persistently down.
  explicit PeerHealth(uint64_t base_cooldown_ms = 2000,
                      uint64_t max_cooldown_ms = 30000)
      : base_cooldown_ms_(base_cooldown_ms), max_cooldown_ms_(max_cooldown_ms) {}

  bool Healthy(const std::string& peer, uint64_t now_ms) const {
    std::lock_guard<std::mutex> lk(mu_);
    checks_.fetch_add(1, std::memory_order_relaxed);
    auto it = until_.find(peer);
    bool ok = it == until_.end() || it->second <= now_ms;
    if (!ok) unhealthy_skips_.fetch_add(1, std::memory_order_relaxed);
    return ok;
  }
  void MarkBad(const std::string& peer, uint64_t now_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = until_.find(peer);
    bool was_ok = it == until_.end() || it->second <= now_ms;
    // Exponential backoff: base << (fails-1), capped at max_cooldown.
    uint32_t fails = ++fail_streak_[peer];
    unsigned shift = fails > 31 ? 31u : (fails - 1);
    uint64_t cd = base_cooldown_ms_ << shift;
    if (cd > max_cooldown_ms_ || cd < base_cooldown_ms_ /*overflow*/) cd = max_cooldown_ms_;
    until_[peer] = now_ms + cd;
    errors_.fetch_add(1, std::memory_order_relaxed);
    if (was_ok) marked_bad_.fetch_add(1, std::memory_order_relaxed);  // healthy->bad edge
    // Bound per-peer cardinality: only track a new peer while under the cap, so a
    // long-lived client churning through many distinct addresses can't grow the
    // map (and its scrape series) without bound. Aggregate errors_ still counts.
    if (peer_errors_.size() < kMaxPeers || peer_errors_.count(peer)) peer_errors_[peer]++;
  }
  // Data-path success: clears the cooldown and the backoff streak. Bumps the
  // served counter (an op completed), so it is NOT for probe-driven recovery.
  void MarkGood(const std::string& peer) {
    std::lock_guard<std::mutex> lk(mu_);
    served_.fetch_add(1, std::memory_order_relaxed);
    ClearLocked(peer);
  }
  // Probe-driven recovery (off the data path): the background prober reached
  // the peer, so clear its cooldown/streak WITHOUT counting a served op. This
  // is what lets recovery happen without the data path ever having to re-dial a
  // dead peer inside its (now backed-off) cooldown.
  void MarkProbeAlive(const std::string& peer) {
    std::lock_guard<std::mutex> lk(mu_);
    ClearLocked(peer);
  }

  // Prometheus client-side metrics text. Aggregates + per-peer error counts.
  std::string Render() const {
    auto m = [](std::string& s, const char* name, const char* type, const char* help,
                uint64_t v) {
      s += "# HELP "; s += name; s += " "; s += help; s += "\n";
      s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
      s += name; s += " "; s += std::to_string(v); s += "\n";
    };
    std::string s;
    m(s, "dfkv_client_ops_served_total", "counter", "Ops with a served response",
      served_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_io_errors_total", "counter", "Transport IO failures (client-observed)",
      errors_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_health_checks_total", "counter", "Peer health checks performed",
      checks_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_unhealthy_skips_total", "counter", "Ops short-circuited (peer in cooldown)",
      unhealthy_skips_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_peer_marked_bad_total", "counter", "Healthy->bad peer transitions",
      marked_bad_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_peer_recovered_total", "counter", "Bad->good peer transitions",
      recovered_.load(std::memory_order_relaxed));
    // per-peer errors (one HELP/TYPE, one series per peer seen)
    std::map<std::string, uint64_t> snap;
    {
      std::lock_guard<std::mutex> lk(mu_);
      snap.insert(peer_errors_.begin(), peer_errors_.end());
    }
    s += "# HELP dfkv_client_peer_errors_total IO failures per peer\n";
    s += "# TYPE dfkv_client_peer_errors_total counter\n";
    for (const auto& [peer, n] : snap)
      s += "dfkv_client_peer_errors_total{peer=\"" + PromLabelEscape(peer) + "\"} " +
           std::to_string(n) + "\n";
    return s;
  }

  // test accessors
  uint64_t served() const { return served_.load(std::memory_order_relaxed); }
  uint64_t errors() const { return errors_.load(std::memory_order_relaxed); }
  uint64_t marked_bad() const { return marked_bad_.load(std::memory_order_relaxed); }
  uint64_t recovered() const { return recovered_.load(std::memory_order_relaxed); }

 private:
  // Clear a peer's cooldown + backoff streak (recovery). Caller holds mu_.
  void ClearLocked(const std::string& peer) {
    fail_streak_.erase(peer);
    if (until_.erase(peer)) recovered_.fetch_add(1, std::memory_order_relaxed);  // bad->good edge
  }

  static constexpr size_t kMaxPeers = 4096;  // per-peer error-map cardinality cap
  mutable std::mutex mu_;
  uint64_t base_cooldown_ms_;
  uint64_t max_cooldown_ms_;
  std::unordered_map<std::string, uint64_t> until_;        // peer -> unhealthy-until (ms)
  std::unordered_map<std::string, uint32_t> fail_streak_;  // peer -> consecutive failures
  std::unordered_map<std::string, uint64_t> peer_errors_;  // peer -> cumulative IO errors
  mutable std::atomic<uint64_t> checks_{0}, unhealthy_skips_{0}, errors_{0},
      marked_bad_{0}, served_{0}, recovered_{0};
};

}  // namespace dfkv

#endif  // DFKV_PEER_HEALTH_H_
