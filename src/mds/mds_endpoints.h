#ifndef DFKV_MDS_ENDPOINTS_H_
#define DFKV_MDS_ENDPOINTS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dfkv {

// Round-robin selection over a static MDS endpoint list with per-endpoint
// failure backoff. There is NO leader concept — the MDS tier is stateless, so
// any reachable MDS is correct (see design doc decision 3). The clock is
// injected (now_ms) so the policy is unit-testable without real time; callers
// pass a monotonic millisecond clock at runtime.
class MdsEndpoints {
 public:
  explicit MdsEndpoints(std::vector<std::string> eps, uint64_t backoff_ms = 2000)
      : eps_(std::move(eps)), backoff_ms_(backoff_ms), until_(eps_.size(), 0) {}

  size_t size() const { return eps_.size(); }

  // Returns the next endpoint whose backoff has expired, advancing the cursor.
  // If ALL are in backoff, returns the one closest to recovery (degrade, never
  // fail). Empty string only when the list itself is empty.
  std::string Pick(uint64_t now_ms) {
    if (eps_.empty()) return {};
    size_t best = 0;
    uint64_t best_until = UINT64_MAX;
    for (size_t i = 0; i < eps_.size(); ++i) {
      size_t idx = (cursor_ + i) % eps_.size();
      if (until_[idx] <= now_ms) {
        cursor_ = (idx + 1) % eps_.size();
        return eps_[idx];
      }
      if (until_[idx] < best_until) { best_until = until_[idx]; best = idx; }
    }
    cursor_ = (best + 1) % eps_.size();
    return eps_[best];
  }

  void MarkFailed(const std::string& ep, uint64_t now_ms) {
    for (size_t i = 0; i < eps_.size(); ++i)
      if (eps_[i] == ep) until_[i] = now_ms + backoff_ms_;
  }

  void MarkOk(const std::string& ep) {
    for (size_t i = 0; i < eps_.size(); ++i)
      if (eps_[i] == ep) until_[i] = 0;
  }

 private:
  std::vector<std::string> eps_;
  uint64_t backoff_ms_;
  std::vector<uint64_t> until_;  // backoff-until (ms) per endpoint; 0 = eligible
  size_t cursor_ = 0;
};

}  // namespace dfkv

#endif  // DFKV_MDS_ENDPOINTS_H_
