/* Rail (RDMA device) selection for the multi-rail client. Pure + header-only so
 * it is unit-testable without RDMA hardware or sysfs. */
#ifndef DFKV_RAIL_SELECT_H_
#define DFKV_RAIL_SELECT_H_

#include <cstddef>
#include <vector>

namespace dfkv {
namespace rdma {

// Choose a device index into `dev_node` (NUMA node per device; -1 = unknown).
// NUMA-aware: when numa_on && caller_node>=0 && some device sits on caller_node,
// round-robin among THOSE devices (by rr_tick); otherwise round-robin across all.
// Returns 0 if dev_node is empty (a sentinel — caller must not index an empty set).
inline size_t PickRail(const std::vector<int>& dev_node, int caller_node,
                       bool numa_on, size_t rr_tick) {
  const size_t n = dev_node.size();
  if (n == 0) return 0;
  if (!numa_on || n == 1 || caller_node < 0) return rr_tick % n;
  std::vector<size_t> local;
  for (size_t i = 0; i < n; ++i)
    if (dev_node[i] == caller_node) local.push_back(i);
  if (local.empty()) return rr_tick % n;        // no NUMA-local rail -> all
  return local[rr_tick % local.size()];
}

}  // namespace rdma
}  // namespace dfkv

#endif  // DFKV_RAIL_SELECT_H_
