/* group_shard: split a per-node Batch* work group into up to N shards so
 * multiple connections drive one node concurrently.
 *
 * A single- or few-node ring otherwise routes a whole batch to ONE (node[,size])
 * group -> one fan-out worker -> one QP, where the server drains it key-by-key
 * (~7 ms/key, ~166 MB/s/conn measured on the phase-8 hit-rate probe -- the L3
 * read-back ceiling; the SG write path has the same single-connection drain).
 * Sharding turns that into up to `maxc` parallel streams per node.
 *
 * Pure logic (no env / no config_dump) so it is unit-testable in isolation; the
 * caller resolves the DFKV_READ_SHARD_KEYS / DFKV_READ_MAX_CONNS knobs once and
 * passes them in. `target` = shard target size (keys per shard); `maxc` = per-node
 * shard/connection cap. Each shard keeps its group's key (`first`) so downstream
 * code is unchanged, and the per-key order within a group is preserved (shards
 * are contiguous slices in the original index order). Groups already at/under one
 * shard pass through untouched (wide rings keep their existing per-node
 * parallelism). Generic over the group vector element: works for the read path's
 * pair<pair<node,size>, idx> and the write path's pair<node, idx> alike.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>

namespace dfkv {

template <class GroupVec>
GroupVec ShardGroups(GroupVec groups, size_t target, size_t maxc) {
  if (maxc <= 1 || target == 0) return groups;
  GroupVec out;
  out.reserve(groups.size());
  for (auto& g : groups) {
    const auto& idx = g.second;
    size_t nsh = (idx.size() + target - 1) / target;
    if (nsh < 1) nsh = 1;
    if (nsh > maxc) nsh = maxc;
    if (nsh <= 1) { out.push_back(std::move(g)); continue; }
    const size_t per = (idx.size() + nsh - 1) / nsh;
    for (size_t s = 0; s < idx.size(); s += per) {
      typename GroupVec::value_type sh;
      sh.first = g.first;
      sh.second.assign(idx.begin() + s,
                       idx.begin() + std::min(idx.size(), s + per));
      out.push_back(std::move(sh));
    }
  }
  return out;
}

}  // namespace dfkv
