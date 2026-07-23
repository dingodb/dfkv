// Pure-logic tests for ShardGroups (client/group_shard.h): the per-node group
// splitter that lets the Batch* read AND SG-write paths drive one node over up to
// DFKV_READ_MAX_CONNS connections instead of a single serial-drain QP. No
// transport / server: it asserts the shard partition (count, contiguity, order
// preservation, boundaries) that the fan-out then turns into parallel connections.
#include "client/group_shard.h"

#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {
// Read-path group key is pair<node,size>; write-path (BatchPutSg) is a bare node
// string. Cover both element shapes so a change to either call site is caught.
using ReadGroup = std::pair<std::pair<std::string, size_t>, std::vector<size_t>>;
using WriteGroup = std::pair<std::string, std::vector<size_t>>;

std::vector<size_t> Iota(size_t n) {
  std::vector<size_t> v(n);
  std::iota(v.begin(), v.end(), 0);
  return v;
}

// Every original index appears exactly once across the shards, in ascending
// order overall (shards are contiguous slices), and each shard is non-empty.
template <class GroupVec>
void AssertCoversInOrder(const GroupVec& out, size_t n) {
  std::vector<size_t> seen;
  for (const auto& g : out) {
    EXPECT_FALSE(g.second.empty());
    for (size_t x : g.second) seen.push_back(x);
  }
  ASSERT_EQ(seen.size(), n);
  for (size_t i = 0; i < n; ++i) EXPECT_EQ(seen[i], i) << "index " << i;
}
}  // namespace

// A group at/under one shard target passes through untouched (wide rings keep
// their existing per-node parallelism; no needless fragmentation).
TEST(ShardGroups, SmallGroupUntouched) {
  std::vector<WriteGroup> in{{"n0", Iota(16)}};
  auto out = ShardGroups(in, /*target=*/16, /*maxc=*/8);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].first, "n0");
  EXPECT_EQ(out[0].second.size(), 16u);
}

// One node with many keys splits into exactly ceil(n/target) shards, capped at
// maxc — this is the single-node-ring case the SG paths care about.
TEST(ShardGroups, OneNodeSplitsToCeil) {
  std::vector<WriteGroup> in{{"n0", Iota(40)}};  // ceil(40/16)=3 <= 8
  auto out = ShardGroups(in, 16, 8);
  EXPECT_EQ(out.size(), 3u);
  for (const auto& g : out) EXPECT_EQ(g.first, "n0");
  AssertCoversInOrder(out, 40);
  // Balanced slices: ceil(40/3)=14, so 14+14+12.
  EXPECT_EQ(out[0].second.size(), 14u);
  EXPECT_EQ(out[1].second.size(), 14u);
  EXPECT_EQ(out[2].second.size(), 12u);
  EXPECT_EQ(out[0].second.front(), 0u);
  EXPECT_EQ(out.back().second.back(), 39u);
}

// Shard count never exceeds maxc no matter how large the group (the per-node
// connection ceiling).
TEST(ShardGroups, CappedAtMaxConns) {
  std::vector<WriteGroup> in{{"n0", Iota(1000)}};
  auto out = ShardGroups(in, 16, 8);
  EXPECT_EQ(out.size(), 8u);
  AssertCoversInOrder(out, 1000);
}

// maxc<=1 disables sharding entirely (the DFKV_READ_MAX_CONNS=1 opt-out).
TEST(ShardGroups, MaxConnsOneIsPassthrough) {
  std::vector<WriteGroup> in{{"n0", Iota(100)}};
  auto out = ShardGroups(in, 16, 1);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].second.size(), 100u);
}

// Independent per-node groups each shard on their own; a small group rides
// through beside a large one that splits.
TEST(ShardGroups, MultiNodeShardsIndependently) {
  std::vector<WriteGroup> in{{"n0", Iota(48)}, {"n1", Iota(4)}};
  auto out = ShardGroups(in, 16, 8);
  size_t n0 = 0, n1 = 0;
  for (const auto& g : out) (g.first == "n0" ? n0 : n1)++;
  EXPECT_EQ(n0, 3u);  // ceil(48/16)=3
  EXPECT_EQ(n1, 1u);  // stays whole
}

// The read-path element shape (pair<pair<node,size>, idx>) shards on idx and
// carries the composite key onto every shard unchanged.
TEST(ShardGroups, ReadGroupKeyShapePreserved) {
  std::vector<ReadGroup> in{{{"n0", 4096}, Iota(40)}};
  auto out = ShardGroups(in, 16, 8);
  EXPECT_EQ(out.size(), 3u);
  for (const auto& g : out) {
    EXPECT_EQ(g.first.first, "n0");
    EXPECT_EQ(g.first.second, 4096u);
  }
  AssertCoversInOrder(out, 40);
}
