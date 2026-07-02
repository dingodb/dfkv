#include "transport/rail_select.h"

#include <gtest/gtest.h>

using dfkv::rdma::PickRail;

TEST(PickRail, NumaOffRoundRobinsAll) {
  std::vector<int> dn{0, 0, 1, 1};
  for (size_t t = 0; t < 6; ++t)
    EXPECT_EQ(PickRail(dn, /*caller=*/1, /*numa_on=*/false, t), t % 4);
}

TEST(PickRail, CallerUnknownRoundRobinsAll) {
  std::vector<int> dn{0, 1};
  EXPECT_EQ(PickRail(dn, -1, true, 0), 0u);
  EXPECT_EQ(PickRail(dn, -1, true, 1), 1u);
  EXPECT_EQ(PickRail(dn, -1, true, 2), 0u);
}

TEST(PickRail, SingleRailAlwaysZero) {
  std::vector<int> dn{-1};
  EXPECT_EQ(PickRail(dn, 0, true, 0), 0u);
  EXPECT_EQ(PickRail(dn, 0, true, 7), 0u);
}

TEST(PickRail, PicksLocalSubsetNode1) {  // dev 2,3 are on NUMA1
  std::vector<int> dn{0, 0, 1, 1};
  EXPECT_EQ(PickRail(dn, 1, true, 0), 2u);
  EXPECT_EQ(PickRail(dn, 1, true, 1), 3u);
  EXPECT_EQ(PickRail(dn, 1, true, 2), 2u);
}

TEST(PickRail, PicksLocalSubsetNode0) {  // dev 0,1 are on NUMA0
  std::vector<int> dn{0, 0, 1, 1};
  EXPECT_EQ(PickRail(dn, 0, true, 0), 0u);
  EXPECT_EQ(PickRail(dn, 0, true, 1), 1u);
  EXPECT_EQ(PickRail(dn, 0, true, 2), 0u);
}

TEST(PickRail, NoLocalRailFallsBackToAll) {  // caller NUMA1 but both devs NUMA0
  std::vector<int> dn{0, 0};
  EXPECT_EQ(PickRail(dn, 1, true, 0), 0u);
  EXPECT_EQ(PickRail(dn, 1, true, 1), 1u);
  EXPECT_EQ(PickRail(dn, 1, true, 2), 0u);
}

TEST(PickRail, AllUnknownFallsBackToAll) {  // single-NUMA box: all -1
  std::vector<int> dn{-1, -1};
  EXPECT_EQ(PickRail(dn, 0, true, 0), 0u);
  EXPECT_EQ(PickRail(dn, 0, true, 1), 1u);
}

TEST(PickRail, EmptyDeviceListReturnsZero) {
  std::vector<int> dn{};
  EXPECT_EQ(PickRail(dn, 0, true, 5), 0u);
  EXPECT_EQ(PickRail(dn, -1, false, 0), 0u);
}
