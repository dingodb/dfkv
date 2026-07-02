#include "utils/numa_util.h"

#include <gtest/gtest.h>

using dfkv::numa::CpuInList;
using dfkv::numa::CurrentNode;

TEST(CpuInList, Ranges) {
  EXPECT_TRUE(CpuInList("0-3,8-11", 2));
  EXPECT_TRUE(CpuInList("0-3,8-11", 0));
  EXPECT_TRUE(CpuInList("0-3,8-11", 3));
  EXPECT_TRUE(CpuInList("0-3,8-11", 8));
  EXPECT_TRUE(CpuInList("0-3,8-11", 11));
  EXPECT_TRUE(CpuInList("0-3,8-11", 9));
  EXPECT_FALSE(CpuInList("0-3,8-11", 5));
  EXPECT_FALSE(CpuInList("0-3,8-11", 12));
}

TEST(CpuInList, SingletonsAndEdges) {
  EXPECT_TRUE(CpuInList("5", 5));
  EXPECT_FALSE(CpuInList("5", 6));
  EXPECT_TRUE(CpuInList("48-95,144-191", 144));
  EXPECT_FALSE(CpuInList("48-95,144-191", 96));
  EXPECT_FALSE(CpuInList("", 0));
  EXPECT_FALSE(CpuInList(nullptr, 0));
}

TEST(CurrentNode, BestEffortStableAndValid) {
  int a = CurrentNode();
  EXPECT_GE(a, -1);          // -1 (unknown/single) or a real node id
  EXPECT_EQ(a, CurrentNode());  // stable across calls on a pinned/idle thread
}
