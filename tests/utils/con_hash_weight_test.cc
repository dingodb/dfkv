#include "utils/con_hash.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <string>
using namespace dfkv;  // NOLINT

// Weight 3 node should receive ~3x the keys of a weight 1 node. Ketama spreads
// 40*weight*4 vnodes, so over many keys the ratio converges. Wide tolerance
// keeps the test deterministic-yet-robust (MD5-hashed fixed key set, no RNG).
TEST(ConHashWeight, DistributionRoughlyProportional) {
  ConHash h;
  h.AddNode("light", 1);
  h.AddNode("heavy", 3);
  h.Build();
  int light = 0, heavy = 0;
  const int N = 20000;
  for (int i = 0; i < N; ++i) {
    std::string node;
    ASSERT_TRUE(h.Lookup("key_" + std::to_string(i), &node));
    if (node == "light") ++light; else ++heavy;
  }
  ASSERT_GT(light, 0);
  double ratio = static_cast<double>(heavy) / light;
  EXPECT_GT(ratio, 2.0) << "heavy=" << heavy << " light=" << light;
  EXPECT_LT(ratio, 4.5) << "heavy=" << heavy << " light=" << light;
}

TEST(ConHashWeight, EqualWeightRoughlyBalanced) {
  ConHash h;
  h.AddNode("a", 1);
  h.AddNode("b", 1);
  h.Build();
  int a = 0, b = 0;
  for (int i = 0; i < 20000; ++i) {
    std::string node;
    h.Lookup("k" + std::to_string(i), &node);
    (node == "a" ? a : b)++;
  }
  double ratio = static_cast<double>(std::max(a, b)) / std::min(a, b);
  EXPECT_LT(ratio, 1.3) << "a=" << a << " b=" << b;
}
