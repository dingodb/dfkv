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

TEST(ConHashWeight, HugeOrNonPositiveWeightStillOnRing) {
  // 40 * weight used to overflow int for huge weights (keys went negative ->
  // ZERO vnodes -> the node silently vanished from the data plane); weight <= 0
  // had the same silent-drop effect. Both must clamp and stay routable.
  dfkv::ConHash h;
  h.AddNode("huge", 2000000000);
  h.AddNode("plain", 1);
  h.Build();
  int hit_huge = 0;
  for (int i = 0; i < 5000; ++i) {
    std::string node;
    ASSERT_TRUE(h.Lookup("key" + std::to_string(i), &node));
    if (node == "huge") ++hit_huge;
  }
  EXPECT_GT(hit_huge, 0) << "huge-weight node dropped from the ring";

  // Zero weight clamps up to 1: paired with an equal-share peer both nodes
  // must take traffic (deterministic — ~50/50 split over 1000 keys).
  dfkv::ConHash h2;
  h2.AddNode("zero", 0);
  h2.AddNode("plain", 1);
  h2.Build();
  int hit_zero = 0;
  for (int i = 0; i < 1000; ++i) {
    std::string node;
    ASSERT_TRUE(h2.Lookup("key" + std::to_string(i), &node));
    if (node == "zero") ++hit_zero;
  }
  EXPECT_GT(hit_zero, 0) << "zero-weight node dropped from the ring";
}
