#include "client/peer_health.h"
#include <gtest/gtest.h>
using namespace dfkv;  // NOLINT

TEST(PeerHealth, UnknownPeerIsHealthy) {
  PeerHealth h(1000);
  EXPECT_TRUE(h.Healthy("a", 0));
}

TEST(PeerHealth, MarkBadCoolsDownThenRecovers) {
  PeerHealth h(/*cooldown_ms=*/1000);
  h.MarkBad("a", 5000);
  EXPECT_FALSE(h.Healthy("a", 5000));
  EXPECT_FALSE(h.Healthy("a", 5999));
  EXPECT_TRUE(h.Healthy("a", 6000));
}

TEST(PeerHealth, MarkGoodClearsImmediately) {
  PeerHealth h(1000);
  h.MarkBad("a", 1000);
  EXPECT_FALSE(h.Healthy("a", 1200));
  h.MarkGood("a");
  EXPECT_TRUE(h.Healthy("a", 1200));
}

TEST(PeerHealth, PeersAreIndependent) {
  PeerHealth h(1000);
  h.MarkBad("a", 1000);
  EXPECT_FALSE(h.Healthy("a", 1500));
  EXPECT_TRUE(h.Healthy("b", 1500));
}
