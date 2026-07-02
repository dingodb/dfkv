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

TEST(PeerHealth, CooldownBacksOffExponentiallyAndCapsAtMax) {
  PeerHealth h(/*base_cooldown_ms=*/100, /*max_cooldown_ms=*/800);
  // 1st failure -> 100ms cooldown: unhealthy at now, healthy at now+100.
  h.MarkBad("p", 1000);
  EXPECT_FALSE(h.Healthy("p", 1050));
  EXPECT_TRUE(h.Healthy("p", 1100));
  // Consecutive failures double the cooldown: 200, 400, 800, then capped at 800.
  h.MarkBad("p", 2000); EXPECT_FALSE(h.Healthy("p", 2199)); EXPECT_TRUE(h.Healthy("p", 2200));
  h.MarkBad("p", 3000); EXPECT_FALSE(h.Healthy("p", 3399)); EXPECT_TRUE(h.Healthy("p", 3400));
  h.MarkBad("p", 4000); EXPECT_FALSE(h.Healthy("p", 4799)); EXPECT_TRUE(h.Healthy("p", 4800));
  h.MarkBad("p", 5000); EXPECT_FALSE(h.Healthy("p", 5799)); EXPECT_TRUE(h.Healthy("p", 5800));  // capped
}

TEST(PeerHealth, MarkGoodResetsBackoffStreak) {
  PeerHealth h(/*base=*/100, /*max=*/10000);
  h.MarkBad("p", 0); h.MarkBad("p", 0); h.MarkBad("p", 0);  // streak -> 400ms
  h.MarkGood("p");                                          // reset
  h.MarkBad("p", 1000);                                     // back to base 100ms
  EXPECT_FALSE(h.Healthy("p", 1099));
  EXPECT_TRUE(h.Healthy("p", 1100));
}

TEST(PeerHealth, MarkProbeAliveRecoversWithoutCountingServed) {
  PeerHealth h(/*base=*/1000, /*max=*/30000);
  h.MarkBad("p", 0);
  EXPECT_FALSE(h.Healthy("p", 500));
  uint64_t served_before = h.served();
  h.MarkProbeAlive("p");                       // off-data-path recovery
  EXPECT_TRUE(h.Healthy("p", 500));            // cooldown cleared
  EXPECT_EQ(h.served(), served_before);        // served counter untouched
  EXPECT_EQ(h.recovered(), 1u);                // bad->good edge counted
  // And the streak was reset: next failure starts at base again.
  h.MarkBad("p", 1000);
  EXPECT_TRUE(h.Healthy("p", 2000));
}
