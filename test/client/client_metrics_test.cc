// TDD — client-side metrics live in PeerHealth (the uniform per-op chokepoint).
#include "client/peer_health.h"

#include <gtest/gtest.h>

#include <string>

using namespace dfkv;  // NOLINT

TEST(ClientMetrics, CountsServedErrorsTransitionsAndRenders) {
  PeerHealth h(/*cooldown_ms=*/1000);
  const std::string p = "10.0.0.1:12000";
  // a healthy check, then a served response
  EXPECT_TRUE(h.Healthy(p, 1000));
  h.MarkGood(p);
  // an IO error marks it bad (healthy->bad edge), then a skip while in cooldown
  h.MarkBad(p, 1000);
  EXPECT_FALSE(h.Healthy(p, 1500));   // still in cooldown -> skip
  // recovery: a later served response clears it (bad->good edge)
  h.MarkGood(p);

  EXPECT_EQ(h.served(), 2u);
  EXPECT_EQ(h.errors(), 1u);
  EXPECT_EQ(h.marked_bad(), 1u);
  EXPECT_EQ(h.recovered(), 1u);

  std::string t = h.Render();
  EXPECT_NE(t.find("dfkv_client_ops_served_total 2"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_client_io_errors_total 1"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_client_unhealthy_skips_total 1"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_client_peer_marked_bad_total 1"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_client_peer_recovered_total 1"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_client_peer_errors_total{peer=\"10.0.0.1:12000\"} 1"), std::string::npos) << t;
}

TEST(ClientMetrics, PerPeerMapIsCardinalityBounded) {
  // A client that churns through many distinct peer addresses must not grow the
  // per-peer error map (and its scrape cardinality) without bound.
  PeerHealth h(1000);
  for (int i = 0; i < 6000; ++i) h.MarkBad("10.0.0." + std::to_string(i) + ":1", 1000);
  std::string t = h.Render();
  size_t n = 0, pos = 0;
  const std::string needle = "dfkv_client_peer_errors_total{peer=";
  while ((pos = t.find(needle, pos)) != std::string::npos) { ++n; pos += needle.size(); }
  EXPECT_LE(n, 4096u) << "per-peer series cardinality not bounded: " << n;
  EXPECT_EQ(h.errors(), 6000u);  // aggregate still counts every error
}
