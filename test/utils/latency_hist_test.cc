// TDD — lock-free sampled latency histogram + 1-in-N sampler.
#include "utils/latency_hist.h"

#include <gtest/gtest.h>

#include <string>

using namespace dfkv;  // NOLINT

TEST(LatencyHist, CountSumAndCumulativeRender) {
  LatencyHist h;
  h.Observe(30e-6);    // bucket 0 (<=50us)
  h.Observe(120e-6);   // bucket 2 (<=250us)
  h.Observe(3e-3);     // bucket 6 (<=5ms)
  h.Observe(500e-3);   // +Inf (>100ms)
  EXPECT_EQ(h.Count(), 4u);
  EXPECT_NEAR(h.Sum(), 30e-6 + 120e-6 + 3e-3 + 500e-3, 1e-3);

  std::string t = h.Render("dfkv_op_latency_seconds", "op=\"get\"");
  EXPECT_NE(t.find("# TYPE dfkv_op_latency_seconds histogram"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_op_latency_seconds_bucket{le=\"5e-05\",op=\"get\"} 1"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_op_latency_seconds_bucket{le=\"+Inf\",op=\"get\"} 4"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_op_latency_seconds_count{op=\"get\"} 4"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_op_latency_seconds_sum{op=\"get\"}"), std::string::npos) << t;
}

TEST(LatencyHist, BucketsAreMonotonicCumulative) {
  LatencyHist h;
  for (int i = 0; i < 10; ++i) h.Observe(1e-6);  // all in bucket 0
  std::string t = h.Render("x", "");
  // le=5e-05 cumulative already 10; le=+Inf also 10
  EXPECT_NE(t.find("x_bucket{le=\"5e-05\"} 10"), std::string::npos) << t;
  EXPECT_NE(t.find("x_bucket{le=\"+Inf\"} 10"), std::string::npos) << t;
  EXPECT_NE(t.find("x_count 10"), std::string::npos) << t;
}

TEST(Sampler, OneInNFiresEveryNthExactly) {
  Sampler s(64);
  int fired = 0;
  bool first = false;
  for (int i = 0; i < 640; ++i) {
    bool hit = s.ShouldSample();
    if (i == 0) first = hit;
    if (hit) fired++;
  }
  EXPECT_TRUE(first);       // first call samples
  EXPECT_EQ(fired, 10);     // exactly 640/64
}
