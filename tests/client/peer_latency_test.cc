#include "client/peer_latency.h"

#include <string>

#include "gtest/gtest.h"

using dfkv::PeerLatency;

namespace {

bool Has(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

TEST(PeerLatencyTest, RecordsPerPeerHistogramAndMax) {
  PeerLatency pl;
  pl.Observe("10.0.0.1:9000", 0.0003);   // 300us -> le=0.0005 bucket
  pl.Observe("10.0.0.1:9000", 0.0012);   // 1.2ms
  pl.Observe("10.0.0.2:9000", 0.002);

  EXPECT_EQ(pl.peers(), 2u);
  std::string s = pl.Render();

  // histogram present, per-peer labeled
  EXPECT_TRUE(Has(s, "# TYPE dfkv_client_peer_latency_seconds histogram"));
  EXPECT_TRUE(Has(s, "dfkv_client_peer_latency_seconds_count{peer=\"10.0.0.1:9000\"} 2"));
  EXPECT_TRUE(Has(s, "dfkv_client_peer_latency_seconds_count{peer=\"10.0.0.2:9000\"} 1"));
  EXPECT_TRUE(Has(s, "dfkv_client_peer_latency_seconds_bucket{le=\"0.0005\",peer=\"10.0.0.1:9000\"} 1"));

  // max gauge present, per-peer
  EXPECT_TRUE(Has(s, "# TYPE dfkv_client_peer_latency_max_seconds gauge"));
  EXPECT_TRUE(Has(s, "dfkv_client_peer_latency_max_seconds{peer=\"10.0.0.1:9000\"} 0.001200"));
  EXPECT_TRUE(Has(s, "dfkv_client_peer_latency_max_seconds{peer=\"10.0.0.2:9000\"} 0.002000"));
}

TEST(PeerLatencyTest, EmptyRendersHeadersOnly) {
  PeerLatency pl;
  std::string s = pl.Render();
  EXPECT_TRUE(Has(s, "# TYPE dfkv_client_peer_latency_seconds histogram"));
  EXPECT_FALSE(Has(s, "peer="));  // no series
}

TEST(PeerLatencyTest, MaxTracksLargest) {
  PeerLatency pl;
  pl.Observe("p", 0.005);
  pl.Observe("p", 0.001);
  pl.Observe("p", 0.003);
  EXPECT_TRUE(Has(pl.Render(), "dfkv_client_peer_latency_max_seconds{peer=\"p\"} 0.005000"));
}

}  // namespace
