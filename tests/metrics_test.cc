// TDD R11 — server metrics counters + Prometheus text + remote Stats op.
#include "kv_node_server.h"
#include "key_map.h"
#include "tcp_transport.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
std::unique_ptr<KvNodeServer> Start(fs::path dir, std::string* addr) {
  fs::remove_all(dir); fs::create_directories(dir);
  auto s = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
  EXPECT_EQ(s->Start(0), Status::kOk);
  *addr = "127.0.0.1:" + std::to_string(s->port());
  return s;
}
}  // namespace

TEST(Metrics, CountersTrackOps) {
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_metrics_a";
  auto s = Start(dir, &addr);
  TcpTransport t;
  std::string v(100, 'm');
  ASSERT_EQ(t.Cache(addr, ToBlockKey("a"), v.data(), v.size()), Status::kOk);
  ASSERT_EQ(t.Cache(addr, ToBlockKey("b"), v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(t.Range(addr, ToBlockKey("a"), 0, v.size(), &out), Status::kOk);   // hit
  ASSERT_EQ(t.Range(addr, ToBlockKey("zzz"), 0, 8, &out), Status::kNotFound);  // miss
  bool e = false;
  ASSERT_EQ(t.Exist(addr, ToBlockKey("a"), &e), Status::kOk); EXPECT_TRUE(e);   // exist hit
  ASSERT_EQ(t.Exist(addr, ToBlockKey("nope"), &e), Status::kOk); EXPECT_FALSE(e); // exist miss

  EXPECT_EQ(s->m_cache_put(), 2u);
  EXPECT_EQ(s->m_cache_hit(), 1u);
  EXPECT_EQ(s->m_cache_miss(), 1u);
  EXPECT_EQ(s->m_exist_hit(), 1u);
  EXPECT_EQ(s->m_exist_miss(), 1u);

  std::string text = s->MetricsText();
  EXPECT_NE(text.find("dfkv_cache_hit_total 1"), std::string::npos) << text;
  EXPECT_NE(text.find("dfkv_cache_put_total 2"), std::string::npos) << text;
  EXPECT_NE(text.find("dfkv_objects 2"), std::string::npos) << text;
  s->Stop();
}

TEST(Metrics, PrometheusFormatAndIdentity) {
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_metrics_c";
  auto s = Start(dir, &addr);
  s->set_identity("n1", "g1");
  std::string text = s->MetricsText();
  // HELP/TYPE metadata present
  EXPECT_NE(text.find("# TYPE dfkv_cache_hit_total counter"), std::string::npos) << text;
  EXPECT_NE(text.find("# TYPE dfkv_used_bytes gauge"), std::string::npos) << text;
  // identity labels applied to series
  EXPECT_NE(text.find("dfkv_cache_hit_total{node=\"n1\",group=\"g1\"} 0"), std::string::npos) << text;
  // build_info + uptime present
  EXPECT_NE(text.find("dfkv_build_info{"), std::string::npos) << text;
  EXPECT_NE(text.find("version=\""), std::string::npos) << text;
  EXPECT_NE(text.find("dfkv_uptime_seconds"), std::string::npos) << text;
  s->Stop();
}

TEST(Metrics, NoIdentityKeepsUnlabeledSeries) {
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_metrics_d";
  auto s = Start(dir, &addr);  // no set_identity
  std::string text = s->MetricsText();
  // back-compat: bare metric line, no label braces
  EXPECT_NE(text.find("dfkv_cache_hit_total 0"), std::string::npos) << text;
  s->Stop();
}

TEST(Metrics, RemoteStatsOp) {
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_metrics_b";
  auto s = Start(dir, &addr);
  TcpTransport t;
  std::string v(50, 'x');
  ASSERT_EQ(t.Cache(addr, ToBlockKey("k"), v.data(), v.size()), Status::kOk);
  std::string text;
  ASSERT_EQ(t.Stats(addr, &text), Status::kOk);
  EXPECT_NE(text.find("dfkv_cache_put_total 1"), std::string::npos) << text;
  s->Stop();
}
