#include "client/kv_client.h"
#include "cache/kv_node_server.h"
#include "mds/mds_server.h"
#include "mds/mds_registrar.h"
#include "common/value_header.h"
#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
const char* EtcdEp() { return std::getenv("DFKV_TEST_ETCD"); }
ValueHeader SelfHdr() {
  return ValueHeader::Make(0x51ULL, 64, 0x46384534u, ValueHeader::kFlagIsMla,
                           8, 0, 78, 1, 576);
}
}  // namespace

TEST(KvClientMds, DiscoversNodesAndRoundTrips) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  fs::path d1 = fs::temp_directory_path() / "dfkv_m3_n1";
  fs::path d2 = fs::temp_directory_path() / "dfkv_m3_n2";
  fs::remove_all(d1); fs::remove_all(d2); fs::create_directories(d1); fs::create_directories(d2);
  KvNodeServer n1(d1.string(), 1ull << 30), n2(d2.string(), 1ull << 30);
  ASSERT_EQ(n1.Start(0), Status::kOk);
  ASSERT_EQ(n2.Start(0), Status::kOk);

  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string mds_ep = "127.0.0.1:" + std::to_string(mds.port());
  std::string group = "m3-e2e-" + std::to_string(mds.port());
  MemberInfo m1{"e1", "127.0.0.1", (uint32_t)n1.port(), 1};
  MemberInfo m2{"e2", "127.0.0.1", (uint32_t)n2.port(), 1};
  MdsRegistrar r1({mds_ep}, group, m1); ASSERT_TRUE(r1.RegisterOnce());
  MdsRegistrar r2({mds_ep}, group, m2); ASSERT_TRUE(r2.RegisterOnce());

  KVClient c({}, SelfHdr());
  c.StartMdsDiscovery({mds_ep}, group, /*poll_ms=*/100);
  bool ready = false;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  std::string probe(64, 'p');
  while (!ready && std::chrono::steady_clock::now() < deadline) {
    if (c.Put("warm/probe_k", probe.data(), probe.size())) ready = true;
    else std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(ready);

  const int N = 200;
  for (int i = 0; i < N; ++i) {
    std::string v = "val_" + std::to_string(i);
    ASSERT_TRUE(c.Put("glm/p_" + std::to_string(i) + "_k", v.data(), v.size())) << i;
  }
  EXPECT_GT(n1.Count(), 0u);
  EXPECT_GT(n2.Count(), 0u);
  for (int i = 0; i < N; ++i) {
    std::string want = "val_" + std::to_string(i), out(want.size(), '\0');
    ASSERT_TRUE(c.Get("glm/p_" + std::to_string(i) + "_k", &out[0], out.size())) << i;
    EXPECT_EQ(out, want);
  }
  mds.Stop(); n1.Stop(); n2.Stop();
  fs::remove_all(d1); fs::remove_all(d2);
}
