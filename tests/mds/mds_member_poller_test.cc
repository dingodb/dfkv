#include "mds/mds_member_poller.h"
#include "mds/mds_server.h"
#include "mds/mds_registrar.h"
#include "common/membership.h"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
using namespace dfkv;  // NOLINT

namespace {
const char* EtcdEp() { return std::getenv("DFKV_TEST_ETCD"); }
}  // namespace

TEST(MdsMemberPoller, PollFiresOnChangeAndDedupsByEpoch) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string mds_ep = "127.0.0.1:" + std::to_string(mds.port());
  std::string group = "m3-poll-" + std::to_string(mds.port());

  MemberInfo a{"pa", "10.2.2.1", 28000, 1}, b{"pb", "10.2.2.2", 28000, 2};
  MdsRegistrar ra({mds_ep}, group, a); ASSERT_TRUE(ra.RegisterOnce());
  MdsRegistrar rb({mds_ep}, group, b); ASSERT_TRUE(rb.RegisterOnce());

  std::mutex mu; std::vector<MemberInfo> last; int fires = 0;
  MdsMemberPoller poller({mds_ep}, group,
      [&](const std::vector<MemberInfo>& ms) {
        std::lock_guard<std::mutex> lk(mu); last = ms; ++fires;
      });
  ASSERT_TRUE(poller.PollOnce());
  { std::lock_guard<std::mutex> lk(mu); EXPECT_EQ(fires, 1); EXPECT_EQ(last.size(), 2u); }
  ASSERT_TRUE(poller.PollOnce());
  { std::lock_guard<std::mutex> lk(mu); EXPECT_EQ(fires, 1); }

  MemberInfo c{"pc", "10.2.2.3", 28000, 1};
  MdsRegistrar rc({mds_ep}, group, c); ASSERT_TRUE(rc.RegisterOnce());
  ASSERT_TRUE(poller.PollOnce());
  { std::lock_guard<std::mutex> lk(mu); EXPECT_EQ(fires, 2); EXPECT_EQ(last.size(), 3u); }
  mds.Stop();
}

TEST(MdsMemberPoller, BackgroundLoopPicksUpMembers) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string mds_ep = "127.0.0.1:" + std::to_string(mds.port());
  std::string group = "m3-bg-" + std::to_string(mds.port());
  MemberInfo a{"qa", "10.2.3.1", 28000, 1};
  MdsRegistrar ra({mds_ep}, group, a); ASSERT_TRUE(ra.RegisterOnce());

  std::atomic<int> seen{0};
  MdsMemberPoller poller({mds_ep}, group,
      [&](const std::vector<MemberInfo>& ms) { seen.store((int)ms.size()); },
      /*poll_ms=*/100);
  poller.Start();
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (seen.load() < 1 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(seen.load(), 1);
  poller.Stop();
  mds.Stop();
}
