#include "mds/mds_registrar.h"
#include "mds/mds_server.h"
#include "mds/mds_proto.h"
#include "common/membership.h"
#include "transport/wire.h"
#include "utils/net_util.h"
#include <gtest/gtest.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
using namespace dfkv;  // NOLINT

namespace {
const char* EtcdEp() { return std::getenv("DFKV_TEST_ETCD"); }

bool ListMembers(int port, const std::string& group, std::vector<MemberInfo>* out) {
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 2000);
  if (fd < 0) return false;
  char pre[kReqPrefix];
  EncodeReq(pre, WireOp::kListMembers, BlockKey{}, 0, 0, group.size());
  bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
            net::WriteAll(fd, group.data(), group.size());
  std::string data;
  if (ok) {
    char rp[kRespPrefix]; Status st; uint64_t dlen = 0;
    ok = net::ReadAll(fd, rp, kRespPrefix) && DecodeResp(rp, &st, &dlen) && st == Status::kOk;
    if (ok) { data.resize(dlen); ok = (dlen == 0) || net::ReadAll(fd, &data[0], dlen); }
  }
  ::close(fd);
  if (!ok) return false;
  uint64_t epoch = 0;
  return DecodeMembers(data.data(), data.size(), out, &epoch);
}

bool WaitForMember(int port, const std::string& group, const MemberInfo& m, int timeout_ms) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    std::vector<MemberInfo> ms;
    if (ListMembers(port, group, &ms))
      for (auto& x : ms) if (x == m) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}
}  // namespace

TEST(MdsRegistrar, RegisterOnceMakesNodeVisible) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string group = "m2-grp-" + std::to_string(mds.port());
  MemberInfo self{"node-x", "10.9.9.9", 28000, 2};
  MdsRegistrar reg({"127.0.0.1:" + std::to_string(mds.port())}, group, self);
  ASSERT_TRUE(reg.RegisterOnce());
  ASSERT_TRUE(reg.HeartbeatOnce());
  std::vector<MemberInfo> ms;
  ASSERT_TRUE(ListMembers(mds.port(), group, &ms));
  ASSERT_EQ(ms.size(), 1u);
  EXPECT_EQ(ms[0], self);
  mds.Stop();
}

TEST(MdsRegistrar, FailoverSkipsDeadEndpoint) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string group = "m2-fo-" + std::to_string(mds.port());
  MemberInfo self{"node-fo", "10.9.9.10", 28000, 1};
  MdsRegistrar reg({"127.0.0.1:1", "127.0.0.1:" + std::to_string(mds.port())}, group, self);
  bool ok = false;
  for (int i = 0; i < 4 && !ok; ++i) ok = reg.RegisterOnce();
  ASSERT_TRUE(ok);
  std::vector<MemberInfo> ms;
  ASSERT_TRUE(ListMembers(mds.port(), group, &ms));
  ASSERT_EQ(ms.size(), 1u);
  EXPECT_EQ(ms[0], self);
  mds.Stop();
}

TEST(MdsRegistrar, BackgroundLoopRegistersAndKeepsAlive) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string group = "m2-bg-" + std::to_string(mds.port());
  MemberInfo self{"node-bg", "10.9.9.11", 28000, 3};
  MdsRegistrar reg({"127.0.0.1:" + std::to_string(mds.port())}, group, self, /*hb_ms=*/200);
  reg.Start();
  EXPECT_TRUE(WaitForMember(mds.port(), group, self, /*timeout_ms=*/5000));
  reg.Stop();
  mds.Stop();
}
