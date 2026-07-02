#include "mds/mds_server.h"
#include "mds/mds_proto.h"
#include "common/membership.h"
#include "transport/wire.h"
#include "utils/net_util.h"
#include <gtest/gtest.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <vector>
using namespace dfkv;  // NOLINT

namespace {
const char* EtcdEp() { return std::getenv("DFKV_TEST_ETCD"); }

bool DoReq(int port, WireOp op, const std::string& payload, Status* st, std::string* data) {
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 2000);
  if (fd < 0) return false;
  char pre[kReqPrefix];
  EncodeReq(pre, op, BlockKey{}, 0, 0, payload.size());
  if (!net::WriteAll(fd, pre, kReqPrefix) ||
      (!payload.empty() && !net::WriteAll(fd, payload.data(), payload.size()))) { ::close(fd); return false; }
  char rp[kRespPrefix];
  if (!net::ReadAll(fd, rp, kRespPrefix)) { ::close(fd); return false; }
  Status s; uint64_t dlen = 0;
  if (!DecodeResp(rp, &s, &dlen)) { ::close(fd); return false; }
  data->resize(dlen);
  if (dlen && !net::ReadAll(fd, &(*data)[0], dlen)) { ::close(fd); return false; }
  *st = s; ::close(fd); return true;
}
}  // namespace

TEST(MdsServer, RegisterThenListRoundTripsThroughEtcd) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  std::string group = "itest-grp-" + std::to_string(port);

  MemberInfo a{"na", "10.1.1.1", 28000, 1};
  MemberInfo b{"nb", "10.1.1.2", 28000, 3};
  Status st; std::string data;
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(group, a), &st, &data));
  EXPECT_EQ(st, Status::kOk);
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(group, b), &st, &data));
  EXPECT_EQ(st, Status::kOk);
  ASSERT_TRUE(DoReq(port, WireOp::kHeartbeat, EncodeMemberReq(group, a), &st, &data));
  EXPECT_EQ(st, Status::kOk);

  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  EXPECT_GT(epoch, 0u);
  ASSERT_EQ(got.size(), 2u);
  bool sawA = false, sawB = false;
  for (auto& m : got) { if (m == a) sawA = true; if (m == b) sawB = true; }
  EXPECT_TRUE(sawA); EXPECT_TRUE(sawB);
  mds.Stop();
}

TEST(MdsServer, ListEmptyGroupOkEmpty) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  Status st; std::string data;
  ASSERT_TRUE(DoReq(mds.port(), WireOp::kListMembers,
                    "empty-grp-" + std::to_string(mds.port()), &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  EXPECT_TRUE(got.empty());
  mds.Stop();
}

TEST(MdsServer, HeartbeatRewritesMemberValueUnderOwnLease) {
  // Regression for the multi-MDS lease-drift bug: the fast path used to only
  // LeaseKeepAlive and never re-Put the key, so (a) a MemberInfo change within
  // the TTL was silently dropped, and (b) with several rotating MDS instances
  // the key stayed attached to another MDS's decaying lease while every
  // heartbeat still returned kOk. Both reduce to the same observable pinned
  // here: a heartbeat must re-write the key's current value.
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  std::string group = "itest-reput-" + std::to_string(port);

  MemberInfo m{"n1", "10.1.1.1", 28000, 1};
  Status st; std::string data;
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(group, m), &st, &data));
  ASSERT_EQ(st, Status::kOk);

  m.ip = "10.9.9.9";  // node reconfigured within the TTL
  m.weight = 7;
  ASSERT_TRUE(DoReq(port, WireOp::kHeartbeat, EncodeMemberReq(group, m), &st, &data));
  ASSERT_EQ(st, Status::kOk);

  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].ip, "10.9.9.9");
  EXPECT_EQ(got[0].weight, 7u);
  mds.Stop();
}

TEST(MdsServer, RejectsPathTraversalGroupAndId) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  Status st; std::string data;

  // A group containing "/members/" would land the key inside group "victim".
  std::string victim = "victim-" + std::to_string(port);
  MemberInfo ghost{"ghost", "6.6.6.6", 28000, 1};
  std::string evil_group = victim + "/members/x/members";
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(evil_group, ghost),
                    &st, &data));
  EXPECT_EQ(st, Status::kInvalid) << "path-traversal group must be rejected";

  // An id with a slash is equally dangerous.
  MemberInfo m{"a/members/x", "1.1.1.1", 28000, 1};
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(victim, m), &st, &data));
  EXPECT_EQ(st, Status::kInvalid) << "path-traversal id must be rejected";

  // The victim group must contain no injected phantom member.
  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, victim, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  EXPECT_TRUE(got.empty()) << "no phantom member should have been injected";

  // A malformed group on ListMembers is rejected outright.
  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, "a/b", &st, &data));
  EXPECT_EQ(st, Status::kInvalid);
  mds.Stop();
}

TEST(MdsServer, ProbeEtcdReflectsReachability) {
  // Dead endpoint: probe must fail fast (used at startup to exit non-zero on a
  // misconfigured --etcd instead of running "up" with silent write failures).
  { MdsServer dead("127.0.0.1:9", /*timeout_ms=*/500);
    EXPECT_FALSE(dead.ProbeEtcd()); }

  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD for the reachable case";
  MdsServer live(ep);
  EXPECT_TRUE(live.ProbeEtcd());
}
