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
