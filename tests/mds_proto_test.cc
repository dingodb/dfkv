#include "mds_proto.h"
#include <gtest/gtest.h>
using namespace dfkv;  // NOLINT

TEST(MdsProto, MemberReqRoundTrip) {
  MemberInfo m{"n1", "10.0.0.7", 28000, 5};
  std::string buf = EncodeMemberReq("group-a", m);
  std::string group; MemberInfo got;
  ASSERT_TRUE(DecodeMemberReq(buf.data(), buf.size(), &group, &got));
  EXPECT_EQ(group, "group-a");
  EXPECT_EQ(got, m);
}

TEST(MdsProto, RejectsTruncated) {
  MemberInfo m{"n1", "10.0.0.7", 28000, 5, 28100};  // tcp_port=28100
  std::string buf = EncodeMemberReq("g", m);
  std::string group; MemberInfo got;
  // Mandatory region (group + member) must be complete; the optional tcp_port tail
  // may be short/absent (older peers). Discover the boundary, assert both sides.
  size_t lo = 0;
  while (lo <= buf.size() && !DecodeMemberReq(buf.data(), lo, &group, &got)) ++lo;
  ASSERT_LE(lo, buf.size());
  for (size_t cut = 0; cut < lo; ++cut)
    EXPECT_FALSE(DecodeMemberReq(buf.data(), cut, &group, &got)) << "cut=" << cut;
  for (size_t cut = lo; cut <= buf.size(); ++cut)
    EXPECT_TRUE(DecodeMemberReq(buf.data(), cut, &group, &got)) << "cut=" << cut;
  ASSERT_TRUE(DecodeMemberReq(buf.data(), buf.size(), &group, &got));
  EXPECT_EQ(got.tcp_port, 28100u);
}
