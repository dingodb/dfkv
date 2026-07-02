#include "common/membership.h"
#include <gtest/gtest.h>
using namespace dfkv;  // NOLINT

TEST(MembershipCodec, RoundTripMultiMember) {
  std::vector<MemberInfo> ms = {
      {"n1", "10.0.0.1", 28000, 1},
      {"n2", "10.0.0.2", 28000, 3},
      {"node-with-long-id", "192.168.123.234", 65535, 100},
  };
  std::string buf = EncodeMembers(ms, /*epoch=*/0xDEADBEEF12345678ULL);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));
  EXPECT_EQ(epoch, 0xDEADBEEF12345678ULL);
  EXPECT_EQ(got, ms);
}

TEST(MembershipCodec, RoundTripEmpty) {
  std::string buf = EncodeMembers({}, 42);
  std::vector<MemberInfo> got{{"stale", "x", 1, 1}};
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));
  EXPECT_EQ(epoch, 42u);
  EXPECT_TRUE(got.empty());
}

TEST(MembershipCodec, RejectsTruncated) {
  std::vector<MemberInfo> ms = {{"n1", "10.0.0.1", 28000, 1, 28100}};  // tcp_port=28100
  std::string buf = EncodeMembers(ms, 1);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  // The mandatory member region must be complete to decode; the OPTIONAL tcp_port
  // tail may be short/absent (older peers send none) -> tolerated, not an error.
  // Discover the region boundary, then require failure below it and success at/above.
  size_t lo = 0;
  while (lo <= buf.size() && !DecodeMembers(buf.data(), lo, &got, &epoch)) ++lo;
  ASSERT_LE(lo, buf.size());
  for (size_t cut = 0; cut < lo; ++cut)
    EXPECT_FALSE(DecodeMembers(buf.data(), cut, &got, &epoch)) << "cut=" << cut;
  for (size_t cut = lo; cut <= buf.size(); ++cut)
    EXPECT_TRUE(DecodeMembers(buf.data(), cut, &got, &epoch)) << "cut=" << cut;
  // Tail absent at the boundary -> tcp_port defaults to 0; fully present -> read back.
  ASSERT_TRUE(DecodeMembers(buf.data(), lo, &got, &epoch));
  EXPECT_EQ(got[0].tcp_port, 0u);
  ASSERT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));
  EXPECT_EQ(got[0].tcp_port, 28100u);
}

TEST(MembersEpoch, OrderIndependentAndContentSensitive) {
  std::vector<MemberInfo> a = {
      {"n1", "10.0.0.1", 28000, 1},
      {"n2", "10.0.0.2", 28000, 3},
      {"n3", "10.0.0.3", 28000, 1},
  };
  std::vector<MemberInfo> reordered = {a[2], a[0], a[1]};
  EXPECT_EQ(MembersEpoch(a), MembersEpoch(reordered)) << "order must not matter";

  // Removing a member changes the epoch.
  std::vector<MemberInfo> fewer = {a[0], a[1]};
  EXPECT_NE(MembersEpoch(a), MembersEpoch(fewer));

  // A content-only change (same ids) changes the epoch.
  std::vector<MemberInfo> b = a;
  b[1].weight = 7;  // was 3
  EXPECT_NE(MembersEpoch(a), MembersEpoch(b));
  std::vector<MemberInfo> c = a;
  c[0].ip = "10.0.0.9";
  EXPECT_NE(MembersEpoch(a), MembersEpoch(c));
  std::vector<MemberInfo> d = a;
  d[2].port = 28001;
  EXPECT_NE(MembersEpoch(a), MembersEpoch(d));

  // Stable across calls; empty set is deterministic.
  EXPECT_EQ(MembersEpoch(a), MembersEpoch(a));
  EXPECT_EQ(MembersEpoch({}), MembersEpoch({}));
}

TEST(MembersEpoch, NoFieldBoundaryCollision) {
  // Length-prefixing must prevent "ab"+"c" hashing the same as "a"+"bc".
  std::vector<MemberInfo> x = {{"ab", "c", 0, 0}};
  std::vector<MemberInfo> y = {{"a", "bc", 0, 0}};
  EXPECT_NE(MembersEpoch(x), MembersEpoch(y));
}
