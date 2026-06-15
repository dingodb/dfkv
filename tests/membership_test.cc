#include "membership.h"
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
  std::string buf = EncodeMembers({{"n1", "10.0.0.1", 28000, 1}}, 1);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  for (size_t cut = 0; cut < buf.size(); ++cut)  // every short prefix must fail
    EXPECT_FALSE(DecodeMembers(buf.data(), cut, &got, &epoch)) << "cut=" << cut;
  EXPECT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));  // full = ok
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
