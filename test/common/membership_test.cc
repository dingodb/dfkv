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

TEST(Membership, ValidGroupOrIdAlphabet) {
  // Accepted: the tokens real deployments use.
  EXPECT_TRUE(IsValidGroupOrId("glm"));
  EXPECT_TRUE(IsValidGroupOrId("glm-a"));
  EXPECT_TRUE(IsValidGroupOrId("gpu2-0045"));
  EXPECT_TRUE(IsValidGroupOrId("itest-grp-12345"));
  EXPECT_TRUE(IsValidGroupOrId("a.b_c-1"));
  EXPECT_TRUE(IsValidGroupOrId(std::string(128, 'a')));  // max length

  // Rejected: empty, overlong, and anything that could escape the key subtree.
  EXPECT_FALSE(IsValidGroupOrId(""));
  EXPECT_FALSE(IsValidGroupOrId(std::string(129, 'a')));
  EXPECT_FALSE(IsValidGroupOrId("a/members/ghost"));  // path traversal
  EXPECT_FALSE(IsValidGroupOrId("a/b"));
  EXPECT_FALSE(IsValidGroupOrId("a b"));   // space
  EXPECT_FALSE(IsValidGroupOrId("a\tb"));
  EXPECT_FALSE(IsValidGroupOrId("a:b"));
  EXPECT_FALSE(IsValidGroupOrId(std::string("a\0b", 3)));  // embedded NUL
}

// --- info extension (NFO1): node self-description on register/heartbeat -------

TEST(MembershipCodec, InfoRoundTripsAlongsideTcpPort) {
  std::vector<MemberInfo> ms = {
      {"n1", "10.0.0.1", 28001, 1, 28100, "ver=1.8.0,engine=slab,cap=5497558138880"},
      {"n2", "10.0.0.2", 28001, 3, 28100, ""},  // one member may have no info
  };
  std::string buf = EncodeMembers(ms, 7);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));
  ASSERT_EQ(got.size(), 2u);
  EXPECT_EQ(got[0].info, "ver=1.8.0,engine=slab,cap=5497558138880");
  EXPECT_EQ(got[0].tcp_port, 28100u);  // TCP1 ext still decoded before NFO1
  EXPECT_EQ(got[1].info, "");
}

TEST(MembershipCodec, OldEncodingWithoutExtsDecodesWithEmptyInfo) {
  // Hand-build a pre-extension encoding (fixed part only): older peers send this.
  MemberInfo m{"n1", "10.0.0.9", 28001, 2};
  std::string full = EncodeMembers({m}, 3);
  size_t fixed = 12 + (4 + m.id.size()) + (4 + m.ip.size()) + 8;  // header + one member
  std::string oldbuf = full.substr(0, fixed);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(oldbuf.data(), oldbuf.size(), &got, &epoch));
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].tcp_port, 0u);
  EXPECT_EQ(got[0].info, "");
}

TEST(MembershipCodec, UnknownFutureExtTagIsIgnored) {
  MemberInfo m{"n1", "10.0.0.9", 28001, 2, 28100, "ver=1.8.0"};
  std::string buf = EncodeMembers({m}, 3);
  buf += std::string("\xAA\xBB\xCC\xDD garbage-future-extension", 30);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));
  EXPECT_EQ(got[0].info, "ver=1.8.0");  // known exts read, future tail ignored
}

TEST(MembershipCodec, TruncatedInfoExtIsBestEffortNotFatal) {
  MemberInfo m{"n1", "10.0.0.9", 28001, 2, 28100, "ver=1.8.0,engine=slab"};
  std::string buf = EncodeMembers({m}, 3);
  // Cut inside the NFO1 payload: members + tcp_port must survive, decode still ok.
  std::string cut = buf.substr(0, buf.size() - 5);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(cut.data(), cut.size(), &got, &epoch));
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].id, "n1");
  EXPECT_EQ(got[0].tcp_port, 28100u);
}

TEST(MembersEpoch, InfoIsExcludedFromEpochAndEquality) {
  // info (like tcp_port) is orthogonal metadata: a change must NOT bump the
  // epoch (no needless client ring rebuilds) and must not affect operator==.
  std::vector<MemberInfo> a = {{"n1", "10.0.0.1", 28001, 1, 28100, "ver=1.7.2"}};
  std::vector<MemberInfo> b = {{"n1", "10.0.0.1", 28001, 1, 28100, "ver=1.8.0,engine=slab"}};
  EXPECT_EQ(MembersEpoch(a), MembersEpoch(b));
  EXPECT_EQ(a[0], b[0]);
}
