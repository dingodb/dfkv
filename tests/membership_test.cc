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
