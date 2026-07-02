#include "utils/base64.h"
#include <gtest/gtest.h>
#include <string>
using namespace dfkv;  // NOLINT

TEST(Base64, KnownVectors) {
  EXPECT_EQ(Base64Encode(""), "");
  EXPECT_EQ(Base64Encode("f"), "Zg==");
  EXPECT_EQ(Base64Encode("fo"), "Zm8=");
  EXPECT_EQ(Base64Encode("foo"), "Zm9v");
  EXPECT_EQ(Base64Encode("foob"), "Zm9vYg==");
  EXPECT_EQ(Base64Encode("fooba"), "Zm9vYmE=");
  EXPECT_EQ(Base64Encode("foobar"), "Zm9vYmFy");
}

TEST(Base64, RoundTripBinaryAndKeys) {
  for (const std::string& s :
       {std::string("/dfkv/v1/groups/g1/members/n1"),
        std::string("\x00\x01\x02\xff\xfe", 5), std::string(257, '\xa5')}) {
    std::string dec;
    ASSERT_TRUE(Base64Decode(Base64Encode(s), &dec));
    EXPECT_EQ(dec, s);
  }
}

TEST(Base64, RejectsInvalidChar) {
  std::string dec;
  EXPECT_FALSE(Base64Decode("not*base64", &dec));
}
