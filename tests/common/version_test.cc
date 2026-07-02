// TDD — version string + --version arg detection.
#include "common/version.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>

using namespace dfkv;  // NOLINT

TEST(Version, NonEmptyAndSemverish) {
  std::string v = Version();
  ASSERT_FALSE(v.empty());
  // CI/release builds stamp DFKV_VERSION (e.g. "1.4.1"); has a digit and a dot.
  EXPECT_NE(v.find('.'), std::string::npos) << v;
  EXPECT_NE(v.find_first_of("0123456789"), std::string::npos) << v;
}

TEST(Version, WantsVersionDetectsFlag) {
  char a0[] = "dfkv_server";
  char vlong[] = "--version";
  char vshort[] = "-V";
  char other[] = "--listen";
  char* with_long[] = {a0, vlong};
  char* with_short[] = {a0, vshort};
  char* without[] = {a0, other};
  EXPECT_TRUE(WantsVersion(2, with_long));
  EXPECT_TRUE(WantsVersion(2, with_short));
  EXPECT_FALSE(WantsVersion(2, without));
  EXPECT_FALSE(WantsVersion(1, with_long));  // argv[0] only
}
