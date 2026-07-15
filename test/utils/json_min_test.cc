#include "utils/json_min.h"
#include <gtest/gtest.h>
using namespace dfkv;  // NOLINT

TEST(JsonMin, ParsesEtcdRangeShape) {
  std::string s = R"({"header":{"revision":"7"},"kvs":[
      {"key":"azE=","value":"djE=","mod_revision":"5"},
      {"key":"azI=","value":"djI=","mod_revision":"6"}],"count":"2"})";
  JsonValue root;
  ASSERT_TRUE(JsonParser::Parse(s, &root));
  ASSERT_EQ(root.type, JsonValue::kObject);
  const JsonValue* hdr = root.find("header");
  ASSERT_NE(hdr, nullptr);
  EXPECT_EQ(hdr->get_str("revision"), "7");
  const JsonValue* kvs = root.find("kvs");
  ASSERT_NE(kvs, nullptr);
  ASSERT_EQ(kvs->type, JsonValue::kArray);
  ASSERT_EQ(kvs->arr.size(), 2u);
  EXPECT_EQ(kvs->arr[0].get_str("key"), "azE=");
  EXPECT_EQ(kvs->arr[1].get_str("value"), "djI=");
}

TEST(JsonMin, MissingKeyAndEmptyArray) {
  JsonValue root;
  ASSERT_TRUE(JsonParser::Parse(R"({"kvs":[],"more":false})", &root));
  EXPECT_EQ(root.find("kvs")->arr.size(), 0u);
  EXPECT_EQ(root.get_str("nope", "DFLT"), "DFLT");
  EXPECT_EQ(root.find("more")->scalar, "false");
}

TEST(JsonMin, HandlesEscapesAndWhitespace) {
  JsonValue root;
  ASSERT_TRUE(JsonParser::Parse("  { \"a\" : \"x\\ny\\\"z\" } ", &root));
  EXPECT_EQ(root.get_str("a"), "x\ny\"z");
}

TEST(JsonMin, RejectsMalformed) {
  JsonValue root;
  EXPECT_FALSE(JsonParser::Parse("{\"a\":}", &root));
  EXPECT_FALSE(JsonParser::Parse("{\"a\":1", &root));
  EXPECT_FALSE(JsonParser::Parse("{\"a\":1}garbage", &root));
  EXPECT_FALSE(JsonParser::Parse("", &root));
}

TEST(JsonMin, DeepNestingFailsCleanlyInsteadOfStackOverflow) {
  // 100k nested arrays previously recursed value->array->value all the way
  // down and crashed the process; the depth cap must reject it as a parse
  // error. A document within the cap still parses.
  JsonValue root;
  const int kDeep = 100000;
  std::string deep(kDeep, '[');
  deep.append(kDeep, ']');
  EXPECT_FALSE(JsonParser::Parse(deep, &root));
  std::string shallow(8, '[');
  shallow += "1";
  shallow.append(8, ']');
  EXPECT_TRUE(JsonParser::Parse(shallow, &root));
}
