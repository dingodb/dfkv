// TDD — Prometheus label-value escaping.
#include "utils/prom_escape.h"

#include <gtest/gtest.h>

using namespace dfkv;  // NOLINT

TEST(PromEscape, EscapesBackslashQuoteNewline) {
  EXPECT_EQ(PromLabelEscape("plain"), "plain");
  EXPECT_EQ(PromLabelEscape("a\"b"), "a\\\"b");          // double-quote escaped
  EXPECT_EQ(PromLabelEscape("a\\b"), "a\\\\b");          // backslash escaped
  EXPECT_EQ(PromLabelEscape("a\nb"), "a\\nb");           // newline escaped
  EXPECT_EQ(PromLabelEscape("/mnt/d\"k/p"), "/mnt/d\\\"k/p");  // disk path with a quote
  // backslash escaped before quote: \" stays one backslash + escaped quote
  EXPECT_EQ(PromLabelEscape("\\\""), "\\\\\\\"");
}
