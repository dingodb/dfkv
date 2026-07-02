// TDD — Prometheus value extractor used by `dfkvctl stat --all`.
#include "utils/prom_parse.h"

#include <gtest/gtest.h>

using namespace dfkv;  // NOLINT

TEST(PromParse, BareAndLabeledAndMissing) {
  std::string text =
      "# HELP dfkv_used_bytes Bytes used on disk\n"
      "# TYPE dfkv_used_bytes gauge\n"
      "dfkv_used_bytes 4096\n"
      "# TYPE dfkv_cache_hit_total counter\n"
      "dfkv_cache_hit_total{node=\"n1\",group=\"g\"} 17\n"
      "dfkv_objects{node=\"n1\"} 3\n";
  EXPECT_EQ(PromMetricValue(text, "dfkv_used_bytes"), 4096u);
  EXPECT_EQ(PromMetricValue(text, "dfkv_cache_hit_total"), 17u);
  EXPECT_EQ(PromMetricValue(text, "dfkv_objects"), 3u);
  EXPECT_EQ(PromMetricValue(text, "dfkv_missing_metric"), 0u);
}

TEST(PromParse, DoesNotMatchPrefixOrHelpLine) {
  std::string text =
      "# HELP dfkv_cache_hit_total help\n"
      "dfkv_cache_hit_total 5\n"
      "dfkv_cache_hit_total_extra 999\n";
  // exact-name match only: must not pick up _extra, must not read the HELP line
  EXPECT_EQ(PromMetricValue(text, "dfkv_cache_hit_total"), 5u);
  EXPECT_EQ(PromMetricValue(text, "dfkv_cache_hit_total_extra"), 999u);
}
