// C ABI guard tests: the FFI boundary must reject null handles/arrays and must
// not feed a null pointer into std::string() (undefined behavior). These run
// against a discovery-only client (empty member list => empty ring), so they
// need no server: routing fails fast and every slot reports miss/failure.
#include "dfkv_c_api.h"

#include <gtest/gtest.h>

namespace {
dfkv_client_t OpenEmpty() {
  // members="" => discovery-only; ring stays empty until MDS discovery runs.
  return dfkv_open("", /*model_hash=*/1, /*page_size=*/64, /*dtype_tag=*/0,
                   /*flags=*/0, /*tp_size=*/1, /*tp_rank=*/0, /*layer_num=*/0,
                   /*head_num=*/0, /*head_dim=*/0);
}
}  // namespace

TEST(CApiGuard, RejectsNullClient) {
  EXPECT_EQ(dfkv_batch_put(nullptr, nullptr, nullptr, nullptr, 0, nullptr), -1);
  EXPECT_EQ(dfkv_batch_get(nullptr, nullptr, nullptr, nullptr, 0, nullptr), -1);
  EXPECT_EQ(dfkv_batch_exist(nullptr, nullptr, 0, nullptr), -1);
  EXPECT_EQ(dfkv_put(nullptr, "k", "v", 1), -1);
  EXPECT_EQ(dfkv_get(nullptr, "k", nullptr, 0), 0);
  EXPECT_EQ(dfkv_exist(nullptr, "k"), 0);
}

TEST(CApiGuard, RejectsNullArraysWithPositiveN) {
  dfkv_client_t c = OpenEmpty();
  ASSERT_NE(c, nullptr);
  int out[2] = {9, 9};
  EXPECT_EQ(dfkv_batch_put(c, nullptr, nullptr, nullptr, 2, out), -1);
  EXPECT_EQ(dfkv_batch_get(c, nullptr, nullptr, nullptr, 2, out), -1);
  EXPECT_EQ(dfkv_batch_exist(c, nullptr, 2, out), -1);
  EXPECT_EQ(dfkv_put(c, nullptr, "v", 1), -1);
  EXPECT_EQ(dfkv_get(c, nullptr, nullptr, 0), 0);
  EXPECT_EQ(dfkv_exist(c, nullptr), 0);
  dfkv_close(c);
}

TEST(CApiGuard, NullElementIsSkippedNotDereferenced) {
  dfkv_client_t c = OpenEmpty();
  ASSERT_NE(c, nullptr);
  const char* keys[2] = {nullptr, "real-key"};
  const void* ptrs[2] = {nullptr, "vv"};
  uint64_t sizes[2] = {0, 2};
  int ok[2] = {9, 9};
  EXPECT_EQ(dfkv_batch_put(c, keys, ptrs, sizes, 2, ok), 0);  // no crash
  EXPECT_EQ(ok[0], 0);  // null key reported as failure, not UB

  void* outs[2] = {nullptr, nullptr};
  int hit[2] = {9, 9};
  EXPECT_EQ(dfkv_batch_get(c, keys, outs, sizes, 2, hit), 0);
  EXPECT_EQ(hit[0], 0);

  int ex[2] = {9, 9};
  EXPECT_EQ(dfkv_batch_exist(c, keys, 2, ex), 0);
  EXPECT_EQ(ex[0], 0);
  dfkv_close(c);
}

TEST(CApiGuard, ZeroLengthBatchIsOk) {
  dfkv_client_t c = OpenEmpty();
  ASSERT_NE(c, nullptr);
  // n==0 with null arrays must be accepted (nothing to do), not rejected.
  EXPECT_EQ(dfkv_batch_put(c, nullptr, nullptr, nullptr, 0, nullptr), 0);
  EXPECT_EQ(dfkv_batch_get(c, nullptr, nullptr, nullptr, 0, nullptr), 0);
  EXPECT_EQ(dfkv_batch_exist(c, nullptr, 0, nullptr), 0);
  dfkv_close(c);
}
