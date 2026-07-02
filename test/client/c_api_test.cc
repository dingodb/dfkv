// C ABI guard tests: the FFI boundary must reject null handles/arrays and must
// not feed a null pointer into std::string() (undefined behavior). These run
// against a discovery-only client (empty member list => empty ring), so they
// need no server: routing fails fast and every slot reports miss/failure.
#include "client/dfkv_c_api.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

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
  EXPECT_EQ(dfkv_set_batch_concurrency(nullptr, 8), -1);
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

// Scatter-gather C ABI: same guard contract as the contiguous batch entrypoints
// (null handle => -1; null arrays with n>0 => -1; n==0 => 0). A discovery-only
// client (empty ring) makes every real route miss, so no server is needed.
TEST(CApiGuard, ScatterGatherNullAndZero) {
  EXPECT_EQ(dfkv_batch_put_sg(nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr), -1);
  EXPECT_EQ(dfkv_batch_get_auto_sg(nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr), -1);

  dfkv_client_t c = OpenEmpty();
  ASSERT_NE(c, nullptr);
  int ok[2] = {9, 9};
  uint64_t len[2] = {9, 9};
  // null nested arrays with n>0 => rejected.
  EXPECT_EQ(dfkv_batch_put_sg(c, nullptr, nullptr, nullptr, nullptr, 2, ok), -1);
  EXPECT_EQ(dfkv_batch_get_auto_sg(c, nullptr, nullptr, nullptr, nullptr, 2, ok, len), -1);
  // zero-length batch is a valid no-op.
  EXPECT_EQ(dfkv_batch_put_sg(c, nullptr, nullptr, nullptr, nullptr, 0, nullptr), 0);
  EXPECT_EQ(dfkv_batch_get_auto_sg(c, nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr), 0);

  // Well-formed call against an empty ring: every key routes nowhere => fail/miss,
  // but the FFI unpacking must not crash on a null keys[i] element.
  const char* keys[2] = {"k0", nullptr};
  const char b0a[8] = {0}, b0b[8] = {0};
  const void* p0[2] = {b0a, b0b};
  const void** ptrs[2] = {p0, nullptr};
  uint64_t s0[2] = {8, 8};
  const uint64_t* sizes[2] = {s0, nullptr};
  int num_bufs[2] = {2, 0};
  EXPECT_EQ(dfkv_batch_put_sg(c, keys, ptrs, sizes, num_bufs, 2, ok), 0);
  EXPECT_EQ(ok[0], 0);  // empty ring => route miss => failure (not a crash)
  EXPECT_EQ(ok[1], 0);  // null key => skipped, reported failed

  char d0a[8], d0b[8];
  void* dp0[2] = {d0a, d0b};
  void** dsts[2] = {dp0, nullptr};
  uint64_t c0[2] = {8, 8};
  const uint64_t* caps[2] = {c0, nullptr};
  int num_dsts[2] = {2, 0};
  EXPECT_EQ(dfkv_batch_get_auto_sg(c, keys, dsts, caps, num_dsts, 2, ok, len), 0);
  EXPECT_EQ(ok[0], 0); EXPECT_EQ(len[0], 0u);
  EXPECT_EQ(ok[1], 0); EXPECT_EQ(len[1], 0u);
  dfkv_close(c);
}

TEST(CApiStats, SnapshotSizingAndContent) {
  dfkv_client_t c = OpenEmpty();
  ASSERT_NE(c, nullptr);
  // a few ops against the empty ring exercise the health/metrics chokepoint
  EXPECT_EQ(dfkv_get(c, "k", nullptr, 0), 0);
  const char* mode = dfkv_transport_mode(c);
  ASSERT_NE(mode, nullptr);
  EXPECT_FALSE(std::string(mode).empty());
  EXPECT_STREQ(dfkv_transport_mode(nullptr), "");
  EXPECT_EQ(dfkv_set_batch_concurrency(c, 4), 0);
  // size query: cap=0 returns the full length without writing
  uint64_t need = dfkv_stats_snapshot(c, nullptr, 0);
  EXPECT_GT(need, 0u);
  // full read: NUL-terminated, contains the client metric family
  std::vector<char> buf(need + 1, 'X');
  uint64_t n = dfkv_stats_snapshot(c, buf.data(), buf.size());
  EXPECT_EQ(n, need);
  EXPECT_EQ(buf[need], '\0');
  EXPECT_NE(std::string(buf.data()).find("dfkv_client_ops_served_total"), std::string::npos);
  // null-client snapshot is 0, no crash
  EXPECT_EQ(dfkv_stats_snapshot(nullptr, buf.data(), buf.size()), 0u);
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
