// TDD R3 — KVStore: the cache-node local store (disk + LRU + cache-only/no-S3).
#include "kv_store.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

using dfkv::BlockKey;
using dfkv::KVStore;
using dfkv::Status;

namespace fs = std::filesystem;

class KVStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("dfkv_store_" + std::to_string(::testing::UnitTest::GetInstance()
                                               ->current_test_info()
                                               ->line()));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }
  KVStore::Options Opts(uint64_t cap = (1ull << 30)) {
    return KVStore::Options{dir_.string(), cap};
  }
  fs::path dir_;
};

TEST_F(KVStoreTest, CacheThenRangeRoundTripAndImmediateVisibility) {
  KVStore s(Opts());
  BlockKey k{111, 0, 1};
  std::string v = "the-kv-bytes-0123456789";
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  // sync visibility: cached immediately after Cache() returns
  EXPECT_TRUE(s.IsCached(k));
  std::string out;
  ASSERT_EQ(s.Range(k, 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
}

TEST_F(KVStoreTest, RangeIntoReadsStraightIntoCallerBuffer) {
  KVStore s(Opts());
  BlockKey k{222, 0, 1};
  std::string v = "zero-copy-server-side-payload-bytes";
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  char buf[64];
  size_t got = 0;
  ASSERT_EQ(s.RangeInto(k, 0, v.size(), buf, sizeof(buf), &got), Status::kOk);
  EXPECT_EQ(got, v.size());
  EXPECT_EQ(std::string(buf, got), v);
  // miss => NotFound, out_len 0
  got = 12345;
  EXPECT_EQ(s.RangeInto(BlockKey{999, 0, 1}, 0, 64, buf, sizeof(buf), &got), Status::kNotFound);
  EXPECT_EQ(got, 0u);
  // dst_cap caps the read
  ASSERT_EQ(s.RangeInto(k, 0, v.size(), buf, 4, &got), Status::kOk);
  EXPECT_EQ(got, 4u);
  EXPECT_EQ(std::string(buf, 4), v.substr(0, 4));
}

TEST_F(KVStoreTest, RangeDirectReturnsSliceInsideAlignedBuffer) {
  KVStore s(Opts());
  BlockKey k{333, 0, 1};
  std::string v(9000, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 13 + 5) & 0xFF);
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);

  void* raw = nullptr;
  ASSERT_EQ(posix_memalign(&raw, 4096, 16 * 1024), 0);
  char* io = static_cast<char*>(raw);
  const char* data = nullptr;
  size_t got = 0;
  ASSERT_EQ(s.RangeDirect(k, 123, 5000, io, 16 * 1024, &data, &got), Status::kOk);
  EXPECT_EQ(got, 5000u);
  EXPECT_GE(data, io);
  EXPECT_LT(data, io + 16 * 1024);
  EXPECT_EQ(std::string(data, got), v.substr(123, 5000));
  std::free(raw);
}

TEST_F(KVStoreTest, RangeMissReturnsNotFoundNoS3) {
  KVStore s(Opts());
  std::string out;
  EXPECT_EQ(s.Range(BlockKey{404, 0, 1}, 0, 16, &out), Status::kNotFound);
  EXPECT_FALSE(s.IsCached(BlockKey{404, 0, 1}));
}

TEST_F(KVStoreTest, IdempotentCacheSkipsRewrite) {
  KVStore s(Opts());
  BlockKey k{7, 0, 1};
  std::string v = "abc";
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);  // second is a no-op skip
  EXPECT_EQ(s.Count(), 1u);
}

TEST_F(KVStoreTest, PartialRange) {
  KVStore s(Opts());
  BlockKey k{9, 0, 1};
  std::string v = "0123456789";
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(k, 3, 4, &out), Status::kOk);
  EXPECT_EQ(out, "3456");
}

TEST_F(KVStoreTest, LargeObjectOver128KiBLocalRead) {
  KVStore s(Opts());
  BlockKey k{42, 0, 1};
  std::string v(300 * 1024, 'x');  // >128KiB: no max_range_size cap on local path
  v[123456] = 'Z';
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(k, 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
}

TEST_F(KVStoreTest, LruEvictsLeastRecentlyUsedWhenOverCapacity) {
  // capacity holds ~2 objects of 1000B
  KVStore s(Opts(/*cap=*/2200));
  auto put = [&](uint64_t id) {
    std::string v(1000, char('a' + (id % 20)));
    return s.Cache(BlockKey{id, 0, 1}, v.data(), v.size());
  };
  ASSERT_EQ(put(1), Status::kOk);
  ASSERT_EQ(put(2), Status::kOk);
  // touch #1 so #2 becomes LRU
  std::string out; ASSERT_EQ(s.Range(BlockKey{1,0,1}, 0, 10, &out), Status::kOk);
  ASSERT_EQ(put(3), Status::kOk);  // over capacity -> evict LRU (#2)
  EXPECT_TRUE(s.IsCached(BlockKey{1, 0, 1}));
  EXPECT_FALSE(s.IsCached(BlockKey{2, 0, 1}));
  EXPECT_TRUE(s.IsCached(BlockKey{3, 0, 1}));
  EXPECT_LE(s.UsedBytes(), 2200u);
}

TEST_F(KVStoreTest, ReloadFromDiskRebuildsIndex) {
  BlockKey k{55, 0, 1};
  std::string v = "persisted";
  {
    KVStore s(Opts());
    ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  }
  KVStore s2(Opts());  // new instance, same dir
  EXPECT_TRUE(s2.IsCached(k));
  std::string out;
  ASSERT_EQ(s2.Range(k, 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
}
