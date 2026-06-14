// TDD R3 — KVStore: the cache-node local store (disk + LRU + cache-only/no-S3).
#include "kv_store.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

using dingofs::cache::kv::BlockKey;
using dingofs::cache::kv::KVStore;
using dingofs::cache::kv::Status;

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
