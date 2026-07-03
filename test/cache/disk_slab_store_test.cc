// DiskSlabStore: extent-file slab store on SlabAllocator + slots.tbl rebuild.
// Covers: put/get/remove, idempotency, eviction, restart WARMTH (rebuild from
// slots.tbl), crash safety (a torn table record reads as free / not resurrected),
// meta mismatch -> clean re-init, oversize rejection.
#include "cache/disk_slab_store.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using dfkv::BlockKey;
using dfkv::DiskSlabStore;
using dfkv::Status;
namespace fs = std::filesystem;

namespace {
class DiskSlabTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("dfkv_slab_" + std::to_string(::testing::UnitTest::GetInstance()
                                              ->current_test_info()->line()));
    fs::remove_all(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  DiskSlabStore::Options Opts(uint64_t cap, uint64_t extent, uint64_t gran) {
    DiskSlabStore::Options o;
    o.dir = dir_.string();
    o.capacity_bytes = cap;
    o.extent_bytes = extent;
    o.slot_granularity = gran;
    return o;
  }
  fs::path dir_;
};
BlockKey K(uint64_t id) { return BlockKey{id, 0, 1}; }
}  // namespace

TEST_F(DiskSlabTest, PutGetRemoveRoundTrip) {
  bool ok = false;
  DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  std::string v = "the-kv-payload-0123456789";
  ASSERT_EQ(s.Cache(K(1), v.data(), v.size()), Status::kOk);
  EXPECT_TRUE(s.IsCached(K(1)));
  EXPECT_EQ(s.Count(), 1u);

  std::string out;
  ASSERT_EQ(s.Range(K(1), 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
  // partial range
  ASSERT_EQ(s.Range(K(1), 4, 5, &out), Status::kOk);
  EXPECT_EQ(out, v.substr(4, 5));
  // RangeInto
  char buf[64];
  size_t got = 0;
  ASSERT_EQ(s.RangeInto(K(1), 0, sizeof(buf), buf, sizeof(buf), &got), Status::kOk);
  EXPECT_EQ(std::string(buf, got), v);

  EXPECT_EQ(s.Range(K(999), 0, 10, &out), Status::kNotFound);
  EXPECT_EQ(s.Cache(K(1), v.data(), v.size()), Status::kOk);  // idempotent
  EXPECT_EQ(s.Count(), 1u);

  ASSERT_EQ(s.Remove(K(1)), Status::kOk);
  EXPECT_FALSE(s.IsCached(K(1)));
  EXPECT_EQ(s.Remove(K(1)), Status::kNotFound);
}

TEST_F(DiskSlabTest, EvictsUnderCapacity) {
  // 1 extent of 4 * 4096 slots. 5 keys -> one eviction.
  bool ok = false;
  DiskSlabStore s(Opts(4 * 4096, 4 * 4096, 4096), &ok);
  ASSERT_TRUE(ok);
  std::string v(4096, 'e');
  for (int i = 0; i < 4; ++i) ASSERT_EQ(s.Cache(K(i), v.data(), v.size()), Status::kOk);
  EXPECT_EQ(s.Count(), 4u);
  ASSERT_EQ(s.Cache(K(4), v.data(), v.size()), Status::kOk);
  EXPECT_EQ(s.Count(), 4u);
  EXPECT_EQ(s.Evictions(), 1u);
  EXPECT_TRUE(s.IsCached(K(4)));
}

TEST_F(DiskSlabTest, RestartRebuildsIndexKeepingWarmth) {
  std::vector<std::string> vals;
  {
    bool ok = false;
    DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
    for (int i = 0; i < 10; ++i) {
      std::string v = "val-" + std::to_string(i) + std::string(100, 'x');
      vals.push_back(v);
      ASSERT_EQ(s.Cache(K(1000 + i), v.data(), v.size()), Status::kOk);
    }
    ASSERT_EQ(s.Count(), 10u);
  }
  // Reopen the same dir: the index must rebuild from slots.tbl and all values
  // read back byte-for-byte (warmth preserved across restart).
  bool ok = false;
  DiskSlabStore s2(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(s2.TableRebuilt(), 10u);
  EXPECT_EQ(s2.Count(), 10u);
  for (int i = 0; i < 10; ++i) {
    std::string out;
    ASSERT_EQ(s2.Range(K(1000 + i), 0, vals[i].size(), &out), Status::kOk) << i;
    EXPECT_EQ(out, vals[i]) << i;
  }
}

TEST_F(DiskSlabTest, TornTableRecordReadsAsFreeNotResurrected) {
  {
    bool ok = false;
    DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
    std::string v(200, 'z');
    ASSERT_EQ(s.Cache(K(7), v.data(), v.size()), Status::kOk);
    ASSERT_EQ(s.Cache(K(8), v.data(), v.size()), Status::kOk);
  }
  // Corrupt the BODY of a real (valid) record so its CRC no longer matches -- a
  // torn write. Records land in high slots (LIFO alloc), so scan for the first
  // "SLTB" magic, then flip a body byte (offset +12, inside the CRC'd region).
  const std::string tbl = (dir_ / "slots.tbl").string();
  int fd = ::open(tbl.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  off_t fsz = ::lseek(fd, 0, SEEK_END);
  bool corrupted = false;
  for (off_t o = 0; o + 64 <= fsz; o += 64) {
    uint8_t m[4];
    if (::pread(fd, m, 4, o) != 4) break;
    // "SLTB" little-endian = 53 54 4C 42
    if (m[0] == 0x53 && m[1] == 0x54 && m[2] == 0x4C && m[3] == 0x42) {
      uint8_t byte = 0;
      ::pread(fd, &byte, 1, o + 12);
      byte ^= 0xFF;
      ::pwrite(fd, &byte, 1, o + 12);
      corrupted = true;
      break;
    }
  }
  ::close(fd);
  ASSERT_TRUE(corrupted) << "expected a valid record to corrupt";

  bool ok = false;
  DiskSlabStore s2(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(s2.TableRebuilt(), 1u) << "the torn record must be skipped";
  EXPECT_EQ(s2.Count(), 1u);
}

TEST_F(DiskSlabTest, MetaMismatchReinitsFresh) {
  {
    bool ok = false;
    DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
    std::string v(50, 'q');
    ASSERT_EQ(s.Cache(K(5), v.data(), v.size()), Status::kOk);
  }
  // Reopen with a DIFFERENT slot_granularity: the layout can't be reused, so the
  // store must re-init fresh (empty) rather than mis-read the old table.
  bool ok = false;
  DiskSlabStore s2(Opts(1 << 20, 1 << 20, 8192), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(s2.Count(), 0u);
  EXPECT_FALSE(s2.IsCached(K(5)));
}

TEST_F(DiskSlabTest, OversizeValueRejected) {
  bool ok = false;
  DiskSlabStore s(Opts(4096, 4096, 4096), &ok);  // one 4096 slot per extent
  ASSERT_TRUE(ok);
  std::string big(4097, 'b');
  EXPECT_EQ(s.Cache(K(1), big.data(), big.size()), Status::kIOError);
  std::string okv(4096, 'o');
  EXPECT_EQ(s.Cache(K(2), okv.data(), okv.size()), Status::kOk);
}

// The data-path I/O runs OUTSIDE the store lock (pin + inflight-count protect
// the slot in the unlocked window). These stress tests exercise the unlocked
// windows under eviction pressure and concurrent Remove -- a read must return
// either the key's FULL correct payload or a clean miss, never torn bytes.
// (Run under the CI TSan job, they also pin down the lock discipline.)
TEST_F(DiskSlabTest, ConcurrentPutGetNeverTorn) {
  bool ok = false;
  // Small pool (4 extents x 256KiB, 4KiB slots) => constant eviction pressure.
  DiskSlabStore s(Opts(1 << 20, 1 << 18, 4096), &ok);
  ASSERT_TRUE(ok);
  constexpr int kThreads = 8, kKeys = 64, kIters = 200;
  auto payload = [](uint64_t id) {
    std::string v(4000, '\0');
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((id * 131 + i) & 0xFF);
    return v;
  };
  std::atomic<int> torn{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&, t] {
      for (int i = 0; i < kIters; ++i) {
        const uint64_t id = (t * 7 + i) % kKeys + 1;
        const std::string want = payload(id);
        if (i % 3 == 0) s.Cache(K(id), want.data(), want.size());
        std::string got;
        Status st = s.Range(K(id), 0, 0, &got);
        if (st == Status::kOk && !got.empty() && got != want) torn.fetch_add(1);
        char buf[4096];
        size_t n = 0;
        st = s.RangeInto(K(id), 0, 0, buf, sizeof(buf), &n);
        if (st == Status::kOk && n == want.size() &&
            std::string(buf, n) != want) torn.fetch_add(1);
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(torn.load(), 0) << "a read observed a torn/foreign payload";
}

TEST_F(DiskSlabTest, RemoveDuringConcurrentAccessStaysConsistent) {
  bool ok = false;
  DiskSlabStore s(Opts(1 << 20, 1 << 18, 4096), &ok);
  ASSERT_TRUE(ok);
  constexpr int kKeys = 16, kIters = 300;
  auto payload = [](uint64_t id) { return std::string(3000, static_cast<char>('a' + id % 26)); };
  std::atomic<int> bad{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 6; ++t) {
    ts.emplace_back([&, t] {
      for (int i = 0; i < kIters; ++i) {
        const uint64_t id = (t + i) % kKeys + 1;
        const std::string want = payload(id);
        switch ((t + i) % 3) {
          case 0: s.Cache(K(id), want.data(), want.size()); break;
          case 1: {
            std::string got;
            if (s.Range(K(id), 0, 0, &got) == Status::kOk && !got.empty() && got != want)
              bad.fetch_add(1);
            break;
          }
          case 2: s.Remove(K(id)); break;
        }
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(bad.load(), 0) << "a read observed another key's bytes after Remove";
  // Steady state: every key must round-trip again (no leaked slots / stuck state).
  for (uint64_t id = 1; id <= kKeys; ++id) {
    const std::string want = payload(id);
    ASSERT_EQ(s.Cache(K(id), want.data(), want.size()), Status::kOk);
    std::string got;
    ASSERT_EQ(s.Range(K(id), 0, 0, &got), Status::kOk) << "key " << id;
    EXPECT_EQ(got, want);
  }
}
