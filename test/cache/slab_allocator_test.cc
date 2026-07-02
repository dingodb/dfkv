// SlabAllocator: media-agnostic slot lifecycle. Pure logic, hermetic tests.
// Covers: alloc/get/remove, idempotent put, size-class reuse vs new, per-class
// CLOCK eviction, extent binding + cross-class steal, pin-blocks-eviction, byte
// accounting, oversize rejection, and a concurrent (TSan) stress.
#include "cache/slab_allocator.h"

#include <gtest/gtest.h>

#include <atomic>
#include <set>
#include <string>
#include <thread>
#include <vector>

using dfkv::SlabAllocator;
using SlotRef = SlabAllocator::SlotRef;

namespace {
SlabAllocator::Options Opts(uint64_t extent_bytes, uint32_t num_extents,
                            uint32_t align = 4096, double waste = 0.25) {
  SlabAllocator::Options o;
  o.extent_bytes = extent_bytes;
  o.num_extents = num_extents;
  o.align = align;
  o.max_waste = waste;
  return o;
}
}  // namespace

TEST(SlabAllocator, PutGetRemoveRoundTrip) {
  SlabAllocator a(Opts(64 * 1024, 4));
  std::vector<std::string> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put("k", 4096, &r, &ev));
  EXPECT_TRUE(r.valid());
  EXPECT_EQ(r.slot_size, 4096u);
  EXPECT_EQ(r.offset, static_cast<uint64_t>(r.slot) * r.slot_size);
  EXPECT_TRUE(ev.empty());
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_EQ(a.UsedBytes(), 4096u);

  SlotRef g;
  ASSERT_TRUE(a.Get("k", &g));
  EXPECT_EQ(g.extent, r.extent);
  EXPECT_EQ(g.slot, r.slot);
  EXPECT_FALSE(a.Get("absent", &g));

  EXPECT_TRUE(a.Remove("k"));
  EXPECT_FALSE(a.Contains("k"));
  EXPECT_EQ(a.Count(), 0u);
  EXPECT_EQ(a.UsedBytes(), 0u);
  EXPECT_FALSE(a.Remove("k"));  // idempotent
}

TEST(SlabAllocator, PutIsIdempotentKeepsSameSlot) {
  SlabAllocator a(Opts(64 * 1024, 4));
  std::vector<std::string> ev;
  SlotRef r1, r2;
  ASSERT_TRUE(a.Put("k", 4096, &r1, &ev));
  ASSERT_TRUE(a.Put("k", 4096, &r2, &ev));  // second put: same slot, no evict
  EXPECT_EQ(r1.extent, r2.extent);
  EXPECT_EQ(r1.slot, r2.slot);
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_TRUE(ev.empty());
}

TEST(SlabAllocator, SizeClassReuseVsNew) {
  SlabAllocator a(Opts(1 << 20, 4, /*align=*/4096, /*waste=*/0.25));
  std::vector<std::string> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put("a", 4096, &r, &ev));       // class 4096
  EXPECT_EQ(a.ClassCount(), 1u);
  ASSERT_TRUE(a.Put("b", 4000, &r, &ev));       // fits 4096 within 25% waste -> reuse
  EXPECT_EQ(a.ClassCount(), 1u);
  EXPECT_EQ(r.slot_size, 4096u);
  ASSERT_TRUE(a.Put("c", 8192, &r, &ev));       // needs a bigger class -> new
  EXPECT_EQ(a.ClassCount(), 2u);
  EXPECT_EQ(r.slot_size, 8192u);
}

TEST(SlabAllocator, EvictsWithinClassUnderPressure) {
  // 1 extent of 4 slots (4 * 4096). A 5th 4096 key must evict one.
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  EXPECT_EQ(a.Count(), 4u);
  EXPECT_TRUE(ev.empty());
  ASSERT_TRUE(a.Put("k4", 4096, &r, &ev));   // full -> evict one
  EXPECT_EQ(a.Count(), 4u);
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_EQ(a.Evictions(), 1u);
  EXPECT_TRUE(a.Contains("k4"));
}

TEST(SlabAllocator, GetGivesSecondChanceInClock) {
  // 4 slots; touch k0 so it survives the first eviction (referenced bit).
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  ASSERT_TRUE(a.Get("k0", &r));  // referenced -> should be spared once
  ASSERT_TRUE(a.Put("k4", 4096, &r, &ev));
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_NE(ev[0], "k0") << "a referenced entry gets a second chance";
  EXPECT_TRUE(a.Contains("k0"));
}

TEST(SlabAllocator, PinBlocksEviction) {
  SlabAllocator a(Opts(2 * 4096, 1));  // 2 slots
  std::vector<std::string> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put("pinned", 4096, &r, &ev));
  ASSERT_TRUE(a.Put("other", 4096, &r, &ev));
  ASSERT_TRUE(a.Pin("pinned"));
  ASSERT_TRUE(a.Put("new", 4096, &r, &ev));  // must evict "other", never "pinned"
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0], "other");
  EXPECT_TRUE(a.Contains("pinned"));

  // With BOTH slots pinned, a further Put has nothing to evict -> fails.
  ASSERT_TRUE(a.Pin("new"));
  ev.clear();
  EXPECT_FALSE(a.Put("nope", 4096, &r, &ev)) << "all slots pinned -> no room";
  EXPECT_TRUE(ev.empty());
  // Unpin frees the path again.
  ASSERT_TRUE(a.Unpin("new"));
  EXPECT_TRUE(a.Put("nope", 4096, &r, &ev));
}

TEST(SlabAllocator, CrossClassStealWhenPoolEmpty) {
  // 2 extents. Fill both with class-A (4096) keys. Then a class-B (8192) key
  // with the pool empty must STEAL a fully-unpinned A extent and rebind it.
  SlabAllocator a(Opts(4 * 4096, 2));  // each extent: 4 A-slots or 2 B-slots
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 8; ++i)  // 8 A-slots = both extents bound to class A
    ASSERT_TRUE(a.Put("a" + std::to_string(i), 4096, &r, &ev));
  EXPECT_EQ(a.BoundExtents(), 2u);
  ev.clear();
  ASSERT_TRUE(a.Put("b0", 8192, &r, &ev));  // needs class B -> steal an A extent
  EXPECT_EQ(r.slot_size, 8192u);
  EXPECT_FALSE(ev.empty()) << "steal evicts the stolen extent's residents";
  EXPECT_TRUE(a.Contains("b0"));
  EXPECT_GE(a.ClassCount(), 2u);
}

TEST(SlabAllocator, OversizeValueRejected) {
  SlabAllocator a(Opts(4096, 2));  // extent holds one 4096 slot
  std::vector<std::string> ev;
  SlotRef r;
  EXPECT_FALSE(a.Put("big", 4097, &r, &ev)) << "value larger than an extent";
  EXPECT_TRUE(a.Put("ok", 4096, &r, &ev));
}

TEST(SlabAllocator, ZeroLenGetsAMinimalSlot) {
  SlabAllocator a(Opts(64 * 1024, 2));
  std::vector<std::string> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put("z", 0, &r, &ev));  // 0-byte anchor -> one aligned slot
  EXPECT_EQ(r.slot_size, 4096u);
  EXPECT_TRUE(a.Contains("z"));
}

TEST(SlabAllocator, ConcurrentPutGetRemoveIsRaceFree) {
  // TSan target: many threads hammer distinct + shared keys. Correctness beyond
  // "no crash / no race" is loose here; the single mutex must serialize cleanly.
  SlabAllocator a(Opts(256 * 4096, 8));
  constexpr int T = 8, N = 2000;
  std::atomic<int> ok_puts{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < T; ++t) {
    ts.emplace_back([&, t] {
      std::vector<std::string> ev;
      SlotRef r;
      for (int i = 0; i < N; ++i) {
        std::string k = "k" + std::to_string((t * N + i) % 5000);
        if (a.Put(k, 4096, &r, &ev)) ok_puts.fetch_add(1);
        a.Get(k, &r);
        if ((i & 3) == 0) a.Pin(k);
        if ((i & 3) == 0) a.Unpin(k);
        if ((i & 7) == 0) a.Remove(k);
        ev.clear();
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_GT(ok_puts.load(), 0);
  EXPECT_LE(a.UsedBytes(), a.Capacity());  // never over-commit the pool
}

TEST(SlabAllocator, RestoreInstallsKeyAtKnownSlot) {
  SlabAllocator a(Opts(4 * 4096, 2));
  // Restore two keys at known slots (as a rebuild would from persistence).
  EXPECT_TRUE(a.Restore("ka", 4096, /*extent=*/0, /*slot=*/1));
  EXPECT_TRUE(a.Restore("kb", 4096, /*extent=*/0, /*slot=*/2));
  EXPECT_EQ(a.Count(), 2u);
  EXPECT_EQ(a.UsedBytes(), 2u * 4096u);
  SlotRef r;
  ASSERT_TRUE(a.Get("ka", &r));
  EXPECT_EQ(r.extent, 0u);
  EXPECT_EQ(r.slot, 1u);
  EXPECT_EQ(r.offset, 1u * 4096u);
  // A subsequent Put on the same extent must use a still-free slot, not clobber
  // the restored ones.
  std::vector<std::string> ev;
  ASSERT_TRUE(a.Put("kc", 4096, &r, &ev));
  EXPECT_TRUE(ev.empty());
  EXPECT_NE(r.slot, 1u);
  EXPECT_NE(r.slot, 2u);
}

TEST(SlabAllocator, RestoreRejectsInconsistentInput) {
  SlabAllocator a(Opts(4 * 4096, 1));
  EXPECT_FALSE(a.Restore("bad_extent", 4096, /*extent=*/9, 0));   // extent out of range
  EXPECT_FALSE(a.Restore("bad_slot", 4096, 0, /*slot=*/99));      // slot out of range
  EXPECT_TRUE(a.Restore("k", 4096, 0, 0));
  EXPECT_FALSE(a.Restore("k", 4096, 0, 1));                       // duplicate key
  EXPECT_FALSE(a.Restore("k2", 4096, 0, 0));                      // slot already taken
  // A second class on the same extent is a persistence inconsistency.
  EXPECT_FALSE(a.Restore("k3", 8192, 0, 0));
}
