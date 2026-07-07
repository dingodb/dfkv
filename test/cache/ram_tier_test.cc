// RamTier: write-through RAM hot tier on SlabAllocator. Hermetic (no RDMA/disk;
// the flush sink is an injected callback). Covers read-after-write, the
// RAM_ONLY->DURABLE state machine via pins, send-in-flight pin blocking
// eviction, backpressure bypass when flush stalls, flush retry/drop, sub-range
// GetPrep, Remove semantics, and a concurrent TSan stress.
#include "cache/ram_tier.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using dfkv::BlockKey;
using dfkv::RamTier;
using namespace std::chrono_literals;

namespace {
BlockKey K(uint64_t id) { return BlockKey{id, 0, 1}; }

// A controllable flush sink: records flushed keys, can be gated (block until
// opened) and can be made to fail.
struct FlushSink {
  std::mutex m;
  std::condition_variable cv;
  std::set<std::string> flushed;
  bool gate_open = true;      // when false, flush blocks
  bool fail = false;          // when true, flush returns false
  int calls = 0;

  RamTier::FlushFn fn() {
    return [this](const BlockKey& k, char*, size_t, size_t) {
      std::unique_lock<std::mutex> lk(m);
      ++calls;
      cv.wait(lk, [this] { return gate_open; });
      if (fail) return false;
      flushed.insert(k.Filename());
      cv.notify_all();
      return true;
    };
  }
  void open() { { std::lock_guard<std::mutex> lk(m); gate_open = true; } cv.notify_all(); }
  void close() { std::lock_guard<std::mutex> lk(m); gate_open = false; }
};

// Poll a predicate up to ~2s.
template <class F>
bool WaitFor(F pred) {
  for (int i = 0; i < 2000; ++i) {
    if (pred()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return pred();
}

RamTier::Options Opts(uint64_t bytes, uint32_t gran = 4096) {
  RamTier::Options o;
  o.bytes = bytes;
  o.slot_granularity = gran;
  return o;
}
}  // namespace

TEST(RamTier, ReadAfterWriteBeforeFlush) {
  FlushSink sink;
  sink.close();  // hold flushes so the slot stays RAM_ONLY
  RamTier rt(Opts(64 * 4096), sink.fn());
  ASSERT_TRUE(rt.ok());
  std::string v = "hot-kv-payload";
  ASSERT_TRUE(rt.Put(K(1), v.data(), v.size()));
  // Visible immediately, even though the flush hasn't run.
  RamTier::Hit h;
  ASSERT_TRUE(rt.GetPrep(K(1), 0, v.size(), &h));
  EXPECT_EQ(std::string(h.ptr, h.len), v);
  EXPECT_EQ(rt.Hits(), 1u);
  rt.Release(h.token);
  sink.open();
}

TEST(RamTier, FlushMakesDurable) {
  FlushSink sink;
  RamTier rt(Opts(64 * 4096), sink.fn());
  std::string v(1000, 'd');
  ASSERT_TRUE(rt.Put(K(2), v.data(), v.size()));
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == 1u; }));
  {
    std::lock_guard<std::mutex> lk(sink.m);
    EXPECT_TRUE(sink.flushed.count(K(2).Filename()));
  }
  EXPECT_EQ(rt.FlushBacklog(), 0u);
}

// Regression (hd04 prod, 2026-07): the RAM tier bound its whole arena to ONE
// SlabAllocator extent, so it could hold only a single size class at a time. A
// real KV working set has assorted value sizes (partial blocks, variable-length
// coalesced chunks); each cross-class Put stole the lone extent and evicted the
// whole tier (ram_objects collapsed to ~1-5 in a 16 GiB arena), so every reuse
// GET missed to slab and load ran at disk speed. With the arena partitioned into
// many extents, distinct size classes coexist and stay resident. Also exercises
// the global-arena-offset math (slots now live in extents > 0).
TEST(RamTier, VaryingSizesCoexistNoChurn) {
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "2097152", 1);  // 2 MiB extents
  FlushSink sink;
  RamTier rt(Opts(256ull << 20), sink.fn());             // 256 MiB arena -> ~128 extents
  ASSERT_TRUE(rt.ok());
  // 6 distinct size classes x 4 keys each (models a real KV mix: a handful of
  // block sizes, many blocks each). 6 classes * kStripeWays(8) = 48 extents of
  // headroom vs 128 available. Each value < the 2 MiB extent.
  const size_t sizes[6] = {40000, 90000, 200000, 380000, 560000, 730000};
  std::vector<std::pair<uint64_t, size_t>> items;
  for (int i = 0; i < 24; ++i) {
    size_t sz = sizes[i % 6];
    std::vector<char> v(sz, static_cast<char>(i + 1));
    ASSERT_TRUE(rt.Put(K(1000 + i), v.data(), v.size())) << "put i=" << i;
    items.emplace_back(1000 + i, sz);
  }
  // All resident (no cross-class eviction) and byte-correct after the global
  // offset resolves each slot's extent.
  EXPECT_EQ(rt.Count(), items.size());
  EXPECT_EQ(rt.Evictions(), 0u);
  for (auto& [id, sz] : items) {
    RamTier::Hit h;
    ASSERT_TRUE(rt.GetPrep(K(id), 0, sz, &h)) << "get id=" << id;
    EXPECT_EQ(h.len, sz);
    EXPECT_EQ(static_cast<unsigned char>(h.ptr[0]),
              static_cast<unsigned char>((id - 1000) + 1));
    rt.Release(h.token);
  }
  ::unsetenv("DFKV_RAM_TIER_EXTENT_BYTES");
}

TEST(RamTier, MissReturnsFalse) {
  FlushSink sink;
  RamTier rt(Opts(64 * 4096), sink.fn());
  RamTier::Hit h;
  EXPECT_FALSE(rt.GetPrep(K(99), 0, 10, &h));
  EXPECT_EQ(rt.Misses(), 1u);
}

TEST(RamTier, SubRangeGetPrep) {
  FlushSink sink;
  RamTier rt(Opts(64 * 4096), sink.fn());
  std::string v = "0123456789";
  ASSERT_TRUE(rt.Put(K(3), v.data(), v.size()));
  RamTier::Hit h;
  ASSERT_TRUE(rt.GetPrep(K(3), 3, 4, &h));
  EXPECT_EQ(std::string(h.ptr, h.len), "3456");
  rt.Release(h.token);
  // offset past end -> zero-length hit (still a hit)
  ASSERT_TRUE(rt.GetPrep(K(3), 100, 10, &h));
  EXPECT_EQ(h.len, 0u);
  rt.Release(h.token);
}

TEST(RamTier, SendPinBlocksEvictionUntilRelease) {
  FlushSink sink;
  RamTier rt(Opts(2 * 4096, 4096), sink.fn());  // exactly 2 slots
  std::string v(4096, 'x');
  ASSERT_TRUE(rt.Put(K(10), v.data(), v.size()));
  ASSERT_TRUE(rt.Put(K(11), v.data(), v.size()));
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == 2u; }));  // both durable

  RamTier::Hit h;
  ASSERT_TRUE(rt.GetPrep(K(10), 0, v.size(), &h));  // send-pin on K10
  // Putting a 3rd must evict the *unpinned* durable one (K11), never K10.
  ASSERT_TRUE(rt.Put(K(12), v.data(), v.size()));
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == 3u; }));
  EXPECT_TRUE(rt.Contains(K(10))) << "send-pinned slot must survive";
  EXPECT_TRUE(rt.Contains(K(12)));
  EXPECT_FALSE(rt.Contains(K(11)));
  EXPECT_GE(rt.Evictions(), 1u);
  rt.Release(h.token);
}

TEST(RamTier, BackpressureBypassWhenFlushStalls) {
  FlushSink sink;
  sink.close();  // stall all flushes -> slots stay RAM_ONLY (flush-pinned)
  RamTier rt(Opts(2 * 4096, 4096), sink.fn());
  std::string v(4096, 'p');
  ASSERT_TRUE(rt.Put(K(20), v.data(), v.size()));
  ASSERT_TRUE(rt.Put(K(21), v.data(), v.size()));
  // Arena full of non-evictable slots -> the next Put is declined (bypass).
  EXPECT_FALSE(rt.Put(K(22), v.data(), v.size()));
  EXPECT_EQ(rt.PutBypass(), 1u);
  // Drain: opening the gate lets flushes land, freeing capacity again.
  sink.open();
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == 2u; }));
  EXPECT_TRUE(rt.Put(K(22), v.data(), v.size()));
}

TEST(RamTier, FlushFailureRetriesThenDrops) {
  FlushSink sink;
  sink.fail = true;  // every flush attempt fails
  RamTier::Options o = Opts(8 * 4096);
  o.flush_retries = 3;
  RamTier rt(o, sink.fn());
  std::string v(500, 'f');
  ASSERT_TRUE(rt.Put(K(30), v.data(), v.size()));
  // After flush_retries failed attempts the entry is dropped from RAM.
  EXPECT_TRUE(WaitFor([&] { return rt.FlushDropped() == 1u; }));
  EXPECT_FALSE(rt.Contains(K(30)));
  {
    std::lock_guard<std::mutex> lk(sink.m);
    EXPECT_GE(sink.calls, 3);
  }
}

TEST(RamTier, RemoveOnlyDropsDurableIdle) {
  FlushSink sink;
  sink.close();
  RamTier rt(Opts(8 * 4096), sink.fn());
  std::string v(200, 'r');
  ASSERT_TRUE(rt.Put(K(40), v.data(), v.size()));
  EXPECT_FALSE(rt.Remove(K(40))) << "non-durable (flushing) -> declined";
  sink.open();
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == 1u; }));
  RamTier::Hit h;
  ASSERT_TRUE(rt.GetPrep(K(40), 0, v.size(), &h));  // now send-in-flight
  EXPECT_FALSE(rt.Remove(K(40))) << "in-flight -> declined";
  rt.Release(h.token);
  EXPECT_TRUE(rt.Remove(K(40))) << "durable + idle -> removed";
  EXPECT_FALSE(rt.Contains(K(40)));
}

TEST(RamTier, ConcurrentPutGetReleaseIsRaceFree) {
  FlushSink sink;
  RamTier rt(Opts(512 * 4096), sink.fn());
  constexpr int T = 8, N = 1500;
  std::atomic<int> hits{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < T; ++t) {
    ts.emplace_back([&, t] {
      std::string v(300, 'c');
      for (int i = 0; i < N; ++i) {
        BlockKey k = K((t * N + i) % 3000);
        rt.Put(k, v.data(), v.size());
        RamTier::Hit h;
        if (rt.GetPrep(k, 0, v.size(), &h)) {
          hits.fetch_add(1);
          rt.Release(h.token);
        }
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_GT(hits.load(), 0);
}
