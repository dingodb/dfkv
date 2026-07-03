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
