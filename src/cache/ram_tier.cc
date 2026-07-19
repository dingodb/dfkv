#include "cache/ram_tier.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "utils/log.h"
#include "utils/thread_name.h"
#include "utils/numa_util.h"

namespace dfkv {

RamTier::RamTier(Options opt, FlushFn flush)
    : opt_(opt), flush_(std::move(flush)) {
  if (opt_.slot_granularity < 4096) opt_.slot_granularity = 4096;  // O_DIRECT floor
  if (opt_.bytes < opt_.slot_granularity) opt_.bytes = opt_.slot_granularity;
  void* p = nullptr;
  // 4096-aligned base => slot addresses (offset is a slot_size multiple, itself
  // a granularity multiple) are O_DIRECT-aligned for the flusher (gap 10.4).
  if (posix_memalign(&p, 4096, opt_.bytes) != 0 || p == nullptr) return;
  arena_ = static_cast<char*>(p);
  // NUMA: interleave the arena across sockets BEFORE first touch. A single
  // pre-faulting thread would otherwise land every page on its own node, and
  // all remote-socket consumers (serve threads, NIC DMA on the far rail) pay
  // the interconnect on every access. Interleave is the balanced default for a
  // shared arena; DFKV_RAM_TIER_NUMA=off opts out (e.g. to combine with
  // DFKV_RDMA_NUMA-pinned single-rail setups).
  const char* nm = std::getenv("DFKV_RAM_TIER_NUMA");
  const bool interleave = !(nm && std::strcmp(nm, "off") == 0);
  const int nodes = numa::OnlineNodeCount();
  if (interleave) numa::InterleaveMemory(arena_, opt_.bytes, nodes);
  // Pre-fault the arena now (one bounded, REPORTED cost at startup) instead of
  // paying first-touch page faults on the PUT hot path; parallel workers keep
  // large arenas to seconds. With RDMA the MR registration pins pages anyway;
  // this covers TCP-only runs and makes the cost explicit either way.
  {
    const auto t0 = std::chrono::steady_clock::now();
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t nt = std::max(1u, std::min(hw ? hw / 4 : 4u, 16u));
    const size_t chunk = (opt_.bytes + nt - 1) / nt;
    std::vector<std::thread> ws;
    for (size_t i = 0; i < nt; ++i) {
      const size_t off = i * chunk;
      if (off >= opt_.bytes) break;
      const size_t n = std::min(chunk, opt_.bytes - off);
      ws.emplace_back([this, off, n] { std::memset(arena_ + off, 0, n); });
    }
    for (auto& w : ws) w.join();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    DFKV_LOG_INFO("ram tier arena pre-faulted: " +
                  std::to_string(opt_.bytes >> 20) + " MiB in " +
                  std::to_string(ms) + " ms (threads=" + std::to_string(ws.size()) +
                  ", numa=" + (interleave && nodes > 1
                                   ? "interleave/" + std::to_string(nodes) + "nodes"
                                   : "default") + ")");
  }

  // Partition the arena into MANY extents. SlabAllocator binds each extent to a
  // single size class; with one giant extent (num_extents=1) the RAM tier could
  // hold only ONE size class at a time, so a real KV working set (block values
  // of assorted sizes -- partial blocks, variable-length coalesced chunks)
  // evicted the whole tier on every cross-class Put (observed on hd04 prod:
  // ram_objects collapsed to ~1-5 in a 16 GiB arena, every reuse GET missed to
  // slab -> load ran at disk speed, ~25x slower than a RAM hit). Many smaller
  // extents let the working set's size classes coexist. Sizing: SlabAllocator
  // keeps up to kStripeWays(8) extents open per size class, so we want
  // num_extents >> num_classes*8 (a real KV mix spans ~15-20 size classes). A
  // 32 MiB extent gives 512 extents on a 16 GiB arena (>> 160) while staying far
  // above any single KV block, so large values still get many slots per extent.
  uint64_t ext = 32ull << 20;
  if (const char* e = std::getenv("DFKV_RAM_TIER_EXTENT_BYTES")) {
    uint64_t v = std::strtoull(e, nullptr, 10);
    if (v >= (1ull << 20)) ext = v;
  }
  if (ext > opt_.bytes) ext = opt_.bytes;
  extent_bytes_ = ext;
  const uint64_t total_extents = std::max<uint64_t>(1, opt_.bytes / ext);

  // Shard count: enough shards to take the single lock off the hot path, few
  // enough that every shard keeps a deep extent pool for class coexistence
  // (>= 32 extents per shard). Small/test arenas degrade to 1 shard (identical
  // to the pre-sharding behavior).
  size_t nshards = 8;
  if (const char* s = std::getenv("DFKV_RAM_TIER_SHARDS")) {
    long v = std::strtol(s, nullptr, 10);
    if (v >= 1 && v <= static_cast<long>(kMaxShards)) nshards = static_cast<size_t>(v);
  }
  nshards = std::min(nshards, kMaxShards);
  while (nshards > 1 && total_extents / nshards < 32) nshards /= 2;
  if (nshards < 1) nshards = 1;

  const uint64_t ext_per_shard = total_extents / nshards;  // remainder unused (<1 extent/shard)
  shards_.reserve(nshards);
  for (size_t s = 0; s < nshards; ++s) {
    auto sh = std::make_unique<Shard>();
    sh->base_off = static_cast<uint64_t>(s) * ext_per_shard * ext;
    SlabAllocator::Options ao;
    ao.extent_bytes = ext;
    ao.num_extents = static_cast<uint32_t>(ext_per_shard);
    ao.align = opt_.slot_granularity;
    ao.max_waste = 0.25;
    sh->alloc = std::make_unique<SlabAllocator>(ao);
    shards_.push_back(std::move(sh));
  }
  if (nshards > 1)
    DFKV_LOG_INFO("ram tier sharded: " + std::to_string(nshards) + " shards x " +
                  std::to_string(ext_per_shard) + " extents");

  // Flush workers: distribute round-robin over shards, at least one per shard.
  const uint32_t nf = std::max<uint32_t>(opt_.flush_threads ? opt_.flush_threads : 1,
                                         static_cast<uint32_t>(nshards));
  flushers_.reserve(nf);
  for (uint32_t i = 0; i < nf; ++i) {
    Shard& s = *shards_[i % nshards];
    flushers_.emplace_back([this, &s, i] { NameThisThread("rt-flush-", i); FlushLoop(s); });
  }
  if (opt_.reclaim_interval_ms > 0) {
    reclaim_thread_ = std::thread([this] {
      NameThisThread("rt-reclaim");
      std::unique_lock<std::mutex> lk(reclaim_mu_);
      for (;;) {
        reclaim_cv_.wait_for(lk, std::chrono::milliseconds(opt_.reclaim_interval_ms),
                             [this] { return reclaim_stop_; });
        if (reclaim_stop_) return;
        lk.unlock();
        for (size_t s = 0; s < shards_.size(); ++s) ReclaimTick(*shards_[s], s);
        lk.lock();
      }
    });
  }
}

RamTier::~RamTier() {
  if (reclaim_thread_.joinable()) {
    { std::lock_guard<std::mutex> lk(reclaim_mu_); reclaim_stop_ = true; }
    reclaim_cv_.notify_all();
    reclaim_thread_.join();
  }
  for (auto& sh : shards_) {
    std::lock_guard<std::mutex> lk(sh->mu);
    sh->stop = true;
  }
  for (auto& sh : shards_) sh->cv.notify_all();
  for (auto& f : flushers_) if (f.joinable()) f.join();
  if (arena_) std::free(arena_);
}

// One reclaimer pass over one shard (mirror of DiskSlabStore::ReclaimTick,
// arena flavor): for every class with new inserts since the last pass, top its
// free slots up to a demand-driven watermark, in small batches per lock hold.
// Victims are unpinned (== durable, no send in flight), so dropping them from
// the index is the same operation Put performs for its own inline evictions.
void RamTier::ReclaimTick(Shard& s, size_t /*shard_idx*/) {
  const auto classes = s.alloc->Classes();
  if (s.reclaim_last_puts.size() < classes.size())
    s.reclaim_last_puts.resize(classes.size(), 0);
  std::vector<uint64_t> delta(classes.size(), 0);
  std::vector<uint32_t> extents(classes.size(), 0);
  for (size_t i = 0; i < classes.size(); ++i) {
    delta[i] = classes[i].puts - s.reclaim_last_puts[i];
    s.reclaim_last_puts[i] = classes[i].puts;
    extents[i] = classes[i].extents;
  }
  // -- GROW -- (mirror of DiskSlabStore::ReclaimTick; see its comment)
  // Runs even while flush-gated, ON PURPOSE: a cold donor's extents hold
  // DURABLE residents, so moving them to the hot class frees admission
  // capacity precisely when the flusher can't -- the shard's pinned mass is
  // the hot class's own unflushed writes, not the donors'.
  for (size_t i = 0; i < classes.size(); ++i) {
    if (delta[i] == 0) continue;
    const auto& c = classes[i];
    size_t free_now = c.free_slots;
    const size_t want = std::max<size_t>(64, static_cast<size_t>(2 * delta[i]));
    if (free_now >= want || c.slots_per_extent == 0 || s.alloc->PoolExtents() > 0)
      continue;
    size_t need_ext =
        (want - free_now + c.slots_per_extent - 1) / c.slots_per_extent;
    need_ext = std::min<size_t>(need_ext, kGrowExtentsPerTick);
    while (need_ext > 0) {
      size_t donor = classes.size();
      for (size_t d = 0; d < classes.size(); ++d) {  // coldest, then biggest
        if (d == i || extents[d] <= SlabAllocator::kStripeWays) continue;
        if (delta[d] != 0 && delta[d] * 4 > delta[i]) continue;  // not cold enough
        if (donor == classes.size() ||
            std::make_pair(delta[d], ~uint64_t(extents[d])) <
                std::make_pair(delta[donor], ~uint64_t(extents[donor])))
          donor = d;
      }
      if (donor == classes.size()) break;
      std::vector<std::string> evicted;
      bool ok;
      {
        std::lock_guard<std::mutex> lk(s.mu);
        ok = s.alloc->StealFrom(donor, i, &evicted);
        for (const auto& ev : evicted) s.index.erase(ev);
      }
      if (!ok) { extents[donor] = 0; continue; }  // donor all pinned: try next
      extents[donor]--;
      rebalanced_.fetch_add(1, std::memory_order_relaxed);
      --need_ext;
    }
  }
  // Flush-gated regime: when the flush queue is deep, the shard is mostly
  // PINNED (not-yet-durable) slots -- free slots are not the admission
  // constraint, and a CLOCK sweep over a pinned-heavy ring just burns lock
  // time skipping entries. Skip the self-eviction phase; it resumes as the
  // flusher catches up. The 4096 budget is per shard.
  {
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.flushq.size() > 4096) return;
  }
  // -- RECLAIM --
  for (size_t i = 0; i < classes.size(); ++i) {
    if (delta[i] == 0) continue;  // no write demand on this class
    const auto& c = classes[i];
    const size_t capacity = c.resident + c.free_slots;
    size_t target = std::max<size_t>(64, static_cast<size_t>(2 * delta[i]));
    target = std::min(target, capacity / 4);
    if (c.free_slots >= target) continue;
    size_t budget = 4096;  // per-tick per-class bound
    for (;;) {
      std::vector<std::string> evicted;
      const size_t batch = std::min<size_t>(64, budget);
      size_t got;
      {
        std::lock_guard<std::mutex> lk(s.mu);
        got = s.alloc->ReclaimClass(i, target, batch, &evicted);
        for (const auto& ev : evicted) s.index.erase(ev);
      }
      if (got > 0) reclaimed_.fetch_add(got, std::memory_order_relaxed);
      budget -= std::min(budget, got);
      // Partial batch = ReclaimClass stopped early (target reached, everything
      // pinned, or its cascade-shrink guard fired) -- don't re-invoke.
      if (got < batch || budget == 0) break;
    }
  }
}

void RamTier::SetArenaMr(void* mr) {
  arena_mr_.store(mr, std::memory_order_release);
}

bool RamTier::Put(const BlockKey& key, const void* data, size_t len) {
  if (!arena_) return false;
  if (data == nullptr && len != 0) return false;
  const std::string fn = key.Filename();
  Shard& s = ShardFor(fn);

  uint64_t offset = 0;
  uint32_t cap = 0;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    // Resident, or another thread is mid-copy on this same (content-addressed)
    // key -> dedup. The in-progress writer produces identical bytes, so returning
    // "accepted" here is correct read-after-write for this key.
    if (s.index.count(fn) || s.writing.count(fn)) return true;
    SlabAllocator::SlotRef ref;
    std::vector<std::string> evicted;
    if (!s.alloc->Put(fn, len, &ref, &evicted)) {
      // Backpressure (gap 10.3): shard full of non-evictable (flushing / in-
      // flight) slots. Decline -> caller does the normal synchronous disk write.
      put_bypass_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    for (const auto& ev : evicted) s.index.erase(ev);  // evicted are durable+idle
    // Flush-pin the slot: RAM_ONLY, not evictable until the async flush lands.
    s.alloc->Pin(fn);
    s.writing.insert(fn);  // reserve the key so a concurrent same-key Put dedups
    offset = s.base_off + ref.extent * extent_bytes_ + ref.offset;  // global offset
    cap = ref.slot_size;                                            // (within-extent)
    // NOT visible yet: index install is deferred until after the copy, so a
    // concurrent GetPrep can't read the slot mid-copy.
  }

  std::memcpy(arena_ + offset, data, len);  // copy outside the lock (MB-scale)

  {
    std::lock_guard<std::mutex> lk(s.mu);
    Entry e;
    e.offset = offset;
    e.len = static_cast<uint32_t>(len);
    e.cap = cap;
    e.durable = false;
    s.index[fn] = e;
    s.writing.erase(fn);  // copy done -> now visible to GetPrep
    s.flushq.push_back(QItem{fn, key, 0});
  }
  s.cv.notify_one();
  puts_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool RamTier::PutDurable(const BlockKey& key, const void* data, size_t len) {
  if (!arena_) return false;
  if (data == nullptr && len != 0) return false;
  const std::string fn = key.Filename();
  Shard& s = ShardFor(fn);

  uint64_t offset = 0;
  uint32_t cap = 0;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.index.count(fn) || s.writing.count(fn)) return true;  // already resident
    SlabAllocator::SlotRef ref;
    std::vector<std::string> evicted;
    if (!s.alloc->Put(fn, len, &ref, &evicted)) return false;  // best-effort skip
    for (const auto& ev : evicted) s.index.erase(ev);
    // Pin only for the copy window: the slot is not in the index yet, but the
    // allocator knows it and an unpinned slot could be CLOCK-evicted mid-copy.
    s.alloc->Pin(fn);
    s.writing.insert(fn);
    offset = s.base_off + ref.extent * extent_bytes_ + ref.offset;
    cap = ref.slot_size;
  }

  std::memcpy(arena_ + offset, data, len);

  {
    std::lock_guard<std::mutex> lk(s.mu);
    Entry e;
    e.offset = offset;
    e.len = static_cast<uint32_t>(len);
    e.cap = cap;
    e.durable = true;  // already on disk: no flushq, no flush-pin
    s.index[fn] = e;
    s.writing.erase(fn);
    s.alloc->Unpin(fn);  // refcount 0 => evictable immediately
  }
  promotes_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool RamTier::GetPrep(const BlockKey& key, uint64_t offset, uint64_t length,
                      Hit* out) {
  if (shards_.empty()) return false;  // failed-construction guard (ok()==false)
  const std::string fn = key.Filename();
  const size_t sidx = std::hash<std::string>{}(fn) % shards_.size();
  Shard& s = *shards_[sidx];
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.index.find(fn);
  if (it == s.index.end()) {
    misses_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  const Entry& e = it->second;
  // Send-pin: the RDMA send reads the shared arena in place; the slot must not
  // be evicted/reused until the send completes (gap 10.1).
  s.alloc->Pin(fn);
  const uint64_t start = std::min<uint64_t>(offset, e.len);
  const uint64_t avail = e.len - start;
  const uint64_t n = std::min(length ? length : avail, avail);
  if (out) {
    out->ptr = arena_ + e.offset + start;
    out->len = static_cast<size_t>(n);
    out->mr = arena_mr_.load(std::memory_order_acquire);
    // Token encodes the shard so Release() routes without a global map.
    out->token = (s.next_token++ << kTokenShardBits) | sidx;
    s.pinned[out->token] = fn;
  }
  hits_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void RamTier::Release(uint64_t token) {
  if (shards_.empty()) return;
  Shard& s = *shards_[(token & (kMaxShards - 1)) % shards_.size()];
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.pinned.find(token);
  if (it == s.pinned.end()) return;
  s.alloc->Unpin(it->second);  // release the send-pin
  s.pinned.erase(it);
}

bool RamTier::Contains(const BlockKey& key) const {
  if (shards_.empty()) return false;
  const std::string fn = key.Filename();
  const Shard& s = ShardFor(fn);
  std::lock_guard<std::mutex> lk(s.mu);
  return s.index.find(fn) != s.index.end();
}

bool RamTier::Remove(const BlockKey& key) {
  if (shards_.empty()) return false;
  const std::string fn = key.Filename();
  Shard& s = ShardFor(fn);
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.index.find(fn);
  if (it == s.index.end()) return false;
  // Best-effort for a cache: only a DURABLE, not-in-flight slot can be freed
  // safely. A still-flushing slot holds the flush-pin (the flusher owns it); an
  // in-flight slot holds a send-pin. Either way, decline -- the caller retries.
  if (!it->second.durable) return false;
  bool in_flight = false;
  for (const auto& kv : s.pinned) if (kv.second == fn) { in_flight = true; break; }
  if (in_flight) return false;
  DropLocked(s, fn);
  return true;
}

void RamTier::DropLocked(Shard& s, const std::string& fn) {
  s.alloc->Remove(fn);   // frees the slot (must be unpinned)
  s.index.erase(fn);
}

void RamTier::FlushLoop(Shard& s) {
  for (;;) {
    // Drain up to kFlushBatchMax queued items in one pass (one worker's batch).
    std::vector<QItem> batch;
    {
      std::unique_lock<std::mutex> lk(s.mu);
      s.cv.wait(lk, [&s] { return s.stop || !s.flushq.empty(); });
      if (s.stop && s.flushq.empty()) return;
      while (!s.flushq.empty() && batch.size() < kFlushBatchMax) {
        batch.push_back(std::move(s.flushq.front()));
        s.flushq.pop_front();
      }
    }

    // Snapshot the slots (guaranteed present: queued items are flush-pinned,
    // so they can't be evicted or Removed). live[] marks items still to flush.
    const size_t B = batch.size();
    std::vector<FlushItem> items(B);
    std::vector<char> live(B, 0);
    {
      std::lock_guard<std::mutex> lk(s.mu);
      for (size_t i = 0; i < B; ++i) {
        auto it = s.index.find(batch[i].fn);
        if (it == s.index.end() || it->second.durable) continue;  // gone/already done
        items[i] = FlushItem{batch[i].key, arena_ + it->second.offset,
                             it->second.len, it->second.cap};
        live[i] = 1;
      }
    }

    // Flush: batched sink when wired (one store visit for the whole dequeue),
    // else the per-item sink. Per-item ok/fail semantics identical either way.
    std::vector<char> ok(B, 0);
    if (flush_batch_) {
      std::vector<FlushItem> sub;
      std::vector<size_t> map;
      for (size_t i = 0; i < B; ++i)
        if (live[i]) { sub.push_back(items[i]); map.push_back(i); }
      if (!sub.empty()) {
        std::vector<bool> r = flush_batch_(sub);
        for (size_t m = 0; m < map.size() && m < r.size(); ++m) ok[map[m]] = r[m] ? 1 : 0;
      }
    } else {
      for (size_t i = 0; i < B; ++i)
        if (live[i])
          ok[i] = (flush_ ? flush_(items[i].key, items[i].data, items[i].len, items[i].cap) : true) ? 1 : 0;
    }
    if (!flush_ && !flush_batch_) for (size_t i = 0; i < B; ++i) ok[i] = live[i];

    {
      std::lock_guard<std::mutex> lk(s.mu);
      for (size_t i = 0; i < B; ++i) {
        if (!live[i]) continue;
        auto it = s.index.find(batch[i].fn);
        if (it == s.index.end()) continue;  // defensive
        if (ok[i]) {
          it->second.durable = true;
          s.alloc->Unpin(batch[i].fn);  // release the flush-pin -> now evictable
          flushed_.fetch_add(1, std::memory_order_relaxed);
        } else if (++batch[i].tries < opt_.flush_retries) {
          s.flushq.push_back(std::move(batch[i]));  // retry later
          s.cv.notify_one();
        } else {
          // Give up: drop from RAM (releases flush-pin + frees slot). GET falls
          // back to a miss (recompute) -- correct cache semantics, no arena leak.
          s.alloc->Unpin(batch[i].fn);
          DropLocked(s, batch[i].fn);
          flush_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  }
}

uint64_t RamTier::Evictions() const {
  uint64_t n = 0;
  for (const auto& sh : shards_) n += sh->alloc->Evictions();
  return n;
}

uint64_t RamTier::UsedBytes() const {
  uint64_t n = 0;
  for (const auto& sh : shards_) n += sh->alloc->UsedBytes();
  return n;
}

size_t RamTier::Count() const {
  size_t n = 0;
  for (const auto& sh : shards_) {
    std::lock_guard<std::mutex> lk(sh->mu);
    n += sh->index.size();
  }
  return n;
}

size_t RamTier::FlushBacklog() const {
  size_t n = 0;
  for (const auto& sh : shards_) {
    std::lock_guard<std::mutex> lk(sh->mu);
    n += sh->flushq.size();
  }
  return n;
}

}  // namespace dfkv
