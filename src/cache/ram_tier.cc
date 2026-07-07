#include "cache/ram_tier.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "utils/log.h"
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
  SlabAllocator::Options ao;
  ao.extent_bytes = ext;
  ao.num_extents = static_cast<uint32_t>(std::max<uint64_t>(1, opt_.bytes / ext));
  ao.align = opt_.slot_granularity;
  ao.max_waste = 0.25;
  alloc_ = std::make_unique<SlabAllocator>(ao);
  const uint32_t nf = opt_.flush_threads ? opt_.flush_threads : 1;
  flushers_.reserve(nf);
  for (uint32_t i = 0; i < nf; ++i)
    flushers_.emplace_back([this] { FlushLoop(); });
}

RamTier::~RamTier() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  flush_cv_.notify_all();
  for (auto& f : flushers_) if (f.joinable()) f.join();
  if (arena_) std::free(arena_);
}

void RamTier::SetArenaMr(void* mr) {
  std::lock_guard<std::mutex> lk(mu_);
  arena_mr_ = mr;
}

bool RamTier::Put(const BlockKey& key, const void* data, size_t len) {
  if (!arena_) return false;
  if (data == nullptr && len != 0) return false;
  const std::string fn = key.Filename();

  uint64_t offset = 0;
  uint32_t cap = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    // Resident, or another thread is mid-copy on this same (content-addressed)
    // key -> dedup. The in-progress writer produces identical bytes, so returning
    // "accepted" here is correct read-after-write for this key.
    if (index_.count(fn) || writing_.count(fn)) return true;
    SlabAllocator::SlotRef ref;
    std::vector<std::string> evicted;
    if (!alloc_->Put(fn, len, &ref, &evicted)) {
      // Backpressure (gap 10.3): arena full of non-evictable (flushing / in-
      // flight) slots. Decline -> caller does the normal synchronous disk write.
      put_bypass_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    for (const auto& ev : evicted) index_.erase(ev);  // evicted are durable+idle
    // Flush-pin the slot: RAM_ONLY, not evictable until the async flush lands.
    alloc_->Pin(fn);
    writing_.insert(fn);  // reserve the key so a concurrent same-key Put dedups
    offset = ref.extent * extent_bytes_ + ref.offset;  // global arena offset
    cap = ref.slot_size;                               // (ref.offset is within-extent)
    // NOT visible yet: index_ install is deferred until after the copy, so a
    // concurrent GetPrep can't read the slot mid-copy.
  }

  std::memcpy(arena_ + offset, data, len);  // copy outside the lock (MB-scale)

  {
    std::lock_guard<std::mutex> lk(mu_);
    Entry e;
    e.offset = offset;
    e.len = static_cast<uint32_t>(len);
    e.cap = cap;
    e.durable = false;
    index_[fn] = e;
    writing_.erase(fn);  // copy done -> now visible to GetPrep
    flushq_.push_back(QItem{fn, key, 0});
  }
  flush_cv_.notify_one();
  puts_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool RamTier::GetPrep(const BlockKey& key, uint64_t offset, uint64_t length,
                      Hit* out) {
  const std::string fn = key.Filename();
  std::lock_guard<std::mutex> lk(mu_);
  auto it = index_.find(fn);
  if (it == index_.end()) {
    misses_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  const Entry& e = it->second;
  // Send-pin: the RDMA send reads the shared arena in place; the slot must not
  // be evicted/reused until the send completes (gap 10.1).
  alloc_->Pin(fn);
  const uint64_t start = std::min<uint64_t>(offset, e.len);
  const uint64_t avail = e.len - start;
  const uint64_t n = std::min(length ? length : avail, avail);
  if (out) {
    out->ptr = arena_ + e.offset + start;
    out->len = static_cast<size_t>(n);
    out->mr = arena_mr_;
    out->token = next_token_++;
    pinned_[out->token] = fn;
  }
  hits_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void RamTier::Release(uint64_t token) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = pinned_.find(token);
  if (it == pinned_.end()) return;
  alloc_->Unpin(it->second);  // release the send-pin
  pinned_.erase(it);
}

bool RamTier::Contains(const BlockKey& key) const {
  std::lock_guard<std::mutex> lk(mu_);
  return index_.find(key.Filename()) != index_.end();
}

bool RamTier::Remove(const BlockKey& key) {
  const std::string fn = key.Filename();
  std::lock_guard<std::mutex> lk(mu_);
  auto it = index_.find(fn);
  if (it == index_.end()) return false;
  // Best-effort for a cache: only a DURABLE, not-in-flight slot can be freed
  // safely. A still-flushing slot holds the flush-pin (the flusher owns it); an
  // in-flight slot holds a send-pin. Either way, decline -- the caller retries.
  if (!it->second.durable) return false;
  bool in_flight = false;
  for (const auto& kv : pinned_) if (kv.second == fn) { in_flight = true; break; }
  if (in_flight) return false;
  DropLocked(fn);
  return true;
}

void RamTier::DropLocked(const std::string& fn) {
  alloc_->Remove(fn);   // frees the slot (must be unpinned)
  index_.erase(fn);
}

void RamTier::FlushLoop() {
  for (;;) {
    QItem item;
    {
      std::unique_lock<std::mutex> lk(mu_);
      flush_cv_.wait(lk, [this] { return stop_ || !flushq_.empty(); });
      if (stop_ && flushq_.empty()) return;
      item = std::move(flushq_.front());
      flushq_.pop_front();
    }

    // Snapshot the slot (guaranteed present: a queued item is flush-pinned, so
    // it can't be evicted or Removed).
    char* data = nullptr;
    size_t len = 0, cap = 0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = index_.find(item.fn);
      if (it == index_.end() || it->second.durable) continue;  // gone/already done
      data = arena_ + it->second.offset;
      len = it->second.len;
      cap = it->second.cap;
    }

    const bool ok = flush_ ? flush_(item.key, data, len, cap) : true;

    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = index_.find(item.fn);
      if (it == index_.end()) continue;  // defensive
      if (ok) {
        it->second.durable = true;
        alloc_->Unpin(item.fn);  // release the flush-pin -> now evictable
        flushed_.fetch_add(1, std::memory_order_relaxed);
      } else if (++item.tries < opt_.flush_retries) {
        flushq_.push_back(std::move(item));  // retry later
        flush_cv_.notify_one();
      } else {
        // Give up: drop from RAM (releases flush-pin + frees slot). GET falls
        // back to a miss (recompute) -- correct cache semantics, no arena leak.
        alloc_->Unpin(item.fn);
        DropLocked(item.fn);
        flush_dropped_.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
}

size_t RamTier::Count() const {
  std::lock_guard<std::mutex> lk(mu_);
  return index_.size();
}

size_t RamTier::FlushBacklog() const {
  std::lock_guard<std::mutex> lk(mu_);
  return flushq_.size();
}

}  // namespace dfkv
