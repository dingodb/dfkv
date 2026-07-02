#include "cache/ram_tier.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

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

  SlabAllocator::Options ao;
  ao.extent_bytes = opt_.bytes;   // the whole arena is one extent
  ao.num_extents = 1;
  ao.align = opt_.slot_granularity;
  ao.max_waste = 0.25;
  alloc_ = std::make_unique<SlabAllocator>(ao);
  flusher_ = std::thread([this] { FlushLoop(); });
}

RamTier::~RamTier() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  flush_cv_.notify_all();
  if (flusher_.joinable()) flusher_.join();
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
    offset = ref.offset;
    // NOT visible yet: index_ install is deferred until after the copy, so a
    // concurrent GetPrep can't read the slot mid-copy.
  }

  std::memcpy(arena_ + offset, data, len);  // copy outside the lock (MB-scale)

  {
    std::lock_guard<std::mutex> lk(mu_);
    Entry e;
    e.offset = offset;
    e.len = static_cast<uint32_t>(len);
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
    const void* data = nullptr;
    size_t len = 0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = index_.find(item.fn);
      if (it == index_.end() || it->second.durable) continue;  // gone/already done
      data = arena_ + it->second.offset;
      len = it->second.len;
    }

    const bool ok = flush_ ? flush_(item.key, data, len) : true;

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
