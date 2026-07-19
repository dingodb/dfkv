/* ReadCoalescer — collapse concurrent identical GETs into one disk read.
 *
 * Why: with TP8 + MLA every rank is a separate process fetching the SAME page
 * (same BlockKey, same range) within the same prefetch window; client-side
 * NodeDedup is per-process so eight copies of every read reach the node and,
 * because the disk path is O_DIRECT with no read cache, hit the NVMe eight
 * times (measured 1:1 disk:wire on replay workloads, E11B 2026-07-20). The
 * write path already absorbs this convoy via the RAM write-through arena; this
 * class is the read-side counterpart, and it registers BOTH read paths:
 *
 *  - Synchronous (TCP kRange, RangeInto, sync RangeDirect): Read() — the first
 *    arrival ("leader") does the disk read into a shared scratch buffer, every
 *    overlapping arrival ("follower") blocks until it completes and memcpy()s
 *    from the scratch.
 *  - Asynchronous (the io_uring serve path, the production default since
 *    v1.27.0): TryRegisterAsync() at prep time creates the flight WITHOUT
 *    performing the read; the serve loop does the read out-of-band and calls
 *    CompleteAsync() with the payload, which hands it to any waiters. A
 *    duplicate GET's prep declines (token 0 / InFlight), falls back to the
 *    sync path, and joins the flight through Read() like any follower.
 *
 * A follower waits at most WaitMs() (DFKV_READ_COALESCE_TIMEOUT_MS, default
 * 500 ms) and falls back to its OWN disk read on timeout or on a failed/
 * abandoned flight — a leader connection can be torn down mid-batch (ring
 * poisoned, batch fail), so a waiter must never be able to hang forever.
 * A follower whose flight was registered by ITS OWN serve thread (a pipelined
 * duplicate on one connection: the leader's read has not been submitted yet)
 * skips waiting entirely and reads itself — waiting would deadlock.
 *
 * Blocking a follower is never worse than the duplicate pread it replaces
 * (both are bounded by one disk read), so this is safe on thread-per-conn
 * TCP and on the RDMA serve loop, which already blocks in RangeDirect.
 *
 * Opt-in via env DFKV_READ_COALESCE=1 (checked by the caller, not here).
 */
#ifndef DFKV_READ_COALESCER_H_
#define DFKV_READ_COALESCER_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common/kv_types.h"
#include "common/status.h"

namespace dfkv {

class ReadCoalescer {
 public:
  struct Key {
    uint64_t id;
    uint32_t index, ksize;
    uint64_t offset, length;
    bool operator==(const Key& o) const {
      return id == o.id && index == o.index && ksize == o.ksize &&
             offset == o.offset && length == o.length;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const {
      size_t h = std::hash<uint64_t>()(k.id);
      h ^= std::hash<uint64_t>()((static_cast<uint64_t>(k.index) << 32) ^ k.ksize) +
           0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h ^= std::hash<uint64_t>()(k.offset ^ (k.length << 1)) + 0x9e3779b97f4a7c15ULL +
           (h << 6) + (h >> 2);
      return h;
    }
  };

  // True if an identical read is currently in flight (used by the async-prep
  // path as a cheap early decline, before the index lookup + fd open).
  bool InFlight(const BlockKey& bk, uint64_t offset, uint64_t length) {
    Key k{bk.id, bk.index, bk.size, offset, length};
    std::lock_guard<std::mutex> lk(mu_);
    return map_.find(k) != map_.end();
  }

  // Register a flight for a read the CALLER will perform out-of-band (io_uring).
  // Returns a non-zero completion token, or 0 if an identical read is already in
  // flight — the caller then declines the async prep and joins via Read().
  // `whole_value` records whether this read covers the entire stored value
  // (offset 0, full length): only such reads are eligible for RAM promotion.
  uint64_t TryRegisterAsync(const BlockKey& bk, uint64_t offset, uint64_t length,
                            bool whole_value) {
    Key k{bk.id, bk.index, bk.size, offset, length};
    std::lock_guard<std::mutex> lk(mu_);
    if (map_.find(k) != map_.end()) return 0;
    auto f = std::make_shared<Flight>();
    f->key = k;
    f->whole = whole_value;
    f->leader_tid = std::this_thread::get_id();
    map_.emplace(k, f);
    const uint64_t t = ++token_seq_;
    tokens_.emplace(t, std::move(f));
    return t;
  }

  // Complete (or abort) an async flight. On st==kOk with data, any waiters are
  // handed a copy of the payload; on any other status (or abort: data==nullptr)
  // waiters fall back to their own disk reads. Returns true if at least one
  // waiter had joined — the fan-in evidence the caller's RAM-promotion
  // admission gate keys on — and fills *key / *whole from registration.
  // Idempotent per token; unknown tokens return false.
  bool CompleteAsync(uint64_t token, Status st, const char* data, size_t len,
                     BlockKey* key = nullptr, bool* whole = nullptr) {
    std::shared_ptr<Flight> f;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = tokens_.find(token);
      if (it == tokens_.end()) return false;
      f = std::move(it->second);
      tokens_.erase(it);
      auto mit = map_.find(f->key);
      if (mit != map_.end() && mit->second == f) map_.erase(mit);
    }
    const bool waiters = f->waiters.load(std::memory_order_acquire) > 0;
    if (waiters && st == Status::kOk && data && len) {
      f->data = AlignedAlloc(len);
      if (f->data) std::memcpy(f->data.get(), data, len);
    }
    {
      std::lock_guard<std::mutex> fl(f->m);
      f->st = st;
      f->len = len;
      f->done = true;
    }
    f->cv.notify_all();
    leaders_.fetch_add(1, std::memory_order_relaxed);
    if (key) *key = BlockKey{f->key.id, f->key.index, f->key.ksize};
    if (whole) *whole = f->whole;
    return waiters;
  }

  // Leader/follower outcome of a Read() call, for the caller's promotion gate
  // on the synchronous path (only meaningful when leader is true).
  struct Outcome {
    bool leader = false;
    bool had_waiters = false;
  };

  // Execute `fill(buf, cap, out_len)` exactly once per concurrent set of
  // identical (key, offset, length) readers; copy the result into every
  // caller's dst. A follower that times out, joins its own thread's pending
  // flight, or sees a failed/aborted flight runs its own fill instead.
  Status Read(const BlockKey& bk, uint64_t offset, uint64_t length, char* dst,
              size_t dst_cap, size_t* out_len,
              const std::function<Status(char*, size_t, size_t*)>& fill,
              Outcome* oc = nullptr) {
    Key k{bk.id, bk.index, bk.size, offset, length};
    std::shared_ptr<Flight> f;
    bool leader = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = map_.find(k);
      if (it == map_.end()) {
        f = std::make_shared<Flight>();
        f->key = k;
        f->leader_tid = std::this_thread::get_id();
        map_.emplace(k, f);
        leader = true;
      } else {
        f = it->second;
      }
    }
    if (leader) {
      // Aligned scratch: the fill may be an O_DIRECT read straight into it.
      f->data = AlignedAlloc(dst_cap);
      size_t n = 0;
      Status st = f->data ? fill(f->data.get(), dst_cap, &n)
                          : fill(dst, dst_cap, &n);  // alloc failed: no sharing
      {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(k);
        if (it != map_.end() && it->second == f) map_.erase(it);
      }
      {
        std::lock_guard<std::mutex> fl(f->m);
        f->st = st;
        f->len = n;
        f->done = true;
      }
      f->cv.notify_all();
      leaders_.fetch_add(1, std::memory_order_relaxed);
      if (oc) {
        oc->leader = true;
        oc->had_waiters = f->waiters.load(std::memory_order_acquire) > 0;
      }
      if (st == Status::kOk && n > 0 && f->data) {
        std::memcpy(dst, f->data.get(), n < dst_cap ? n : dst_cap);
        if (out_len) *out_len = n < dst_cap ? n : dst_cap;
      } else if (st == Status::kOk && out_len && !f->data) {
        // fill wrote dst directly; *out_len was set by fill via &n above.
        *out_len = n;
      }
      return st;
    }
    // A pipelined duplicate on the flight's own serve thread: the leader's read
    // has not been submitted yet (async prep registers before the uring batch
    // runs), so waiting here would deadlock. Read it ourselves.
    if (f->leader_tid == std::this_thread::get_id())
      return fill(dst, dst_cap, out_len);
    // Follower: wait (bounded) for the leader, then copy from the shared
    // buffer; on timeout / failure / abort, fall back to our own read.
    f->waiters.fetch_add(1, std::memory_order_acq_rel);
    {
      std::unique_lock<std::mutex> fl(f->m);
      if (!f->cv.wait_for(fl, std::chrono::milliseconds(WaitMs()),
                          [&] { return f->done; })) {
        timeouts_.fetch_add(1, std::memory_order_relaxed);
        fl.unlock();
        return fill(dst, dst_cap, out_len);
      }
    }
    if (f->st != Status::kOk || !f->data) return fill(dst, dst_cap, out_len);
    const size_t c = f->len < dst_cap ? f->len : dst_cap;
    if (c) std::memcpy(dst, f->data.get(), c);
    if (out_len) *out_len = c;
    coalesced_.fetch_add(1, std::memory_order_relaxed);
    return Status::kOk;
  }

  size_t leaders() const { return leaders_.load(std::memory_order_relaxed); }
  size_t coalesced() const { return coalesced_.load(std::memory_order_relaxed); }
  size_t timeouts() const { return timeouts_.load(std::memory_order_relaxed); }

 private:
  struct Flight {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    Status st = Status::kInvalid;
    size_t len = 0;
    std::shared_ptr<char> data;  // 4096-aligned scratch (leader's read target)
    Key key{};
    bool whole = false;
    std::thread::id leader_tid;
    std::atomic<int> waiters{0};
  };

  static std::shared_ptr<char> AlignedAlloc(size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, 4096, n ? n : 4096) != 0 || !p) return nullptr;
    return std::shared_ptr<char>(static_cast<char*>(p),
                                 [](char* q) { std::free(q); });
  }

  // Follower wait bound. A leader connection can die mid-batch with the flight
  // never completed (every known path aborts it, but a waiter must not bet its
  // liveness on that); 500 ms is ~2 orders above a healthy NVMe read.
  static int WaitMs() {
    static const int ms = [] {
      const char* e = std::getenv("DFKV_READ_COALESCE_TIMEOUT_MS");
      if (e && *e) {
        long v = std::strtol(e, nullptr, 10);
        if (v >= 10 && v <= 60000) return static_cast<int>(v);
      }
      return 500;
    }();
    return ms;
  }

  std::mutex mu_;
  std::unordered_map<Key, std::shared_ptr<Flight>, KeyHash> map_;
  std::unordered_map<uint64_t, std::shared_ptr<Flight>> tokens_;
  uint64_t token_seq_ = 0;
  std::atomic<size_t> leaders_{0}, coalesced_{0}, timeouts_{0};
};

}  // namespace dfkv

#endif  // DFKV_READ_COALESCER_H_
