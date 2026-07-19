/* ReadCoalescer — collapse concurrent identical GETs into one disk read.
 *
 * Why: with TP8 + MLA every rank is a separate process fetching the SAME page
 * (same BlockKey, same range) within the same prefetch window; client-side
 * NodeDedup is per-process so eight copies of every read reach the node and,
 * because the disk path is O_DIRECT with no read cache, hit the NVMe eight
 * times (measured 1:1 disk:wire on replay workloads, E11B 2026-07-20). The
 * write path already absorbs this convoy via the RAM write-through arena; this
 * class is the read-side counterpart: the first arrival ("leader") does the
 * disk read into a shared scratch buffer, every overlapping arrival
 * ("follower") blocks until it completes and memcpy()s from the scratch.
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
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

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
  // path to decline io_uring and fall back to the synchronous path, which
  // then joins the flight instead of issuing a duplicate disk read).
  bool InFlight(const BlockKey& bk, uint64_t offset, uint64_t length) {
    Key k{bk.id, bk.index, bk.size, offset, length};
    std::lock_guard<std::mutex> lk(mu_);
    return map_.find(k) != map_.end();
  }

  // Execute `fill(buf, cap, out_len)` exactly once per concurrent set of
  // identical (key, offset, length) readers; copy the result into every
  // caller's dst. Returns the fill status (shared verbatim by followers).
  Status Read(const BlockKey& bk, uint64_t offset, uint64_t length, char* dst,
              size_t dst_cap, size_t* out_len,
              const std::function<Status(char*, size_t, size_t*)>& fill) {
    Key k{bk.id, bk.index, bk.size, offset, length};
    std::shared_ptr<Flight> f;
    bool leader = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = map_.find(k);
      if (it == map_.end()) {
        f = std::make_shared<Flight>();
        map_.emplace(k, f);
        leader = true;
      } else {
        f = it->second;
      }
    }
    if (leader) {
      f->data = std::make_shared<std::vector<char>>(dst_cap);
      size_t n = 0;
      Status st = fill(f->data->data(), dst_cap, &n);
      {
        std::lock_guard<std::mutex> lk(mu_);
        map_.erase(k);  // late arrivals after this start their own flight
      }
      {
        std::lock_guard<std::mutex> fl(f->m);
        f->st = st;
        f->len = n;
        f->done = true;
      }
      f->cv.notify_all();
      leaders_.fetch_add(1, std::memory_order_relaxed);
      if (st == Status::kOk && n > 0) {
        std::memcpy(dst, f->data->data(), n < dst_cap ? n : dst_cap);
        if (out_len) *out_len = n < dst_cap ? n : dst_cap;
      }
      return st;
    }
    // Follower: wait for the leader's read, then copy from the shared buffer.
    std::unique_lock<std::mutex> fl(f->m);
    f->cv.wait(fl, [&] { return f->done; });
    Status st = f->st;
    size_t n = f->len;
    if (st == Status::kOk && n > 0) {
      size_t c = n < dst_cap ? n : dst_cap;
      std::memcpy(dst, f->data->data(), c);
      if (out_len) *out_len = c;
    }
    coalesced_.fetch_add(1, std::memory_order_relaxed);
    return st;
  }

  size_t leaders() const { return leaders_.load(std::memory_order_relaxed); }
  size_t coalesced() const { return coalesced_.load(std::memory_order_relaxed); }

 private:
  struct Flight {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    Status st = Status::kInvalid;
    size_t len = 0;
    std::shared_ptr<std::vector<char>> data;
  };
  std::mutex mu_;
  std::unordered_map<Key, std::shared_ptr<Flight>, KeyHash> map_;
  std::atomic<size_t> leaders_{0}, coalesced_{0};
};

}  // namespace dfkv

#endif  // DFKV_READ_COALESCER_H_
