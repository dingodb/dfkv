/* NodeDedup — same-host GET rendezvous over POSIX shared memory.
 *
 * Motivation (phase 5): TP-replicated KV (MLA/DSA latent) is written by one
 * rank but read back by EVERY rank — 8x fabric traffic and 8x server read
 * load per L3 prefix load (measured on GLM-5.2 TP8). The ranks are same-host
 * processes requesting the SAME key set within milliseconds (measured 12 ms
 * spread vs ~44 ms per fetch batch), so a plain look-aside cache misses: by
 * the time rank N checks, rank 0's fetch is still in flight. What dedups the
 * fetch is a RENDEZVOUS: the first arrival claims a key (FETCHING), fetches
 * it remotely and publishes the payload; the others wait briefly and copy —
 * with a hard deadline that falls back to a direct fetch, so no failure mode
 * is worse than today's 8x behavior.
 *
 * Correctness model (lock-free, crash-robust):
 *  - Index slot state machine EMPTY -> FETCHING -> READY guarded by CAS; a
 *    FETCHING older than takeover_ms (fetcher died/hung) can be re-claimed.
 *  - Payload arena is a ring with a MONOTONIC allocation cursor; readers
 *    validate "my payload's lap has not been overwritten" (cursor - alloc_seq
 *    <= arena - len) before AND after the copy, plus a per-slot seqlock
 *    generation for slot reuse. A failed validation is just a miss.
 *  - Nothing here is load-bearing for correctness of the cache itself: every
 *    exit path (no slot, timeout, torn read, size mismatch) degrades to the
 *    caller's normal remote fetch.
 *
 * SCOPE: host-memory destinations only (the SGLang HiCache path). GPUDirect
 * destinations (vLLM connector) must NOT enable this — memcpy to a device VA
 * is invalid; that path needs CUDA IPC and stays with the Phase-2b design.
 * Off by default (DFKV_CLIENT_NODE_DEDUP=1 opts in). */
#ifndef DFKV_NODE_DEDUP_H_
#define DFKV_NODE_DEDUP_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "common/kv_types.h"

namespace dfkv {

class NodeDedup {
 public:
  struct Options {
    std::string name;              // shm name (leading '/'); see MakeName()
    uint64_t arena_bytes = 512ull << 20;   // DFKV_NODE_DEDUP_ARENA_MB
    uint32_t slots = 65536;                // DFKV_NODE_DEDUP_SLOTS (pow2 forced)
    int wait_ms = 500;                     // DFKV_NODE_DEDUP_WAIT_MS
    int takeover_ms = 2000;                // DFKV_NODE_DEDUP_TAKEOVER_MS
    int ttl_ms = 5000;                     // DFKV_NODE_DEDUP_TTL_MS
  };

  // Builds Options from the environment; returns nullptr (feature off) unless
  // DFKV_CLIENT_NODE_DEDUP=1. model_hash namespaces the segment so distinct
  // keyspaces on one host never share entries.
  static std::unique_ptr<NodeDedup> FromEnv(uint64_t model_hash);
  // Opens (or creates) the shm segment. Returns nullptr on any failure — the
  // caller just runs without dedup. A parameter mismatch with an existing
  // segment also returns nullptr (never reinterpret another config's layout).
  static std::unique_ptr<NodeDedup> Open(const Options& opt);
  ~NodeDedup();

  NodeDedup(const NodeDedup&) = delete;
  NodeDedup& operator=(const NodeDedup&) = delete;

  // Per-key verdict for a batch pass.
  enum class Role {
    kHit,    // payload copied into dst (done — skip the remote fetch)
    kFetch,  // caller owns the fetch; MUST call Publish or Abort afterwards
    kWait,   // someone else is fetching; call WaitCopy after issuing own batch
  };

  // Classify one key. n = the exact byte count this caller will read; only a
  // published payload of exactly n bytes is a hit (lockstep peers request the
  // same n; anything else falls back to a direct fetch).
  Role Claim(const BlockKey& key, size_t n, char* dst);

  // Fetch outcomes for keys claimed as kFetch.
  void Publish(const BlockKey& key, const char* data, size_t n);
  void Abort(const BlockKey& key, size_t n);

  // Wait (poll, bounded by wait_ms) for a kWait key and copy it into dst.
  // false = timeout/torn/mismatch — caller fetches directly.
  bool WaitCopy(const BlockKey& key, size_t n, char* dst);

  // diagnostics (relaxed; process-local)
  uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
  uint64_t fetches() const { return fetches_.load(std::memory_order_relaxed); }
  uint64_t wait_hits() const { return wait_hits_.load(std::memory_order_relaxed); }
  uint64_t wait_timeouts() const { return wait_timeouts_.load(std::memory_order_relaxed); }

 private:
  struct Slot;    // 64-byte shm index slot
  struct Header;  // shm header
  NodeDedup() = default;

  Slot* Find(const BlockKey& key, size_t n) const;      // matching live slot or null
  Slot* Reserve(const BlockKey& key, size_t n);         // claim a slot as FETCHING
  bool CopyOut(Slot* s, size_t n, char* dst) const;     // seqlock+lap validated copy
  static uint64_t NowMs();

  Header* hdr_ = nullptr;
  Slot* slots_ = nullptr;
  char* arena_ = nullptr;
  uint64_t arena_bytes_ = 0;
  uint32_t nslots_ = 0;       // power of two
  int wait_ms_ = 500;
  int takeover_ms_ = 2000;
  int ttl_ms_ = 5000;
  void* map_base_ = nullptr;
  size_t map_len_ = 0;

  std::atomic<uint64_t> hits_{0}, fetches_{0}, wait_hits_{0}, wait_timeouts_{0};
};

}  // namespace dfkv

#endif  // DFKV_NODE_DEDUP_H_
