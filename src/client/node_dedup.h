/* NodeDedup — same-host GET/EXIST rendezvous over POSIX shared memory.
 *
 * Motivation (phase 5/6): TP-replicated KV (MLA/DSA latent) is written by one
 * rank but read back by EVERY rank — 8x fabric traffic and 8x server read
 * load per L3 prefix load (measured on GLM-5.2 TP8). The ranks are same-host
 * processes requesting the SAME key set within milliseconds (measured 12 ms
 * spread vs ~44 ms per fetch batch), so a plain look-aside cache misses: by
 * the time rank N checks, rank 0's fetch is still in flight. What dedups the
 * fetch is a RENDEZVOUS: the first arrival claims a key (FETCHING), fetches
 * it remotely and publishes; the others wait briefly and copy — with a hard
 * deadline that falls back to a direct fetch, so no failure mode is worse
 * than today's 8x behavior. EXIST probes (2 per page per rank) ride the same
 * machinery with a 1-byte payload; a stale exist answer is safe either way
 * (stale absent -> recompute, stale present -> GET miss -> recompute).
 *
 * Correctness model (lock-free, crash-robust):
 *  - Index slot identity is (key, kind); the payload length is an attribute
 *    (strict readers require len == n, auto readers accept len <= cap).
 *  - State machine EMPTY -> FETCHING -> READY guarded by CAS; a FETCHING
 *    older than takeover_ms (fetcher died/hung) can be re-claimed.
 *  - Payload arena is a ring with a MONOTONIC allocation cursor; readers
 *    validate "my payload's lap has not been overwritten" (cursor - alloc_seq
 *    <= arena - len) before AND after the copy, plus a per-slot seqlock
 *    generation for slot reuse. A failed validation is just a miss.
 *  - Nothing here is load-bearing: every exit path (no slot, timeout, torn
 *    read, size mismatch) degrades to the caller's normal remote op.
 *
 * SCOPE: host-memory destinations (the SGLang HiCache path). GPUDirect
 * destinations are ROUTED AWAY from this class by the batch wrappers (a
 * memcpy to a device VA is invalid) and served by the CUDA-IPC flavor in
 * node_dedup_gpu.h (phase 2b, DFKV_CLIENT_NODE_DEDUP_GPU=1 opts in on top).
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
    std::string name;              // shm name (leading '/'); see FromEnv()
    uint64_t arena_bytes = 512ull << 20;   // DFKV_NODE_DEDUP_ARENA_MB
    uint32_t slots = 65536;                // DFKV_NODE_DEDUP_SLOTS (pow2 forced)
    int wait_ms = 500;                     // DFKV_NODE_DEDUP_WAIT_MS
    int takeover_ms = 2000;                // DFKV_NODE_DEDUP_TAKEOVER_MS
    int ttl_ms = 5000;                     // DFKV_NODE_DEDUP_TTL_MS
    // Cooperative fetch (DFKV_CLIENT_NODE_DEDUP=coop): keys are partitioned
    // deterministically by hash(key) % world, and only the owning rank ever
    // claims a fetch (others go straight to the wait path). Motivation: the
    // classic claim RACE collapses under a real inference batch — the ranks'
    // arrival spread (~12 ms) is far below a fetch batch (~44 ms), so the
    // first-arriving rank claims essentially the WHOLE batch and the other 7
    // serialize behind ONE fetcher (measured live: only r0/r1 ever touched the
    // server; hot rounds went net-negative, v1.28.0 release notes). The
    // partition keeps the 1x fabric traffic of the rendezvous AND N-way fetch
    // parallelism: every rank fetches 1/N of the batch concurrently, publishes
    // its shard, and copies the rest from shm.
    bool coop = false;
    int rank = -1;                 // this process's TP rank (coop only)
    int world = 0;                 // TP size (coop only; coop needs world >= 2)
  };

  // Entry namespaces sharing one index: a key's data payload and its exist
  // answer are distinct entries.
  enum class Kind : uint32_t { kData = 1, kExist = 2 };

  // Builds Options from the environment; returns nullptr (feature off) unless
  // DFKV_CLIENT_NODE_DEDUP is "1" (classic claim-race) or "coop" (cooperative
  // partition; needs rank/world from the client identity — with world < 2 or an
  // unknown rank it degrades to classic). model_hash namespaces the segment so
  // distinct keyspaces on one host never share entries.
  static std::unique_ptr<NodeDedup> FromEnv(uint64_t model_hash, int rank = -1,
                                            int world = 0);
  // The env-derived segment name. The shm LAYOUT VERSION is part of the name:
  // a layout change must never collide with a segment an older lib left
  // behind — v1.23.0 shipped a layout bump behind the same name, the header
  // check "safely" refused the stale segment, and the feature silently
  // disabled itself across the whole upgrade (canary re-measured 8x). The
  // header magic stays as a backstop only. Exposed so tests clean up the
  // exact segment FromEnv would use.
  static std::string EnvSegmentName(uint64_t model_hash);
  // Opens (or creates) the shm segment. Returns nullptr on any failure — the
  // caller just runs without dedup. A parameter mismatch with an existing
  // segment also returns nullptr (never reinterpret another config's layout).
  static std::unique_ptr<NodeDedup> Open(const Options& opt);
  ~NodeDedup();

  NodeDedup(const NodeDedup&) = delete;
  NodeDedup& operator=(const NodeDedup&) = delete;

  // Per-key verdict for a batch pass.
  enum class Role {
    kHit,    // result delivered (skip the remote op)
    kFetch,  // caller owns the remote op; MUST call Publish or Abort afterwards
    kWait,   // someone else is fetching; call the matching Wait* afterwards
  };

  // Strict GET (BatchGet semantics: exact n bytes). A published payload of a
  // different length is not a hit (lockstep peers request the same n).
  Role Claim(const BlockKey& key, size_t n, char* dst);
  bool WaitCopy(const BlockKey& key, size_t n, char* dst);

  // Variable-size GET (BatchGetAuto semantics): hit iff published len <= cap;
  // *got receives the payload length.
  Role ClaimAuto(const BlockKey& key, size_t cap, char* dst, size_t* got);
  bool WaitCopyAuto(const BlockKey& key, size_t cap, char* dst, size_t* got);

  // EXIST probe (1-byte answer). Stale answers are bounded by ttl_ms and safe
  // in both directions (miss -> recompute; false present -> GET miss).
  Role ClaimExist(const BlockKey& key, bool* val);
  bool WaitExist(const BlockKey& key, bool* val);

  // ---- cooperative partition (Options::coop) ----
  bool coop() const { return coop_; }
  // Deterministic key ownership: FNV-1a(key filename) % world == rank. Every
  // rank computes the same answer, so exactly one rank fetches each key.
  bool Owns(const BlockKey& key) const;
  // Non-owner claim: READY -> kHit, missing/FETCHING -> kWait (never reserves
  // a slot; the owner may not have arrived yet and the wait path covers that).
  // The one exception: a FETCHING slot stale beyond takeover_ms is taken over
  // (kFetch) so a crashed owner cannot park its keys forever.
  Role ClaimPassive(const BlockKey& key, size_t n, char* dst);
  Role ClaimAutoPassive(const BlockKey& key, size_t cap, char* dst, size_t* got);
  Role ClaimExistPassive(const BlockKey& key, bool* val);

  // Fetch outcomes for keys claimed as kFetch (kind selects the namespace;
  // exist publishes a 1-byte payload).
  void Publish(const BlockKey& key, Kind kind, const char* data, size_t n);
  void Abort(const BlockKey& key, Kind kind);

  // diagnostics (relaxed; process-local)
  uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
  uint64_t fetches() const { return fetches_.load(std::memory_order_relaxed); }
  uint64_t wait_hits() const { return wait_hits_.load(std::memory_order_relaxed); }
  uint64_t wait_timeouts() const { return wait_timeouts_.load(std::memory_order_relaxed); }

 private:
  struct Slot;    // 64-byte shm index slot
  struct Header;  // shm header
  NodeDedup() = default;

  Slot* Find(const BlockKey& key, Kind kind) const;
  Slot* Reserve(const BlockKey& key, Kind kind);
  // Validated copy. strict_n: exact length required (SIZE_MAX = any <= cap).
  // Returns payload length via *got on success.
  bool CopyOut(Slot* s, size_t cap, size_t strict_n, char* dst, size_t* got) const;
  Role ClaimImpl(const BlockKey& key, Kind kind, size_t cap, size_t strict_n,
                 char* dst, size_t* got);
  Role ClaimPassiveImpl(const BlockKey& key, Kind kind, size_t cap,
                        size_t strict_n, char* dst, size_t* got);
  bool WaitImpl(const BlockKey& key, Kind kind, size_t cap, size_t strict_n,
                char* dst, size_t* got);
  static uint64_t NowMs();

  Header* hdr_ = nullptr;
  Slot* slots_ = nullptr;
  char* arena_ = nullptr;
  uint64_t arena_bytes_ = 0;
  uint32_t nslots_ = 0;       // power of two
  int wait_ms_ = 500;
  int takeover_ms_ = 2000;
  int ttl_ms_ = 5000;
  bool coop_ = false;
  int rank_ = -1;
  int world_ = 0;
  void* map_base_ = nullptr;
  size_t map_len_ = 0;

  std::atomic<uint64_t> hits_{0}, fetches_{0}, wait_hits_{0}, wait_timeouts_{0};
};

}  // namespace dfkv

#endif  // DFKV_NODE_DEDUP_H_
