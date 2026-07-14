/* SlabAllocator — media-agnostic slot lifecycle manager for the slab store.
 *
 * It owns the LAYOUT (which extent, which offset) of fixed-size slots, NOT the
 * bytes: the caller (DiskSlabStore, or the P3 RamTier) maps a returned SlotRef
 * to its physical medium (an extent file's offset, or a RAM arena offset) and
 * does the I/O. Keeping this pure-logic makes it hermetically unit-testable and
 * lets one implementation back both media (dfkv's slab-store rework and the RAM
 * hot tier share the same size-class + CLOCK + pin machinery).
 *
 * Model (memcached-style slab): a fixed pool of equal-size EXTENTS; each extent
 * is bound on demand to ONE size CLASS and carved into uniform slots. A value of
 * `len` bytes goes into the smallest class whose slot_size >= align_up(len) with
 * bounded internal waste (else a new class is created). Reclamation is per-class
 * CLOCK second-chance; when a needy class has no slots and the pool is empty, an
 * entirely-unpinned extent is stolen from another class (its slots evicted) and
 * re-bound. A pinned slot (in-flight RDMA send) is never evicted.
 *
 * Concurrency: one mutex. The ops are O(1) in-memory; the slow medium I/O runs
 * OUTSIDE this class in the caller. (Per-shard striping is a later perf option;
 * the single lock is correct and TSan-clean.) */
#ifndef DFKV_SLAB_ALLOCATOR_H_
#define DFKV_SLAB_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace dfkv {

class SlabAllocator {
 public:
  // Physical placement of a key's value. `offset` is within extent `extent`.
  struct SlotRef {
    uint32_t cls = 0;         // size-class index (diagnostic)
    uint32_t extent = 0;      // extent index [0, num_extents)
    uint32_t slot = 0;        // slot index within the extent
    uint32_t slot_size = 0;   // class slot size in bytes (>= requested, aligned)
    uint64_t offset = 0;      // byte offset within the extent (= slot * slot_size)
    bool valid() const { return slot_size != 0; }
  };

  struct Options {
    uint64_t extent_bytes = (1ull << 30);  // bytes per extent (e.g. 1 GiB)
    uint32_t num_extents = 8;              // total extents in the pool
    uint32_t align = 4096;                 // slot-size alignment (O_DIRECT friendly)
    // Reuse an existing (larger) class instead of creating a new one when the
    // internal waste (slot_size - aligned_len) stays under this fraction.
    double max_waste = 0.25;
    // Fired (under the allocator lock) whenever a RUNTIME bind hands extent E to
    // a class -- initial bind, steal, or pool re-bind. A persistent caller MUST
    // wipe E's stale metadata here: records left by a PREVIOUS class survive a
    // rebind (they sit at slot-grid positions the new class never overwrites),
    // and a later rebuild can resurrect an old key pointing at the new
    // occupant's bytes. NOT fired on Restore's rebuild-time binds (those records
    // are the source of truth being read). Optional; RamTier (no persistence)
    // leaves it empty.
    std::function<void(uint32_t extent)> on_extent_bind;
  };

  // Target write-parallelism per class: Put keeps up to this many extents open
  // (bound with free slots) so concurrent writers stripe across inodes. Public
  // because it is also the natural per-class capacity FLOOR for external
  // rebalance policy (never shrink a donor below its striping width).
  static constexpr size_t kStripeWays = 8;

  explicit SlabAllocator(Options opt);

  // Reserve a slot for `key` holding `len` bytes.
  // - Idempotent: if `key` is already resident, *out is its current slot and no
  //   eviction happens (returns true).
  // - Under capacity pressure, evicts unpinned slots to make room; the evicted
  //   keys are appended to `*evicted` (caller drops them from its own index / I/O).
  // - Returns false only if `len` exceeds one extent, or every candidate slot is
  //   pinned (no room can be freed). *out is left invalid on false.
  bool Put(const std::string& key, size_t len, SlotRef* out,
           std::vector<std::string>* evicted);

  // Look up `key`; on hit sets *out and marks it referenced (CLOCK second chance).
  bool Get(const std::string& key, SlotRef* out);
  // Get + Pin in ONE lock acquisition -- the read path does both back-to-back on
  // every GET, and at high op rates the second acquisition is pure contention.
  bool GetAndPin(const std::string& key, SlotRef* out);
  bool Contains(const std::string& key) const;

  // Crash-recovery only: install a key KNOWN (from persistence) to occupy
  // (extent, slot) in a class of `slot_size`. Binds the extent to that class if
  // needed and reserves the exact slot. Must be called before any Put, and the
  // (extent, slot) triples restored must be internally consistent (one class per
  // extent). Returns false on an inconsistency (mixed class on an extent, or a
  // slot out of range) -- the caller then discards that record.
  bool Restore(const std::string& key, uint32_t slot_size, uint32_t extent,
               uint32_t slot);

  // Drop `key`. true if present. CONTRACT: the slot is freed for reuse
  // IMMEDIATELY, even if the key is pinned -- pins only block EVICTION, not an
  // explicit Remove. Callers that do I/O against a slot outside their own lock
  // must therefore never Remove an in-flight key: DiskSlabStore defers such
  // Removes to the last releaser (inflight_/deferred_remove_), and RamTier's
  // Remove declines while a flush-pin or send-pin is held. (An earlier version
  // of this comment promised deferred reuse; the implementation never did that,
  // and the guards belong at the caller layer, where in-flight state lives.)
  // A pinned Remove logs a WARN: in-tree callers make it unreachable, so one
  // firing means a new caller is missing its guard.
  bool Remove(const std::string& key);

  // Pin/unpin a resident key: a pinned slot is never chosen for eviction (used
  // while an RDMA send reads the slot). Balanced calls; Pin on an absent key is
  // a no-op returning false.
  bool Pin(const std::string& key);
  bool Unpin(const std::string& key);

  // Per-class occupancy snapshot for an external reclaimer thread. Indices are
  // stable (classes_ only grows), so a caller can diff `puts` across polls to
  // detect write demand per class.
  struct ClassStat {
    uint32_t slot_size = 0;
    size_t free_slots = 0;   // free slots across this class's bound extents
    size_t resident = 0;     // resident keys
    uint64_t puts = 0;       // cumulative new-key inserts (idempotent hits excluded)
    uint32_t extents = 0;    // extents currently bound to this class
    uint32_t slots_per_extent = 0;
  };
  std::vector<ClassStat> Classes() const;
  uint32_t PoolExtents() const;  // unbound extents in the shared pool

  // Evict up to `max_victims` unpinned entries (CLOCK order) from class
  // `cls_index` until it holds >= `target_free` free slots. Runs the same sweep
  // Put would run inline, but from a background thread in bounded batches, so
  // the Put fast path stays pop-free-slot. No-op while the shared pool still
  // has unbound extents (Put grows the class by binding one -- cheap, no
  // residency loss; eviction only earns its keep on a full store). Stops early
  // if an eviction returned a fully-free extent to the pool (free count went
  // DOWN: reclaiming further would cascade-shrink the class, and the pool
  // gained capacity anyway). Returns the number evicted; keys are appended to
  // *evicted for the caller to drop from its own index.
  size_t ReclaimClass(size_t cls_index, size_t target_free, size_t max_victims,
                      std::vector<std::string>* evicted);

  // Move ONE entirely-unpinned extent from `donor_cls` to `target_cls`: evict
  // the donor extent's residents (appended to *evicted), unbind, re-bind to the
  // target. The rebalance mechanism for a demand shift on a full store -- the
  // POLICY (which class is hot, which donor is cold, floors, rate) lives in the
  // caller's reclaim tick, which is the layer that sees per-interval demand.
  // Prefers the donor's emptiest eligible extent. Returns false when donor ==
  // target, either index is invalid, the target's slot exceeds an extent, or
  // the donor has no fully-unpinned extent.
  bool StealFrom(size_t donor_cls, size_t target_cls,
                 std::vector<std::string>* evicted);

  // stats (diagnostic)
  size_t Count() const;
  uint64_t UsedBytes() const;       // sum of slot_size over resident keys
  uint64_t Capacity() const;        // num_extents * extent_bytes
  uint64_t Evictions() const;
  uint64_t Steals() const;          // cross-class extent steals (capacity churn signal)
  uint64_t ExtentReturns() const;   // fully-free extents unbound back to the pool
  size_t ClassCount() const;
  uint32_t BoundExtents() const;    // extents currently carved to a class

 private:
  static constexpr int kUnbound = -1;

  struct Slot { uint32_t extent; uint32_t slot; };
  struct Entry {
    SlotRef ref;
    uint32_t refs = 0;                       // pin count
    bool referenced = false;                 // CLOCK bit
    std::list<std::string>::iterator ring_it;  // this key's node in its class ring
    // This key's node in its extent's resident list (points at the index_ map
    // key, which is node-stable). Lets StealExtentFor enumerate one extent's
    // residents in O(residents) instead of scanning the whole index.
    std::list<const std::string*>::iterator ext_it;
  };
  struct Class {
    uint32_t slot_size = 0;
    uint32_t slots_per_extent = 0;
    // Free slots BUCKETED PER EXTENT with a round-robin cursor, so consecutive
    // Puts land on different extents (different files). A single free stack
    // hands out one extent's slots back-to-back, funneling every concurrent
    // writer into one inode -- and buffered writes to one file serialize on the
    // kernel's per-inode lock (measured 2.1 vs 18.2 GB/s at 8 writers on XFS).
    std::unordered_map<uint32_t, std::vector<uint32_t>> free_by_ext;
    std::vector<uint32_t> ext_rr;              // extents with >=1 free slot
    size_t rr_next = 0;                        // rotation cursor into ext_rr
    size_t free_count = 0;                     // total free slots (== sum over free_by_ext)
    uint32_t bound_extents = 0;                // extents currently bound to this class
    uint64_t puts = 0;                         // cumulative new-key inserts (demand signal)
    std::list<std::string> ring;               // CLOCK ring of resident keys
    std::list<std::string>::iterator hand;     // persistent eviction cursor
    Class() : hand(ring.end()) {}
  };
  struct ExtentMeta {
    int cls = kUnbound;      // class index bound to, or kUnbound (in pool)
    uint32_t free_slots = 0; // free slots (== total when fully empty)
    uint32_t total_slots = 0;
    uint32_t pinned = 0;     // resident pinned slots in this extent (steal guard)
    // Keys resident in this extent (pointers into index_'s node-stable keys);
    // per-key node handle lives in Entry::ext_it. Empty iff fully free.
    std::list<const std::string*> residents;
  };

  size_t ClassForLen(size_t aligned_len);  // returns class index (creates if needed)
  size_t ClassForExactSize(uint32_t slot_size);  // find/create a class of exactly slot_size
  bool BindFreeExtent(size_t cls);         // bind a pool extent to cls; false if none
  bool EvictOneFrom(size_t cls, std::vector<std::string>* evicted);  // CLOCK evict 1
  // Steal an entirely-unpinned extent from another class and rebind it to cls.
  // min_donor_extents > 0 restricts donors to classes holding MORE than that
  // many extents (the growth-first path uses kStripeWays: two under-provisioned
  // classes must not ping-pong extents off each other). 0 = any donor (the
  // last-resort semantics this function always had).
  bool StealExtentFor(size_t cls, std::vector<std::string>* evicted,
                      size_t min_donor_extents = 0); // rebind a full extent
  // Shared steal core: evict extent E's residents, unbind E, re-bind a pool
  // extent to target_cls. E must be fully unpinned. With mu_ held.
  bool StealExtentLocked(uint32_t E, size_t target_cls,
                         std::vector<std::string>* evicted);
  void FreeSlotLocked(const std::string& key, Entry& e);  // internal removal
  // Per-extent free-slot bookkeeping (all with mu_ held).
  void PushFreeLocked(Class& C, uint32_t ext, uint32_t slot);
  bool PopFreeLocked(Class& C, Slot* out);                    // round-robin across extents
  bool TakeFreeSlotLocked(Class& C, uint32_t ext, uint32_t slot);  // Restore: exact slot
  void DropExtentFreeLocked(Class& C, uint32_t ext);          // Steal: forget an extent
  void DropFromRotationLocked(Class& C, uint32_t ext);        // remove ext from ext_rr

  Options opt_;
  mutable std::mutex mu_;
  // unique_ptr so a Class (holding a std::list ring + a persistent CLOCK `hand`
  // iterator, with per-key ring_it iterators stored in index_) NEVER moves when
  // the vector grows -- a moved std::list leaves those iterators dangling
  // (same reason KVStore boxes its Shards).
  std::vector<std::unique_ptr<Class>> classes_;
  std::vector<ExtentMeta> extents_;
  std::unordered_map<std::string, Entry> index_;
  uint32_t unbound_ = 0;   // pool extents not bound to any class (skip bind scans at 0)
  uint64_t used_bytes_ = 0;
  uint64_t evictions_ = 0;
  uint64_t steals_ = 0;
  uint64_t extent_returns_ = 0;
};

}  // namespace dfkv

#endif  // DFKV_SLAB_ALLOCATOR_H_
