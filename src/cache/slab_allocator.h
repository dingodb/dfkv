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
  };

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
  bool Contains(const std::string& key) const;

  // Crash-recovery only: install a key KNOWN (from persistence) to occupy
  // (extent, slot) in a class of `slot_size`. Binds the extent to that class if
  // needed and reserves the exact slot. Must be called before any Put, and the
  // (extent, slot) triples restored must be internally consistent (one class per
  // extent). Returns false on an inconsistency (mixed class on an extent, or a
  // slot out of range) -- the caller then discards that record.
  bool Restore(const std::string& key, uint32_t slot_size, uint32_t extent,
               uint32_t slot);

  // Drop `key` (frees its slot). true if present. A pinned key is still removed
  // from the index but its slot is not reused until Unpin brings refs to 0.
  bool Remove(const std::string& key);

  // Pin/unpin a resident key: a pinned slot is never chosen for eviction (used
  // while an RDMA send reads the slot). Balanced calls; Pin on an absent key is
  // a no-op returning false.
  bool Pin(const std::string& key);
  bool Unpin(const std::string& key);

  // stats (diagnostic)
  size_t Count() const;
  uint64_t UsedBytes() const;       // sum of slot_size over resident keys
  uint64_t Capacity() const;        // num_extents * extent_bytes
  uint64_t Evictions() const;
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
  };
  struct Class {
    uint32_t slot_size = 0;
    uint32_t slots_per_extent = 0;
    std::vector<Slot> free_slots;              // available slots (stack)
    std::list<std::string> ring;               // CLOCK ring of resident keys
    std::list<std::string>::iterator hand;     // persistent eviction cursor
    Class() : hand(ring.end()) {}
  };
  struct ExtentMeta {
    int cls = kUnbound;      // class index bound to, or kUnbound (in pool)
    uint32_t free_slots = 0; // free slots (== total when fully empty)
    uint32_t total_slots = 0;
    uint32_t pinned = 0;     // resident pinned slots in this extent (steal guard)
  };

  size_t ClassForLen(size_t aligned_len);  // returns class index (creates if needed)
  size_t ClassForExactSize(uint32_t slot_size);  // find/create a class of exactly slot_size
  bool BindFreeExtent(size_t cls);         // bind a pool extent to cls; false if none
  bool EvictOneFrom(size_t cls, std::vector<std::string>* evicted);  // CLOCK evict 1
  bool StealExtentFor(size_t cls, std::vector<std::string>* evicted); // rebind a full extent
  void FreeSlotLocked(const std::string& key, Entry& e);  // internal removal

  Options opt_;
  mutable std::mutex mu_;
  // unique_ptr so a Class (holding a std::list ring + a persistent CLOCK `hand`
  // iterator, with per-key ring_it iterators stored in index_) NEVER moves when
  // the vector grows -- a moved std::list leaves those iterators dangling
  // (same reason KVStore boxes its Shards).
  std::vector<std::unique_ptr<Class>> classes_;
  std::vector<ExtentMeta> extents_;
  std::unordered_map<std::string, Entry> index_;
  uint64_t used_bytes_ = 0;
  uint64_t evictions_ = 0;
};

}  // namespace dfkv

#endif  // DFKV_SLAB_ALLOCATOR_H_
