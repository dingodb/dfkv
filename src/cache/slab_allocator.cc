#include "cache/slab_allocator.h"

#include <algorithm>

#include "utils/log.h"  // WARN guard on Remove-while-pinned (see header contract)

namespace dfkv {

namespace {
uint64_t AlignUp(uint64_t x, uint64_t a) {
  return a <= 1 ? x : ((x + (a - 1)) / a) * a;
}
}  // namespace

SlabAllocator::SlabAllocator(Options opt) : opt_(opt) {
  if (opt_.align == 0) opt_.align = 1;
  if (opt_.num_extents == 0) opt_.num_extents = 1;
  extents_.assign(opt_.num_extents, ExtentMeta{});  // all kUnbound (in pool)
  unbound_ = opt_.num_extents;
}

void SlabAllocator::PushFreeLocked(Class& C, uint32_t ext, uint32_t slot) {
  auto& v = C.free_by_ext[ext];
  if (v.empty()) C.ext_rr.push_back(ext);
  v.push_back(slot);
  C.free_count++;
}

bool SlabAllocator::PopFreeLocked(Class& C, Slot* out) {
  if (C.ext_rr.empty()) return false;
  if (C.rr_next >= C.ext_rr.size()) C.rr_next = 0;
  const uint32_t ext = C.ext_rr[C.rr_next];
  auto vit = C.free_by_ext.find(ext);
  auto& v = vit->second;
  const uint32_t slot = v.back();
  v.pop_back();
  C.free_count--;
  if (v.empty()) {
    C.free_by_ext.erase(vit);
    // Swap-erase keeps rr_next pointing at the moved-in extent: no skip, no advance.
    C.ext_rr[C.rr_next] = C.ext_rr.back();
    C.ext_rr.pop_back();
  } else {
    ++C.rr_next;  // rotate: the NEXT Put goes to a different extent (inode)
  }
  if (out) *out = Slot{ext, slot};
  return true;
}

void SlabAllocator::DropFromRotationLocked(Class& C, uint32_t ext) {
  for (size_t i = 0; i < C.ext_rr.size(); ++i) {
    if (C.ext_rr[i] != ext) continue;
    C.ext_rr[i] = C.ext_rr.back();
    C.ext_rr.pop_back();
    if (C.rr_next > i) --C.rr_next;
    return;
  }
}

bool SlabAllocator::TakeFreeSlotLocked(Class& C, uint32_t ext, uint32_t slot) {
  auto vit = C.free_by_ext.find(ext);
  if (vit == C.free_by_ext.end()) return false;
  auto& v = vit->second;
  auto it = std::find(v.begin(), v.end(), slot);
  if (it == v.end()) return false;
  *it = v.back();
  v.pop_back();
  C.free_count--;
  if (v.empty()) { C.free_by_ext.erase(vit); DropFromRotationLocked(C, ext); }
  return true;
}

void SlabAllocator::DropExtentFreeLocked(Class& C, uint32_t ext) {
  auto vit = C.free_by_ext.find(ext);
  if (vit == C.free_by_ext.end()) return;
  C.free_count -= vit->second.size();
  C.free_by_ext.erase(vit);
  DropFromRotationLocked(C, ext);
}

size_t SlabAllocator::ClassForLen(size_t aligned_len) {
  // Smallest existing class whose slot fits with bounded waste; else new class.
  size_t best = classes_.size();
  uint32_t best_size = 0;
  for (size_t i = 0; i < classes_.size(); ++i) {
    uint32_t ss = classes_[i]->slot_size;
    if (ss < aligned_len) continue;
    if (static_cast<double>(ss - aligned_len) > opt_.max_waste * ss) continue;
    if (best == classes_.size() || ss < best_size) { best = i; best_size = ss; }
  }
  if (best != classes_.size()) return best;
  auto c = std::make_unique<Class>();
  c->slot_size = static_cast<uint32_t>(aligned_len);
  c->slots_per_extent = static_cast<uint32_t>(opt_.extent_bytes / c->slot_size);
  classes_.push_back(std::move(c));
  return classes_.size() - 1;
}

size_t SlabAllocator::ClassForExactSize(uint32_t slot_size) {
  for (size_t i = 0; i < classes_.size(); ++i)
    if (classes_[i]->slot_size == slot_size) return i;
  auto c = std::make_unique<Class>();
  c->slot_size = slot_size;
  c->slots_per_extent = slot_size ? static_cast<uint32_t>(opt_.extent_bytes / slot_size) : 0;
  classes_.push_back(std::move(c));
  return classes_.size() - 1;
}

bool SlabAllocator::Restore(const std::string& key, uint32_t slot_size,
                            uint32_t extent, uint32_t slot) {
  std::lock_guard<std::mutex> lk(mu_);
  if (slot_size == 0 || extent >= extents_.size() || index_.count(key)) return false;
  const size_t cls = ClassForExactSize(slot_size);
  Class& C = *classes_[cls];
  if (C.slots_per_extent == 0 || slot >= C.slots_per_extent) return false;
  ExtentMeta& m = extents_[extent];
  if (m.cls == kUnbound) {  // first restored slot on this extent: bind + cut slots
    m.cls = static_cast<int>(cls);
    m.total_slots = C.slots_per_extent;
    m.free_slots = C.slots_per_extent;
    m.pinned = 0;
    C.bound_extents++;
    --unbound_;
    for (uint32_t s = 0; s < m.total_slots; ++s) PushFreeLocked(C, extent, s);
  } else if (m.cls != static_cast<int>(cls)) {
    return false;  // persistence inconsistency: two classes on one extent
  }
  // Reserve the exact slot: pull it out of the extent's free bucket.
  if (!TakeFreeSlotLocked(C, extent, slot)) return false;  // taken (duplicate record)
  Entry e;
  e.ref.cls = static_cast<uint32_t>(cls);
  e.ref.extent = extent;
  e.ref.slot = slot;
  e.ref.slot_size = slot_size;
  e.ref.offset = static_cast<uint64_t>(slot) * slot_size;
  C.ring.push_front(key);
  e.ring_it = C.ring.begin();
  m.free_slots--;
  used_bytes_ += slot_size;
  auto res = index_.emplace(key, std::move(e));
  m.residents.push_front(&res.first->first);
  res.first->second.ext_it = m.residents.begin();
  return true;
}

bool SlabAllocator::BindFreeExtent(size_t cls) {
  if (unbound_ == 0) return false;  // O(1) fast path for the per-Put stripe top-up
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    if (extents_[e].cls != kUnbound) continue;
    Class& C = *classes_[cls];
    const uint32_t total = C.slots_per_extent;
    if (total == 0) return false;  // slot larger than an extent (guarded in Put)
    extents_[e].cls = static_cast<int>(cls);
    extents_[e].total_slots = total;
    extents_[e].free_slots = total;
    extents_[e].pinned = 0;
    C.bound_extents++;
    --unbound_;
    // Wipe-before-use: stale persisted records from a previous binding must be
    // gone before any slot of this extent can be handed out (same lock hold).
    if (opt_.on_extent_bind) opt_.on_extent_bind(e);
    for (uint32_t s = 0; s < total; ++s) PushFreeLocked(C, e, s);
    return true;
  }
  return false;
}

void SlabAllocator::FreeSlotLocked(const std::string& key, Entry& e) {
  const uint32_t cls = e.ref.cls, ext = e.ref.extent, slot = e.ref.slot;
  Class& C = *classes_[cls];
  if (C.hand != C.ring.end() && C.hand == e.ring_it) C.hand = std::next(C.hand);
  C.ring.erase(e.ring_it);
  extents_[ext].residents.erase(e.ext_it);
  PushFreeLocked(C, ext, slot);
  extents_[ext].free_slots++;
  if (e.refs > 0 && extents_[ext].pinned > 0) extents_[ext].pinned--;
  used_bytes_ -= e.ref.slot_size;
  index_.erase(key);
  // Capacity flows back between classes: once an extent is ENTIRELY free and
  // the class keeps at least kStripeWays other extents in rotation (write
  // parallelism preserved), unbind it to the shared pool -- a workload shift
  // (new value size => new class) then re-binds it instead of stealing, which
  // would evict another class's residents.
  ExtentMeta& m = extents_[ext];
  if (m.free_slots == m.total_slots && C.ext_rr.size() > kStripeWays) {
    DropExtentFreeLocked(C, ext);
    C.bound_extents--;
    m.cls = kUnbound;
    m.total_slots = 0;
    m.free_slots = 0;
    m.pinned = 0;
    ++unbound_;
    ++extent_returns_;
  }
}

bool SlabAllocator::EvictOneFrom(size_t cls, std::vector<std::string>* evicted) {
  Class& C = *classes_[cls];
  if (C.ring.empty()) return false;
  // CLOCK second-chance sweep, bounded to ~2 cycles; skip pinned entries. A
  // referenced entry is cleared and given a second chance; the first unreferenced
  // unpinned entry is evicted. If the bound is hit with no unreferenced victim,
  // evict the first unpinned entry regardless of its bit (forward progress).
  const size_t limit = 2 * C.ring.size() + 2;
  std::list<std::string>::iterator fallback = C.ring.end();
  for (size_t steps = 0; steps < limit; ++steps) {
    if (C.hand == C.ring.end()) C.hand = C.ring.begin();
    if (C.ring.empty()) break;
    auto cur = C.hand;
    auto it = index_.find(*cur);
    if (it == index_.end()) { C.hand = std::next(cur); C.ring.erase(cur); continue; }
    Entry& e = it->second;
    if (e.refs > 0) { C.hand = std::next(cur); continue; }  // pinned: skip
    if (fallback == C.ring.end()) fallback = cur;           // first unpinned seen
    if (e.referenced) { e.referenced = false; C.hand = std::next(cur); continue; }
    // unpinned + unreferenced -> evict.
    std::string victim = *cur;
    FreeSlotLocked(victim, e);
    evicted->push_back(std::move(victim));
    evictions_++;
    return true;
  }
  if (fallback != C.ring.end()) {  // bound hit: force-evict an unpinned entry
    std::string victim = *fallback;
    auto it = index_.find(victim);
    if (it != index_.end()) {
      FreeSlotLocked(victim, it->second);
      evicted->push_back(std::move(victim));
      evictions_++;
      return true;
    }
  }
  return false;  // every entry in this class is pinned
}

bool SlabAllocator::StealExtentFor(size_t cls, std::vector<std::string>* evicted,
                                   size_t min_donor_extents) {
  // Cold path: the target class has no slot and the pool is empty. Reclaim an
  // entirely-unpinned extent bound to ANOTHER class, evict its residents, unbind
  // it, and re-bind to the target class. Prefer the emptiest such extent.
  int best = -1;
  uint32_t best_free = 0;
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& m = extents_[e];
    if (m.cls == kUnbound || m.cls == static_cast<int>(cls)) continue;
    if (m.pinned != 0) continue;  // can't evict a pinned slot
    if (min_donor_extents > 0 &&
        classes_[m.cls]->bound_extents <= min_donor_extents)
      continue;  // donor floor: never shrink a class below its striping width
    if (best < 0 || m.free_slots > best_free) { best = static_cast<int>(e); best_free = m.free_slots; }
  }
  if (best < 0) return false;
  return StealExtentLocked(static_cast<uint32_t>(best), cls, evicted);
}

bool SlabAllocator::StealExtentLocked(uint32_t E, size_t target_cls,
                                      std::vector<std::string>* evicted) {
  ExtentMeta& m = extents_[E];
  const int old_cls = m.cls;
  // Evict every resident key living in extent E, walked off its resident list:
  // O(residents in E), not O(whole index) -- steal happens on every extent
  // hand-off during a size-class mix shift, so it must not scan the store.
  while (!m.residents.empty()) {
    const std::string victim = *m.residents.front();
    auto it = index_.find(victim);
    if (it == index_.end()) { m.residents.pop_front(); continue; }  // stale node (defensive)
    FreeSlotLocked(victim, it->second);  // erases the front resident node
    evicted->push_back(victim);
    evictions_++;
    // The last eviction can auto-return E to the pool (FreeSlotLocked's
    // fully-free unbind); residents is empty then, so the loop just ends.
  }
  // Unbind E unless FreeSlotLocked's return-to-pool already did.
  if (m.cls != kUnbound) {
    DropExtentFreeLocked(*classes_[old_cls], E);
    classes_[old_cls]->bound_extents--;
    m.cls = kUnbound;
    m.total_slots = 0;
    m.free_slots = 0;
    m.pinned = 0;
    ++unbound_;
  }
  ++steals_;
  return BindFreeExtent(target_cls);  // now picks the freshly-unbound E
}

bool SlabAllocator::StealFrom(size_t donor_cls, size_t target_cls,
                              std::vector<std::string>* evicted) {
  std::lock_guard<std::mutex> lk(mu_);
  if (donor_cls == target_cls) return false;
  if (donor_cls >= classes_.size() || target_cls >= classes_.size()) return false;
  if (classes_[target_cls]->slots_per_extent == 0) return false;  // slot > extent
  // The donor's emptiest fully-unpinned extent (fewest residents to evict).
  int best = -1;
  uint32_t best_free = 0;
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& m = extents_[e];
    if (m.cls != static_cast<int>(donor_cls) || m.pinned != 0) continue;
    if (best < 0 || m.free_slots > best_free) { best = static_cast<int>(e); best_free = m.free_slots; }
  }
  if (best < 0) return false;  // donor has no eligible extent
  return StealExtentLocked(static_cast<uint32_t>(best), target_cls, evicted);
}

bool SlabAllocator::Put(const std::string& key, size_t len, SlotRef* out,
                        std::vector<std::string>* evicted) {
  std::lock_guard<std::mutex> lk(mu_);
  auto existing = index_.find(key);
  if (existing != index_.end()) {  // idempotent: keep the current slot
    existing->second.referenced = true;
    if (out) *out = existing->second.ref;
    return true;
  }
  uint64_t aligned = AlignUp(len, opt_.align);
  if (aligned == 0) aligned = opt_.align;              // 0-len value -> one min slot
  if (aligned > opt_.extent_bytes) return false;       // larger than an extent

  const size_t cls = ClassForLen(static_cast<size_t>(aligned));
  Slot got{0, 0};
  for (;;) {
    Class& C = *classes_[cls];
    // Stripe top-up: keep up to kStripeWays extents in this class's rotation so
    // concurrent writers land on different files (cheap no-op once the pool is
    // exhausted -- BindFreeExtent bails at unbound_ == 0).
    while (C.ext_rr.size() < kStripeWays && BindFreeExtent(cls)) {}
    if (PopFreeLocked(C, &got)) break;
    // Growth-first while under the striping width: a class below kStripeWays
    // extents is still bootstrapping -- self-evicting here pins it at birth
    // size forever (its growth would be left to accidental all-pinned moments
    // or the background rebalance tick; this closes the intra-tick window).
    // Donor floor keeps two bootstrapping classes from ping-ponging.
    if (C.bound_extents < kStripeWays &&
        StealExtentFor(cls, evicted, /*min_donor_extents=*/kStripeWays))
      continue;
    if (EvictOneFrom(cls, evicted)) continue;
    if (StealExtentFor(cls, evicted)) continue;
    return false;  // nothing to free: all candidate slots are pinned
  }

  Entry e;
  e.ref.cls = static_cast<uint32_t>(cls);
  e.ref.extent = got.extent;
  e.ref.slot = got.slot;
  e.ref.slot_size = classes_[cls]->slot_size;
  e.ref.offset = static_cast<uint64_t>(got.slot) * classes_[cls]->slot_size;
  e.refs = 0;
  e.referenced = false;
  Class& C = *classes_[cls];
  C.ring.push_front(key);
  e.ring_it = C.ring.begin();
  C.puts++;
  extents_[got.extent].free_slots--;
  used_bytes_ += e.ref.slot_size;
  auto res = index_.emplace(key, std::move(e));
  ExtentMeta& m = extents_[got.extent];
  m.residents.push_front(&res.first->first);
  res.first->second.ext_it = m.residents.begin();
  if (out) *out = res.first->second.ref;
  return true;
}

bool SlabAllocator::Get(const std::string& key, SlotRef* out) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return false;
  it->second.referenced = true;  // CLOCK second chance
  if (out) *out = it->second.ref;
  return true;
}

bool SlabAllocator::GetAndPin(const std::string& key, SlotRef* out) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return false;
  it->second.referenced = true;  // CLOCK second chance
  if (it->second.refs == 0) extents_[it->second.ref.extent].pinned++;
  it->second.refs++;
  if (out) *out = it->second.ref;
  return true;
}

bool SlabAllocator::Contains(const std::string& key) const {
  std::lock_guard<std::mutex> lk(mu_);
  return index_.find(key) != index_.end();
}

bool SlabAllocator::Remove(const std::string& key) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return false;
  if (it->second.refs > 0)
    DFKV_LOG_WARN("SlabAllocator::Remove on a PINNED key (refs=" +
                  std::to_string(it->second.refs) +
                  "): its slot is freed for reuse NOW -- the caller is missing "
                  "an in-flight guard (see header contract)");
  FreeSlotLocked(key, it->second);
  return true;
}

bool SlabAllocator::Pin(const std::string& key) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return false;
  if (it->second.refs == 0) extents_[it->second.ref.extent].pinned++;
  it->second.refs++;
  return true;
}

bool SlabAllocator::Unpin(const std::string& key) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = index_.find(key);
  if (it == index_.end() || it->second.refs == 0) return false;
  it->second.refs--;
  if (it->second.refs == 0 && extents_[it->second.ref.extent].pinned > 0)
    extents_[it->second.ref.extent].pinned--;
  return true;
}

std::vector<SlabAllocator::ClassStat> SlabAllocator::Classes() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<ClassStat> out;
  out.reserve(classes_.size());
  for (const auto& c : classes_) {
    ClassStat s;
    s.slot_size = c->slot_size;
    s.free_slots = c->free_count;
    s.resident = c->ring.size();
    s.puts = c->puts;
    s.extents = c->bound_extents;
    s.slots_per_extent = c->slots_per_extent;
    out.push_back(s);
  }
  return out;
}

size_t SlabAllocator::ReclaimClass(size_t cls_index, size_t target_free,
                                   size_t max_victims,
                                   std::vector<std::string>* evicted) {
  std::lock_guard<std::mutex> lk(mu_);
  if (cls_index >= classes_.size()) return 0;
  Class& C = *classes_[cls_index];
  size_t n = 0;
  while (n < max_victims && C.free_count < target_free) {
    // While the shared pool still has extents, Put grows the class by binding
    // one (cheap, no residency loss) -- evicting now would only shrink the
    // cache for headroom the pool provides for free. Reclaim earns its keep
    // exactly when the pool is exhausted and Put's only inline recourse is the
    // CLOCK sweep this thread exists to pre-run.
    if (unbound_ > 0) break;
    const size_t before = C.free_count;
    if (!EvictOneFrom(cls_index, evicted)) break;  // all pinned / empty
    ++n;
    // An eviction normally gains one free slot. If free_count did NOT grow, the
    // freed slot's extent went back to the shared pool (fully-free unbind):
    // stop -- more reclaiming would cascade-shrink this class, and the pool
    // just gained a whole extent for whoever needs it.
    if (C.free_count <= before) break;
  }
  return n;
}

size_t SlabAllocator::Count() const {
  std::lock_guard<std::mutex> lk(mu_);
  return index_.size();
}
uint64_t SlabAllocator::UsedBytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  return used_bytes_;
}
uint64_t SlabAllocator::Capacity() const {
  return static_cast<uint64_t>(opt_.num_extents) * opt_.extent_bytes;
}
uint64_t SlabAllocator::Evictions() const {
  std::lock_guard<std::mutex> lk(mu_);
  return evictions_;
}
uint64_t SlabAllocator::Steals() const {
  std::lock_guard<std::mutex> lk(mu_);
  return steals_;
}
uint64_t SlabAllocator::ExtentReturns() const {
  std::lock_guard<std::mutex> lk(mu_);
  return extent_returns_;
}
size_t SlabAllocator::ClassCount() const {
  std::lock_guard<std::mutex> lk(mu_);
  return classes_.size();
}
uint32_t SlabAllocator::PoolExtents() const {
  std::lock_guard<std::mutex> lk(mu_);
  return unbound_;
}
uint32_t SlabAllocator::BoundExtents() const {
  std::lock_guard<std::mutex> lk(mu_);
  uint32_t n = 0;
  for (const auto& m : extents_) if (m.cls != kUnbound) ++n;
  return n;
}

}  // namespace dfkv
