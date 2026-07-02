#include "cache/slab_allocator.h"

#include <algorithm>

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
    C.free_slots.reserve(C.free_slots.size() + m.total_slots);
    for (uint32_t s = 0; s < m.total_slots; ++s) C.free_slots.push_back(Slot{extent, s});
  } else if (m.cls != static_cast<int>(cls)) {
    return false;  // persistence inconsistency: two classes on one extent
  }
  // Reserve the exact slot: pull it out of the class free stack.
  auto& fs = C.free_slots;
  auto it = std::find_if(fs.begin(), fs.end(), [extent, slot](const Slot& s) {
    return s.extent == extent && s.slot == slot;
  });
  if (it == fs.end()) return false;  // already taken (duplicate record)
  fs.erase(it);
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
  index_.emplace(key, std::move(e));
  return true;
}

bool SlabAllocator::BindFreeExtent(size_t cls) {
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    if (extents_[e].cls != kUnbound) continue;
    Class& C = *classes_[cls];
    const uint32_t total = C.slots_per_extent;
    if (total == 0) return false;  // slot larger than an extent (guarded in Put)
    extents_[e].cls = static_cast<int>(cls);
    extents_[e].total_slots = total;
    extents_[e].free_slots = total;
    extents_[e].pinned = 0;
    C.free_slots.reserve(C.free_slots.size() + total);
    for (uint32_t s = 0; s < total; ++s) C.free_slots.push_back(Slot{e, s});
    return true;
  }
  return false;
}

void SlabAllocator::FreeSlotLocked(const std::string& key, Entry& e) {
  const uint32_t cls = e.ref.cls, ext = e.ref.extent, slot = e.ref.slot;
  Class& C = *classes_[cls];
  if (C.hand != C.ring.end() && C.hand == e.ring_it) C.hand = std::next(C.hand);
  C.ring.erase(e.ring_it);
  C.free_slots.push_back(Slot{ext, slot});
  extents_[ext].free_slots++;
  if (e.refs > 0 && extents_[ext].pinned > 0) extents_[ext].pinned--;
  used_bytes_ -= e.ref.slot_size;
  index_.erase(key);
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

bool SlabAllocator::StealExtentFor(size_t cls, std::vector<std::string>* evicted) {
  // Cold path: the target class has no slot and the pool is empty. Reclaim an
  // entirely-unpinned extent bound to ANOTHER class, evict its residents, unbind
  // it, and re-bind to the target class. Prefer the emptiest such extent.
  int best = -1;
  uint32_t best_free = 0;
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& m = extents_[e];
    if (m.cls == kUnbound || m.cls == static_cast<int>(cls)) continue;
    if (m.pinned != 0) continue;  // can't evict a pinned slot
    if (best < 0 || m.free_slots > best_free) { best = static_cast<int>(e); best_free = m.free_slots; }
  }
  if (best < 0) return false;
  const uint32_t E = static_cast<uint32_t>(best);
  const int old_cls = extents_[E].cls;
  // Evict every resident key living in extent E (scan the index -- steal is rare).
  std::vector<std::string> victims;
  for (const auto& kv : index_)
    if (kv.second.ref.extent == E) victims.push_back(kv.first);
  for (const auto& v : victims) {
    auto it = index_.find(v);
    if (it != index_.end()) { FreeSlotLocked(v, it->second); evicted->push_back(v); evictions_++; }
  }
  // Drop E's now-free slots from the old class's free list, then unbind E.
  auto& oldfree = classes_[old_cls]->free_slots;
  oldfree.erase(std::remove_if(oldfree.begin(), oldfree.end(),
                               [E](const Slot& s) { return s.extent == E; }),
                oldfree.end());
  extents_[E].cls = kUnbound;
  extents_[E].total_slots = 0;
  extents_[E].free_slots = 0;
  extents_[E].pinned = 0;
  return BindFreeExtent(cls);  // now picks the freshly-unbound E
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
    if (!C.free_slots.empty()) { got = C.free_slots.back(); C.free_slots.pop_back(); break; }
    if (BindFreeExtent(cls)) continue;
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
  extents_[got.extent].free_slots--;
  used_bytes_ += e.ref.slot_size;
  auto res = index_.emplace(key, std::move(e));
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

bool SlabAllocator::Contains(const std::string& key) const {
  std::lock_guard<std::mutex> lk(mu_);
  return index_.find(key) != index_.end();
}

bool SlabAllocator::Remove(const std::string& key) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return false;
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
size_t SlabAllocator::ClassCount() const {
  std::lock_guard<std::mutex> lk(mu_);
  return classes_.size();
}
uint32_t SlabAllocator::BoundExtents() const {
  std::lock_guard<std::mutex> lk(mu_);
  uint32_t n = 0;
  for (const auto& m : extents_) if (m.cls != kUnbound) ++n;
  return n;
}

}  // namespace dfkv
