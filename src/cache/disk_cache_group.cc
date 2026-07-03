#include "cache/disk_cache_group.h"

#include <cstdlib>

#include "cache/disk_slab_store.h"

namespace dfkv {

DiskCacheGroup::DiskCacheGroup(Options opt) {
  size_t n = opt.cache_dirs.empty() ? 1 : opt.cache_dirs.size();
  uint64_t per_disk = opt.capacity_bytes / n;
  if (per_disk == 0) per_disk = opt.capacity_bytes;  // tiny-cap safety
  std::string engine = opt.engine;
  if (engine.empty()) {
    const char* e = std::getenv("DFKV_STORE_ENGINE");
    engine = (e && *e) ? e : "file";
  }
  const bool use_slab = (engine == "slab");
  engine_ = use_slab ? "slab" : "file";  // resolved truth, reported via EngineName
  // Slab write mode: DFKV_SLAB_WRITE=direct switches the aligned CacheDirect
  // path to O_DIRECT extent writes (see DiskSlabStore::Options::direct_writes).
  const char* wm = std::getenv("DFKV_SLAB_WRITE");
  const bool slab_direct = wm && std::string(wm) == "direct";
  for (const auto& dir : opt.cache_dirs) {
    std::unique_ptr<StoreEngine> store;
    if (use_slab) {
      DiskSlabStore::Options so;
      so.dir = dir;
      so.capacity_bytes = per_disk;
      so.direct_writes = slab_direct;
      store = std::make_unique<DiskSlabStore>(so);
    } else {
      store = std::make_unique<KVStore>(KVStore::Options{dir, per_disk});
    }
    by_id_[dir] = store.get();
    disks_.push_back(std::move(store));
    ring_.AddNode(dir);  // disk id = its dir path
  }
  ring_.Build();
}

StoreEngine* DiskCacheGroup::Route(const BlockKey& key) const {
  if (disks_.size() == 1) return disks_[0].get();
  std::string id;
  if (!ring_.Lookup(key.Filename(), &id)) return nullptr;
  auto it = by_id_.find(id);
  return it == by_id_.end() ? nullptr : it->second;
}

Status DiskCacheGroup::Cache(const BlockKey& key, const void* data, size_t len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Cache(key, data, len);
}

Status DiskCacheGroup::Remove(const BlockKey& key) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Remove(key);
}

Status DiskCacheGroup::CacheDirect(const BlockKey& key, char* data, size_t len,
                                   size_t cap) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->CacheDirect(key, data, len, cap);
}

Status DiskCacheGroup::Range(const BlockKey& key, uint64_t offset,
                             uint64_t length, std::string* out) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Range(key, offset, length, out);
}

Status DiskCacheGroup::RangeInto(const BlockKey& key, uint64_t offset,
                                 uint64_t length, char* dst, size_t dst_cap,
                                 size_t* out_len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeInto(key, offset, length, dst, dst_cap, out_len);
}

Status DiskCacheGroup::RangeDirect(const BlockKey& key, uint64_t offset,
                                   uint64_t length, char* io_buf, size_t io_cap,
                                   const char** out_data, size_t* out_len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeDirect(key, offset, length, io_buf, io_cap, out_data, out_len);
}

Status DiskCacheGroup::RangeDirectPrep(const BlockKey& key, uint64_t offset,
                                       uint64_t length, size_t io_cap,
                                       KVStore::RangePrep* out) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  Status st = d->RangeDirectPrep(key, offset, length, io_cap, out);
  // Brand the engine's token with the disk index so RangeRelease -- which has
  // no key to route by -- finds its way back (0 stays 0 = nothing to release).
  if (st == Status::kOk && out && out->token != 0) {
    for (size_t i = 0; i < disks_.size(); ++i) {
      if (disks_[i].get() != d) continue;
      out->token = (static_cast<uint64_t>(i + 1) << 56) | (out->token & kTokenMask);
      break;
    }
  }
  return st;
}

void DiskCacheGroup::RangeRelease(uint64_t token) {
  const size_t i = static_cast<size_t>(token >> 56);
  if (i == 0 || i > disks_.size()) return;
  disks_[i - 1]->RangeRelease(token & kTokenMask);
}

bool DiskCacheGroup::IsCached(const BlockKey& key) const {
  StoreEngine* d = Route(key);
  return d != nullptr && d->IsCached(key);
}

uint64_t DiskCacheGroup::UsedBytes() const {
  uint64_t t = 0;
  for (const auto& d : disks_) t += d->UsedBytes();
  return t;
}

size_t DiskCacheGroup::Count() const {
  size_t t = 0;
  for (const auto& d : disks_) t += d->Count();
  return t;
}

uint64_t DiskCacheGroup::Evictions() const {
  uint64_t t = 0;
  for (const auto& d : disks_) t += d->Evictions();
  return t;
}

uint64_t DiskCacheGroup::EvictedBytes() const {
  uint64_t t = 0;
  for (const auto& d : disks_) t += d->EvictedBytes();
  return t;
}

}  // namespace dfkv
