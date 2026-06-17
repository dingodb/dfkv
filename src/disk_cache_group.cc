#include "disk_cache_group.h"

namespace dfkv {

DiskCacheGroup::DiskCacheGroup(Options opt) {
  size_t n = opt.cache_dirs.empty() ? 1 : opt.cache_dirs.size();
  uint64_t per_disk = opt.capacity_bytes / n;
  if (per_disk == 0) per_disk = opt.capacity_bytes;  // tiny-cap safety
  for (const auto& dir : opt.cache_dirs) {
    auto store = std::make_unique<KVStore>(KVStore::Options{dir, per_disk});
    by_id_[dir] = store.get();
    disks_.push_back(std::move(store));
    ring_.AddNode(dir);  // disk id = its dir path
  }
  ring_.Build();
}

KVStore* DiskCacheGroup::Route(const BlockKey& key) const {
  if (disks_.size() == 1) return disks_[0].get();
  std::string id;
  if (!ring_.Lookup(key.Filename(), &id)) return nullptr;
  auto it = by_id_.find(id);
  return it == by_id_.end() ? nullptr : it->second;
}

Status DiskCacheGroup::Cache(const BlockKey& key, const void* data, size_t len) {
  KVStore* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Cache(key, data, len);
}

Status DiskCacheGroup::CacheDirect(const BlockKey& key, char* data, size_t len,
                                   size_t cap) {
  KVStore* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->CacheDirect(key, data, len, cap);
}

Status DiskCacheGroup::Range(const BlockKey& key, uint64_t offset,
                             uint64_t length, std::string* out) {
  KVStore* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Range(key, offset, length, out);
}

Status DiskCacheGroup::RangeInto(const BlockKey& key, uint64_t offset,
                                 uint64_t length, char* dst, size_t dst_cap,
                                 size_t* out_len) {
  KVStore* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeInto(key, offset, length, dst, dst_cap, out_len);
}

Status DiskCacheGroup::RangeDirect(const BlockKey& key, uint64_t offset,
                                   uint64_t length, char* io_buf, size_t io_cap,
                                   const char** out_data, size_t* out_len) {
  KVStore* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeDirect(key, offset, length, io_buf, io_cap, out_data, out_len);
}

bool DiskCacheGroup::IsCached(const BlockKey& key) const {
  KVStore* d = Route(key);
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
