#include "kv_store.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace dfkv {

const char* StatusName(Status s) {
  switch (s) {
    case Status::kOk: return "Ok";
    case Status::kNotFound: return "NotFound";
    case Status::kCacheFull: return "CacheFull";
    case Status::kIOError: return "IOError";
    case Status::kInvalid: return "Invalid";
  }
  return "?";
}

KVStore::KVStore(Options opt) : opt_(std::move(opt)) {
  fs::create_directories(opt_.cache_dir);
  RebuildIndex();
}

void KVStore::RebuildIndex() {
  std::lock_guard<std::mutex> lk(mu_);
  index_.clear();
  lru_.clear();
  used_bytes_ = 0;
  std::error_code ec;
  if (!fs::exists(opt_.cache_dir, ec)) return;
  for (auto it = fs::recursive_directory_iterator(opt_.cache_dir, ec);
       !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (!it->is_regular_file(ec)) continue;
    std::string fname = it->path().filename().string();
    if (fname.size() >= 4 && fname.substr(fname.size() - 4) == ".tmp") continue;
    uint64_t sz = static_cast<uint64_t>(fs::file_size(it->path(), ec));
    lru_.push_front(fname);
    index_[fname] = Entry{it->path().string(), sz, lru_.begin()};
    used_bytes_ += sz;
  }
}

void KVStore::TouchLocked(const std::string& fname) {
  auto it = index_.find(fname);
  if (it == index_.end()) return;
  lru_.erase(it->second.lru_it);
  lru_.push_front(fname);
  it->second.lru_it = lru_.begin();
}

void KVStore::EvictIfNeededLocked() {
  while (used_bytes_ > opt_.capacity_bytes && !lru_.empty()) {
    std::string victim = lru_.back();
    auto it = index_.find(victim);
    if (it == index_.end()) { lru_.pop_back(); continue; }
    std::error_code ec;
    fs::remove(it->second.path, ec);
    used_bytes_ -= it->second.size;
    lru_.pop_back();
    index_.erase(it);
  }
}

Status KVStore::Cache(const BlockKey& key, const void* data, size_t len) {
  if (data == nullptr && len != 0) return Status::kInvalid;
  std::lock_guard<std::mutex> lk(mu_);
  const std::string fname = key.Filename();
  if (index_.count(fname)) return Status::kOk;  // idempotent skip

  fs::path full = fs::path(opt_.cache_dir) / key.StoreKey();
  std::error_code ec;
  fs::create_directories(full.parent_path(), ec);
  fs::path tmp = full;
  tmp += ".tmp";
  {
    std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
    if (!ofs) return Status::kIOError;
    if (len) ofs.write(static_cast<const char*>(data), static_cast<std::streamsize>(len));
    if (!ofs) return Status::kIOError;
  }
  fs::rename(tmp, full, ec);  // atomic publish
  if (ec) { fs::remove(tmp, ec); return Status::kIOError; }

  lru_.push_front(fname);
  index_[fname] = Entry{full.string(), len, lru_.begin()};
  used_bytes_ += len;
  EvictIfNeededLocked();
  return Status::kOk;
}

Status KVStore::Range(const BlockKey& key, uint64_t offset, uint64_t length,
                      std::string* out) {
  // The mutex protects the in-memory index/LRU, NOT the bulk file read. Holding
  // it across the whole read serializes all reads to this disk (the GET hot path
  // for KVCache). Instead: look up + open the file + touch LRU under the lock —
  // the open fd pins the inode, so a concurrent eviction (fs::remove) can't pull
  // the bytes out from under us (POSIX unlink keeps the inode alive until close)
  // — then release the lock and do the 2.74 MiB read concurrently.
  std::ifstream ifs;
  uint64_t fsize = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    const std::string fname = key.Filename();
    auto it = index_.find(fname);
    if (it == index_.end()) return Status::kNotFound;  // cache-only: clean miss
    ifs.open(it->second.path, std::ios::binary);
    if (!ifs) return Status::kIOError;
    fsize = it->second.size;
    TouchLocked(fname);
  }
  if (offset > fsize) return Status::kInvalid;
  uint64_t end = offset + length;
  if (end > fsize) end = fsize;
  uint64_t n = end - offset;
  out->resize(n);
  ifs.seekg(static_cast<std::streamoff>(offset));
  if (n) ifs.read(&(*out)[0], static_cast<std::streamsize>(n));
  if (!ifs && n) return Status::kIOError;
  return Status::kOk;
}

bool KVStore::IsCached(const BlockKey& key) const {
  std::lock_guard<std::mutex> lk(mu_);
  return index_.count(key.Filename()) != 0;
}

uint64_t KVStore::UsedBytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  return used_bytes_;
}

size_t KVStore::Count() const {
  std::lock_guard<std::mutex> lk(mu_);
  return index_.size();
}

}  // namespace dfkv
