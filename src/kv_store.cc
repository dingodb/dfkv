#include "kv_store.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

namespace dfkv {

namespace {

// All block payloads go through O_DIRECT, which requires the file offset, the
// transfer length, AND the memory buffer to be aligned to (at least) the
// device's logical sector size. 4096 is a safe superset for xfs/NVMe (and any
// 512-byte-sector device), so we align everything to it.
constexpr size_t kDioAlign = 4096;

inline uint64_t AlignDown(uint64_t x) { return x & ~(uint64_t{kDioAlign} - 1); }

// Rounds up to a kDioAlign multiple; returns false on overflow.
inline bool AlignUp(uint64_t x, uint64_t* out) {
  if (x > UINT64_MAX - (kDioAlign - 1)) return false;
  *out = (x + (kDioAlign - 1)) & ~(uint64_t{kDioAlign} - 1);
  return true;
}

// Move-only RAII file descriptor (close on scope exit).
class Fd {
 public:
  Fd() = default;
  explicit Fd(int fd) : fd_(fd) {}
  ~Fd() { reset(); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
  Fd& operator=(Fd&& o) noexcept {
    if (this != &o) { reset(); fd_ = o.fd_; o.fd_ = -1; }
    return *this;
  }
  int get() const { return fd_; }
  bool valid() const { return fd_ >= 0; }
  void reset(int fd = -1) {
    // close() can fail with EINTR, but on Linux the fd is freed regardless, so a
    // retry would close an unrelated fd. Close once and discard the result.
    if (fd_ >= 0) { int r = ::close(fd_); (void)r; }
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

// RAII aligned heap buffer for O_DIRECT transfers (posix_memalign + free).
class AlignedBuf {
 public:
  // n must be > 0 and a multiple of kDioAlign (the only way we call it).
  explicit AlignedBuf(size_t n) {
    // posix_memalign returns the error code (e.g. ENOMEM) and does NOT set errno.
    if (posix_memalign(&p_, kDioAlign, n) != 0) p_ = nullptr;
  }
  ~AlignedBuf() { free(p_); }
  AlignedBuf(const AlignedBuf&) = delete;
  AlignedBuf& operator=(const AlignedBuf&) = delete;
  bool valid() const { return p_ != nullptr; }
  char* data() const { return static_cast<char*>(p_); }

 private:
  void* p_ = nullptr;
};

// Writes [data, data+len) to `path` via O_DIRECT: fallocate the (aligned) space,
// write an aligned superset from a bounce buffer, then ftruncate to the exact
// `len`. Returns true on success. The caller owns tmp-file cleanup on failure.
bool WriteFileDirect(const std::string& path, const void* data, size_t len) {
  Fd fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644));
  if (!fd.valid()) return false;
  if (len == 0) return true;  // empty file already created + truncated by O_TRUNC

  uint64_t alen64 = 0;
  if (!AlignUp(len, &alen64)) return false;
  const size_t alen = static_cast<size_t>(alen64);

  // Reserve the space first (user-requested). fallocate is a pure optimization;
  // if the FS doesn't support it the O_DIRECT write below still extends the file,
  // so tolerate "unsupported"; treat other errors (e.g. ENOSPC) as fatal.
  if (::fallocate(fd.get(), 0, 0, static_cast<off_t>(alen)) != 0 &&
      errno != EOPNOTSUPP && errno != ENOSYS && errno != EINVAL) {
    return false;
  }

  AlignedBuf buf(alen);
  if (!buf.valid()) return false;
  std::memcpy(buf.data(), data, len);
  std::memset(buf.data() + len, 0, alen - len);  // zero the pad (no stale heap on disk)

  size_t done = 0;
  while (done < alen) {
    ssize_t w = ::pwrite(fd.get(), buf.data() + done, alen - done,
                         static_cast<off_t>(done));
    if (w < 0) { if (errno == EINTR) continue; return false; }
    if (w == 0) return false;  // not expected for a regular file
    done += static_cast<size_t>(w);
    // A sub-block resume offset would make the next O_DIRECT pwrite EINVAL.
    if (done < alen && (done % kDioAlign) != 0) return false;
  }

  if (::ftruncate(fd.get(), static_cast<off_t>(len)) != 0) return false;  // exact size
  return true;
}

// Reads exactly [offset, offset+n) from the already-open O_DIRECT fd into `dst`
// (n > 0, dst holds >= n bytes). Reads the aligned superset into a bounce buffer
// and trims out the requested slice. Returns true on success.
bool PreadRangeDirect(int fd, uint64_t offset, size_t n, char* dst) {
  const uint64_t astart = AlignDown(offset);
  uint64_t aend = 0;
  if (!AlignUp(offset + n, &aend)) return false;  // offset+n can't overflow: both bounded by fsize
  const size_t alen = static_cast<size_t>(aend - astart);

  AlignedBuf buf(alen);
  if (!buf.valid()) return false;

  size_t total = 0;
  while (total < alen) {
    ssize_t r = ::pread(fd, buf.data() + total, alen - total,
                        static_cast<off_t>(astart + total));
    if (r < 0) { if (errno == EINTR) continue; return false; }
    if (r == 0) break;  // EOF (the aligned tail can legitimately extend past EOF)
    total += static_cast<size_t>(r);
  }

  const size_t head = static_cast<size_t>(offset - astart);
  if (total < head + n) return false;  // file shrank/corrupt under us
  std::memcpy(dst, buf.data() + head, n);
  return true;
}

}  // namespace

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
  const std::string fname = key.Filename();
  {  // best-effort early idempotent skip (avoids a needless 2.74 MiB write)
    std::lock_guard<std::mutex> lk(mu_);
    if (index_.count(fname)) return Status::kOk;
  }
  // Write the payload OUTSIDE the lock so concurrent writes to the same disk run
  // in parallel (the mutex only guards the in-memory index/LRU). A unique tmp
  // name per writer avoids collisions; the index re-check below resolves races.
  fs::path full = fs::path(opt_.cache_dir) / key.StoreKey();
  std::error_code ec;
  fs::create_directories(full.parent_path(), ec);
  fs::path tmp = full;
  tmp += "." + std::to_string(tmp_seq_.fetch_add(1, std::memory_order_relaxed)) + ".tmp";
  if (!WriteFileDirect(tmp.string(), data, len)) {
    fs::remove(tmp, ec);
    return Status::kIOError;
  }
  std::lock_guard<std::mutex> lk(mu_);
  if (index_.count(fname)) { fs::remove(tmp, ec); return Status::kOk; }  // lost the race; keep first
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
  // — then release the lock and do the 2.74 MiB O_DIRECT read concurrently.
  Fd fd;
  uint64_t fsize = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    const std::string fname = key.Filename();
    auto it = index_.find(fname);
    if (it == index_.end()) return Status::kNotFound;  // cache-only: clean miss
    fd.reset(::open(it->second.path.c_str(), O_RDONLY | O_DIRECT));
    if (!fd.valid()) return Status::kIOError;  // entry exists; open failure is I/O
    fsize = it->second.size;
    TouchLocked(fname);
  }
  if (offset > fsize) return Status::kInvalid;
  // Clamp by subtraction so offset + length can't overflow uint64_t.
  uint64_t n = length;
  if (n > fsize - offset) n = fsize - offset;
  out->resize(n);
  if (n == 0) return Status::kOk;
  if (!PreadRangeDirect(fd.get(), offset, static_cast<size_t>(n), &(*out)[0]))
    return Status::kIOError;
  return Status::kOk;
}

Status KVStore::RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                          char* dst, size_t dst_cap, size_t* out_len) {
  *out_len = 0;
  Fd fd;
  uint64_t fsize = 0;
  {  // index lookup + open under the lock; bulk read outside (see Range)
    std::lock_guard<std::mutex> lk(mu_);
    const std::string fname = key.Filename();
    auto it = index_.find(fname);
    if (it == index_.end()) return Status::kNotFound;
    fd.reset(::open(it->second.path.c_str(), O_RDONLY | O_DIRECT));
    if (!fd.valid()) return Status::kIOError;
    fsize = it->second.size;
    TouchLocked(fname);
  }
  if (offset > fsize) return Status::kInvalid;
  // Clamp by subtraction so offset + length can't overflow uint64_t.
  uint64_t n = length;
  if (n > fsize - offset) n = fsize - offset;
  if (n > dst_cap) n = dst_cap;
  if (n == 0) return Status::kOk;  // *out_len already 0
  if (!PreadRangeDirect(fd.get(), offset, static_cast<size_t>(n), dst))
    return Status::kIOError;
  *out_len = static_cast<size_t>(n);
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
