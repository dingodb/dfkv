#include "cache/disk_slab_store.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <vector>

#include "utils/net_util.h"  // PutU32/PutU64/GetU32/GetU64 (host-endian codec)

namespace fs = std::filesystem;

namespace dfkv {

namespace {
// Table-record + meta magics.
constexpr uint32_t kRecMagic = 0x424C5453u;   // "SLTB"
constexpr uint32_t kMetaMagic = 0x424C534Du;  // "SLBM"
constexpr uint32_t kFormatVersion = 1;

// Small CRC32 (IEEE, reflected) over a byte range; enough to catch a torn record.
uint32_t Crc32(const uint8_t* p, size_t n) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    crc ^= p[i];
    for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return crc ^ 0xFFFFFFFFu;
}

bool PwriteAll(int fd, const void* buf, size_t n, uint64_t off) {
  const char* p = static_cast<const char*>(buf);
  size_t done = 0;
  while (done < n) {
    ssize_t w = ::pwrite(fd, p + done, n - done, static_cast<off_t>(off + done));
    if (w < 0) { if (errno == EINTR) continue; return false; }
    if (w == 0) return false;
    done += static_cast<size_t>(w);
  }
  return true;
}

bool PreadAll(int fd, void* buf, size_t n, uint64_t off) {
  char* p = static_cast<char*>(buf);
  size_t done = 0;
  while (done < n) {
    ssize_t r = ::pread(fd, p + done, n - done, static_cast<off_t>(off + done));
    if (r < 0) { if (errno == EINTR) continue; return false; }
    if (r == 0) return false;  // short/EOF
    done += static_cast<size_t>(r);
  }
  return true;
}
}  // namespace

DiskSlabStore::DiskSlabStore(Options opt, bool* ok) : opt_(std::move(opt)) {
  if (opt_.extent_bytes == 0) opt_.extent_bytes = (1ull << 30);
  if (opt_.slot_granularity == 0) opt_.slot_granularity = (1ull << 20);
  if (opt_.capacity_bytes < opt_.extent_bytes) opt_.capacity_bytes = opt_.extent_bytes;
  num_extents_ = static_cast<uint32_t>(opt_.capacity_bytes / opt_.extent_bytes);
  if (num_extents_ == 0) num_extents_ = 1;
  max_slots_per_extent_ =
      static_cast<uint32_t>(opt_.extent_bytes / opt_.slot_granularity);
  if (max_slots_per_extent_ == 0) max_slots_per_extent_ = 1;

  SlabAllocator::Options ao;
  ao.extent_bytes = opt_.extent_bytes;
  ao.num_extents = num_extents_;
  ao.align = static_cast<uint32_t>(opt_.slot_granularity);  // slot_size is a granularity multiple
  ao.max_waste = 0.25;
  alloc_ = std::make_unique<SlabAllocator>(ao);

  ok_ = OpenOrInit();
  if (ok_) Rebuild();
  if (ok) *ok = ok_;
}

DiskSlabStore::~DiskSlabStore() {
  for (int fd : extent_fds_) if (fd >= 0) ::close(fd);
  if (table_fd_ >= 0) ::close(table_fd_);
}

bool DiskSlabStore::OpenOrInit() {
  std::error_code ec;
  fs::create_directories(opt_.dir, ec);
  fs::create_directories(fs::path(opt_.dir) / "extents", ec);
  const std::string meta_path = (fs::path(opt_.dir) / "slab_meta").string();

  // Meta = magic, version, extent_bytes, slot_granularity, num_extents. A magic
  // mismatch = corruption (refuse); a config mismatch = re-init fresh (the old
  // layout can't be reused). Cache semantics make a fresh start safe.
  uint8_t meta[64];
  bool fresh = true;
  int mfd = ::open(meta_path.c_str(), O_RDONLY);
  if (mfd >= 0) {
    if (PreadAll(mfd, meta, sizeof(meta), 0) && net::GetU32(reinterpret_cast<char*>(meta)) == kMetaMagic) {
      const uint32_t ver = net::GetU32(reinterpret_cast<char*>(meta) + 4);
      const uint64_t eb = net::GetU64(reinterpret_cast<char*>(meta) + 8);
      const uint64_t sg = net::GetU64(reinterpret_cast<char*>(meta) + 16);
      const uint32_t ne = net::GetU32(reinterpret_cast<char*>(meta) + 24);
      fresh = !(ver == kFormatVersion && eb == opt_.extent_bytes &&
                sg == opt_.slot_granularity && ne == num_extents_);
    }
    ::close(mfd);
  }

  const std::string tbl_path = (fs::path(opt_.dir) / "slots.tbl").string();
  const uint64_t tbl_bytes =
      static_cast<uint64_t>(num_extents_) * max_slots_per_extent_ * kRecBytes;

  if (fresh) {
    // (Re)write meta, zero the table, and (re)create extent files at full size.
    std::memset(meta, 0, sizeof(meta));
    net::PutU32(reinterpret_cast<char*>(meta), kMetaMagic);
    net::PutU32(reinterpret_cast<char*>(meta) + 4, kFormatVersion);
    net::PutU64(reinterpret_cast<char*>(meta) + 8, opt_.extent_bytes);
    net::PutU64(reinterpret_cast<char*>(meta) + 16, opt_.slot_granularity);
    net::PutU32(reinterpret_cast<char*>(meta) + 24, num_extents_);
    int wfd = ::open(meta_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (wfd < 0 || !PwriteAll(wfd, meta, sizeof(meta), 0)) { if (wfd >= 0) ::close(wfd); return false; }
    ::close(wfd);
    // Truncate the table to size (sparse zeros == all-free records).
    int tfd = ::open(tbl_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (tfd < 0 || ::ftruncate(tfd, static_cast<off_t>(tbl_bytes)) != 0) {
      if (tfd >= 0) ::close(tfd);
      return false;
    }
    table_fd_ = tfd;
  } else {
    int tfd = ::open(tbl_path.c_str(), O_RDWR);
    if (tfd < 0) return false;
    ::ftruncate(tfd, static_cast<off_t>(tbl_bytes));  // ensure full size (idempotent)
    table_fd_ = tfd;
  }

  // Open (create + size) every extent file, keep the fd resident.
  extent_fds_.assign(num_extents_, -1);
  for (uint32_t e = 0; e < num_extents_; ++e) {
    char name[32];
    std::snprintf(name, sizeof(name), "E%05u", e);
    const std::string ep = (fs::path(opt_.dir) / "extents" / name).string();
    int fd = ::open(ep.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) return false;
    off_t sz = ::lseek(fd, 0, SEEK_END);
    if (sz < static_cast<off_t>(opt_.extent_bytes))
      ::ftruncate(fd, static_cast<off_t>(opt_.extent_bytes));
    extent_fds_[e] = fd;
  }
  return true;
}

void DiskSlabStore::Rebuild() {
  // Scan slots.tbl; each CRC-valid, state==1 record reinstalls its key into the
  // allocator (Restore) at its recorded slot. A torn/free record is skipped.
  uint8_t rec[kRecBytes];
  for (uint32_t e = 0; e < num_extents_; ++e) {
    for (uint32_t s = 0; s < max_slots_per_extent_; ++s) {
      if (!PreadAll(table_fd_, rec, kRecBytes, TableOffset(e, s))) return;
      if (net::GetU32(reinterpret_cast<char*>(rec)) != kRecMagic) continue;
      const uint32_t crc = net::GetU32(reinterpret_cast<char*>(rec) + 4);
      if (Crc32(rec + 8, kRecBytes - 8) != crc) continue;  // torn
      if (rec[8] != 1) continue;                            // not valid
      const uint32_t slot_size = net::GetU32(reinterpret_cast<char*>(rec) + 12);
      BlockKey key;
      key.id = net::GetU64(reinterpret_cast<char*>(rec) + 16);
      key.index = net::GetU32(reinterpret_cast<char*>(rec) + 24);
      key.size = net::GetU32(reinterpret_cast<char*>(rec) + 28);
      const uint32_t payload_len = net::GetU32(reinterpret_cast<char*>(rec) + 32);
      const std::string fn = key.Filename();
      if (alloc_->Restore(fn, slot_size, e, s)) {
        payload_len_[fn] = payload_len;
        ++table_rebuilt_;
      }
    }
  }
}

bool DiskSlabStore::WriteRecord(const SlabAllocator::SlotRef& r, const BlockKey& key,
                                uint32_t payload_len, bool valid) {
  uint8_t rec[kRecBytes];
  std::memset(rec, 0, sizeof(rec));
  net::PutU32(reinterpret_cast<char*>(rec), kRecMagic);
  rec[8] = valid ? 1 : 0;
  net::PutU32(reinterpret_cast<char*>(rec) + 12, r.slot_size);
  net::PutU64(reinterpret_cast<char*>(rec) + 16, key.id);
  net::PutU32(reinterpret_cast<char*>(rec) + 24, key.index);
  net::PutU32(reinterpret_cast<char*>(rec) + 28, key.size);
  net::PutU32(reinterpret_cast<char*>(rec) + 32, payload_len);
  const uint32_t crc = Crc32(rec + 8, kRecBytes - 8);
  net::PutU32(reinterpret_cast<char*>(rec) + 4, crc);
  return PwriteAll(table_fd_, rec, kRecBytes, TableOffset(r.extent, r.slot));
}

bool DiskSlabStore::WritePayload(const SlabAllocator::SlotRef& r, const void* data,
                                 size_t len) {
  if (len == 0) return true;
  return PwriteAll(extent_fds_[r.extent], data, len, r.offset);
}

Status DiskSlabStore::Cache(const BlockKey& key, const void* data, size_t len) {
  if (data == nullptr && len != 0) return Status::kInvalid;
  if (!ok_) return Status::kIOError;
  const std::string fn = key.Filename();
  std::lock_guard<std::mutex> lk(mu_);
  if (alloc_->Contains(fn)) return Status::kOk;  // idempotent

  SlabAllocator::SlotRef ref;
  std::vector<std::string> evicted;
  if (!alloc_->Put(fn, len, &ref, &evicted)) return Status::kIOError;  // too big / all pinned
  // Drop evicted keys from the runtime payload map. Their table records are left
  // as-is: a slot's record always reflects its LAST occupant, so this Put's
  // WriteRecord overwrites the reused slot's record, and any other evicted slot
  // that isn't reused before a crash simply "resurrects" its (still-valid,
  // content-addressed) key on restart -- correct cache data, never corruption
  // (design's resurrectable-remove semantics).
  for (const auto& ev : evicted) payload_len_.erase(ev);
  if (!WritePayload(ref, data, len) ||
      !WriteRecord(ref, key, static_cast<uint32_t>(len), /*valid=*/true)) {
    alloc_->Remove(fn);  // roll back the reservation on I/O failure
    return Status::kIOError;
  }
  payload_len_[fn] = static_cast<uint32_t>(len);
  return Status::kOk;
}

Status DiskSlabStore::Range(const BlockKey& key, uint64_t offset, uint64_t length,
                            std::string* out) {
  if (!ok_) return Status::kIOError;
  const std::string fn = key.Filename();
  std::lock_guard<std::mutex> lk(mu_);
  SlabAllocator::SlotRef ref;
  if (!alloc_->Get(fn, &ref)) return Status::kNotFound;
  auto it = payload_len_.find(fn);
  if (it == payload_len_.end()) return Status::kNotFound;
  const uint32_t plen = it->second;
  if (offset >= plen) { out->clear(); return Status::kOk; }
  const uint64_t avail = plen - offset;
  const uint64_t n = std::min(length ? length : avail, avail);
  out->resize(static_cast<size_t>(n));
  if (n && !PreadAll(extent_fds_[ref.extent], &(*out)[0], static_cast<size_t>(n),
                     ref.offset + offset))
    return Status::kIOError;
  return Status::kOk;
}

Status DiskSlabStore::RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                                char* dst, size_t dst_cap, size_t* out_len) {
  if (!ok_) return Status::kIOError;
  const std::string fn = key.Filename();
  std::lock_guard<std::mutex> lk(mu_);
  SlabAllocator::SlotRef ref;
  if (!alloc_->Get(fn, &ref)) return Status::kNotFound;
  auto it = payload_len_.find(fn);
  if (it == payload_len_.end()) return Status::kNotFound;
  const uint32_t plen = it->second;
  if (offset >= plen) { if (out_len) *out_len = 0; return Status::kOk; }
  uint64_t n = std::min(length ? length : (plen - offset), static_cast<uint64_t>(plen - offset));
  if (n > dst_cap) n = dst_cap;
  if (n && !PreadAll(extent_fds_[ref.extent], dst, static_cast<size_t>(n), ref.offset + offset))
    return Status::kIOError;
  if (out_len) *out_len = static_cast<size_t>(n);
  return Status::kOk;
}

bool DiskSlabStore::IsCached(const BlockKey& key) const {
  if (!ok_) return false;
  std::lock_guard<std::mutex> lk(mu_);
  return alloc_->Contains(key.Filename());
}

Status DiskSlabStore::Remove(const BlockKey& key) {
  if (!ok_) return Status::kIOError;
  const std::string fn = key.Filename();
  std::lock_guard<std::mutex> lk(mu_);
  SlabAllocator::SlotRef ref;
  if (!alloc_->Get(fn, &ref)) return Status::kNotFound;
  WriteRecord(ref, key, 0, /*valid=*/false);  // free the durable record first
  alloc_->Remove(fn);
  payload_len_.erase(fn);
  return Status::kOk;
}

size_t DiskSlabStore::Count() const { return alloc_->Count(); }
uint64_t DiskSlabStore::UsedBytes() const { return alloc_->UsedBytes(); }
uint64_t DiskSlabStore::Capacity() const { return alloc_->Capacity(); }
uint64_t DiskSlabStore::Evictions() const { return alloc_->Evictions(); }

}  // namespace dfkv
