/*
 * DingoFS KV-cache for SGLang HiCache — value header.
 *
 * Every stored KV value is prefixed with this fixed 48-byte header so a read
 * whose model / page-size / dtype / layer geometry / CRC does not match the
 * reader's expectation is treated as a MISS (recompute) instead of silently
 * returning wrong bytes. For MLA models the latent is TP-layout invariant, so
 * tp_size/tp_rank are NOT part of the match (flags bit0 = is_mla).
 *
 * Portable (no brpc/MDS deps) so it builds & is unit-tested standalone.
 * Serialization is host-endian (memcpy); all targets are x86_64 LE. A read on
 * a different-endian host fails the magic/version check => safe MISS (recompute),
 * never silent corruption. Cross-arch sharing would need explicit LE fields.
 */
#ifndef DFKV_VALUE_HEADER_H_
#define DFKV_VALUE_HEADER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dfkv {

// Integrity checksum = CRC32C (Castagnoli, reflected poly 0x82F63B78, init/xorout
// 0xFFFFFFFF) — chosen over IEEE 0xEDB88320 because x86-64 has a hardware crc32c
// instruction (SSE4.2) that runs ~10-20 GB/s vs ~0.3 GB/s for the byte-at-a-time
// table. At the GLM-5.1 MLA page size (2.74 MiB) the table CRC was the dominant
// per-op cost (~7 ms); hardware crc32c removes it. We runtime-dispatch on
// __builtin_cpu_supports so the binary stays portable; the scalar table is the
// fallback and the two must agree bit-for-bit (see value_header_test).
namespace detail {

// Scalar Castagnoli table (thread-safe magic static; no init race in batch fan-out).
inline uint32_t Crc32cScalar(const void* data, size_t len) {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
      t[i] = c;
    }
    return t;
  }();
  uint32_t c = 0xFFFFFFFFu;
  const auto* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; ++i) c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

#if defined(__x86_64__) || defined(__i386__)
// Hardware crc32c via SSE4.2; target attribute lets the rest of the TU compile
// without -msse4.2 while this function uses the instruction (runtime-gated below).
__attribute__((target("sse4.2"))) inline uint32_t Crc32cHw(const void* data, size_t len) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint64_t c = 0xFFFFFFFFu;
  while (len >= 8) { uint64_t v; std::memcpy(&v, p, 8); c = __builtin_ia32_crc32di(c, v); p += 8; len -= 8; }
  while (len) { c = __builtin_ia32_crc32qi(static_cast<uint32_t>(c), *p++); --len; }
  return static_cast<uint32_t>(c) ^ 0xFFFFFFFFu;
}
#endif

}  // namespace detail

inline uint32_t Crc32(const void* data, size_t len) {
#if defined(__x86_64__) || defined(__i386__)
  static const bool kHw = __builtin_cpu_supports("sse4.2");
  if (kHw) return detail::Crc32cHw(data, len);
#endif
  return detail::Crc32cScalar(data, len);
}

#pragma pack(push, 1)
struct ValueHeader {
  static constexpr size_t kSize = 48;
  static constexpr uint32_t kMagic = 0x444F4B56u;  // 'DFKV'
  static constexpr uint16_t kVersion = 2;  // v2: integrity checksum is CRC32C (was IEEE crc32 in v1)
  enum Flag : uint16_t { kFlagIsMla = 0x1 };

  uint32_t magic = kMagic;
  uint16_t version = kVersion;
  uint16_t flags = 0;            // bit0: is_mla
  uint64_t model_hash = 0;
  uint32_t page_size = 0;        // tokens per page
  uint32_t dtype_tag = 0;        // e.g. 'F8E4'
  uint16_t tp_size = 0;
  uint16_t tp_rank = 0;
  uint16_t layer_num = 0;
  uint16_t head_num = 0;
  uint16_t head_dim = 0;
  uint16_t reserved = 0;
  uint64_t payload_len = 0;
  uint32_t crc32 = 0;            // CRC32 over payload bytes

  static ValueHeader Make(uint64_t model_hash, uint32_t page_size,
                          uint32_t dtype_tag, uint16_t flags, uint16_t tp_size,
                          uint16_t tp_rank, uint16_t layer_num, uint16_t head_num,
                          uint16_t head_dim) {
    ValueHeader h;
    h.magic = kMagic;
    h.version = kVersion;
    h.flags = flags;
    h.model_hash = model_hash;
    h.page_size = page_size;
    h.dtype_tag = dtype_tag;
    h.tp_size = tp_size;
    h.tp_rank = tp_rank;
    h.layer_num = layer_num;
    h.head_num = head_num;
    h.head_dim = head_dim;
    h.payload_len = 0;
    h.crc32 = 0;
    return h;
  }

  void SetPayload(const void* p, size_t n) {
    payload_len = n;
    crc32 = Crc32(p, n);
  }

  bool is_mla() const { return (flags & kFlagIsMla) != 0; }

  void Serialize(char* out /*[kSize]*/) const { std::memcpy(out, this, kSize); }

  static bool Parse(const char* in, size_t n, ValueHeader* out) {
    if (in == nullptr || n < kSize) return false;
    ValueHeader tmp;
    std::memcpy(&tmp, in, kSize);
    if (tmp.magic != kMagic) return false;
    *out = tmp;
    return true;
  }
};
#pragma pack(pop)

static_assert(sizeof(ValueHeader) == ValueHeader::kSize, "header must be 48B");

// Identity match: every geometry field must agree. tp_size/tp_rank are only
// compared for non-MLA (MHA/GQA) values; MLA latent is TP-layout invariant.
inline bool HeaderMatches(const ValueHeader& a, const ValueHeader& b) {
  if (a.magic != b.magic || a.version != b.version || a.flags != b.flags) return false;
  if (a.model_hash != b.model_hash || a.page_size != b.page_size ||
      a.dtype_tag != b.dtype_tag || a.layer_num != b.layer_num ||
      a.head_num != b.head_num || a.head_dim != b.head_dim) {
    return false;
  }
  if (!a.is_mla()) {
    if (a.tp_size != b.tp_size || a.tp_rank != b.tp_rank) return false;
  }
  return true;
}

}  // namespace dfkv

#endif  // DFKV_VALUE_HEADER_H_
