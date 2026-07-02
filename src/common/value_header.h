/*
 * DingoFS KV-cache for SGLang HiCache — value header.
 *
 * Every stored KV value is prefixed with this fixed 48-byte header so a read
 * whose model / page-size / dtype / layer geometry does not match the reader's
 * expectation is treated as a MISS (recompute) instead of silently returning
 * wrong bytes. For MLA models the latent is TP-layout invariant, so
 * tp_size/tp_rank are NOT part of the match (flags bit0 = is_mla). Payload
 * integrity is left to the transport (RC RDMA / RoCE carry a hardware ICRC); the
 * header no longer carries a payload checksum (was CRC32C through v2) — this also
 * keeps the zero-copy datapath zero-TOUCH (the CPU never reads the payload).
 *
 * Portable (no brpc/MDS deps) so it builds & is unit-tested standalone.
 * Serialization is host-endian (memcpy); all targets are x86_64 LE. A read on
 * a different-endian host fails the magic/version check => safe MISS (recompute),
 * never silent corruption. Cross-arch sharing would need explicit LE fields.
 */
#ifndef DFKV_VALUE_HEADER_H_
#define DFKV_VALUE_HEADER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dfkv {

#pragma pack(push, 1)
struct ValueHeader {
  static constexpr size_t kSize = 48;
  static constexpr uint32_t kMagic = 0x444F4B56u;  // 'DFKV'
  static constexpr uint16_t kVersion = 3;  // v3: dropped payload checksum (was CRC32C in v2, IEEE in v1)
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
  uint32_t reserved2 = 0;        // was CRC32 over payload (v2); reserved for future use

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
    h.reserved2 = 0;
    return h;
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
