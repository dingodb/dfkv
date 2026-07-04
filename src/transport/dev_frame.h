/* DCP1 ("Declared Caps") -- a capacity declaration hidden in the RDMA
 * bootstrap device-name frame, using the same deterministic-zero-pad trick as
 * DPQ1: the client has always memset the 32-byte frame before writing the
 * NUL-terminated device name, and the server has always stopped reading at
 * the first NUL. So the tail is a compatible extension area:
 *   [ name bytes | \0 | magic "DCP1" u32 | max_block_bytes u64 | zeros... ]
 * max_block_bytes is the largest value payload the client will ever PUT or
 * GET on this connection (connectors know their block geometry exactly); the
 * server sizes that connection's per-slot buffers to it instead of the global
 * worst case (issue #110). 0 / absent / no room after a long name =
 * undeclared = worst-case sizing (exactly the old behavior).
 * Header-only and verbs-free so the codec is unit-testable in non-RDMA
 * builds; rdma_verbs.h re-exports it for the transport/server. */
#ifndef DFKV_TRANSPORT_DEV_FRAME_H_
#define DFKV_TRANSPORT_DEV_FRAME_H_

#include <cstdint>
#include <cstring>
#include <string>

namespace dfkv::rdma {

constexpr size_t kDevNameBytes = 32;
constexpr uint32_t kDevCapsMagic = 0x31504344u;  // ASCII "DCP1" (LE)

// Writes name + optional DCP1 tail. max_block_bytes==0 or a name too long to
// leave 13 tail bytes -> plain legacy frame.
inline void EncodeDevFrame(const std::string& dev, uint64_t max_block_bytes,
                           char out[kDevNameBytes]) {
  std::memset(out, 0, kDevNameBytes);
  const size_t n = dev.size() < kDevNameBytes - 1 ? dev.size() : kDevNameBytes - 1;
  std::memcpy(out, dev.data(), n);
  if (max_block_bytes == 0 || n + 1 + 4 + 8 > kDevNameBytes) return;
  std::memcpy(out + n + 1, &kDevCapsMagic, 4);
  std::memcpy(out + n + 5, &max_block_bytes, 8);
}

// Returns the declared max block bytes, or 0 when absent/legacy/garbled.
inline uint64_t ParseDevFrameCaps(const char in[kDevNameBytes]) {
  size_t nul = 0;
  while (nul < kDevNameBytes && in[nul] != '\0') ++nul;
  if (nul + 1 + 4 + 8 > kDevNameBytes) return 0;
  uint32_t magic = 0;
  std::memcpy(&magic, in + nul + 1, 4);
  if (magic != kDevCapsMagic) return 0;
  uint64_t v = 0;
  std::memcpy(&v, in + nul + 5, 8);
  return v;
}

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_DEV_FRAME_H_
