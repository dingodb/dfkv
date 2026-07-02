/* key_map: SGLang page-hash string -> deterministic BlockKey.
 * F1 fix: BlockKey.size is a FIXED constant (never payload length) so Put/Get/
 * Exist build identical Filename() and route to the same node.
 *
 * Block identity is 96 bits: id = MD5[0..8) (LE u64), index = MD5[8..12) (LE
 * u32). The 64-bit id alone gave a birthday-collision probability of a few
 * percent at the ~1e9 lifetime-write scale of a 5 TiB x N-node ring -- and a
 * collision is NOT a miss but a SILENT cross-key read (the geometry header only
 * checks model/page/dtype/layer, so two same-model pages that collide pass
 * validation and one KV block is served under the other's key). Extending
 * identity into the (previously always-0) index field -- which Filename()/
 * StoreKey()/operator== already incorporate -- cuts collision probability by
 * 2^32 with no wire or storage-path change. id stays byte-identical to the old
 * Md5_64, so routing (a separate Md5_32 hash) is unaffected; only the on-disk
 * key changes, so a mixed old/new fleet cross-misses once (clean cache warm-up,
 * never corruption). */
#ifndef DFKV_KEY_MAP_H_
#define DFKV_KEY_MAP_H_

#include <cstdint>
#include <string>

#include "common/kv_types.h"
#include "utils/md5.h"

namespace dfkv {

// Identity-only size constant. Real payload length lives in the value header.
inline constexpr uint32_t kKvFixedSize = 1;

inline BlockKey ToBlockKey(const std::string& key) {
  uint8_t d[16];
  Md5(key.data(), key.size(), d);  // one hash; split into id + index
  uint64_t id = 0;
  for (int i = 0; i < 8; ++i) id |= uint64_t(d[i]) << (8 * i);         // bytes 0..7 (== Md5_64)
  uint32_t index = uint32_t(d[8]) | (uint32_t(d[9]) << 8) |
                   (uint32_t(d[10]) << 16) | (uint32_t(d[11]) << 24);   // bytes 8..11
  return BlockKey{id, index, kKvFixedSize};
}

}  // namespace dfkv

#endif  // DFKV_KEY_MAP_H_
