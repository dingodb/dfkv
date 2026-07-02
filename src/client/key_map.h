/* key_map: SGLang page-hash string -> deterministic BlockKey.
 * F1 fix: BlockKey.size is a FIXED constant (never payload length) so Put/Get/
 * Exist build identical Filename() and route to the same node. */
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
  return BlockKey{Md5_64(key), 0u, kKvFixedSize};
}

}  // namespace dfkv

#endif  // DFKV_KEY_MAP_H_
