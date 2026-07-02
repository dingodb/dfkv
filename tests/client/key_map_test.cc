// TDD R2 — key_map: sglang page-hash string -> deterministic BlockKey.
// CRITICAL (F1): size must be a FIXED constant so Put/Get/Exist build the SAME
// identity key (Filename) and route identically. Payload length must NOT enter
// BlockKey.size.
#include "client/key_map.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

using dfkv::BlockKey;
using dfkv::ToBlockKey;
using dfkv::kKvFixedSize;

TEST(KeyMap, DeterministicSameKeySameId) {
  EXPECT_EQ(ToBlockKey("glm-5.1/abc123_k").id, ToBlockKey("glm-5.1/abc123_k").id);
}

TEST(KeyMap, SizeIsFixedConstantRegardlessOfCaller) {
  // F1 regression: identity size is constant; never the payload length.
  BlockKey a = ToBlockKey("k1");
  BlockKey b = ToBlockKey("k1");
  EXPECT_EQ(a.size, kKvFixedSize);
  EXPECT_EQ(b.size, kKvFixedSize);
  EXPECT_EQ(a.index, 0u);
  EXPECT_EQ(a.Filename(), b.Filename());  // Put/Get/Exist must agree
}

TEST(KeyMap, DifferentKeysGiveDifferentIdsNoCollisionInSample) {
  std::set<uint64_t> ids;
  for (int i = 0; i < 20000; ++i) {
    ids.insert(ToBlockKey("glm-5.1/page_" + std::to_string(i) + "_k").id);
  }
  EXPECT_EQ(ids.size(), 20000u);  // 64-bit hash: no collision in 20k sample
}

TEST(KeyMap, FilenameAndStoreKeyFormat) {
  BlockKey k{123456789ULL, 0u, kKvFixedSize};
  EXPECT_EQ(k.Filename(), "123456789_0_" + std::to_string(kKvFixedSize));
  // StoreKey buckets by id/1e6 then id/1e3, mirroring dingofs layout.
  EXPECT_EQ(k.StoreKey(),
            "blocks/123/123456/" + k.Filename());
}
