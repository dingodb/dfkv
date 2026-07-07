// TDD R2 — key_map: sglang page-hash string -> deterministic BlockKey.
// CRITICAL (F1): size must be a FIXED constant so Put/Get/Exist build the SAME
// identity key (Filename) and route identically. Payload length must NOT enter
// BlockKey.size.
#include "client/key_map.h"
#include "utils/md5.h"

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
  // F1 regression: identity size is constant; never the payload length. The
  // index now carries MD5[8..12) identity bits (was always 0), so Put/Get/Exist
  // still agree because they all derive it deterministically from the same key.
  BlockKey a = ToBlockKey("k1");
  BlockKey b = ToBlockKey("k1");
  EXPECT_EQ(a.size, kKvFixedSize);
  EXPECT_EQ(b.size, kKvFixedSize);
  EXPECT_EQ(a.index, b.index);            // deterministic
  EXPECT_EQ(a.Filename(), b.Filename());  // Put/Get/Exist must agree
}

TEST(KeyMap, IdUnchangedFromMd5_64ButIndexNowCarriesIdentity) {
  // id must stay byte-identical to the old Md5_64 so ring routing is unaffected;
  // index is the newly-added identity (MD5 bytes 8..11, little-endian).
  const std::string key = "glm-5.2/page_777_k";
  BlockKey k = ToBlockKey(key);
  EXPECT_EQ(k.id, dfkv::Md5_64(key)) << "id must equal the legacy 64-bit hash";
  uint8_t d[16];
  dfkv::Md5(key.data(), key.size(), d);
  uint32_t want_index = uint32_t(d[8]) | (uint32_t(d[9]) << 8) |
                        (uint32_t(d[10]) << 16) | (uint32_t(d[11]) << 24);
  EXPECT_EQ(k.index, want_index);
}

TEST(KeyMap, IndexParticipatesInIdentity96Bit) {
  // Distinct keys differ in the full 96-bit identity (id,index); index actually
  // carries hash bits (non-zero for essentially every key).
  std::set<std::pair<uint64_t, uint32_t>> ids;
  size_t nonzero_index = 0;
  for (int i = 0; i < 50000; ++i) {
    BlockKey k = ToBlockKey("glm-5.2/pg_" + std::to_string(i) + "_k");
    ids.insert({k.id, k.index});
    if (k.index != 0) ++nonzero_index;
  }
  EXPECT_EQ(ids.size(), 50000u);
  EXPECT_GT(nonzero_index, 49900u) << "index must actually carry hash bits";
}

TEST(KeyMap, DifferentKeysGiveDifferentIdsNoCollisionInSample) {
  std::set<uint64_t> ids;
  for (int i = 0; i < 20000; ++i) {
    ids.insert(ToBlockKey("glm-5.1/page_" + std::to_string(i) + "_k").id);
  }
  EXPECT_EQ(ids.size(), 20000u);  // 64-bit hash: no collision in 20k sample
}

TEST(KeyMap, ModelHashZeroIsLegacyRawKeyHash) {
  // model_hash==0 (unset / "no isolation") must stay byte-identical to the old
  // library so existing data and a mixed old/new fleet are unaffected.
  const std::string key = "glm-5.2/page_777_k";
  EXPECT_EQ(ToBlockKey(key, 0).id, dfkv::Md5_64(key));
  EXPECT_EQ(ToBlockKey(key).id, dfkv::Md5_64(key));  // default arg == 0
}

TEST(KeyMap, DifferentModelHashGivesDifferentKey) {
  // The fix: a different model_hash for the SAME content yields a different
  // BlockKey, so EXIST/GET/PUT/REMOVE are scoped by model_hash at the key layer
  // (EXIST no longer cross-matches another model_hash's key). Deterministic per
  // (key, model_hash).
  const std::string key = "glm-5.2/pg_42_k";
  BlockKey a1 = ToBlockKey(key, 111111);
  BlockKey a2 = ToBlockKey(key, 111111);
  BlockKey b = ToBlockKey(key, 222222);
  EXPECT_EQ(a1.Filename(), a2.Filename());          // deterministic per model_hash
  EXPECT_NE(a1.Filename(), b.Filename());           // different model_hash -> different key
  EXPECT_NE(ToBlockKey(key, 0).Filename(), b.Filename());  // and different from legacy
}

TEST(KeyMap, NoCrossModelHashCollisionInSample) {
  // Same content under many model_hashes must all land on distinct identities.
  std::set<std::pair<uint64_t, uint32_t>> ids;
  const std::string key = "glm-5.2/shared_content_k";
  for (uint64_t mh = 1; mh <= 20000; ++mh) {
    BlockKey k = ToBlockKey(key, mh);
    ids.insert({k.id, k.index});
  }
  EXPECT_EQ(ids.size(), 20000u);
}

TEST(KeyMap, FilenameAndStoreKeyFormat) {
  BlockKey k{123456789ULL, 0u, kKvFixedSize};
  EXPECT_EQ(k.Filename(), "123456789_0_" + std::to_string(kKvFixedSize));
  // StoreKey buckets by id/1e6 then id/1e3, mirroring dingofs layout.
  EXPECT_EQ(k.StoreKey(),
            "blocks/123/123456/" + k.Filename());
}
