// TDD R1 — value_header: fixed 48B header guarding against silent wrong-reads.
#include "value_header.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>

using dfkv::ValueHeader;
using dfkv::Crc32;
using dfkv::HeaderMatches;

namespace {

ValueHeader MakeSelf() {
  // identity = (model, page_size, dtype, tp_size, tp_rank, layer, head, head_dim)
  return ValueHeader::Make(/*model_hash=*/0xABCD1234ULL, /*page_size=*/64,
                           /*dtype_tag=*/0x46384534u /*'F8E4'*/, /*flags=*/1,
                           /*tp_size=*/8, /*tp_rank=*/0, /*layer_num=*/78,
                           /*head_num=*/1, /*head_dim=*/576);
}

}  // namespace

TEST(ValueHeader, IsExactly48Bytes) {
  EXPECT_EQ(sizeof(ValueHeader), 48u);
  EXPECT_EQ(ValueHeader::kSize, 48u);
}

TEST(ValueHeader, SerializeParseRoundTripPreservesFields) {
  std::string payload = "hello-kv-cache-payload";
  ValueHeader h = MakeSelf();
  h.SetPayload(payload.data(), payload.size());

  char buf[ValueHeader::kSize];
  h.Serialize(buf);

  ValueHeader got;
  ASSERT_TRUE(ValueHeader::Parse(buf, sizeof(buf), &got));
  EXPECT_EQ(got.model_hash, h.model_hash);
  EXPECT_EQ(got.page_size, h.page_size);
  EXPECT_EQ(got.dtype_tag, h.dtype_tag);
  EXPECT_EQ(got.tp_size, h.tp_size);
  EXPECT_EQ(got.tp_rank, h.tp_rank);
  EXPECT_EQ(got.layer_num, h.layer_num);
  EXPECT_EQ(got.head_num, h.head_num);
  EXPECT_EQ(got.head_dim, h.head_dim);
  EXPECT_EQ(got.flags, h.flags);
  EXPECT_EQ(got.payload_len, payload.size());
  EXPECT_EQ(got.crc32, h.crc32);
}

TEST(ValueHeader, CrcMatchesPayloadAndDetectsCorruption) {
  std::string payload = "deadbeef-kv";
  ValueHeader h = MakeSelf();
  h.SetPayload(payload.data(), payload.size());
  EXPECT_EQ(h.crc32, Crc32(payload.data(), payload.size()));

  std::string corrupt = payload;
  corrupt[0] ^= 0xFF;
  EXPECT_NE(h.crc32, Crc32(corrupt.data(), corrupt.size()));
}

TEST(Crc32c, MatchesKnownCastagnoliCheckVector) {
  // CRC32C (Castagnoli) standard check value for the ASCII string "123456789".
  const char* s = "123456789";
  EXPECT_EQ(Crc32(s, 9), 0xE3069283u);
  EXPECT_EQ(Crc32("", 0), 0u);  // CRC of empty == 0 (init ^ xorout)
}

TEST(Crc32c, HardwareMatchesScalarOnRandomBuffers) {
  // The SSE4.2 path and the table fallback must produce identical CRC32C.
  std::string buf;
  uint32_t x = 0x12345678u;
  for (size_t len : {0u, 1u, 7u, 8u, 9u, 63u, 64u, 1000u, 65537u}) {
    buf.resize(len);
    for (size_t i = 0; i < len; ++i) { x = x * 1103515245u + 12345u; buf[i] = char(x >> 16); }
#if defined(__x86_64__) || defined(__i386__)
    EXPECT_EQ(dfkv::detail::Crc32cScalar(buf.data(), len),
              dfkv::detail::Crc32cHw(buf.data(), len)) << "len=" << len;
#endif
    EXPECT_EQ(Crc32(buf.data(), len), dfkv::detail::Crc32cScalar(buf.data(), len)) << "len=" << len;
  }
}

TEST(ValueHeader, ParseRejectsBadMagicAndShortBuffer) {
  ValueHeader h = MakeSelf();
  char buf[ValueHeader::kSize];
  h.Serialize(buf);

  ValueHeader got;
  // short buffer
  EXPECT_FALSE(ValueHeader::Parse(buf, ValueHeader::kSize - 1, &got));
  // corrupt magic
  buf[0] ^= 0xFF;
  EXPECT_FALSE(ValueHeader::Parse(buf, sizeof(buf), &got));
}

TEST(ValueHeader, MatchesIsTrueForSameIdentityFalseOnAnyDrift) {
  ValueHeader a = MakeSelf();
  ValueHeader b = MakeSelf();
  EXPECT_TRUE(HeaderMatches(a, b));

  // each identity field drift => mismatch
  b = MakeSelf(); b.page_size = 32;     EXPECT_FALSE(HeaderMatches(a, b));
  b = MakeSelf(); b.model_hash = 0xDEAD; EXPECT_FALSE(HeaderMatches(a, b));
  b = MakeSelf(); b.dtype_tag = 0xBF16;  EXPECT_FALSE(HeaderMatches(a, b));
  b = MakeSelf(); b.layer_num = 80;      EXPECT_FALSE(HeaderMatches(a, b));
  b = MakeSelf(); b.head_dim = 512;      EXPECT_FALSE(HeaderMatches(a, b));
  b = MakeSelf(); b.version += 1;        EXPECT_FALSE(HeaderMatches(a, b));
}

TEST(ValueHeader, MlaIgnoresTpRankInMatch) {
  // MLA latent is rank-invariant: tp_size/tp_rank must NOT gate a match when is_mla(flag bit0).
  ValueHeader a = MakeSelf();  // flags=1 => is_mla
  ValueHeader b = MakeSelf();
  b.tp_rank = 3;
  b.tp_size = 4;
  EXPECT_TRUE(HeaderMatches(a, b));
}
