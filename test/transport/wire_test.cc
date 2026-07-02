// Wire-framing encode/decode: round-trips plus the negative paths a hostile or
// version-skewed peer can take (bad version byte, oversized declared length).
#include "transport/wire.h"

#include <cstring>

#include <gtest/gtest.h>

using namespace dfkv;

TEST(Wire, ReqRoundTrip) {
  char buf[kReqPrefix];
  BlockKey k{0x1122334455667788ull, 0xAABBCCDD, 0x12345678};
  EncodeReq(buf, WireOp::kCache, k, 4096, 8192, 65536);
  ReqFields rq;
  ASSERT_TRUE(DecodeReq(buf, &rq));
  EXPECT_EQ(rq.op, static_cast<uint8_t>(WireOp::kCache));
  EXPECT_EQ(rq.id, k.id);
  EXPECT_EQ(rq.index, k.index);
  EXPECT_EQ(rq.size, k.size);
  EXPECT_EQ(rq.offset, 4096u);
  EXPECT_EQ(rq.length, 8192u);
  EXPECT_EQ(rq.payload_len, 65536u);
}

TEST(Wire, RespRoundTrip) {
  char buf[kRespPrefix];
  EncodeResp(buf, Status::kOk, 1u << 20);
  Status st = Status::kInvalid;
  uint64_t dlen = 0;
  ASSERT_TRUE(DecodeResp(buf, &st, &dlen));
  EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(dlen, 1u << 20);
}

TEST(Wire, ReqRejectsBadVersion) {
  char buf[kReqPrefix];
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, 0);
  buf[0] = static_cast<char>(kProtoVersionV2 + 1);  // unknown version (>v2)
  ReqFields rq;
  EXPECT_FALSE(DecodeReq(buf, &rq));
}

TEST(Wire, RespRejectsBadVersion) {
  char buf[kRespPrefix];
  EncodeResp(buf, Status::kOk, 0);
  buf[0] = static_cast<char>(kProtoVersionV2 + 1);  // unknown version (>v2)
  Status st = Status::kInvalid;
  uint64_t dlen = 0;
  EXPECT_FALSE(DecodeResp(buf, &st, &dlen));
}

TEST(Wire, ReqRejectsOversizedPayload) {
  char buf[kReqPrefix];
  // A garbage/hostile 64-bit length must NOT decode (else it drives a huge alloc).
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, kMaxFrameLen + 1);
  ReqFields rq;
  EXPECT_FALSE(DecodeReq(buf, &rq));               // rejected by default ceiling
  EXPECT_TRUE(DecodeReq(buf, &rq, kMaxFrameLen + 1));  // explicit higher bound accepts
  EXPECT_EQ(rq.payload_len, kMaxFrameLen + 1);
}

TEST(Wire, ReqAcceptsPayloadAtCeiling) {
  char buf[kReqPrefix];
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, kMaxFrameLen);
  ReqFields rq;
  EXPECT_TRUE(DecodeReq(buf, &rq));  // exactly at the ceiling is allowed
  EXPECT_EQ(rq.payload_len, kMaxFrameLen);
}

TEST(Wire, ReqRejectsOverTighterBound) {
  char buf[kReqPrefix];
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, 4097);
  ReqFields rq;
  EXPECT_FALSE(DecodeReq(buf, &rq, 4096));  // caller-supplied tighter cap enforced
  EXPECT_TRUE(DecodeReq(buf, &rq, 4097));
}

TEST(Wire, RespRejectsOversizedData) {
  char buf[kRespPrefix];
  EncodeResp(buf, Status::kOk, kMaxFrameLen + 1);
  Status st = Status::kInvalid;
  uint64_t dlen = 0;
  EXPECT_FALSE(DecodeResp(buf, &st, &dlen));
  EXPECT_TRUE(DecodeResp(buf, &st, &dlen, kMaxFrameLen + 1));
  EXPECT_EQ(dlen, kMaxFrameLen + 1);
}

TEST(Wire, ReqV2RoundTripCarriesSeq) {
  char buf[kReqPrefixV2];
  BlockKey k{0x1122334455667788ull, 0xAABBCCDD, 0x12345678};
  EncodeReqV2(buf, WireOp::kRange, k, 4096, 8192, 65536, 0xDEADBEEFCAFEull);
  ReqFields rq;
  EXPECT_EQ(DecodeReq(buf, &rq), kProtoVersionV2);
  EXPECT_EQ(rq.op, static_cast<uint8_t>(WireOp::kRange));
  EXPECT_EQ(rq.id, k.id);
  EXPECT_EQ(rq.payload_len, 65536u);
  EXPECT_EQ(rq.seq, 0xDEADBEEFCAFEull);
}

TEST(Wire, ReqV1DecodesWithZeroSeqAndVersionOne) {
  char buf[kReqPrefix];
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, 0);
  ReqFields rq;
  EXPECT_EQ(DecodeReq(buf, &rq), kProtoVersion);
  EXPECT_EQ(rq.seq, 0u) << "a v1 frame has no seq";
}

TEST(Wire, RespV2RoundTripEchoesSeq) {
  char buf[kRespPrefixV2];
  EncodeRespV2(buf, Status::kOk, 1u << 20, 0x0102030405060708ull);
  Status st = Status::kInvalid;
  uint64_t dlen = 0, seq = 0;
  EXPECT_EQ(DecodeResp(buf, &st, &dlen, kMaxFrameLen, &seq), kProtoVersionV2);
  EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(dlen, 1u << 20);
  EXPECT_EQ(seq, 0x0102030405060708ull);
}

TEST(Wire, RespV1DecodesWithVersionOneAndZeroSeq) {
  char buf[kRespPrefix];
  EncodeResp(buf, Status::kNotFound, 0);
  Status st = Status::kInvalid;
  uint64_t dlen = 0, seq = 0xff;
  EXPECT_EQ(DecodeResp(buf, &st, &dlen, kMaxFrameLen, &seq), kProtoVersion);
  EXPECT_EQ(st, Status::kNotFound);
  EXPECT_EQ(seq, 0u);
}

TEST(Wire, V2OversizePayloadStillRejected) {
  char buf[kReqPrefixV2];
  EncodeReqV2(buf, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, kMaxFrameLen + 1, 7);
  ReqFields rq;
  EXPECT_EQ(DecodeReq(buf, &rq), 0u);  // oversize -> rejected even for v2
}
