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
  buf[0] = static_cast<char>(kProtoVersion + 1);  // version skew
  ReqFields rq;
  EXPECT_FALSE(DecodeReq(buf, &rq));
}

TEST(Wire, RespRejectsBadVersion) {
  char buf[kRespPrefix];
  EncodeResp(buf, Status::kOk, 0);
  buf[0] = static_cast<char>(kProtoVersion + 1);
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
