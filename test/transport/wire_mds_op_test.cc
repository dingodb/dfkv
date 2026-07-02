#include "transport/wire.h"
#include <gtest/gtest.h>
using namespace dfkv;  // NOLINT

// The MDS ops reuse the existing request framing; only the op byte is new.
TEST(WireMdsOp, NewOpsRoundTripThroughReqPrefix) {
  for (WireOp op : {WireOp::kRegister, WireOp::kHeartbeat, WireOp::kListMembers}) {
    char buf[kReqPrefix];
    EncodeReq(buf, op, BlockKey{}, /*offset=*/0, /*length=*/0, /*payload_len=*/7);
    ReqFields f{};
    ASSERT_TRUE(DecodeReq(buf, &f));
    EXPECT_EQ(f.op, static_cast<uint8_t>(op));
    EXPECT_EQ(f.payload_len, 7u);
  }
}

TEST(WireMdsOp, OpValuesAreStableAndDistinct) {
  EXPECT_EQ(static_cast<uint8_t>(WireOp::kRegister), 6);
  EXPECT_EQ(static_cast<uint8_t>(WireOp::kHeartbeat), 7);
  EXPECT_EQ(static_cast<uint8_t>(WireOp::kListMembers), 8);
}
