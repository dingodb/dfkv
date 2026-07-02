// Server dual-accept (wire v2): a KvNodeServer and an MdsServer must decode both
// v1 and v2 request frames and reply in the SAME version (echoing the v2 seq),
// so a v1 and a v2 client share one server with no flag-day upgrade.
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "cache/kv_node_server.h"
#include "mds/mds_server.h"
#include "transport/wire.h"
#include "utils/net_util.h"

using namespace dfkv;  // NOLINT
namespace fs = std::filesystem;

namespace {

// Send a raw request frame (v1 or v2 per `seq_or_v1`) and read the reply prefix;
// returns the reply version, and fills status / data_len / echoed seq.
uint8_t RoundTripRaw(int port, WireOp op, const BlockKey& k, uint64_t offset,
                     uint64_t length, const std::string& payload, bool v2,
                     uint64_t req_seq, Status* st, uint64_t* dlen, uint64_t* seq_out,
                     std::string* data_out) {
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 2000);
  if (fd < 0) return 0;
  char pre[kReqPrefixV2];
  size_t plen;
  if (v2) { EncodeReqV2(pre, op, k, offset, length, payload.size(), req_seq); plen = kReqPrefixV2; }
  else { EncodeReq(pre, op, k, offset, length, payload.size()); plen = kReqPrefix; }
  if (!net::WriteAll(fd, pre, plen) ||
      (!payload.empty() && !net::WriteAll(fd, payload.data(), payload.size()))) {
    ::close(fd); return 0;
  }
  char rp[kRespPrefixV2];
  if (!net::ReadAll(fd, rp, kRespPrefix)) { ::close(fd); return 0; }
  if (static_cast<uint8_t>(rp[0]) == kProtoVersionV2 &&
      !net::ReadAll(fd, rp + kRespPrefix, kRespPrefixV2 - kRespPrefix)) { ::close(fd); return 0; }
  uint8_t ver = DecodeResp(rp, st, dlen, kMaxFrameLen, seq_out);
  if (ver && *dlen && data_out) { data_out->resize(*dlen); net::ReadAll(fd, &(*data_out)[0], *dlen); }
  ::close(fd);
  return ver;
}

const char* EtcdEp() { return std::getenv("DFKV_TEST_ETCD"); }

}  // namespace

TEST(WireV2Server, KvNodeServerRepliesInRequestVersion) {
  fs::path dir = fs::temp_directory_path() / "dfkv_wirev2_node";
  fs::remove_all(dir);
  fs::create_directories(dir);
  KvNodeServer srv(dir.string(), 1ull << 30);
  ASSERT_EQ(srv.Start(0), Status::kOk);
  int port = srv.port();

  std::string val = "wire-v2-value";
  BlockKey k{0x777, 0, 1};
  Status st; uint64_t dlen = 0, seq = 0; std::string data;

  // v2 PUT then v2 GET: reply is v2 and echoes the request seq.
  EXPECT_EQ(RoundTripRaw(port, WireOp::kCache, k, 0, 0, val, /*v2=*/true, 0x1234,
                         &st, &dlen, &seq, &data), kProtoVersionV2);
  EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(seq, 0x1234u) << "v2 reply must echo the request seq";

  EXPECT_EQ(RoundTripRaw(port, WireOp::kRange, k, 0, val.size(), "", /*v2=*/true,
                         0x9999, &st, &dlen, &seq, &data), kProtoVersionV2);
  EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(seq, 0x9999u);
  EXPECT_EQ(data, val);

  // v1 GET on the same server: reply is v1 (10-byte prefix, no seq).
  EXPECT_EQ(RoundTripRaw(port, WireOp::kRange, k, 0, val.size(), "", /*v2=*/false,
                         0, &st, &dlen, &seq, &data), kProtoVersion);
  EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(data, val);

  EXPECT_EQ(srv.m_wire_v2(), 2u);  // the two v2 requests
  EXPECT_GE(srv.m_wire_v1(), 1u);  // the v1 request (+ any Start-time noise: none)
  srv.Stop();
  fs::remove_all(dir);
}

TEST(WireV2Server, MdsServerRepliesInRequestVersion) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  std::string group = "wirev2-" + std::to_string(port);

  Status st; uint64_t dlen = 0, seq = 0; std::string data;
  // v2 kListMembers (empty group is a clean kOk empty list) echoes the seq.
  uint8_t ver = RoundTripRaw(port, WireOp::kListMembers, BlockKey{}, 0, 0, group,
                             /*v2=*/true, 0x5150, &st, &dlen, &seq, &data);
  EXPECT_EQ(ver, kProtoVersionV2);
  EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(seq, 0x5150u);

  // v1 on the same MDS still works and replies v1.
  ver = RoundTripRaw(port, WireOp::kListMembers, BlockKey{}, 0, 0, group,
                     /*v2=*/false, 0, &st, &dlen, &seq, &data);
  EXPECT_EQ(ver, kProtoVersion);
  EXPECT_EQ(st, Status::kOk);
  mds.Stop();
}
