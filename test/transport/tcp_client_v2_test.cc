// TCP client wire v2: a client with DFKV_WIRE_VERSION=2 sends v2 frames (seq)
// and validates the echoed seq against a live dual-accept KvNodeServer. Also
// checks the default (v1) client still works, and that a seq-mismatching peer
// is rejected (connection dropped, not a misattributed reply).
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#include "cache/kv_node_server.h"
#include "transport/tcp_transport.h"
#include "transport/wire.h"
#include "utils/net_util.h"

using namespace dfkv;  // NOLINT
namespace fs = std::filesystem;

namespace {
// A KvNodeServer over a temp dir; RAII teardown.
struct Node {
  fs::path dir;
  KvNodeServer srv;
  explicit Node(const std::string& tag)
      : dir(fs::temp_directory_path() / ("dfkv_tcpv2_" + tag)),
        srv((fs::remove_all(dir), fs::create_directories(dir), dir.string()),
            1ull << 30) {
    EXPECT_EQ(srv.Start(0), Status::kOk);
  }
  ~Node() { srv.Stop(); fs::remove_all(dir); }
  std::string node() const { return "127.0.0.1:" + std::to_string(srv.port()); }
};

// NOTE: ClientWireVersion() caches DFKV_WIRE_VERSION on first use, so a single
// test binary can't flip it at runtime. This test asserts the value the harness
// was launched with (see the CMake test env below); it exercises v2 when the
// env selects it and v1 otherwise, and the wire-level v2 behavior is covered
// hermetically by wire_v2_server_test regardless.
}  // namespace

TEST(TcpClientV2, RoundTripsAgainstDualAcceptServer) {
  Node n("rt");
  TcpTransport t;
  t.set_timeouts(2000, 3000);

  std::string v(1000, 'v');
  ASSERT_EQ(t.Cache(n.node(), BlockKey{0x1234, 0, 1}, v.data(), v.size()),
            Status::kOk);
  std::string out;
  ASSERT_EQ(t.Range(n.node(), BlockKey{0x1234, 0, 1}, 0, v.size(), &out),
            Status::kOk);
  EXPECT_EQ(out, v);
  bool e = false;
  ASSERT_EQ(t.Exist(n.node(), BlockKey{0x1234, 0, 1}, &e), Status::kOk);
  EXPECT_TRUE(e);

  // If launched with DFKV_WIRE_VERSION=2, the server must have counted v2
  // requests; otherwise v1. Either way the round trips above succeeded.
  const char* env = std::getenv("DFKV_WIRE_VERSION");
  const bool want_v2 = env && std::string(env) == "2";
  EXPECT_EQ(ClientWireVersion(), want_v2 ? kProtoVersionV2 : kProtoVersion);
  if (want_v2) {
    EXPECT_GT(n.srv.m_wire_v2(), 0u);
    EXPECT_EQ(n.srv.m_wire_v1(), 0u);
  } else {
    EXPECT_GT(n.srv.m_wire_v1(), 0u);
    EXPECT_EQ(n.srv.m_wire_v2(), 0u);
  }
}

// A fake server that echoes the WRONG seq in a v2 reply must be rejected by a
// v2 client (the round trip fails), and accepted by a v1 client (no seq check).
TEST(TcpClientV2, SeqMismatchIsRejectedOnV2Only) {
  int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(lfd, 0);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ::bind(lfd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  ::listen(lfd, 4);
  socklen_t sl = sizeof(sa);
  ::getsockname(lfd, reinterpret_cast<sockaddr*>(&sa), &sl);
  int port = ntohs(sa.sin_port);

  std::thread srv([lfd] {
    int fd = ::accept(lfd, nullptr, nullptr);
    if (fd < 0) return;
    char pre[kReqPrefixV2];
    if (net::ReadAll(fd, pre, kReqPrefix)) {
      const bool v2 = static_cast<uint8_t>(pre[0]) == kProtoVersionV2;
      if (v2) net::ReadAll(fd, pre + kReqPrefix, kReqPrefixV2 - kReqPrefix);
      ReqFields rq;
      DecodeReq(pre, &rq);
      if (rq.payload_len) { std::string s(rq.payload_len, 0); net::ReadAll(fd, &s[0], rq.payload_len); }
      char rp[kRespPrefixV2];
      size_t rlen;
      if (v2) { EncodeRespV2(rp, Status::kOk, 0, rq.seq ^ 0xdeadbeef); rlen = kRespPrefixV2; }  // WRONG seq
      else { EncodeResp(rp, Status::kOk, 0); rlen = kRespPrefix; }
      net::WriteAll(fd, rp, rlen);
    }
    ::close(fd);
  });

  TcpTransport t;
  t.set_timeouts(2000, 3000);
  std::string node = "127.0.0.1:" + std::to_string(port);
  // Cache is a status-only op; a v2 client must reject the bad-seq reply
  // (kIOError), while a v1 client accepts it (no seq to check).
  Status st = t.Cache(node, BlockKey{1, 0, 1}, "x", 1);
  if (ClientWireVersion() == kProtoVersionV2)
    EXPECT_EQ(st, Status::kIOError) << "v2 client must reject a mismatched seq";
  else
    EXPECT_EQ(st, Status::kOk) << "v1 client has no seq to validate";

  srv.join();
  ::close(lfd);
}
