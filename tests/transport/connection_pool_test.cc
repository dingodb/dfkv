// TDD R8 — TcpTransport connection pooling + server keep-alive.
// Many sequential requests over one transport must REUSE connections (not dial
// per call), and survive a stale pooled connection.
#include "cache/kv_node_server.h"
#include "client/key_map.h"
#include "transport/tcp_transport.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
struct Node {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::string addr;
};
std::unique_ptr<Node> Start(const std::string& tag) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_pool_" + tag);
  fs::remove_all(n->dir);
  fs::create_directories(n->dir);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}
}  // namespace

TEST(ConnectionPool, SequentialRequestsReuseOneConnection) {
  auto n = Start("reuse");
  TcpTransport t;
  std::string v(256, 'p');
  const int N = 100;
  for (int i = 0; i < N; ++i) {
    BlockKey bk = ToBlockKey("k" + std::to_string(i));
    ASSERT_EQ(t.Cache(n->addr, bk, v.data(), v.size()), Status::kOk);
  }
  for (int i = 0; i < N; ++i) {
    BlockKey bk = ToBlockKey("k" + std::to_string(i));
    std::string out;
    ASSERT_EQ(t.Range(n->addr, bk, 0, v.size(), &out), Status::kOk);
    ASSERT_EQ(out, v);
  }
  // 200 ops over one transport -> connections reused, far fewer than 200 accepts
  EXPECT_LE(n->srv->AcceptCount(), 4u) << "accepts=" << n->srv->AcceptCount();
  n->srv->Stop();
}

TEST(ConnectionPool, CorrectnessAcrossManyKeys) {
  auto n = Start("correct");
  TcpTransport t;
  for (int i = 0; i < 50; ++i) {
    BlockKey bk = ToBlockKey("c" + std::to_string(i));
    std::string v = "val_" + std::to_string(i);
    ASSERT_EQ(t.Cache(n->addr, bk, v.data(), v.size()), Status::kOk);
  }
  for (int i = 0; i < 50; ++i) {
    BlockKey bk = ToBlockKey("c" + std::to_string(i));
    bool e = false;
    ASSERT_EQ(t.Exist(n->addr, bk, &e), Status::kOk);
    EXPECT_TRUE(e);
    std::string out;
    ASSERT_EQ(t.Range(n->addr, bk, 0, 64, &out), Status::kOk);
    EXPECT_EQ(out, "val_" + std::to_string(i));
  }
  n->srv->Stop();
}

TEST(ConnectionPool, SurvivesStalePooledConnection) {
  // Put via transport, restart the server (old pooled fd is now stale), then a
  // new request must transparently reconnect (retry once) — not hard-fail.
  auto n = Start("stale");
  TcpTransport t;
  BlockKey bk = ToBlockKey("warm");
  std::string v(64, 'w');
  ASSERT_EQ(t.Cache(n->addr, bk, v.data(), v.size()), Status::kOk);  // pools a conn
  // bounce the server on the SAME port to invalidate the pooled connection
  int port = n->srv->port();
  n->srv->Stop();
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  ASSERT_EQ(n->srv->Start(port), Status::kOk);
  // existing data reloaded from disk; request must reconnect through stale fd
  std::string out;
  EXPECT_EQ(t.Range(n->addr, bk, 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
  n->srv->Stop();
}
