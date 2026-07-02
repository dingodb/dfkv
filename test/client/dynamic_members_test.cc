// TDD R12 — dynamic membership: KVClient.SetMembers() rebuilds the ring at
// runtime so adding a node re-routes new keys without recreating the client.
#include "client/kv_client.h"
#include "cache/kv_node_server.h"
#include "common/value_header.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
ValueHeader Hdr() {
  return ValueHeader::Make(0x51, 64, 0x46384534u, ValueHeader::kFlagIsMla, 8, 0, 78, 1, 576);
}
struct Node { fs::path dir; std::unique_ptr<KvNodeServer> srv; std::string addr; };
std::unique_ptr<Node> Start(const std::string& tag) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_dyn_" + tag);
  fs::remove_all(n->dir); fs::create_directories(n->dir);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}
}  // namespace

TEST(DynamicMembers, AddingNodeReroutesNewKeys) {
  auto a = Start("a"); auto b = Start("b");
  KVClient c({{"a", a->addr}}, Hdr());  // start with only node a
  std::string v(64, 'v');
  for (int i = 0; i < 60; ++i)
    ASSERT_TRUE(c.Put("p1_" + std::to_string(i), v.data(), v.size()));
  EXPECT_EQ(b->srv->Count(), 0u);  // b not in the ring yet

  c.SetMembers(std::vector<std::pair<std::string,std::string>>{{"a", a->addr}, {"b", b->addr}});  // hot add node b
  for (int i = 0; i < 60; ++i)
    ASSERT_TRUE(c.Put("p2_" + std::to_string(i), v.data(), v.size()));
  EXPECT_GT(b->srv->Count(), 0u);  // some new keys now land on b

  // keys still readable (those routed to their current owner)
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c.Get("p2_0", &out[0], out.size()));
  a->srv->Stop(); b->srv->Stop();
}
