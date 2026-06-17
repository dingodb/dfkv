// RDMA datapath test over a loopback device (Soft-RoCE / rdma_rxe in CI, or any
// real RDMA NIC). Exercises the native-verbs transport + server + versioned wire
// frame + zero-copy RangeInto + (with depth>1) the pipelined worker pool — the
// code that is otherwise only validated on real 400G hardware. Skips cleanly when
// no RDMA device is present. Built only when DFKV_WITH_RDMA is defined. Run under
// ThreadSanitizer to exercise the worker-pool / QP concurrency.
#include "kv_client.h"
#include "key_map.h"
#include "kv_node_server.h"
#include "rdma_server.h"
#include "rdma_transport.h"
#include "rdma_verbs.h"
#include "value_header.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

// Small per-buffer cap so the test stays well under a modest RLIMIT_MEMLOCK
// (RDMA pins registered memory); the test values are a few KB.
constexpr size_t kMaxMsg = 256 * 1024;

ValueHeader SelfHdr() {
  return ValueHeader::Make(0x51ULL, 64, 0x46384534u, ValueHeader::kFlagIsMla,
                           8, 0, 78, 1, 576);
}

// A cache node serving RDMA: KvNodeServer owns the DiskCacheGroup; RdmaServer
// bootstraps QPs and routes requests to it (generic handler + zero-copy range).
struct RdmaNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;  // bootstrap "ip:port" for the client member list

  explicit RdmaNode(const std::string& tag, size_t max_msg = kMaxMsg) {
    dir = fs::temp_directory_path() / ("dfkv_rdma_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);  // TCP listener owns the cache group
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, uint64_t id, uint32_t idx, uint32_t ks, uint64_t off,
               uint64_t len, const char* pl, uint64_t pll, std::string* out) {
          return srv->ProcessRequest(op, id, idx, ks, off, len, pl, pll, out);
        },
        max_msg);
    rsrv->set_range_handler(
        [this](uint64_t id, uint32_t idx, uint32_t ks, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data, size_t* out_len) {
          return srv->RangeDirect(id, idx, ks, off, len, io_buf, cap, out_data, out_len);
        });
    rsrv->set_cache_direct_handler(
        [this](uint64_t id, uint32_t idx, uint32_t ks, char* data, size_t len,
               size_t cap) {
          return srv->CacheDirect(id, idx, ks, data, len, cap);
        });
    EXPECT_EQ(rsrv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(rsrv->port());
  }
  ~RdmaNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    fs::remove_all(dir);
  }
};

bool HaveRdma() { return RdmaTransport::Available(); }

// Last value of a single-line Prometheus counter (skips the # HELP/# TYPE lines
// via rfind, which lands on the value line emitted after them).
long CounterVal(const std::string& text, const std::string& name) {
  auto p = text.rfind(name);
  if (p == std::string::npos) return -1;
  auto sp = text.find(' ', p);
  if (sp == std::string::npos) return -1;
  try { return std::stol(text.substr(sp + 1)); } catch (...) { return -1; }
}

}  // namespace

// Direct transport ExistMany: windowed batch existence probe on one connection.
// Must be correct across multiple send windows (N > depth) with mixed hit/miss.
TEST(RdmaLoopback, ExistManyWindowedMixedHitMiss) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  RdmaNode node("exm");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 80;  // exceeds a single send window (depth) to exercise looping
  for (int i = 0; i < N; ++i) {
    std::string v = "v" + std::to_string(i);
    ASSERT_TRUE(c.Put("p" + std::to_string(i), v.data(), v.size())) << i;
  }
  // Interleave present (even) and absent (odd) keys.
  std::vector<BlockKey> keys;
  for (int i = 0; i < N; ++i) {
    keys.push_back(ToBlockKey("p" + std::to_string(i)));      // present
    keys.push_back(ToBlockKey("absent" + std::to_string(i))); // miss
  }
  std::vector<char> exists;
  auto sts = rt.ExistMany(node.addr, keys, &exists);
  ASSERT_EQ(exists.size(), keys.size());
  ASSERT_EQ(sts.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    bool want = (i % 2 == 0);
    EXPECT_EQ(exists[i] != 0, want) << "i=" << i;
  }
}

// Client BatchExist over RDMA must pipeline on the per-node pooled connection
// instead of fanning out one round trip per key. Guards the perf fix: after a
// warm-up that pools a connection, a large BatchExist opens no new connections.
TEST(RdmaLoopback, BatchExistReusesPooledConn) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bex");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 64;  // > batch_concurrency (8): the old per-key fan-out opened many
  for (int i = 0; i < N; ++i) {
    std::string v = "v" + std::to_string(i);
    ASSERT_TRUE(c.Put("e" + std::to_string(i), v.data(), v.size())) << i;
  }
  std::vector<std::string> probe;
  for (int i = 0; i < N; ++i) {
    probe.push_back("e" + std::to_string(i));        // present
    probe.push_back("e" + std::to_string(i) + "_x"); // absent
  }

  // Warm the pool so exactly one connection is parked for the node.
  EXPECT_TRUE(c.Exist("e0"));
  long before = CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_GE(before, 1);

  auto er = c.BatchExist(probe);
  ASSERT_EQ(er.size(), probe.size());
  for (size_t i = 0; i < probe.size(); ++i)
    EXPECT_EQ((bool)er[i], (i % 2 == 0)) << probe[i];

  long after = CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_LE(after - before, 1)
      << "BatchExist opened " << (after - before)
      << " new conns; pipelined path should reuse the pooled connection";
}

TEST(RdmaLoopback, PutGetExistMissOverRdma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  RdmaNode node("pgem");
  RdmaTransport rt(kMaxMsg);  // first device (rxe0 in CI)
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string v(4096, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 31 + 7) & 0xFF);
  ASSERT_TRUE(c.Put("k1", v.data(), v.size()));
  EXPECT_TRUE(c.Exist("k1"));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("k1", &out[0], out.size()));
  EXPECT_EQ(out, v);

  // miss: absent key
  std::string m(v.size(), '\0');
  EXPECT_FALSE(c.Get("absent", &m[0], m.size()));
  EXPECT_FALSE(c.Exist("absent"));
}

// Observability counters: the server tallies request completions and the client
// transport tallies connections opened (+ per-rail) and MR regions.
TEST(RdmaLoopback, MetricsCountersTrackOps) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("mco");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::vector<char> pool(64 * 1024);
  c.RegisterMemory(pool.data(), pool.size());
  std::string v(2048, 'z');
  ASSERT_TRUE(c.Put("m1", v.data(), v.size()));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("m1", &out[0], out.size()));

  // server: at least the PUT + GET requests were completed
  EXPECT_GE(node.rsrv->Completions(), 2u);
  std::string srv_text = node.rsrv->MetricsText();
  EXPECT_NE(srv_text.find("dfkv_rdma_completions_total"), std::string::npos) << srv_text;

  // client transport: a connection was opened and the MR region declared
  std::string cli_text = rt.MetricsText();
  EXPECT_NE(cli_text.find("dfkv_rdma_client_conns_opened_total"), std::string::npos) << cli_text;
  EXPECT_NE(cli_text.find("dfkv_rdma_client_rail_conns_total{dev="), std::string::npos) << cli_text;
  EXPECT_NE(cli_text.find("dfkv_rdma_client_mr_regions 1"), std::string::npos) << cli_text;

  // and the client snapshot folds transport metrics in after the health metrics
  std::string snap = c.MetricsSnapshot();
  EXPECT_NE(snap.find("dfkv_client_ops_served_total"), std::string::npos) << snap;
  EXPECT_NE(snap.find("dfkv_rdma_client_conns_opened_total"), std::string::npos) << snap;
}

TEST(RdmaLoopback, BatchZeroCopyRoundtrip) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bzc");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 20;
  const size_t sz = 4096;
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "b" + std::to_string(i);
    vals[i].assign(sz, static_cast<char>((i * 13 + 1) & 0xFF));
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) EXPECT_TRUE(pr[i]) << i;

  // GET into fresh buffers (RDMA scatters payload straight in = zero copy).
  std::vector<std::string> outs(N);
  std::vector<KvGetItem> gets(N);
  for (int i = 0; i < N; ++i) { outs[i].assign(sz, '\0'); gets[i] = {keys[i], &outs[i][0], sz}; }
  auto gr = c.BatchGet(gets);
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(gr[i]) << i;
    EXPECT_EQ(outs[i], vals[i]) << i;
  }
}

// Single Put/Get always take the zero-copy fast path on RDMA (register caller
// buffer + scatter-send / scatter-recv) — no size threshold.
TEST(RdmaLoopback, SingleZeroCopyPutGet) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("szc");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string v(8192, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 37 + 11) & 0xFF);
  // Put twice into the SAME source buffer to exercise the MR-cache hit on re-put.
  ASSERT_TRUE(c.Put("z1", v.data(), v.size()));
  ASSERT_TRUE(c.Put("z1", v.data(), v.size()));

  // Get into the SAME dst buffer twice (scatter-recv straight in; MR-cache hit).
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("z1", &out[0], out.size()));
  EXPECT_EQ(out, v);
  out.assign(v.size(), '\0');
  ASSERT_TRUE(c.Get("z1", &out[0], out.size()));
  EXPECT_EQ(out, v);

  // Miss on the zero-copy Get path must be a clean miss (kNotFound), not an error.
  std::string m(v.size(), '\0');
  EXPECT_FALSE(c.Get("absent", &m[0], m.size()));
  // Size mismatch (stored payload_len != requested n) => miss, not corruption.
  std::string shorter(v.size() / 2, '\0');
  EXPECT_FALSE(c.Get("z1", &shorter[0], shorter.size()));
}

// A registered memory region (RegisterMemory) is registered once; every buffer
// inside it resolves to that one MR with no per-op ibv_reg_mr. Verified directly
// at the endpoint: two distinct sub-buffers return the SAME MR; an outside buffer
// registers ad-hoc (a different MR).
// Many endpoints on the same device must all Open via the shared per-device
// ibv_context+PD registry (#6 fix: no per-connection ibv_open_device thrash).
// Opening + closing N in waves exercises the refcount get-or-create/free path;
// a leak or double-free here would surface as an Open failure or crash.
TEST(RdmaLoopback, ManyEndpointsShareDeviceContext) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  constexpr int N = 24;  // > typical t16; well past the 1-2 conn happy path
  {
    std::vector<std::unique_ptr<rdma::RcEndpoint>> eps;
    for (int i = 0; i < N; ++i) {
      eps.push_back(std::make_unique<rdma::RcEndpoint>());
      ASSERT_TRUE(eps.back()->Open(nullptr, 64 * 1024, 1)) << "endpoint " << i;
    }
    eps.clear();  // all close -> registry refcount returns to 0, frees ctx+pd
  }
  // A fresh endpoint after the registry drained must still open (re-creates the
  // shared device cleanly).
  rdma::RcEndpoint again;
  ASSERT_TRUE(again.Open(nullptr, 64 * 1024, 1));
}

TEST(RdmaLoopback, PoolMrSharedAcrossBuffers) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint ep;
  ASSERT_TRUE(ep.Open(nullptr, 64 * 1024, 1));
  std::vector<char> region(256 * 1024);
  ASSERT_TRUE(ep.AddPoolMr(region.data(), region.size()));
  ibv_mr* a = ep.RegisterUser(region.data() + 4096, 4096);
  ibv_mr* b = ep.RegisterUser(region.data() + 200000, 4096);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a, b);  // both inside the pool -> one shared MR, no per-op registration
  std::vector<char> outside(4096);
  ibv_mr* c = ep.RegisterUser(outside.data(), outside.size());
  ASSERT_NE(c, nullptr);
  EXPECT_NE(c, a);  // outside the pool -> ad-hoc registration
}

// End-to-end: register the host pool once, then Put from / Get into sub-buffers
// of it. All transfers hit the pool MR (no per-op registration) and round-trip.
TEST(RdmaLoopback, RegisterMemoryRoundtrip) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("rmr");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<char> pool(128 * 1024);
  c.RegisterMemory(pool.data(), pool.size());  // one region covers every page below

  const size_t sz = 4096;
  for (int i = 0; i < 8; ++i) {
    char* src = pool.data() + i * sz * 2;            // distinct sub-buffer per page
    char* dst = pool.data() + i * sz * 2 + sz;       // get into a neighbouring slot
    for (size_t k = 0; k < sz; ++k) src[k] = static_cast<char>((i * 17 + k) & 0xFF);
    std::string key = "rm" + std::to_string(i);
    ASSERT_TRUE(c.Put(key, src, sz)) << i;
    ASSERT_TRUE(c.Get(key, dst, sz)) << i;
    EXPECT_EQ(0, std::memcmp(src, dst, sz)) << i;
  }
}

// An empty (n==0) value in a batch exercises the PostSendScatter 1-SGE degrade
// (no payload SGE / no MR registration) alongside non-empty items.
TEST(RdmaLoopback, BatchPutEmptyValue) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bev");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string nonempty(2048, 'x');
  std::vector<KvPutItem> puts = {
      {"e_full", nonempty.data(), nonempty.size()},
      {"e_empty", nullptr, 0},
  };
  auto pr = c.BatchPut(puts);
  ASSERT_TRUE(pr[0]);
  ASSERT_TRUE(pr[1]);

  EXPECT_TRUE(c.Exist("e_empty"));
  std::string out(nonempty.size(), '\0');
  ASSERT_TRUE(c.Get("e_full", &out[0], out.size()));
  EXPECT_EQ(out, nonempty);
  // Empty value round-trips: a 0-byte Get is a hit.
  EXPECT_TRUE(c.Get("e_empty", nullptr, 0));
}

TEST(RdmaLoopback, PipelinedPoolDepth) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // depth>1 enables client pipelining + the server GET worker pool (the most
  // concurrency-heavy path). Env is read by both the client ctor and the server
  // serve loop in this single process. Run under TSan to catch races.
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  ::setenv("DFKV_RDMA_WORKERS", "4", 1);
  RdmaNode node("ppd");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 64;
  const size_t sz = 4096;
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "p" + std::to_string(i);
    vals[i].assign(sz, static_cast<char>((i * 7 + 3) & 0xFF));
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;

  std::vector<std::string> outs(N);
  std::vector<KvGetItem> gets(N);
  for (int i = 0; i < N; ++i) { outs[i].assign(sz, '\0'); gets[i] = {keys[i], &outs[i][0], sz}; }
  auto gr = c.BatchGet(gets);
  int hits = 0;
  for (int i = 0; i < N; ++i) { if (gr[i]) { ++hits; EXPECT_EQ(outs[i], vals[i]) << i; } }
  EXPECT_EQ(hits, N);

  ::unsetenv("DFKV_RDMA_DEPTH");
  ::unsetenv("DFKV_RDMA_WORKERS");
}

// Regression for the conn-thread leak (#3). A server Serve thread blocks in
// WaitComp forever after a silent client disconnect (a torn-down RC peer yields
// no completion), so without an idle timeout a long-running server accumulates
// one live thread per lifetime connection — Stop() is the only reaper. The fix:
// an idle timeout reclaims the connection (the thread returns), and ReapDoneLocked
// joins the finished thread on the next accept. This test sets a short idle window
// and verifies the live count drains back to the baseline; without the fix it
// would stay pinned near N and time out.
TEST(RdmaLoopback, ReclaimsAndReapsIdleConnThreads) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_IDLE_MS", "120", 1);  // reclaim idle conns fast (test only)
  constexpr size_t kSmallMsg = 16 * 1024;   // stay well under an 8 MiB memlock
  RdmaNode node("reap", kSmallMsg);          // server buffers must be small too
  // One long-lived client endpoint holds the shared per-device ibv_context open
  // (its server side may be reclaimed when idle and is transparently re-dialed).
  RdmaTransport keep(kSmallMsg);
  KVClient ckeep({{"n", node.addr}}, SelfHdr(), &keep);
  std::string kk(64, 'k');
  ASSERT_TRUE(ckeep.Put("keep", kk.data(), kk.size()));

  const int N = 30;
  for (int i = 0; i < N; ++i) {
    RdmaTransport rt(kSmallMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    std::string v(512, static_cast<char>(i & 0xFF));
    ASSERT_TRUE(c.Put("k" + std::to_string(i), v.data(), v.size())) << "i=" << i;
    // rt + c leave scope here -> transient client gone -> server conn goes idle.
  }

  // Each round: wait past the idle window so every prior connection's Serve thread
  // has exited, then make ONE connection whose accept runs ReapDoneLocked to join
  // all the finished threads. Only that single in-flight drain thread should then
  // remain. Without reclaim+reap the N transient threads stay in conns_ and the
  // count never falls (the loop exhausts its rounds and the assert fails).
  size_t live = 0;
  for (int round = 0; round < 6; ++round) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));  // > idle (120 ms)
    { RdmaTransport rt(kSmallMsg); KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
      std::string v(16, 'x'); c.Put("drain", v.data(), v.size()); }
    live = node.rsrv->live_conn_count();
    if (live <= 2) break;
  }
  EXPECT_LE(live, 2u) << "idle conn threads were not reclaimed + reaped (leak)";
  ::unsetenv("DFKV_RDMA_IDLE_MS");
}
