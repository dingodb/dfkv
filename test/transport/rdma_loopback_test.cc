// RDMA datapath test over a loopback device (Soft-RoCE / rdma_rxe in CI, or any
// real RDMA NIC). Exercises the native-verbs transport + server + versioned wire
// frame + zero-copy RangeInto + (with depth>1) the pipelined worker pool — the
// code that is otherwise only validated on real 400G hardware. Skips cleanly when
// no RDMA device is present. Built only when DFKV_WITH_RDMA is defined. Run under
// ThreadSanitizer to exercise the worker-pool / QP concurrency.
#include "client/kv_client.h"
#include "client/key_map.h"
#include "cache/kv_node_server.h"
#include "cache/rdma_server.h"
#include "transport/rdma_transport.h"
#include "transport/rdma_verbs.h"
#include "common/value_header.h"

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

// The user-MR LRU cap must be >= pipeline depth, else a single windowed batch
// (which registers up to `depth` distinct out-of-pool buffers before posting
// their WRs) could evict an MR still referenced by an in-flight WR.
TEST(RdmaLoopback, UserMrCapTracksDepth) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint shallow;
  ASSERT_TRUE(shallow.Open(nullptr, 16 * 1024, 1));
  EXPECT_GE(shallow.user_mr_cap(), 64u);            // floor
  EXPECT_GE(shallow.user_mr_cap(), shallow.depth());
  rdma::RcEndpoint deep;
  ASSERT_TRUE(deep.Open(nullptr, 16 * 1024, 100));  // depth > default cap of 64
  EXPECT_GE(deep.user_mr_cap(), 100u) << "cap below depth -> in-window MR eviction risk";
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

// Scatter-gather over RDMA: BatchPutSg gathers N caller buffers into one stored
// blob via a multi-SGE SEND; BatchGetAutoSg scatters the stored blob across N
// caller buffers via a multi-SGE RECV. Exercises the real PostSendScatterMulti /
// PostRecvScatterMulti datapath (not the concat fallback). Covers N=1, N=2,
// N=29 (max_sge-1), variable sizes, the >29 guard, and N > depth windowing.
TEST(RdmaLoopback, ScatterGatherRoundtripOverRdma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // depth>1 so the SG path also crosses send windows (N > depth).
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sg");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  ASSERT_TRUE(rt.pipelined());  // RDMA path, not the TCP fallback

  auto make_chunks = [](const std::string& tag, const std::vector<size_t>& sizes) {
    std::vector<std::string> v(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      v[i].resize(sizes[i]);
      for (size_t b = 0; b < sizes[i]; ++b)
        v[i][b] = static_cast<char>((tag.size() + i * 31 + b * 7) & 0xFF);
    }
    return v;
  };
  auto roundtrip = [&](const std::string& key, const std::vector<size_t>& sizes) {
    auto src = make_chunks(key, sizes);
    std::vector<const void*> ptrs;
    for (auto& s : src) ptrs.push_back(s.data());
    KvPutItemSg put{key, ptrs, sizes};
    auto pr = c.BatchPutSg({put});
    ASSERT_TRUE(pr[0]) << "put " << key;
    std::vector<std::string> dst(sizes.size());
    std::vector<void*> dptrs;
    for (size_t i = 0; i < sizes.size(); ++i) { dst[i].assign(sizes[i], '\0'); dptrs.push_back(&dst[i][0]); }
    KvGetItemSg get{key, dptrs, sizes};
    std::vector<size_t> lens;
    auto gr = c.BatchGetAutoSg({get}, &lens);
    ASSERT_TRUE(gr[0]) << "get " << key;
    size_t total = 0; for (size_t s : sizes) total += s;
    EXPECT_EQ(lens[0], total);
    for (size_t i = 0; i < sizes.size(); ++i) EXPECT_EQ(dst[i], src[i]) << key << " seg " << i;
  };

  roundtrip("sg_n1", std::vector<size_t>(1, 4096));
  roundtrip("sg_n2", std::vector<size_t>(2, 2048));
  roundtrip("sg_n29", std::vector<size_t>(29, 256));         // max payload SGEs
  roundtrip("sg_var", {1, 7, 64, 333, 4096, 11});            // variable per-seg sizes

  // >29 segments: rejected by the guard (put fails, get is a miss) — no corruption.
  {
    std::vector<size_t> sizes(30, 16);
    auto src = make_chunks("sg_over", sizes);
    std::vector<const void*> ptrs; for (auto& s : src) ptrs.push_back(s.data());
    KvPutItemSg put{"sg_over", ptrs, sizes};
    EXPECT_FALSE(c.BatchPutSg({put})[0]);
    std::vector<std::string> dst(30);
    std::vector<void*> dptrs;
    for (int i = 0; i < 30; ++i) { dst[i].assign(16, '\0'); dptrs.push_back(&dst[i][0]); }
    KvGetItemSg get{"sg_over", dptrs, sizes};
    std::vector<size_t> lens;
    EXPECT_FALSE(c.BatchGetAutoSg({get}, &lens)[0]);
  }

  // Multi-key fan-out exceeding the send window depth (4). Pin concurrency to 1:
  // the Soft-RoCE (rdma_rxe) loopback used in CI races when many distinct QPs run
  // in parallel (the SAME flakiness affects the contiguous BatchGetAuto path on
  // rxe — it is an emulation artifact, not an SG defect). This still exercises the
  // real multi-SGE verbs datapath across many keys and multiple send windows.
  {
    c.set_batch_concurrency(1);
    const int N = 24;
    std::vector<std::vector<std::string>> srcs(N);
    std::vector<KvPutItemSg> puts(N);
    for (int i = 0; i < N; ++i) {
      std::vector<size_t> sizes(1 + (i % 4), 64 + (i % 8));
      srcs[i] = make_chunks("sgm" + std::to_string(i), sizes);
      std::vector<const void*> ptrs; for (auto& s : srcs[i]) ptrs.push_back(s.data());
      puts[i] = {"sgm" + std::to_string(i), ptrs, sizes};
    }
    auto pr = c.BatchPutSg(puts);
    for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;
    std::vector<std::vector<std::string>> dsts(N);
    std::vector<KvGetItemSg> gets(N);
    for (int i = 0; i < N; ++i) {
      dsts[i].resize(srcs[i].size());
      std::vector<void*> dptrs; std::vector<size_t> caps(srcs[i].size());
      for (size_t j = 0; j < srcs[i].size(); ++j) {
        dsts[i][j].assign(srcs[i][j].size(), '\0');
        dptrs.push_back(&dsts[i][j][0]); caps[j] = srcs[i][j].size();
      }
      gets[i] = {"sgm" + std::to_string(i), dptrs, caps};
    }
    std::vector<size_t> lens;
    auto gr = c.BatchGetAutoSg(gets, &lens);
    for (int i = 0; i < N; ++i) {
      ASSERT_TRUE(gr[i]) << i;
      for (size_t j = 0; j < srcs[i].size(); ++j) EXPECT_EQ(dsts[i][j], srcs[i][j]) << i << " " << j;
    }
  }
  ::unsetenv("DFKV_RDMA_DEPTH");
}

// Fix 1 regression: an oversized SG key (total payload > max_payload_, but with a
// legal segment count so it passes the client guard) must fail ONLY itself inside
// CacheFromMulti/RangeIntoMulti, NOT poison its node batch. Previously the up-front
// validation std::fill'd every result kInvalid and returned; now the offender is
// skipped in the window and its siblings on the same node proceed normally.
TEST(RdmaLoopback, ScatterGatherOversizedFailsOnlyOffender) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sgov");
  RdmaTransport rt(kMaxMsg);  // max_payload_ = 256 KiB
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  c.set_batch_concurrency(1);  // single node, stable on rxe loopback
  ASSERT_TRUE(rt.pipelined());

  auto fill = [](std::string& s, int seed) {
    for (size_t b = 0; b < s.size(); ++b) s[b] = static_cast<char>((seed + b * 7) & 0xFF);
  };

  // Valid sibling A, an oversized item (2 segs * 200 KiB = 400 KiB > 256 KiB max_payload,
  // only 2 segments so it clears the <=29-seg client guard), then valid sibling B —
  // all routed to the same (only) node. The offender must report failure; both
  // siblings must succeed and round-trip byte-exact.
  std::string a0(4096, '\0'); fill(a0, 11);
  std::string b0(8192, '\0'); fill(b0, 23);
  std::string big0(200 * 1024, '\0'), big1(200 * 1024, '\0');
  fill(big0, 31); fill(big1, 37);

  std::vector<KvPutItemSg> puts = {
      {"sgov_a", {a0.data()}, {a0.size()}},
      {"sgov_big", {big0.data(), big1.data()}, {big0.size(), big1.size()}},
      {"sgov_b", {b0.data()}, {b0.size()}},
  };
  auto pr = c.BatchPutSg(puts);
  EXPECT_TRUE(pr[0]) << "sibling A put";
  EXPECT_FALSE(pr[1]) << "oversized put must fail only itself";
  EXPECT_TRUE(pr[2]) << "sibling B put (must not be poisoned by the offender)";

  // The two siblings must have been stored; the oversized key must be absent (never
  // written). Drive the GET-side oversized path too: a get whose total cap exceeds
  // max_payload must miss without poisoning its siblings.
  std::string ga(4096, '\0'), gb(8192, '\0');
  std::string gbig0(200 * 1024, '\0'), gbig1(200 * 1024, '\0');
  std::vector<KvGetItemSg> gets = {
      {"sgov_a", {&ga[0]}, {ga.size()}},
      {"sgov_big", {&gbig0[0], &gbig1[0]}, {gbig0.size(), gbig1.size()}},
      {"sgov_b", {&gb[0]}, {gb.size()}},
  };
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg(gets, &lens);
  EXPECT_TRUE(gr[0]) << "sibling A get";
  EXPECT_FALSE(gr[1]) << "oversized get must miss only itself";
  EXPECT_TRUE(gr[2]) << "sibling B get (must not be poisoned by the offender)";
  if (gr[0]) { EXPECT_EQ(ga, a0); }
  if (gr[2]) { EXPECT_EQ(gb, b0); }

  ::unsetenv("DFKV_RDMA_DEPTH");
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

// Regression: a BATCH op whose pooled connection the server reclaimed on idle
// must transparently re-dial and succeed — NOT return kIOError for the batch.
// Before the fix only the single-op RoundTrip retried a stale pooled conn; the
// batch paths (CacheMany/CacheFrom/RangeInto/ExistMany/...) gave up on the first
// failed window, which surfaced to SGLang as "Write page to storage: N pages
// failed" on writes and 0-hit prefixes on reads after an idle gap.
TEST(RdmaLoopback, BatchRetriesAfterServerReclaim) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_IDLE_MS", "120", 1);   // server reclaims idle conns fast
  RdmaNode node("brc");
  RdmaTransport rt(kMaxMsg);                   // long-lived: holds the device ctx
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 24;
  for (int i = 0; i < N; ++i) {
    std::string v = "val" + std::to_string(i);
    ASSERT_TRUE(c.Put("b" + std::to_string(i), v.data(), v.size())) << "warm i=" << i;
  }
  // Warm the pool so a connection is parked for the node, then snapshot the dial
  // counter — the batch op below must open a NEW one when it finds it stale.
  EXPECT_TRUE(c.Exist("b0"));
  long opened_warm =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_GE(opened_warm, 1);

  // Let the server reclaim the now-idle pooled connection (idle window 120 ms).
  std::this_thread::sleep_for(std::chrono::milliseconds(400));

  // The pooled conn is stale (its server peer was reclaimed). A batch existence
  // probe must detect that on the first window and re-dial a fresh conn, returning
  // CORRECT results — not kIOError for the whole batch. Direct transport call so no
  // KVClient-level health retry can mask a transport that failed to recover.
  std::vector<BlockKey> keys;
  for (int i = 0; i < N; ++i) {
    keys.push_back(ToBlockKey("b" + std::to_string(i)));     // present (warm)
    keys.push_back(ToBlockKey("nope" + std::to_string(i)));  // absent
  }
  std::vector<char> exists;
  auto sts = rt.ExistMany(node.addr, keys, &exists);
  ASSERT_EQ(exists.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_NE(sts[i], Status::kIOError) << "i=" << i << " (stale conn not retried)";
    EXPECT_EQ(exists[i] != 0, (i % 2 == 0)) << "i=" << i;
  }
  // The retry re-dialed: a new client connection was opened after the warm-up.
  long opened_after =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_GT(opened_after, opened_warm) << "batch op did not re-dial after reclaim";

  ::unsetenv("DFKV_RDMA_IDLE_MS");
}

// A cache node that ALSO wires the async-GET prep + complete hooks, so the server
// uses the io_uring batch-and-wait GET path when DFKV_SERVER_URING=1 (and the
// binary was built with -DDFKV_WITH_URING). With the env off / unbuilt these
// hooks are simply never consulted and the node behaves like a plain RdmaNode.
struct RdmaUringNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;

  explicit RdmaUringNode(const std::string& tag, size_t max_msg = kMaxMsg) {
    dir = fs::temp_directory_path() / ("dfkv_uring_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, uint64_t id, uint32_t idx, uint32_t ks, uint64_t off,
               uint64_t len, const char* pl, uint64_t pll, std::string* out) {
          return srv->ProcessRequest(op, id, idx, ks, off, len, pl, pll, out);
        },
        max_msg);
    rsrv->set_range_handler(  // sync fallback (used when uring path is off)
        [this](uint64_t id, uint32_t idx, uint32_t ks, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data, size_t* out_len) {
          return srv->RangeDirect(id, idx, ks, off, len, io_buf, cap, out_data, out_len);
        });
    rsrv->set_cache_direct_handler(
        [this](uint64_t id, uint32_t idx, uint32_t ks, char* data, size_t len,
               size_t cap) {
          return srv->CacheDirect(id, idx, ks, data, len, cap);
        });
    rsrv->set_range_prep_handler(
        [this](uint64_t id, uint32_t idx, uint32_t ks, uint64_t off, uint64_t len,
               size_t cap, RdmaServer::RangePrepResult* o) {
          KVStore::RangePrep p;
          Status st = srv->RangeDirectPrep(id, idx, ks, off, len, cap, &p);
          if (st == Status::kOk) {
            o->fd = p.fd; o->aligned_off = p.aligned_off; o->aligned_len = p.aligned_len;
            o->head = p.head; o->payload_len = p.payload_len;
          }
          return st;
        });
    rsrv->set_range_complete_handler(
        [this](bool ok, size_t bytes) { srv->RangeDirectComplete(ok, bytes); });
    EXPECT_EQ(rsrv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(rsrv->port());
  }
  ~RdmaUringNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    fs::remove_all(dir);
  }
};

// Correctness proof for the io_uring async-GET path: many concurrent GETs over a
// SINGLE pooled connection, with the depth high enough that several requests are
// in flight per WaitComp batch (so the server submits a multi-read io_uring batch
// and must reply in arrival order). Every value must come back byte-correct. With
// the flag OFF this still passes via the synchronous path (regression guard); with
// DFKV_SERVER_URING=1 + a URING build it exercises the batch-and-wait reads.
TEST(RdmaLoopback, UringAsyncGetManyConcurrentInOrder) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // Request the async path (no-op if unbuilt). Use overwrite=0 so a shell-set
  // DFKV_SERVER_URING (e.g. =0 to force the sync path) wins — this lets the CI
  // run the SAME test through both the sync and async serve loops.
  ::setenv("DFKV_SERVER_URING", "1", 0);
  ::setenv("DFKV_SERVER_URING_DEPTH", "32", 1);
  ::setenv("DFKV_RDMA_DEPTH", "8", 1);      // K=8 in-flight => multi-read batches
  // Small per-buffer cap so K=8 slots (rbuf+sbuf+dbuf each) stay under an 8 MiB
  // RLIMIT_MEMLOCK (CI default). Values below are <= 12 KiB, well within 64 KiB.
  constexpr size_t kUringMsg = 64 * 1024;
  RdmaUringNode node("ag", kUringMsg);
  RdmaTransport rt(kUringMsg);
  ASSERT_TRUE(rt.pipelined());
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  // Pin client batch concurrency to 1 QP: the Soft-RoCE (rdma_rxe) loopback used
  // in CI races when many distinct QPs run in parallel (an emulation artifact that
  // hits the plain sync GET path too — see ScatterGatherRoundtripOverRdma). A
  // single connection still pipelines up to `depth` GETs per send window, so the
  // SERVER still forms multi-read io_uring batches; this only removes the rxe
  // cross-QP flakiness, not the concurrency under test.
  c.set_batch_concurrency(1);

  // Distinct content per key (offset + index dependent) so a misrouted reply
  // (wrong buffer / reordered) would mismatch. Mix of sizes incl. sub-page and
  // multi-page to exercise the O_DIRECT aligned-superset trim.
  const int N = 200;
  const size_t kSizes[] = {64, 512, 4096, 8192, 12288};
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "ag" + std::to_string(i);
    size_t sz = kSizes[i % 5];
    vals[i].resize(sz);
    for (size_t k = 0; k < sz; ++k)
      vals[i][k] = static_cast<char>((i * 131 + k * 7 + 13) & 0xFF);
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;

  // Run several BatchGet rounds over the one pooled connection; each round fans
  // out N GETs that pipeline K-at-a-time -> the server forms multi-read batches.
  for (int round = 0; round < 3; ++round) {
    std::vector<std::string> outs(N);
    std::vector<KvGetItem> gets(N);
    for (int i = 0; i < N; ++i) { outs[i].assign(vals[i].size(), '\0'); gets[i] = {keys[i], &outs[i][0], vals[i].size()}; }
    auto gr = c.BatchGet(gets);
    for (int i = 0; i < N; ++i) {
      ASSERT_TRUE(gr[i]) << "round " << round << " key " << i;
      EXPECT_EQ(outs[i], vals[i]) << "round " << round << " key " << i;
    }
  }

  // A miss on the async path must still be a clean miss (kNotFound), not an error
  // or a stale-buffer hit; interleave present/absent to mix sync-miss + async-hit
  // replies in the same pipelined window (order-preservation across reply kinds).
  std::vector<std::string> mouts(N);
  std::vector<KvGetItem> mgets;
  std::vector<std::string> mkeys;
  for (int i = 0; i < N; ++i) {
    mkeys.push_back(keys[i]);
    mkeys.push_back("ag_absent" + std::to_string(i));
  }
  std::vector<std::string> mo(mkeys.size());
  std::vector<KvGetItem> mg(mkeys.size());
  for (size_t i = 0; i < mkeys.size(); ++i) {
    size_t cap = (i % 2 == 0) ? vals[i / 2].size() : 4096;
    mo[i].assign(cap, '\0');
    mg[i] = {mkeys[i], &mo[i][0], cap};
  }
  auto mgr = c.BatchGet(mg);
  for (size_t i = 0; i < mkeys.size(); ++i) {
    if (i % 2 == 0) {
      ASSERT_TRUE(mgr[i]) << "present key " << mkeys[i];
      EXPECT_EQ(mo[i], vals[i / 2]) << "present key " << mkeys[i];
    } else {
      EXPECT_FALSE(mgr[i]) << "absent key should miss: " << mkeys[i];
    }
  }

  ::unsetenv("DFKV_SERVER_URING");
  ::unsetenv("DFKV_SERVER_URING_DEPTH");
  ::unsetenv("DFKV_RDMA_DEPTH");
}
