/* dfkv_bench — throughput / latency benchmark for a dfkv cluster.
 *
 *   dfkv_bench (--members "n=ip:port,..." | --mds "ip:port,..." [--group g])
 *              [--size BYTES] [--count N] [--threads T] [--batch B]
 *              [--op put|get|both] [--key-seed S] [--ready-timeout SECS]
 *
 * Ring membership is EITHER a static --members list OR --mds discovery (exactly
 * one). With --mds the client populates the ring from the MDS group and waits
 * (up to --ready-timeout, default 30s) for a warmup Put to succeed before the
 * measured phases start, so you can point bench straight at a production ring
 * without hand-listing members from `dfkvctl ring`.
 *
 * Transport is chosen by the same env switch as the client: DFKV_RDMA=1 (+ device
 * DFKV_RDMA_DEV) uses RDMA, else TCP. --batch B issues B keys per BatchPut/
 * BatchGet call, which on RDMA pipelines up to B requests in flight on one
 * connection (cap by DFKV_RDMA_DEPTH); B=1 is the per-op path. Each op uses a
 * unique key ("dfkv-bench-<i>") so the write-once cache never collides. Reports
 * aggregate GB/s and per-call p50/p99/max latency for the PUT and GET phases.
 *
 * --key-seed S pins the key namespace to S instead of the default pid, so the
 * phases can run in DIFFERENT processes or on DIFFERENT nodes: `--op put
 * --key-seed x` on node A, then `--op get --key-seed x` (same --size/--count)
 * on node B measures the cross-node read path (the PD "prefill writes, decode
 * reads" shape). With an explicit seed, re-running PUT with the same seed+size
 * hits the write-once dedup and measures nothing -- pick a fresh seed per
 * experiment (the pid default exists precisely to make reruns collision-free). */
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "client/kv_client.h"
#include "transport/transport_factory.h"
#include "common/value_header.h"
#include "common/version.h"

using namespace dfkv;  // NOLINT
using Clock = std::chrono::steady_clock;

static std::vector<std::pair<std::string, std::string>> ParseMembers(const std::string& s) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = 0;
  while (i <= s.size()) {
    size_t c = s.find(',', i);
    if (c == std::string::npos) c = s.size();
    std::string tok = s.substr(i, c - i);
    size_t eq = tok.find('=');
    if (eq != std::string::npos) out.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
    if (c == s.size()) break;
    i = c + 1;
  }
  return out;
}

// Comma-split MDS endpoints ("ip:port,ip:port") — no name= prefix, unlike members.
static std::vector<std::string> SplitComma(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i <= s.size()) {
    size_t c = s.find(',', i);
    if (c == std::string::npos) c = s.size();
    std::string tok = s.substr(i, c - i);
    if (!tok.empty()) out.push_back(tok);
    if (c == s.size()) break;
    i = c + 1;
  }
  return out;
}

static double Pct(std::vector<double>& v, double p) {
  if (v.empty()) return 0;
  size_t k = static_cast<size_t>(p * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// Run `units` work-items across `threads`. fn(u) does one unit (a batch) and
// returns the number of FAILED ops in it. Records per-unit latency in ms.
static double RunPhase(size_t units, size_t threads,
                       const std::function<size_t(size_t)>& fn,
                       std::vector<double>* lat, std::atomic<size_t>* fails) {
  lat->assign(units, 0.0);
  std::atomic<size_t> next{0};
  auto worker = [&] {
    for (size_t u = next.fetch_add(1); u < units; u = next.fetch_add(1)) {
      auto t0 = Clock::now();
      size_t f = fn(u);
      auto t1 = Clock::now();
      (*lat)[u] = std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (f) fails->fetch_add(f);
    }
  };
  auto start = Clock::now();
  std::vector<std::thread> ts;
  for (size_t w = 0; w < threads; ++w) ts.emplace_back(worker);
  for (auto& t : ts) t.join();
  return std::chrono::duration<double>(Clock::now() - start).count();
}

static void Report(const char* phase, size_t count, size_t size, size_t threads,
                   size_t batch, double secs, std::vector<double>& lat, size_t fails) {
  double total_gb = double(count) * double(size) / 1e9;
  std::printf("%-5s n=%zu size=%zu threads=%zu batch=%zu | %.3fs  %.2f GB/s  %.0f ops/s  "
              "call-lat ms p50=%.3f p99=%.3f max=%.3f  fails=%zu\n",
              phase, count, size, threads, batch, secs, total_gb / secs,
              secs > 0 ? count / secs : 0, Pct(lat, 0.50), Pct(lat, 0.99), Pct(lat, 1.0), fails);
}

int main(int argc, char** argv) {
  if (dfkv::WantsVersion(argc, argv)) { std::printf("dfkv_bench %s\n", dfkv::Version()); return 0; }
  std::string members, mds, group = "default", op = "both", key_seed;
  size_t size = 2752512, count = 2000, threads = 8, batch = 1, bc = 0;
  size_t ready_timeout_s = 30, mds_poll_ms = 1000;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--members")) members = argv[i + 1];
    else if (!std::strcmp(argv[i], "--mds")) mds = argv[i + 1];       // MDS discovery endpoints
    else if (!std::strcmp(argv[i], "--group")) group = argv[i + 1];   // MDS ring group
    else if (!std::strcmp(argv[i], "--ready-timeout")) ready_timeout_s = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--size")) size = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--count")) count = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--threads")) threads = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--batch")) batch = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--bc")) bc = std::stoull(argv[i + 1]);  // KVClient internal batch_concurrency
    else if (!std::strcmp(argv[i], "--op")) op = argv[i + 1];
    else if (!std::strcmp(argv[i], "--key-seed")) key_seed = argv[i + 1];
  }
  if (batch < 1) batch = 1;
  // Ring membership: exactly one of a static --members list or --mds discovery.
  auto mem = ParseMembers(members);
  if (mem.empty() == mds.empty()) {
    std::fprintf(stderr,
                 "need exactly one of --members name=ip:port,... "
                 "or --mds ip:port[,...] [--group g]\n");
    return 2;
  }

  std::string reason;
  MakeClientTransport(&reason);
  if (mds.empty())
    std::printf("dfkv_bench transport=%s members=%zu\n", reason.c_str(), mem.size());
  else
    std::printf("dfkv_bench transport=%s mds=%s group=%s\n",
                reason.c_str(), mds.c_str(), group.c_str());

  // GLM-5.1 / MLA geometry (matches dfkv_smoke / dfkvctl defaults)
  ValueHeader hdr = ValueHeader::Make(0x51, 64, 0x46384534u, ValueHeader::kFlagIsMla,
                                      8, 0, 78, 1, 576);
  KVClient c(mem, hdr);   // empty members when discovering via MDS
  if (!mds.empty()) {
    // Populate the ring from MDS, then wait until it's usable (a warmup Put
    // succeeds) so the measured phases don't race discovery. Mirrors
    // dfkv_discover_smoke's readiness probe.
    c.StartMdsDiscovery(SplitComma(mds), group, static_cast<int>(mds_poll_ms));
    std::vector<char> probe(size, 'w');
    auto deadline = Clock::now() + std::chrono::seconds(ready_timeout_s);
    bool ready = false;
    while (!ready && Clock::now() < deadline) {
      if (c.Put("dfkv_bench/warmup_k", probe.data(), probe.size())) ready = true;
      else std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!ready) {
      std::fprintf(stderr,
                   "dfkv_bench: ring not ready after %zus (mds=%s group=%s)\n",
                   ready_timeout_s, mds.c_str(), group.c_str());
      return 2;
    }
  }
  // External --threads already provide concurrency; with multi-node members the
  // client's per-call internal RunParallel(batch_concurrency) would nest under
  // each external thread (threads x nodes x bc) and can exhaust threads/connections.
  // --bc 1 disables internal nesting so external threads drive load cleanly.
  if (bc) c.set_batch_concurrency(bc);

  std::string val(size, '\0');
  for (size_t i = 0; i < size; ++i) val[i] = static_cast<char>(i & 0xFF);
  // Unique key namespace per run (pid + size): dfkv is write-once (idempotent
  // Cache), so a fixed key set would make reruns at a different --size skip the
  // PUT (measuring nothing) and miss the GET (stored size != requested size) =>
  // false fails. pid+size guarantees fresh keys each invocation. --key-seed
  // replaces the pid so a GET in another process/node can find a prior PUT's
  // keys (cross-node read benchmarks); the caller owns collision avoidance.
  const std::string run_id =
      "dfkv-bench-" +
      (key_seed.empty() ? std::to_string(static_cast<long>(getpid())) : key_seed) +
      "-" + std::to_string(size) + "-";
  auto key = [&run_id](size_t i) { return run_id + std::to_string(i); };
  const size_t units = (count + batch - 1) / batch;

  std::vector<double> lat;
  std::atomic<size_t> fails{0};

  if (op == "put" || op == "both") {
    fails = 0;
    double s = RunPhase(units, threads, [&](size_t u) {
      size_t base = u * batch, w = std::min(batch, count - base);
      std::vector<KvPutItem> items(w);
      for (size_t j = 0; j < w; ++j) items[j] = {key(base + j), val.data(), val.size()};
      auto oks = c.BatchPut(items);
      size_t f = 0; for (bool ok : oks) if (!ok) ++f; return f;
    }, &lat, &fails);
    Report("PUT", count, size, threads, batch, s, lat, fails.load());
  }
  if (op == "get" || op == "both") {
    fails = 0;
    // One contiguous, page-aligned destination arena for ALL threads, declared
    // once via RegisterMemory: every dst then resolves to the pre-registered
    // pool MR (the production connectors' registered-host-pool shape). The old
    // per-thread std::string buffers exceeded the per-connection user-MR cache
    // (threads*batch addrs vs cap ~64) once calls rotate over pooled
    // connections, so under load nearly every GET paid an ad-hoc 4MiB
    // ibv_reg_mr+dereg (gup pins + mmap_lock) — measured as second-scale call
    // tails that capped throughput. Falls back to the old buffers if the
    // allocation fails.
    const size_t stride = (size + 4095) & ~size_t{4095};
    char* arena = static_cast<char*>(std::aligned_alloc(4096, threads * batch * stride));
    if (arena) c.RegisterMemory(arena, threads * batch * stride);
    std::atomic<size_t> arena_slot{0};
    double s = RunPhase(units, threads, [&](size_t u) {
      size_t base = u * batch, w = std::min(batch, count - base);
      thread_local char* mybuf = nullptr;
      thread_local std::vector<std::string> outs;
      if (arena) {
        if (!mybuf) mybuf = arena + arena_slot.fetch_add(1) * batch * stride;
      } else if (outs.size() < batch) {
        outs.resize(batch); for (auto& o : outs) o.resize(size);
      }
      std::vector<KvGetItem> items(w);
      for (size_t j = 0; j < w; ++j)
        items[j] = {key(base + j), arena ? mybuf + j * stride : &outs[j][0], size};
      // DFKV_BENCH_STALL_MS: log wall-clock timestamps of slow calls to stderr
      // so stalls can be time-correlated across processes/nodes (diagnostics).
      static const long stall_ms = [] {
        const char* e = std::getenv("DFKV_BENCH_STALL_MS");
        return e && *e ? std::strtol(e, nullptr, 10) : 0;
      }();
      const auto st0 = std::chrono::system_clock::now();
      auto hits = c.BatchGet(items);
      if (stall_ms > 0) {
        const auto st1 = std::chrono::system_clock::now();
        double ms = std::chrono::duration<double, std::milli>(st1 - st0).count();
        if (ms >= static_cast<double>(stall_ms))
          std::fprintf(stderr, "STALL %.3f dur=%.1fms u=%zu\n",
                       std::chrono::duration<double>(st1.time_since_epoch()).count(), ms, u);
      }
      size_t f = 0; for (bool h : hits) if (!h) ++f; return f;
    }, &lat, &fails);
    Report("GET", count, size, threads, batch, s, lat, fails.load());
    std::free(arena);
  }
  return 0;
}
