/* dfkv_bench — throughput / latency benchmark for a dfkv cluster.
 *
 *   dfkv_bench --members "n=ip:port,..." [--size BYTES] [--count N]
 *              [--threads T] [--batch B] [--op put|get|both]
 *
 * Transport is chosen by the same env switch as the client: DFKV_RDMA=1 (+ device
 * DFKV_RDMA_DEV) uses RDMA, else TCP. --batch B issues B keys per BatchPut/
 * BatchGet call, which on RDMA pipelines up to B requests in flight on one
 * connection (cap by DFKV_RDMA_DEPTH); B=1 is the per-op path. Each op uses a
 * unique key ("dfkv-bench-<i>") so the write-once cache never collides. Reports
 * aggregate GB/s and per-call p50/p99/max latency for the PUT and GET phases. */
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
  std::string members, op = "both";
  size_t size = 2752512, count = 2000, threads = 8, batch = 1, bc = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--members")) members = argv[i + 1];
    else if (!std::strcmp(argv[i], "--size")) size = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--count")) count = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--threads")) threads = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--batch")) batch = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--bc")) bc = std::stoull(argv[i + 1]);  // KVClient internal batch_concurrency
    else if (!std::strcmp(argv[i], "--op")) op = argv[i + 1];
  }
  if (batch < 1) batch = 1;
  auto mem = ParseMembers(members);
  if (mem.empty()) { std::fprintf(stderr, "need --members name=ip:port,...\n"); return 2; }

  std::string reason;
  MakeClientTransport(&reason);
  std::printf("dfkv_bench transport=%s members=%zu\n", reason.c_str(), mem.size());

  // GLM-5.1 / MLA geometry (matches dfkv_smoke / dfkvctl defaults)
  ValueHeader hdr = ValueHeader::Make(0x51, 64, 0x46384534u, ValueHeader::kFlagIsMla,
                                      8, 0, 78, 1, 576);
  KVClient c(mem, hdr);
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
  // false fails. pid+size guarantees fresh keys each invocation.
  const std::string run_id =
      "dfkv-bench-" + std::to_string(static_cast<long>(getpid())) + "-" + std::to_string(size) + "-";
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
    double s = RunPhase(units, threads, [&](size_t u) {
      size_t base = u * batch, w = std::min(batch, count - base);
      // Reuse per-thread destination buffers across calls so the RDMA MR cache
      // hits (models SGLang HiCache's stable registered host pool / zero-copy).
      thread_local std::vector<std::string> outs;
      if (outs.size() < batch) { outs.resize(batch); for (auto& o : outs) o.resize(size); }
      std::vector<KvGetItem> items(w);
      for (size_t j = 0; j < w; ++j) items[j] = {key(base + j), &outs[j][0], size};
      auto hits = c.BatchGet(items);
      size_t f = 0; for (bool h : hits) if (!h) ++f; return f;
    }, &lat, &fails);
    Report("GET", count, size, threads, batch, s, lat, fails.load());
  }
  return 0;
}
