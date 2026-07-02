/* Standalone cache-node binary for the harness / pre-prod functional bring-up.
 * Usage: dfkv_server --dir <p1[,p2,p3...]> --port <p|0> --cap <bytes>
 *   --dir accepts comma-separated NVMe SSD paths (like dingo-cache --cache_dir);
 *   --cap is the TOTAL capacity, split evenly across the disks.
 * Prints "PORT <port>" on stdout once listening, then runs until SIGTERM/SIGINT. */
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <memory>
#include <vector>

#include "utils/args.h"
#include "cache/kv_node_server.h"
#include "utils/wire_limits.h"
#include "utils/log.h"
#include "mds/mds_registrar.h"
#include "utils/metrics_http.h"
#include "common/version.h"
#ifdef DFKV_WITH_RDMA
#include "cache/rdma_server.h"
#endif

using dfkv::KvNodeServer;
using dfkv::Status;

static volatile sig_atomic_t g_stop = 0;
static void OnSig(int) { g_stop = 1; }

int main(int argc, char** argv) {
  if (dfkv::WantsVersion(argc, argv)) { std::printf("dfkv_server %s\n", dfkv::Version()); return 0; }
  // No args or --help: print usage and exit, rather than silently starting a
  // server with default config (the old behavior — a foot-gun).
  if (argc == 1 || dfkv::WantsHelp(argc, argv)) {
    bool help = dfkv::WantsHelp(argc, argv);
    std::FILE* out = help ? stdout : stderr;
    std::fprintf(out,
      "dfkv_server %s — dfkv cache-node daemon\n"
      "Usage: dfkv_server --dir <p1[,p2,...]> [--port <p|0>] --cap <bytes> [options]\n\n"
      "  --dir <paths>        comma-separated NVMe paths; --cap is the TOTAL, split evenly\n"
      "  --cap <bytes>        total cache capacity (LRU self-limits)\n"
      "  --port <p>           TCP bootstrap/data port (0 = ephemeral)\n"
      "  --rdma-port <p>      RDMA QP-bootstrap port (enables RDMA data path)\n"
      "  --rdma-dev <name>    RDMA device by name (comma list = multi-rail)\n"
      "  --mds <ip:port,...>  MDS endpoints to register into (with --group/--id/--advertise)\n"
      "  --group <g>          membership group name (default \"default\")\n"
      "  --id <id>            node id; --advertise <ip:port>  address peers reach (rdma-port)\n"
      "  --weight <n>         consistent-hash weight (default 1)\n"
      "  --metrics-port <p>   enable Prometheus /metrics (omit = off); --metrics-bind <addr>\n"
      "  --version, -V        print version and exit\n"
      "  --help, -h           print this help and exit\n",
      dfkv::Version());
    return help ? 0 : 1;
  }
  dfkv::Args args(argc, argv,
                  {"--dir", "--port", "--cap", "--rdma-port", "--rdma-dev",
                   "--mds", "--group", "--id", "--advertise", "--weight",
                   "--metrics-port", "--metrics-bind"});
  std::string dir = args.Get("--dir", "/tmp/dfkv_node");
  std::string rdma_dev = args.Get("--rdma-dev", "");
  std::string mds = args.Get("--mds", "");
  std::string group = args.Get("--group", "default");
  std::string node_id = args.Get("--id", "");
  std::string advertise = args.Get("--advertise", "");
  std::string metrics_bind = args.Get("--metrics-bind", "");
  int weight = args.GetInt("--weight", 1);
  int port = args.GetInt("--port", 0);
  int rdma_port = args.GetInt("--rdma-port", -1);
  int metrics_port = args.GetInt("--metrics-port", -1);
  unsigned long long cap = args.GetU64("--cap", 1ull << 30);
  if (!args.ok()) {
    std::fprintf(stderr, "dfkv_server: %s\n(run with --help for usage)\n",
                 args.error().c_str());
    return 2;
  }
  // A mistyped --advertise (no ':') used to wrap rfind(':')+1 into a garbage
  // port; reject it up front rather than register an unreachable address.
  if (!advertise.empty() && !dfkv::IsValidHostPort(advertise)) {
    std::fprintf(stderr, "dfkv_server: --advertise must be host:port (1..65535), "
                 "got '%s'\n", advertise.c_str());
    return 2;
  }
  (void)rdma_port;
  std::signal(SIGINT, OnSig);
  std::signal(SIGTERM, OnSig);

  // split --dir on commas into one entry per NVMe SSD
  std::vector<std::string> dirs;
  for (size_t i = 0, j; i <= dir.size(); i = j + 1) {
    j = dir.find(',', i);
    if (j == std::string::npos) j = dir.size();
    if (j > i) dirs.push_back(dir.substr(i, j - i));
  }
  if (dirs.empty()) dirs.push_back(dir);

  KvNodeServer srv(dirs, cap);
  // TCP request frames share the RDMA path's resolved max-value bound (OOM-DoS
  // guard; see utils/wire_limits.h).
  srv.set_max_request_payload(dfkv::wire_limits::MaxRequestPayload());
  // Identity for Prometheus labels: reuse the MDS --id/--group when present.
  if (!node_id.empty() || group != "default") srv.set_identity(node_id, group);
  if (srv.Start(port) != Status::kOk) {
    std::fprintf(stderr, "failed to start on port %d\n", port);
    return 1;
  }
  std::printf("PORT %d\n", srv.port());
  std::fflush(stdout);

  // Optional Prometheus /metrics endpoint (declared here, started after the RDMA
  // server below so its render callback can fold in the RDMA server counters).
  std::unique_ptr<dfkv::MetricsHttpServer> mhttp;
  DFKV_LOG_INFO("dfkv_server listening on port " + std::to_string(srv.port()) + ", dir=" + dir);

  // Announce the transport build mode loudly so a TCP-only binary (built without
  // -DDFKV_WITH_RDMA=ON) can never be mistaken for an RDMA one at deploy time.
#ifdef DFKV_WITH_RDMA
  DFKV_LOG_INFO("dfkv build: transport=RDMA (libibverbs) + TCP fallback");
#else
  DFKV_LOG_INFO("dfkv build: transport=TCP-only (built WITHOUT -DDFKV_WITH_RDMA=ON)");
  if (rdma_port >= 0 || !rdma_dev.empty())
    DFKV_LOG_WARN("--rdma-port/--rdma-dev IGNORED: this binary has no RDMA support; "
                  "rebuild with -DDFKV_WITH_RDMA=ON (needs libibverbs-dev)");
#endif

#ifdef DFKV_WITH_RDMA
  std::unique_ptr<dfkv::RdmaServer> rsrv;
  if (rdma_port >= 0) {
    rsrv = std::make_unique<dfkv::RdmaServer>(
        [&srv](uint8_t op, uint64_t id, uint32_t idx, uint32_t ks, uint64_t off,
               uint64_t len, const char* pl, uint64_t pll, std::string* out) {
          return srv.ProcessRequest(op, id, idx, ks, off, len, pl, pll, out);
        },
        /*max_msg=*/64u << 20, rdma_dev);
    rsrv->set_range_handler(  // server-side direct GET: disk -> registered dbuf -> RDMA scatter
        [&srv](uint64_t id, uint32_t idx, uint32_t ks, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data, size_t* out_len) {
          return srv.RangeDirect(id, idx, ks, off, len, io_buf, cap, out_data, out_len);
        });
    rsrv->set_cache_direct_handler(  // server-side direct PUT: RDMA dbuf -> O_DIRECT write
        [&srv](uint64_t id, uint32_t idx, uint32_t ks, char* data, size_t len,
               size_t cap) {
          return srv.CacheDirect(id, idx, ks, data, len, cap);
        });
    // Async-GET hooks (used only when built -DDFKV_WITH_URING and started with
    // DFKV_SERVER_URING=1; otherwise the serve loop ignores these and uses the
    // synchronous range_handler above verbatim).
    rsrv->set_range_prep_handler(
        [&srv](uint64_t id, uint32_t idx, uint32_t ks, uint64_t off, uint64_t len,
               size_t cap, dfkv::RdmaServer::RangePrepResult* out) {
          dfkv::KVStore::RangePrep p;
          Status st = srv.RangeDirectPrep(id, idx, ks, off, len, cap, &p);
          if (st == Status::kOk) {
            out->fd = p.fd;
            out->aligned_off = p.aligned_off;
            out->aligned_len = p.aligned_len;
            out->head = p.head;
            out->payload_len = p.payload_len;
          }
          return st;
        });
    rsrv->set_range_complete_handler(
        [&srv](bool ok, size_t bytes_read) {
          srv.RangeDirectComplete(ok, bytes_read);
        });
    if (rsrv->Start(rdma_port) == Status::kOk)
      DFKV_LOG_INFO("dfkv_server RDMA listening (TCP bootstrap) on port " +
                    std::to_string(rsrv->port()) +
                    (rdma_dev.empty() ? "" : ", dev=" + rdma_dev));
    else
      DFKV_LOG_WARN("dfkv_server RDMA listener failed to start (no device?)");
  }
#endif

  // Start /metrics now that the (optional) RDMA server exists: the render
  // callback combines the cache-node metrics with the RDMA server counters.
  if (metrics_port >= 0) {
    mhttp = std::make_unique<dfkv::MetricsHttpServer>([&] {
      std::string s = srv.MetricsText();
#ifdef DFKV_WITH_RDMA
      if (rsrv) s += rsrv->MetricsText();
#endif
      return s;
    });
    if (mhttp->Start(metrics_port, metrics_bind) == Status::kOk)
      DFKV_LOG_INFO("dfkv_server /metrics on port " + std::to_string(mhttp->port()));
    else
      DFKV_LOG_WARN("dfkv_server /metrics failed to start on port " + std::to_string(metrics_port));
  }

  std::unique_ptr<dfkv::MdsRegistrar> registrar;
  if (!mds.empty()) {
    if (node_id.empty() || advertise.empty()) {
      std::fprintf(stderr, "--mds requires --id and --advertise <ip:port>\n");
      srv.Stop();
      return 1;
    }
    size_t colon = advertise.rfind(':');
    std::string aip = advertise.substr(0, colon);
    uint32_t aport = static_cast<uint32_t>(std::atoi(advertise.c_str() + colon + 1));
    std::vector<std::string> eps;
    for (size_t i = 0, j; i <= mds.size(); i = j + 1) {
      j = mds.find(',', i);
      if (j == std::string::npos) j = mds.size();
      if (j > i) eps.push_back(mds.substr(i, j - i));
    }
    dfkv::MemberInfo self{node_id, aip, aport,
                          static_cast<uint32_t>(weight < 1 ? 1 : weight)};
    self.tcp_port = static_cast<uint32_t>(srv.port());  // TCP wire/stat port -> `dfkvctl stat`
    registrar = std::make_unique<dfkv::MdsRegistrar>(std::move(eps), group, self);
    registrar->Start();
    DFKV_LOG_INFO("dfkv_server registered with MDS group=" + group + " id=" + node_id +
                  " advertise=" + advertise);
  }

  while (!g_stop) { struct timespec ts{0, 50 * 1000 * 1000}; nanosleep(&ts, nullptr); }
  DFKV_LOG_INFO("dfkv_server shutting down");
  if (mhttp) mhttp->Stop();
  if (registrar) registrar->Stop();
#ifdef DFKV_WITH_RDMA
  if (rsrv) rsrv->Stop();
#endif
  srv.Stop();
  return 0;
}
