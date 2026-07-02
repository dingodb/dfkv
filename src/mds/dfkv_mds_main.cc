#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>

#include "mds/mds_server.h"
#include "utils/metrics_http.h"
#include "common/version.h"

namespace {
volatile sig_atomic_t g_stop = 0;
void OnSig(int) { g_stop = 1; }  // async-signal-safe: just set a flag
}  // namespace

int main(int argc, char** argv) {
  if (dfkv::WantsVersion(argc, argv)) { std::printf("dfkv_mds %s\n", dfkv::Version()); return 0; }
  // No args or --help: print usage and exit, rather than starting with defaults.
  if (argc == 1 || dfkv::WantsHelp(argc, argv)) {
    bool help = dfkv::WantsHelp(argc, argv);
    std::FILE* out = help ? stdout : stderr;
    std::fprintf(out,
      "dfkv_mds %s — dfkv Membership Directory Service daemon (stateless; state in etcd)\n"
      "Usage: dfkv_mds --listen <port> [--etcd <host:port>] [options]\n\n"
      "  --listen <port>      TCP port to serve MDS requests on\n"
      "  --etcd <host:port>   etcd endpoint (default 127.0.0.1:2379)\n"
      "  --metrics-port <p>   enable Prometheus /metrics (omit = off); --metrics-bind <addr>\n"
      "  --version, -V        print version and exit\n"
      "  --help, -h           print this help and exit\n",
      dfkv::Version());
    return help ? 0 : 1;
  }
  std::string etcd = "127.0.0.1:2379", metrics_bind;
  int port = 0, metrics_port = -1;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--etcd")) etcd = argv[i + 1];
    else if (!std::strcmp(argv[i], "--listen")) port = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--metrics-port")) metrics_port = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--metrics-bind")) metrics_bind = argv[i + 1];
  }
  dfkv::MdsServer srv(etcd);
  std::signal(SIGINT, OnSig);
  std::signal(SIGTERM, OnSig);
  if (srv.Start(port) != dfkv::Status::kOk) {
    std::fprintf(stderr, "dfkv_mds: failed to listen on %d\n", port);
    return 1;
  }
  std::printf("dfkv_mds listening on %d, etcd=%s\n", srv.port(), etcd.c_str());
  std::fflush(stdout);
  // Optional Prometheus /metrics on a dedicated port/thread. Absent => no listener.
  std::unique_ptr<dfkv::MetricsHttpServer> mhttp;
  if (metrics_port >= 0) {
    mhttp = std::make_unique<dfkv::MetricsHttpServer>([&srv] { return srv.MetricsText(); });
    if (mhttp->Start(metrics_port, metrics_bind) == dfkv::Status::kOk)
      std::printf("dfkv_mds /metrics on port %d\n", mhttp->port());
    std::fflush(stdout);
  }
  // poll flag from normal context (matches dfkv_server_main.cc)
  while (!g_stop) { struct timespec ts{0, 50 * 1000 * 1000}; nanosleep(&ts, nullptr); }
  if (mhttp) mhttp->Stop();
  srv.Stop();  // called from the main thread, not the signal handler
  return 0;
}
