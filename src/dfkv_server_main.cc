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

#include "kv_node_server.h"
#include "log.h"

using dfkv::KvNodeServer;
using dfkv::Status;

static volatile sig_atomic_t g_stop = 0;
static void OnSig(int) { g_stop = 1; }

int main(int argc, char** argv) {
  std::string dir = "/tmp/dfkv_node";
  int port = 0;
  unsigned long long cap = 1ull << 30;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--dir")) dir = argv[i + 1];
    else if (!std::strcmp(argv[i], "--port")) port = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--cap")) cap = std::strtoull(argv[i + 1], nullptr, 10);
  }
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
  if (srv.Start(port) != Status::kOk) {
    std::fprintf(stderr, "failed to start on port %d\n", port);
    return 1;
  }
  std::printf("PORT %d\n", srv.port());
  std::fflush(stdout);
  DFKV_LOG_INFO("dfkv_server listening on port " + std::to_string(srv.port()) + ", dir=" + dir);
  while (!g_stop) { struct timespec ts{0, 50 * 1000 * 1000}; nanosleep(&ts, nullptr); }
  DFKV_LOG_INFO("dfkv_server shutting down");
  srv.Stop();
  return 0;
}
