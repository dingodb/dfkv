#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <unistd.h>

#include "mds_server.h"

namespace {
dfkv::MdsServer* g_srv = nullptr;
void OnSig(int) { if (g_srv) g_srv->Stop(); }
}  // namespace

int main(int argc, char** argv) {
  std::string etcd = "127.0.0.1:2379";
  int port = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--etcd")) etcd = argv[i + 1];
    else if (!std::strcmp(argv[i], "--listen")) port = std::atoi(argv[i + 1]);
  }
  dfkv::MdsServer srv(etcd);
  g_srv = &srv;
  std::signal(SIGINT, OnSig);
  std::signal(SIGTERM, OnSig);
  if (srv.Start(port) != dfkv::Status::kOk) {
    std::fprintf(stderr, "dfkv_mds: failed to listen on %d\n", port);
    return 1;
  }
  std::printf("dfkv_mds listening on %d, etcd=%s\n", srv.port(), etcd.c_str());
  std::fflush(stdout);
  pause();  // until SIGINT/SIGTERM -> OnSig calls Stop()
  return 0;
}
