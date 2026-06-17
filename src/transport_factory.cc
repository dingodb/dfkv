#include "transport_factory.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "tcp_transport.h"

#ifdef DFKV_WITH_RDMA
#include "rdma_transport.h"  // native verbs: device-by-name, 400G-capable
#endif

namespace dfkv {

namespace {
bool EnvTruthy(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::strcmp(v, "0") != 0 &&
         std::strcmp(v, "false") != 0 && std::strcmp(v, "no") != 0;
}

bool RequireRdma() { return EnvTruthy("DFKV_REQUIRE_RDMA"); }
}  // namespace

std::unique_ptr<Transport> MakeClientTransport(std::string* reason) {
#ifdef DFKV_WITH_RDMA
  if (EnvTruthy("DFKV_RDMA")) {
    if (RdmaTransport::Available()) {  // native verbs (device-by-name, 400G)
      if (reason) *reason = "rdma";
      return std::make_unique<RdmaTransport>();
    }
    if (RequireRdma()) {
      if (reason) *reason = "rdma-required-but-no-device";
      return nullptr;
    }
    if (reason) *reason = "tcp(rdma-requested-but-no-device)";
    return std::make_unique<TcpTransport>();
  }
  if (RequireRdma()) {
    if (reason) *reason = "rdma-required-but-DFKV_RDMA-not-set";
    return nullptr;
  }
  if (reason) *reason = "tcp(rdma-not-requested)";
#else
  if (RequireRdma()) {
    if (reason) {
      *reason = EnvTruthy("DFKV_RDMA") ? "rdma-required-but-not-built"
                                       : "rdma-required-but-DFKV_RDMA-not-set";
    }
    return nullptr;
  }
  if (reason) *reason = EnvTruthy("DFKV_RDMA") ? "tcp(rdma-not-built)" : "tcp";
#endif
  return std::make_unique<TcpTransport>();
}

}  // namespace dfkv
