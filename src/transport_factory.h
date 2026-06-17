/* Client transport selection: RDMA when built with DFKV_WITH_RDMA, requested
 * (env DFKV_RDMA=1), and an RDMA device is usable; otherwise TCP. By default it
 * falls back to TCP; with DFKV_REQUIRE_RDMA=1 it returns nullptr instead of
 * silently falling back. */
#ifndef DFKV_TRANSPORT_FACTORY_H_
#define DFKV_TRANSPORT_FACTORY_H_

#include <memory>

#include "transport.h"

namespace dfkv {

// reason (optional) receives "rdma", "tcp(...)", or "rdma-required-..."
// describing what was chosen or why no transport was returned.
std::unique_ptr<Transport> MakeClientTransport(std::string* reason = nullptr);

}  // namespace dfkv

#endif  // DFKV_TRANSPORT_FACTORY_H_
