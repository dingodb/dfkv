#ifndef DFKV_ETCD_CLIENT_H_
#define DFKV_ETCD_CLIENT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "utils/http_client.h"

namespace dfkv {

struct RangeResult {
  std::vector<std::pair<std::string, std::string>> kvs;  // decoded key, value
  int64_t revision = 0;  // etcd header revision — used as the membership epoch
};

// Thin etcd v3 client over the gRPC-gateway JSON API. Holds a non-owning
// HttpTransport so it is unit-testable with a fake. Every call is one POST.
class EtcdClient {
 public:
  explicit EtcdClient(HttpTransport* http) : http_(http) {}

  std::optional<int64_t> LeaseGrant(int ttl_seconds);
  bool LeaseKeepAlive(int64_t lease_id);
  bool Put(const std::string& key, const std::string& value, int64_t lease_id);
  std::optional<RangeResult> RangePrefix(const std::string& prefix);

 private:
  HttpTransport* http_;  // not owned
};

}  // namespace dfkv

#endif  // DFKV_ETCD_CLIENT_H_
