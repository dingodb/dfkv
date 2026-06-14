/* KVClient — the standalone DingoFS KV cache client used by the SGLang HiCache
 * plugin. Routes keys via consistent hash, wraps values with a ValueHeader
 * (model/page/dtype/layer + CRC), and speaks to cache nodes via a Transport.
 *  - Put  -> SyncCache (durable-visible write, header-wrapped)
 *  - Get  -> Range + header/CRC verify (mismatch or corruption => miss)
 *  - Exist-> local existence check on the owning node */
#ifndef DINGOFS_SRC_CACHE_KVCLIENT_KV_CLIENT_H_
#define DINGOFS_SRC_CACHE_KVCLIENT_KV_CLIENT_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "con_hash.h"
#include "transport.h"
#include "value_header.h"

namespace dingofs {
namespace cache {
namespace kv {

class KVClient {
 public:
  // members: (node_name, "ip:port"). self_hdr: this engine's geometry identity.
  // transport defaults to TcpTransport (owned) when nullptr.
  KVClient(std::vector<std::pair<std::string, std::string>> members,
           ValueHeader self_hdr, Transport* transport = nullptr);

  bool Put(const std::string& key, const void* value, size_t n);
  bool Get(const std::string& key, void* out, size_t n);  // true = hit
  bool Exist(const std::string& key);

 private:
  std::string Route(const std::string& key) const;

  ConHash ring_;
  std::map<std::string, std::string> addr_;  // name -> ip:port
  ValueHeader self_hdr_;
  std::unique_ptr<Transport> owned_;
  Transport* t_;
};

}  // namespace kv
}  // namespace cache
}  // namespace dingofs

#endif  // DINGOFS_SRC_CACHE_KVCLIENT_KV_CLIENT_H_
