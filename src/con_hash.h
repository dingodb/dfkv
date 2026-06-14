/* Ketama consistent hash for client-side cache-node routing.
 * Mirrors dingofs's MD5-based KetamaConHash behaviour (the real SDK reuses
 * dingofs PeerGroup; this portable copy powers the standalone multi-node test). */
#ifndef DINGOFS_SRC_CACHE_KVCLIENT_CON_HASH_H_
#define DINGOFS_SRC_CACHE_KVCLIENT_CON_HASH_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dingofs {
namespace cache {
namespace kv {

class ConHash {
 public:
  void AddNode(const std::string& name, int weight = 1);
  void Build();
  // Returns false when the ring is empty.
  bool Lookup(const std::string& key, std::string* node) const;
  size_t NodeCount() const { return nodes_.size(); }

 private:
  std::vector<std::pair<std::string, int>> nodes_;  // (name, weight)
  std::map<uint32_t, std::string> ring_;            // point -> node
};

}  // namespace kv
}  // namespace cache
}  // namespace dingofs

#endif  // DINGOFS_SRC_CACHE_KVCLIENT_CON_HASH_H_
