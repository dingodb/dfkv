#include "utils/con_hash.h"

#include "utils/md5.h"

namespace dfkv {

void ConHash::AddNode(const std::string& name, int weight) {
  if (weight < 1) weight = 1;
  nodes_.emplace_back(name, weight);
}

void ConHash::Build() {
  ring_.clear();
  // Ketama: 40 keys * 4 points = 160 vnodes per weight unit.
  for (const auto& [name, weight] : nodes_) {
    int keys = 40 * weight;
    for (int k = 0; k < keys; ++k) {
      uint8_t d[16];
      std::string s = name + "-" + std::to_string(k);
      Md5(s.data(), s.size(), d);
      for (int h = 0; h < 4; ++h) {
        uint32_t point = uint32_t(d[h * 4]) | (uint32_t(d[h * 4 + 1]) << 8) |
                         (uint32_t(d[h * 4 + 2]) << 16) |
                         (uint32_t(d[h * 4 + 3]) << 24);
        ring_[point] = name;  // last-writer-wins on the rare collision
      }
    }
  }
}

std::map<std::string, size_t> ConHash::NodePointCounts() const {
  std::map<std::string, size_t> counts;
  for (const auto& [point, name] : ring_) counts[name]++;
  return counts;
}

bool ConHash::Lookup(const std::string& key, std::string* node) const {
  if (ring_.empty()) return false;
  uint32_t h = Md5_32(key);
  auto it = ring_.lower_bound(h);
  if (it == ring_.end()) it = ring_.begin();  // wrap around the ring
  *node = it->second;
  return true;
}

}  // namespace dfkv
