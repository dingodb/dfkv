#include "kv_client.h"

#include <cstring>
#include <vector>

#include "key_map.h"
#include "tcp_transport.h"

namespace dingofs {
namespace cache {
namespace kv {

KVClient::KVClient(std::vector<std::pair<std::string, std::string>> members,
                   ValueHeader self_hdr, Transport* transport)
    : self_hdr_(self_hdr) {
  for (auto& [name, addr] : members) {
    ring_.AddNode(name);
    addr_[name] = addr;
  }
  ring_.Build();
  if (transport) {
    t_ = transport;
  } else {
    owned_ = std::make_unique<TcpTransport>();
    t_ = owned_.get();
  }
}

std::string KVClient::Route(const std::string& key) const {
  std::string name;
  if (!ring_.Lookup(key, &name)) return "";
  auto it = addr_.find(name);
  return it == addr_.end() ? "" : it->second;
}

bool KVClient::Put(const std::string& key, const void* value, size_t n) {
  std::string node = Route(key);
  if (node.empty()) return false;

  ValueHeader h = self_hdr_;
  h.SetPayload(value, n);  // sets payload_len + crc32
  std::vector<char> buf(ValueHeader::kSize + n);
  h.Serialize(buf.data());
  if (n) std::memcpy(buf.data() + ValueHeader::kSize, value, n);

  BlockKey bk = ToBlockKey(key);
  return t_->Cache(node, bk, buf.data(), buf.size()) == Status::kOk;
}

bool KVClient::Get(const std::string& key, void* out, size_t n) {
  std::string node = Route(key);
  if (node.empty()) return false;

  BlockKey bk = ToBlockKey(key);
  std::string raw;
  if (t_->Range(node, bk, 0, ValueHeader::kSize + n, &raw) != Status::kOk)
    return false;
  if (raw.size() < ValueHeader::kSize) return false;

  ValueHeader h;
  if (!ValueHeader::Parse(raw.data(), raw.size(), &h)) return false;
  if (!HeaderMatches(self_hdr_, h)) return false;          // geometry drift => miss
  if (h.payload_len != n) return false;
  if (raw.size() < ValueHeader::kSize + n) return false;

  const char* payload = raw.data() + ValueHeader::kSize;
  if (Crc32(payload, n) != h.crc32) return false;          // corruption => miss
  std::memcpy(out, payload, n);
  return true;
}

bool KVClient::Exist(const std::string& key) {
  std::string node = Route(key);
  if (node.empty()) return false;
  bool e = false;
  if (t_->Exist(node, ToBlockKey(key), &e) != Status::kOk) return false;
  return e;
}

}  // namespace kv
}  // namespace cache
}  // namespace dingofs
