#include "kv_client.h"

#include <atomic>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include "key_map.h"
#include "tcp_transport.h"

namespace dfkv {

namespace {
// Run fn(i) for i in [0,n) across up to `workers` threads (atomic work-steal).
void RunParallel(size_t n, size_t workers, const std::function<void(size_t)>& fn) {
  if (n == 0) return;
  if (workers > n) workers = n;
  if (workers <= 1) { for (size_t i = 0; i < n; ++i) fn(i); return; }
  std::atomic<size_t> next{0};
  std::vector<std::thread> ts;
  ts.reserve(workers);
  for (size_t w = 0; w < workers; ++w) {
    ts.emplace_back([&] {
      for (size_t i = next.fetch_add(1); i < n; i = next.fetch_add(1)) fn(i);
    });
  }
  for (auto& t : ts) t.join();
}
}  // namespace

KVClient::KVClient(std::vector<std::pair<std::string, std::string>> members,
                   ValueHeader self_hdr, Transport* transport)
    : self_hdr_(self_hdr) {
  SetMembers(std::move(members));
  if (transport) {
    t_ = transport;
  } else {
    owned_ = std::make_unique<TcpTransport>();
    t_ = owned_.get();
  }
}

void KVClient::SetMembers(std::vector<std::pair<std::string, std::string>> members) {
  ConHash ring;
  std::map<std::string, std::string> addr;
  for (auto& [name, a] : members) {
    ring.AddNode(name);
    addr[name] = a;
  }
  ring.Build();
  std::lock_guard<std::mutex> lk(ring_mu_);
  ring_ = std::move(ring);
  addr_ = std::move(addr);
}

std::string KVClient::Route(const std::string& key) const {
  std::lock_guard<std::mutex> lk(ring_mu_);
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

bool KVClient::GetAuto(const std::string& key, std::string* out, size_t max_bytes) {
  std::string node = Route(key);
  if (node.empty()) return false;
  std::string raw;
  if (t_->Range(node, ToBlockKey(key), 0, ValueHeader::kSize + max_bytes, &raw) != Status::kOk)
    return false;
  if (raw.size() < ValueHeader::kSize) return false;
  ValueHeader h;
  if (!ValueHeader::Parse(raw.data(), raw.size(), &h)) return false;
  if (!HeaderMatches(self_hdr_, h)) return false;
  if (raw.size() < ValueHeader::kSize + h.payload_len) return false;
  const char* p = raw.data() + ValueHeader::kSize;
  if (Crc32(p, h.payload_len) != h.crc32) return false;
  out->assign(p, h.payload_len);
  return true;
}

bool KVClient::Exist(const std::string& key) {
  std::string node = Route(key);
  if (node.empty()) return false;
  bool e = false;
  if (t_->Exist(node, ToBlockKey(key), &e) != Status::kOk) return false;
  return e;
}

std::vector<bool> KVClient::BatchPut(const std::vector<KvPutItem>& items) {
  std::vector<char> ok(items.size(), 0);  // char (not vector<bool>) for thread-safe writes
  RunParallel(items.size(), batch_concurrency_, [&](size_t i) {
    ok[i] = Put(items[i].key, items[i].value, items[i].n) ? 1 : 0;
  });
  return std::vector<bool>(ok.begin(), ok.end());
}

std::vector<bool> KVClient::BatchGet(const std::vector<KvGetItem>& items) {
  std::vector<char> hit(items.size(), 0);
  RunParallel(items.size(), batch_concurrency_, [&](size_t i) {
    hit[i] = Get(items[i].key, items[i].out, items[i].n) ? 1 : 0;
  });
  return std::vector<bool>(hit.begin(), hit.end());
}

std::vector<bool> KVClient::BatchExist(const std::vector<std::string>& keys) {
  std::vector<char> e(keys.size(), 0);
  RunParallel(keys.size(), batch_concurrency_, [&](size_t i) {
    e[i] = Exist(keys[i]) ? 1 : 0;
  });
  return std::vector<bool>(e.begin(), e.end());
}

}  // namespace dfkv
