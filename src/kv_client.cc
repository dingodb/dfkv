#include "kv_client.h"

#include <atomic>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include "key_map.h"
#include "transport_factory.h"

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
    owned_ = MakeClientTransport();  // RDMA if available+requested, else TCP
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

void KVClient::SetMembers(const std::vector<MemberInfo>& members) {
  ConHash ring;
  std::map<std::string, std::string> addr;
  for (const auto& m : members) {
    ring.AddNode(m.id, m.weight < 1 ? 1 : static_cast<int>(m.weight));
    addr[m.id] = m.ip + ":" + std::to_string(m.port);
  }
  ring.Build();
  std::lock_guard<std::mutex> lk(ring_mu_);
  ring_ = std::move(ring);
  addr_ = std::move(addr);
}

void KVClient::StartMdsDiscovery(std::vector<std::string> mds_eps,
                                 const std::string& group, int poll_ms) {
  StopMdsDiscovery();
  poller_ = std::make_unique<MdsMemberPoller>(
      std::move(mds_eps), group,
      [this](const std::vector<MemberInfo>& ms) { SetMembers(ms); }, poll_ms);
  poller_->Start();
}

void KVClient::StopMdsDiscovery() {
  if (poller_) { poller_->Stop(); poller_.reset(); }
}

KVClient::~KVClient() { StopMdsDiscovery(); }

bool KVClient::RefreshMembers(const std::string& seed_addr) {
  std::string text;
  if (t_->Members(seed_addr, &text) != Status::kOk) return false;
  std::vector<std::pair<std::string, std::string>> members;
  for (size_t i = 0; i <= text.size();) {  // parse "name=ip:port,name=ip:port,..."
    size_t c = text.find(',', i);
    if (c == std::string::npos) c = text.size();
    std::string tok = text.substr(i, c - i);
    size_t eq = tok.find('=');
    if (eq != std::string::npos && eq + 1 < tok.size())
      members.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
    if (c == text.size()) break;
    i = c + 1;
  }
  if (members.empty()) return false;
  SetMembers(std::move(members));
  return true;
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
  const size_t N = items.size();
  std::vector<char> ok(N, 0);  // char (not vector<bool>) for thread-safe writes
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, batch_concurrency_, [&](size_t i) {
      ok[i] = Put(items[i].key, items[i].value, items[i].n) ? 1 : 0;
    });
    return std::vector<bool>(ok.begin(), ok.end());
  }
  // RDMA: group by node, wrap once, pipeline each node's writes on one connection.
  std::vector<std::vector<char>> bufs(N);
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    std::string node = Route(items[i].key);
    if (node.empty()) continue;
    ValueHeader h = self_hdr_;
    h.SetPayload(items[i].value, items[i].n);
    bufs[i].resize(ValueHeader::kSize + items[i].n);
    h.Serialize(bufs[i].data());
    if (items[i].n)
      std::memcpy(bufs[i].data() + ValueHeader::kSize, items[i].value, items[i].n);
    by_node[node].push_back(i);
  }
  std::vector<std::pair<std::string, std::vector<size_t>>> groups(by_node.begin(), by_node.end());
  RunParallel(groups.size(), batch_concurrency_, [&](size_t g) {
    const std::string& node = groups[g].first;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<CacheItem> ci;
    ci.reserve(idx.size());
    for (size_t k : idx) ci.push_back(CacheItem{ToBlockKey(items[k].key), bufs[k].data(), bufs[k].size()});
    std::vector<Status> sts = t_->CacheMany(node, ci);
    for (size_t m = 0; m < idx.size(); ++m) ok[idx[m]] = (sts[m] == Status::kOk) ? 1 : 0;
  });
  return std::vector<bool>(ok.begin(), ok.end());
}

std::vector<bool> KVClient::BatchGet(const std::vector<KvGetItem>& items) {
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, batch_concurrency_, [&](size_t i) {
      hit[i] = Get(items[i].key, items[i].out, items[i].n) ? 1 : 0;
    });
    return std::vector<bool>(hit.begin(), hit.end());
  }
  // RDMA: group by (node, n) so each group shares the Range length, then pipeline.
  std::map<std::pair<std::string, size_t>, std::vector<size_t>> by;
  for (size_t i = 0; i < N; ++i) {
    std::string node = Route(items[i].key);
    if (node.empty()) continue;
    by[{node, items[i].n}].push_back(i);
  }
  std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>> groups(by.begin(), by.end());
  RunParallel(groups.size(), batch_concurrency_, [&](size_t g) {
    const std::string& node = groups[g].first.first;
    const size_t n = groups[g].first.second;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> keys;
    std::vector<RangeDst> dsts;
    keys.reserve(idx.size());
    dsts.reserve(idx.size());
    for (size_t k : idx) {
      keys.push_back(ToBlockKey(items[k].key));
      dsts.push_back(RangeDst{items[k].out, n});
    }
    // Zero-copy: the payload lands straight in items[].out; hdrs[] gets the value
    // header for geometry/CRC verification (CRC is computed over the dst buffer).
    std::vector<std::string> hdrs;
    std::vector<Status> sts = t_->RangeInto(node, keys, dsts, ValueHeader::kSize, &hdrs);
    for (size_t m = 0; m < idx.size(); ++m) {
      if (sts[m] != Status::kOk || hdrs[m].size() < ValueHeader::kSize) continue;
      ValueHeader h;
      if (!ValueHeader::Parse(hdrs[m].data(), hdrs[m].size(), &h)) continue;
      if (!HeaderMatches(self_hdr_, h)) continue;
      if (h.payload_len != n) continue;
      if (Crc32(items[idx[m]].out, n) != h.crc32) continue;  // verify in-place
      hit[idx[m]] = 1;
    }
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
