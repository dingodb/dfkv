#include "client/kv_client.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "client/key_map.h"
#include "utils/log.h"
#include "transport/transport_factory.h"

namespace dfkv {

namespace {
// Max payload segments per scatter-gather key. The RDMA QP carries max_sge SGEs
// per work request; SGE0 is the header, leaving max_sge-1 payload SGEs. The
// device cap on hd04 ConnectX is 30 -> 29 payload segments. The connector
// guarantees <=29; a key over this is reported failed instead of corrupted.
//
// KNOWN LIMITATION (heterogeneous hardware): this is a fixed 29, not the live
// negotiated ep.max_sge()-1. On an HCA reporting max_sge < 30, a key whose seg
// count is between (max_sge-1) and 29 passes this client guard but trips the
// transport's per-window check (CacheFromMulti/RangeIntoMulti). That now fails
// only the OFFENDING item (per-item kInvalid), not the whole node batch, so it
// is graceful fail-soft (the connector recomputes just those blocks; siblings
// still hit; no data corruption) — it only costs the cache benefit for the
// oversized keys on small-max_sge devices. Remaining TODO for full cross-HCA
// efficiency: derive the SG chunk size from the live max_sge instead of this
// constant so even those keys cache.
constexpr size_t kSgMaxPayloadSegs = 29;

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
    transport_reason_ = "injected";
  } else {
    std::string reason;
    owned_ = MakeClientTransport(&reason);  // RDMA if available+requested, else TCP
    transport_reason_ = reason.empty() ? std::string("unknown") : reason;
    if (!owned_) {
      DFKV_LOG_ERROR("dfkv client transport unavailable: " + transport_reason_);
      throw std::runtime_error("dfkv client transport unavailable: " + transport_reason_);
    }
    t_ = owned_.get();
    DFKV_LOG_INFO("dfkv client transport=" + transport_reason_);
  }
  // Opt-in active per-peer latency probe. Off unless DFKV_PROBE_INTERVAL_MS>0,
  // so default behavior is unchanged (no background traffic).
  if (const char* pe = std::getenv("DFKV_PROBE_INTERVAL_MS")) {
    int ms = std::atoi(pe);
    if (ms > 0) StartProbe(ms);
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

KVClient::~KVClient() { StopProbe(); StopMdsDiscovery(); }

void KVClient::StartProbe(int interval_ms) {
  if (interval_ms <= 0 || probe_running_.load(std::memory_order_relaxed)) return;
  probe_interval_ms_ = interval_ms;
  probe_running_.store(true, std::memory_order_relaxed);
  probe_th_ = std::thread([this] { ProbeLoop(); });
}

void KVClient::StopProbe() {
  if (!probe_running_.exchange(false, std::memory_order_relaxed)) return;
  probe_cv_.notify_all();
  if (probe_th_.joinable()) probe_th_.join();
}

void KVClient::ProbeLoop() {
  // A sentinel key that no real workload writes; kExist of it is a cheap round
  // trip that measures the node's responsiveness (kNotFound is a healthy reply).
  const BlockKey probe = ToBlockKey("__dfkv_probe__");
  while (probe_running_.load(std::memory_order_relaxed)) {
    std::vector<std::string> addrs;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      addrs.reserve(addr_.size());
      for (const auto& [name, a] : addr_) addrs.push_back(a);
    }
    for (const auto& a : addrs) {
      if (!probe_running_.load(std::memory_order_relaxed)) break;
      bool e = false;
      auto t0 = std::chrono::steady_clock::now();
      Status st = t_->Exist(a, probe, &e);
      double sec = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
      // Only record real round trips; a transport IO error (e.g. connect refused)
      // is fast and would skew latency, and is already counted by PeerHealth.
      if (st != Status::kIOError) peer_lat_.Observe(a, sec);
    }
    std::unique_lock<std::mutex> lk(probe_mu_);
    probe_cv_.wait_for(lk, std::chrono::milliseconds(probe_interval_ms_),
                       [this] { return !probe_running_.load(std::memory_order_relaxed); });
  }
}

std::string KVClient::MetricsSnapshot() const {
  std::string s;
  s += "# HELP dfkv_client_transport_info Client transport selected by dfkv_open\n";
  s += "# TYPE dfkv_client_transport_info gauge\n";
  s += "dfkv_client_transport_info{transport=\"" + transport_reason_ + "\"} 1\n";
  s += health_.Render();
  s += peer_lat_.Render();
  s += op_stats_.Render();
  if (t_) s += t_->MetricsText();
  return s;
}

std::vector<bool> KVClient::RecordBatch(OpMetrics::Op op,
                                        std::chrono::steady_clock::time_point t0,
                                        const std::vector<char>& flags,
                                        uint64_t bytes) {
  uint64_t hits = 0;
  for (char c : flags) if (c) ++hits;
  double s = std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - t0).count();
  op_stats_.Record(op, flags.size(), hits, bytes, s);
  return std::vector<bool>(flags.begin(), flags.end());
}

uint64_t KVClient::NowMs() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

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
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;

  ValueHeader h = self_hdr_;
  h.payload_len = n;  // geometry comes from self_hdr_; no payload checksum (v3)
  BlockKey bk = ToBlockKey(key);
  Status st;
  if (t_->pipelined()) {
    // Zero copy: build only the 48B header, scatter-send [header|value] with the
    // value gathered straight from the caller's registered buffer. `value` must
    // stay stable until this returns (CacheFrom is synchronous). For a buffer in a
    // RegisterMemory'd pool this does no per-op MR registration.
    std::array<char, ValueHeader::kSize> hdr;
    h.Serialize(hdr.data());
    std::vector<CacheSrc> srcs{CacheSrc{bk, hdr.data(), ValueHeader::kSize, value, n}};
    st = t_->CacheFrom(node, srcs)[0];
  } else {
    // TCP (non-pipelined): wrap into one buffer and Cache.
    std::vector<char> buf(ValueHeader::kSize + n);
    h.Serialize(buf.data());
    if (n) std::memcpy(buf.data() + ValueHeader::kSize, value, n);
    st = t_->Cache(node, bk, buf.data(), buf.size());
  }
  if (st == Status::kIOError) { health_.MarkBad(node, now); return false; }
  health_.MarkGood(node);
  return st == Status::kOk;
}

bool KVClient::Get(const std::string& key, void* out, size_t n) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;

  BlockKey bk = ToBlockKey(key);
  if (t_->pipelined()) {
    // Zero copy AND zero touch: the payload scatters straight into `out` and the
    // CPU never reads it; only the 48B header comes back (hdrs[0]) for geometry verify.
    std::vector<BlockKey> keys{bk};
    std::vector<RangeDst> dsts{RangeDst{out, n}};
    std::vector<std::string> hdrs;
    Status st = t_->RangeInto(node, keys, dsts, ValueHeader::kSize, &hdrs)[0];
    // A miss decodes to kNotFound (a healthy response), not kIOError — only a
    // real transport error marks the node bad.
    if (st == Status::kIOError) { health_.MarkBad(node, now); return false; }
    health_.MarkGood(node);
    if (st != Status::kOk || hdrs[0].size() < ValueHeader::kSize) return false;
    ValueHeader h;
    if (!ValueHeader::Parse(hdrs[0].data(), hdrs[0].size(), &h)) return false;
    if (!HeaderMatches(self_hdr_, h)) return false;        // geometry drift => miss
    if (h.payload_len != n) return false;
    return true;
  }

  // TCP (non-pipelined): response into a string, then copy to out.
  std::string raw;
  Status st = t_->Range(node, bk, 0, ValueHeader::kSize + n, &raw);
  if (st == Status::kIOError) { health_.MarkBad(node, now); return false; }
  health_.MarkGood(node);
  if (st != Status::kOk) return false;
  if (raw.size() < ValueHeader::kSize) return false;

  ValueHeader h;
  if (!ValueHeader::Parse(raw.data(), raw.size(), &h)) return false;
  if (!HeaderMatches(self_hdr_, h)) return false;          // geometry drift => miss
  if (h.payload_len != n) return false;
  if (raw.size() < ValueHeader::kSize + n) return false;

  const char* payload = raw.data() + ValueHeader::kSize;
  std::memcpy(out, payload, n);
  return true;
}

bool KVClient::GetAuto(const std::string& key, std::string* out, size_t max_bytes) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  std::string raw;
  Status st = t_->Range(node, ToBlockKey(key), 0, ValueHeader::kSize + max_bytes, &raw);
  if (st == Status::kIOError) { health_.MarkBad(node, now); return false; }
  health_.MarkGood(node);
  if (st != Status::kOk) return false;
  if (raw.size() < ValueHeader::kSize) return false;
  ValueHeader h;
  if (!ValueHeader::Parse(raw.data(), raw.size(), &h)) return false;
  if (!HeaderMatches(self_hdr_, h)) return false;
  if (raw.size() < ValueHeader::kSize + h.payload_len) return false;
  const char* p = raw.data() + ValueHeader::kSize;
  out->assign(p, h.payload_len);
  return true;
}

bool KVClient::GetAuto(const std::string& key, void* out, size_t cap, size_t* out_len) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  // Read [header | up-to-cap payload]; the header tells us the true payload_len.
  std::string raw;
  Status st = t_->Range(node, ToBlockKey(key), 0, ValueHeader::kSize + cap, &raw);
  if (st == Status::kIOError) { health_.MarkBad(node, now); return false; }
  health_.MarkGood(node);
  if (st != Status::kOk) return false;
  if (raw.size() < ValueHeader::kSize) return false;
  ValueHeader h;
  if (!ValueHeader::Parse(raw.data(), raw.size(), &h)) return false;
  if (!HeaderMatches(self_hdr_, h)) return false;        // geometry drift => miss
  if (h.payload_len > cap) return false;                 // won't fit caller buffer
  if (raw.size() < ValueHeader::kSize + h.payload_len) return false;
  if (h.payload_len) std::memcpy(out, raw.data() + ValueHeader::kSize, h.payload_len);
  if (out_len) *out_len = h.payload_len;
  return true;
}

bool KVClient::Exist(const std::string& key) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  bool e = false;
  Status st = t_->Exist(node, ToBlockKey(key), &e);
  if (st == Status::kIOError) { health_.MarkBad(node, now); return false; }
  health_.MarkGood(node);
  return st == Status::kOk && e;
}

bool KVClient::Remove(const std::string& key) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  Status st = t_->Remove(node, ToBlockKey(key));
  if (st == Status::kIOError) { health_.MarkBad(node, now); return false; }
  health_.MarkGood(node);
  // kNotFound is a success for the caller: the goal (key not present) holds.
  return st == Status::kOk || st == Status::kNotFound;
}

std::vector<bool> KVClient::BatchPut(const std::vector<KvPutItem>& items) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> ok(N, 0);  // char (not vector<bool>) for thread-safe writes
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, batch_concurrency_.load(std::memory_order_relaxed), [&](size_t i) {
      ok[i] = Put(items[i].key, items[i].value, items[i].n) ? 1 : 0;
    });
    return std::vector<bool>(ok.begin(), ok.end());
  }
  // RDMA: group by node and scatter-send [header|value] — zero copy of the value.
  // Build only the 48B header per item (just stamps payload_len; no payload pass);
  // hdrs must outlive the RunParallel lambdas that reference it.
  std::vector<std::array<char, ValueHeader::kSize>> hdrs(N);
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    std::string node = Route(items[i].key);
    if (node.empty()) continue;
    ValueHeader h = self_hdr_;
    h.payload_len = items[i].n;
    h.Serialize(hdrs[i].data());
    by_node[node].push_back(i);
  }
  std::vector<std::pair<std::string, std::vector<size_t>>> groups(by_node.begin(), by_node.end());
  RunParallel(groups.size(), batch_concurrency_.load(std::memory_order_relaxed), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<CacheSrc> srcs;
    srcs.reserve(idx.size());
    for (size_t k : idx)
      srcs.push_back(CacheSrc{ToBlockKey(items[k].key), hdrs[k].data(),
                              ValueHeader::kSize, items[k].value, items[k].n});
    std::vector<Status> sts = t_->CacheFrom(node, srcs);
    for (size_t m = 0; m < idx.size(); ++m) ok[idx[m]] = (sts[m] == Status::kOk) ? 1 : 0;
    bool resp = false, ioerr = false;
    for (Status s : sts) { if (s == Status::kIOError) ioerr = true; else resp = true; }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, now);
  });
  // RDMA path records the batch once (TCP path above is counted per-key by Put()).
  uint64_t bytes = 0;
  for (size_t i = 0; i < N; ++i) if (ok[i]) bytes += items[i].n;
  return RecordBatch(OpMetrics::kPut, t0, ok, bytes);
}

std::vector<bool> KVClient::BatchGet(const std::vector<KvGetItem>& items) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, batch_concurrency_.load(std::memory_order_relaxed), [&](size_t i) {
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
  RunParallel(groups.size(), batch_concurrency_.load(std::memory_order_relaxed), [&](size_t g) {
    const std::string& node = groups[g].first.first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
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
    // Zero-copy + zero-touch: the payload lands straight in items[].out (CPU never
    // reads it); hdrs[] gets the value header for geometry verification only.
    std::vector<std::string> hdrs;
    std::vector<Status> sts = t_->RangeInto(node, keys, dsts, ValueHeader::kSize, &hdrs);
    bool resp = false, ioerr = false;
    for (Status s : sts) { if (s == Status::kIOError) ioerr = true; else resp = true; }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, now);
    for (size_t m = 0; m < idx.size(); ++m) {
      if (sts[m] != Status::kOk || hdrs[m].size() < ValueHeader::kSize) continue;
      ValueHeader h;
      if (!ValueHeader::Parse(hdrs[m].data(), hdrs[m].size(), &h)) continue;
      if (!HeaderMatches(self_hdr_, h)) continue;
      if (h.payload_len != n) continue;
      hit[idx[m]] = 1;
    }
  });
  uint64_t bytes = 0;
  for (size_t i = 0; i < N; ++i) if (hit[i]) bytes += items[i].n;
  return RecordBatch(OpMetrics::kGet, t0, hit, bytes);
}

std::vector<bool> KVClient::BatchGetAuto(const std::vector<KvGetItem>& items,
                                         std::vector<size_t>* out_lens) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  std::vector<size_t> lens(N, 0);  // distinct indices => thread-safe writes
  if (!t_->pipelined()) {  // TCP: parallelize per item with our own threads.
    RunParallel(N, batch_concurrency_.load(std::memory_order_relaxed), [&](size_t i) {
      size_t got = 0;
      if (GetAuto(items[i].key, items[i].out, items[i].n, &got)) { hit[i] = 1; lens[i] = got; }
    });
    if (out_lens) *out_lens = std::move(lens);
    return std::vector<bool>(hit.begin(), hit.end());
  }
  // RDMA: group by (node, cap) so each group shares the Range length, then
  // pipeline. Same zero-copy datapath as BatchGet; the only difference is we
  // accept any payload_len <= cap (instead of == n) and report the true length.
  std::map<std::pair<std::string, size_t>, std::vector<size_t>> by;
  for (size_t i = 0; i < N; ++i) {
    std::string node = Route(items[i].key);
    if (node.empty()) continue;
    by[{node, items[i].n}].push_back(i);
  }
  std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>> groups(by.begin(), by.end());
  RunParallel(groups.size(), batch_concurrency_.load(std::memory_order_relaxed), [&](size_t g) {
    const std::string& node = groups[g].first.first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const size_t cap = groups[g].first.second;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> keys;
    std::vector<RangeDst> dsts;
    keys.reserve(idx.size());
    dsts.reserve(idx.size());
    for (size_t k : idx) {
      keys.push_back(ToBlockKey(items[k].key));
      dsts.push_back(RangeDst{items[k].out, cap});
    }
    std::vector<std::string> hdrs;
    std::vector<Status> sts = t_->RangeInto(node, keys, dsts, ValueHeader::kSize, &hdrs);
    bool resp = false, ioerr = false;
    for (Status s : sts) { if (s == Status::kIOError) ioerr = true; else resp = true; }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, now);
    for (size_t m = 0; m < idx.size(); ++m) {
      if (sts[m] != Status::kOk || hdrs[m].size() < ValueHeader::kSize) continue;
      ValueHeader h;
      if (!ValueHeader::Parse(hdrs[m].data(), hdrs[m].size(), &h)) continue;
      if (!HeaderMatches(self_hdr_, h)) continue;
      if (h.payload_len > cap) continue;  // doesn't fit caller buffer => miss
      lens[idx[m]] = h.payload_len;
      hit[idx[m]] = 1;
    }
  });
  uint64_t bytes = 0;
  for (size_t x : lens) bytes += x;
  if (out_lens) *out_lens = std::move(lens);
  return RecordBatch(OpMetrics::kGet, t0, hit, bytes);
}

std::vector<bool> KVClient::BatchExist(const std::vector<std::string>& keys) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = keys.size();
  std::vector<char> e(N, 0);
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, batch_concurrency_.load(std::memory_order_relaxed), [&](size_t i) {
      e[i] = Exist(keys[i]) ? 1 : 0;
    });
    return std::vector<bool>(e.begin(), e.end());
  }
  // RDMA: group by node so each node's keys pipeline kExist on a single pooled
  // connection (one Acquire/node) instead of one round trip — and one connection
  // bootstrap under contention — per key. Nodes still fan out in parallel.
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    std::string node = Route(keys[i]);
    if (node.empty()) continue;
    by_node[node].push_back(i);
  }
  std::vector<std::pair<std::string, std::vector<size_t>>> groups(by_node.begin(), by_node.end());
  RunParallel(groups.size(), batch_concurrency_.load(std::memory_order_relaxed), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> bkeys;
    bkeys.reserve(idx.size());
    for (size_t k : idx) bkeys.push_back(ToBlockKey(keys[k]));
    std::vector<char> ex;
    std::vector<Status> sts = t_->ExistMany(node, bkeys, &ex);
    bool resp = false, ioerr = false;
    for (Status s : sts) { if (s == Status::kIOError) ioerr = true; else resp = true; }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, now);
    for (size_t m = 0; m < idx.size(); ++m) e[idx[m]] = (m < ex.size() && ex[m]) ? 1 : 0;
  });
  return RecordBatch(OpMetrics::kExist, t0, e, 0);
}

std::vector<bool> KVClient::BatchRemove(const std::vector<std::string>& keys) {
  const size_t N = keys.size();
  std::vector<char> ok(N, 0);
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, batch_concurrency_.load(std::memory_order_relaxed), [&](size_t i) {
      ok[i] = Remove(keys[i]) ? 1 : 0;
    });
    return std::vector<bool>(ok.begin(), ok.end());
  }
  // RDMA: group by node and RemoveMany per node (sequential within a node since
  // eviction is off the hot path; nodes still fan out in parallel).
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    std::string node = Route(keys[i]);
    if (node.empty()) continue;
    by_node[node].push_back(i);
  }
  std::vector<std::pair<std::string, std::vector<size_t>>> groups(by_node.begin(), by_node.end());
  RunParallel(groups.size(), batch_concurrency_.load(std::memory_order_relaxed), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> bkeys;
    bkeys.reserve(idx.size());
    for (size_t k : idx) bkeys.push_back(ToBlockKey(keys[k]));
    std::vector<Status> sts = t_->RemoveMany(node, bkeys);
    bool resp = false, ioerr = false;
    for (Status s : sts) { if (s == Status::kIOError) ioerr = true; else resp = true; }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, now);
    for (size_t m = 0; m < idx.size(); ++m)
      ok[idx[m]] = (m < sts.size() &&
                    (sts[m] == Status::kOk || sts[m] == Status::kNotFound)) ? 1 : 0;
  });
  return std::vector<bool>(ok.begin(), ok.end());
}

std::vector<bool> KVClient::BatchPutSg(const std::vector<KvPutItemSg>& items) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> ok(N, 0);  // char (not vector<bool>) for thread-safe writes

  // Guard: a key with more payload segments than one RDMA work request can carry
  // (max_sge-1) is reported failed up front and excluded from the wire (so it
  // never corrupts a shared connection). Done transport-independently so the
  // contract holds on TCP too. Empty-segment keys are valid (header-only).
  std::vector<char> over(N, 0);
  for (size_t i = 0; i < N; ++i) {
    if (items[i].key.empty() ||  // null/empty key: skip (no header-only blob written)
        items[i].ptrs.size() != items[i].sizes.size() ||
        items[i].ptrs.size() > kSgMaxPayloadSegs)
      over[i] = 1;
  }

  // Header per item (stamps total payload_len = sum of segment sizes). hdrs must
  // outlive the RunParallel lambdas that reference it.
  std::vector<std::array<char, ValueHeader::kSize>> hdrs(N);
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    if (over[i]) continue;
    std::string node = Route(items[i].key);
    if (node.empty()) continue;
    size_t total = 0;
    for (size_t s : items[i].sizes) total += s;
    ValueHeader h = self_hdr_;
    h.payload_len = total;
    h.Serialize(hdrs[i].data());
    by_node[node].push_back(i);
  }
  std::vector<std::pair<std::string, std::vector<size_t>>> groups(by_node.begin(), by_node.end());
  RunParallel(groups.size(), batch_concurrency_.load(std::memory_order_relaxed), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<CacheSrcMulti> srcs;
    srcs.reserve(idx.size());
    for (size_t k : idx) {
      CacheSrcMulti s;
      s.key = ToBlockKey(items[k].key);
      s.header = hdrs[k].data();
      s.header_len = ValueHeader::kSize;
      s.payloads.reserve(items[k].ptrs.size());
      for (size_t j = 0; j < items[k].ptrs.size(); ++j)
        s.payloads.emplace_back(items[k].ptrs[j], items[k].sizes[j]);
      srcs.push_back(std::move(s));
    }
    std::vector<Status> sts = t_->CacheFromMulti(node, srcs);
    for (size_t m = 0; m < idx.size(); ++m) ok[idx[m]] = (sts[m] == Status::kOk) ? 1 : 0;
    bool resp = false, ioerr = false;
    for (Status s : sts) { if (s == Status::kIOError) ioerr = true; else resp = true; }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, now);
  });
  uint64_t bytes = 0;
  for (size_t i = 0; i < N; ++i) if (ok[i]) for (size_t s : items[i].sizes) bytes += s;
  return RecordBatch(OpMetrics::kPut, t0, ok, bytes);
}

std::vector<bool> KVClient::BatchGetAutoSg(const std::vector<KvGetItemSg>& items,
                                           std::vector<size_t>* out_lens) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  std::vector<size_t> lens(N, 0);  // distinct indices => thread-safe writes

  // Guard (same as put): a key with more destination segments than one RDMA work
  // request can scatter into is reported a miss up front.
  std::vector<char> over(N, 0);
  for (size_t i = 0; i < N; ++i) {
    if (items[i].key.empty() ||  // null/empty key: skip (no wasted GET issued)
        items[i].dsts.size() != items[i].caps.size() ||
        items[i].dsts.size() > kSgMaxPayloadSegs)
      over[i] = 1;
  }

  // Group by (node, total_cap): a group shares the Range length (= sum of caps)
  // so the existing RangeInto windowing/pipelining applies unchanged.
  std::map<std::pair<std::string, size_t>, std::vector<size_t>> by;
  for (size_t i = 0; i < N; ++i) {
    if (over[i]) continue;
    std::string node = Route(items[i].key);
    if (node.empty()) continue;
    size_t cap = 0;
    for (size_t c : items[i].caps) cap += c;
    by[{node, cap}].push_back(i);
  }
  std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>> groups(by.begin(), by.end());
  RunParallel(groups.size(), batch_concurrency_.load(std::memory_order_relaxed), [&](size_t g) {
    const std::string& node = groups[g].first.first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> keys;
    std::vector<RangeDstMulti> dsts;
    keys.reserve(idx.size());
    dsts.reserve(idx.size());
    for (size_t k : idx) {
      keys.push_back(ToBlockKey(items[k].key));
      RangeDstMulti d;
      d.payloads.reserve(items[k].dsts.size());
      for (size_t j = 0; j < items[k].dsts.size(); ++j)
        d.payloads.emplace_back(items[k].dsts[j], items[k].caps[j]);
      dsts.push_back(std::move(d));
    }
    std::vector<std::string> hdrs;
    std::vector<size_t> tlens;
    std::vector<Status> sts =
        t_->RangeIntoMulti(node, keys, dsts, ValueHeader::kSize, &hdrs, &tlens);
    bool resp = false, ioerr = false;
    for (Status s : sts) { if (s == Status::kIOError) ioerr = true; else resp = true; }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, now);
    for (size_t m = 0; m < idx.size(); ++m) {
      size_t cap = groups[g].first.second;
      if (sts[m] != Status::kOk || hdrs[m].size() < ValueHeader::kSize) continue;
      ValueHeader h;
      if (!ValueHeader::Parse(hdrs[m].data(), hdrs[m].size(), &h)) continue;
      if (!HeaderMatches(self_hdr_, h)) continue;
      if (h.payload_len > cap) continue;  // doesn't fit caller buffers => miss
      lens[idx[m]] = h.payload_len;       // authoritative length from header
      hit[idx[m]] = 1;
    }
  });
  uint64_t bytes = 0;
  for (size_t x : lens) bytes += x;
  if (out_lens) *out_lens = std::move(lens);
  return RecordBatch(OpMetrics::kGet, t0, hit, bytes);
}

}  // namespace dfkv
