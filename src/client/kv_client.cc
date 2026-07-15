#include "client/kv_client.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "client/key_map.h"
#include "utils/log.h"
#include "utils/thread_name.h"
#include "transport/transport_factory.h"

namespace dfkv {

namespace {
// Max payload segments per scatter-gather key: the guards below use the LIVE
// Transport::MaxSgPayloadSegs() (negotiated max_sge - 1 on RDMA; 29 on TCP),
// and the connector reads the same value through dfkv_max_sg_segs, so its
// chunking matches whatever HCA the client actually opened. A key exceeding
// the guard is reported failed up front instead of corrupted; a stale caller
// still hard-coding 29 on a smaller-max_sge device degrades per-item (the
// transport's window check fails only the offending key), never per-batch.

// Shared fan-out pool for the Batch* paths. The old RunParallel created and
// joined up to 32 std::threads PER CALL; a connector issuing batches at high
// rate paid ~10-30 us of thread create/teardown per worker on every batch,
// plus the scheduler churn. Workers here are created lazily once (cap 32,
// process lifetime) and assist parallel-for jobs; the caller thread always
// participates, so a job never waits on pool availability to start.
class FanoutPool {
 public:
  // Leaky singleton: workers block in cv-wait at process exit; a static
  // destructor joining them would hang shutdown, so the pool is never
  // destroyed (threads are detached and die with the process).
  static FanoutPool& Instance() {
    static FanoutPool* p = new FanoutPool();
    return *p;
  }

  void Run(size_t n, size_t workers, const std::function<void(size_t)>& fn) {
    if (n == 0) return;
    if (workers > n) workers = n;
    if (workers > kMaxThreads) workers = kMaxThreads;
    if (workers <= 1) {  // serial path: exceptions propagate as before
      for (size_t i = 0; i < n; ++i) fn(i);
      return;
    }
    auto job = std::make_shared<Job>();
    job->n = n;
    job->fn = fn;  // one std::function copy per batch (vs a thread per worker)
    const size_t helpers = workers - 1;
    EnsureThreads(helpers);
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (size_t i = 0; i < helpers; ++i) queue_.push_back(job);
    }
    cv_.notify_all();
    Work(*job);  // caller participates; late helpers find no indices and drop out
    std::unique_lock<std::mutex> lk(job->m);
    job->done_cv.wait(lk, [&] { return job->done.load(std::memory_order_acquire) == job->n; });
  }

 private:
  struct Job {
    size_t n = 0;
    std::function<void(size_t)> fn;
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    std::mutex m;
    std::condition_variable done_cv;
  };

  static void Work(Job& j) {
    for (size_t i = j.next.fetch_add(1); i < j.n; i = j.next.fetch_add(1)) {
      // An escaping exception previously hit the std::thread entry and called
      // std::terminate (while the serial path propagated it) — a cache client
      // must degrade the ITEM, not kill the process. The item's slot keeps its
      // pre-initialized failure value (miss/failed-put), which the connector
      // already handles.
      try { j.fn(i); } catch (...) {}
      if (j.done.fetch_add(1, std::memory_order_acq_rel) + 1 == j.n) {
        std::lock_guard<std::mutex> lk(j.m);  // pairs with the waiter's wait
        j.done_cv.notify_all();
      }
    }
  }

  void EnsureThreads(size_t want) {
    std::lock_guard<std::mutex> lk(mu_);
    while (threads_ < want && threads_ < kMaxThreads) {
      std::thread([this, i = threads_] { NameThisThread("kv-fan-", i); Loop(); }).detach();
      ++threads_;
    }
  }

  void Loop() {
    for (;;) {
      std::shared_ptr<Job> j;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return !queue_.empty(); });
        j = std::move(queue_.front());
        queue_.pop_front();
      }
      Work(*j);
    }
  }

  static constexpr size_t kMaxThreads = 32;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> queue_;
  size_t threads_ = 0;  // guarded by mu_
};

// Run fn(i) for i in [0,n) across up to `workers` threads (atomic work-steal,
// shared persistent pool).
void RunParallel(size_t n, size_t workers, const std::function<void(size_t)>& fn) {
  FanoutPool::Instance().Run(n, workers, fn);
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
  // Same-host GET rendezvous (phase 5). Off unless DFKV_CLIENT_NODE_DEDUP=1;
  // namespaced by model_hash so distinct keyspaces never share entries.
  dedup_ = NodeDedup::FromEnv(self_hdr_.model_hash);
  // Batch fan-out workers override (0/unset = auto; see set_batch_concurrency).
  if (const char* bc = std::getenv("DFKV_BATCH_CONCURRENCY")) {
    long x = std::strtol(bc, nullptr, 10);
    if (x > 0) batch_concurrency_.store(static_cast<size_t>(x), std::memory_order_relaxed);
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

void KVClient::StartClientRegistration(std::vector<std::string> mds_eps,
                                       const std::string& group, MemberInfo self,
                                       int heartbeat_ms) {
  StopClientRegistration();
  // is_client=true routes the registrar to kClientRegister/kClientHeartbeat,
  // which the MDS writes under /clients/<id> (never the placement ring). No
  // stats provider: a consumer has no ring-level stats to report.
  client_registrar_ = std::make_unique<MdsRegistrar>(
      std::move(mds_eps), group, self, heartbeat_ms, /*io_timeout_ms=*/2000,
      /*is_client=*/true);
  client_registrar_->Start();
}

void KVClient::StopClientRegistration() {
  if (client_registrar_) { client_registrar_->Stop(); client_registrar_.reset(); }
}

KVClient::~KVClient() { StopProbe(); StopClientRegistration(); StopMdsDiscovery(); }

void KVClient::StartProbe(int interval_ms) {
  if (interval_ms <= 0 || probe_running_.load(std::memory_order_relaxed)) return;
  probe_interval_ms_ = interval_ms;
  probe_running_.store(true, std::memory_order_relaxed);
  probe_th_ = std::thread([this] { NameThisThread("kv-probe"); ProbeLoop(); });
}

void KVClient::StopProbe() {
  if (!probe_running_.exchange(false, std::memory_order_relaxed)) return;
  probe_cv_.notify_all();
  if (probe_th_.joinable()) probe_th_.join();
}

void KVClient::ProbeLoop() {
  // A sentinel key that no real workload writes; kExist of it is a cheap round
  // trip that measures the node's responsiveness (kNotFound is a healthy reply).
  const BlockKey probe = ToBlockKey("__dfkv_probe__", self_hdr_.model_hash);
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
      // A non-IO reply also means the peer is reachable: clear its data-path
      // cooldown here (off the data path) so recovery never depends on a data
      // request re-dialing a peer that is still inside its backed-off cooldown.
      if (st != Status::kIOError) {
        peer_lat_.Observe(a, sec);
        health_.MarkProbeAlive(a);
      }
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
  if (dedup_) {
    s += "# HELP dfkv_client_dedup_hits_total Rendezvous results served from shm\n";
    s += "# TYPE dfkv_client_dedup_hits_total counter\n";
    s += "dfkv_client_dedup_hits_total " + std::to_string(dedup_->hits()) + "\n";
    s += "# HELP dfkv_client_dedup_fetches_total Keys this process fetched for the rendezvous\n";
    s += "# TYPE dfkv_client_dedup_fetches_total counter\n";
    s += "dfkv_client_dedup_fetches_total " + std::to_string(dedup_->fetches()) + "\n";
    s += "# HELP dfkv_client_dedup_wait_hits_total Waits resolved by a peer's publish\n";
    s += "# TYPE dfkv_client_dedup_wait_hits_total counter\n";
    s += "dfkv_client_dedup_wait_hits_total " + std::to_string(dedup_->wait_hits()) + "\n";
    s += "# HELP dfkv_client_dedup_wait_timeouts_total Waits that fell back to a direct fetch\n";
    s += "# TYPE dfkv_client_dedup_wait_timeouts_total counter\n";
    s += "dfkv_client_dedup_wait_timeouts_total " + std::to_string(dedup_->wait_timeouts()) + "\n";
  }
  if (const GpuNodeDedup* gd = gpu_dedup_raw_.load(std::memory_order_acquire)) {
    s += "# HELP dfkv_client_gpu_dedup_hits_total GPU rendezvous results served via CUDA IPC\n";
    s += "# TYPE dfkv_client_gpu_dedup_hits_total counter\n";
    s += "dfkv_client_gpu_dedup_hits_total " + std::to_string(gd->hits()) + "\n";
    s += "# HELP dfkv_client_gpu_dedup_fetches_total Keys this process fetched for the GPU rendezvous\n";
    s += "# TYPE dfkv_client_gpu_dedup_fetches_total counter\n";
    s += "dfkv_client_gpu_dedup_fetches_total " + std::to_string(gd->fetches()) + "\n";
    s += "# HELP dfkv_client_gpu_dedup_wait_hits_total Waits resolved by a peer's publish\n";
    s += "# TYPE dfkv_client_gpu_dedup_wait_hits_total counter\n";
    s += "dfkv_client_gpu_dedup_wait_hits_total " + std::to_string(gd->wait_hits()) + "\n";
    s += "# HELP dfkv_client_gpu_dedup_wait_timeouts_total Waits that fell back to a direct fetch\n";
    s += "# TYPE dfkv_client_gpu_dedup_wait_timeouts_total counter\n";
    s += "dfkv_client_gpu_dedup_wait_timeouts_total " + std::to_string(gd->wait_timeouts()) + "\n";
  }
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
  BlockKey bk = ToBlockKey(key, self_hdr_.model_hash);
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
  // Fresh clock: `now` was taken before a round trip that can burn the whole
  // io_timeout, which would shorten (or void) the cooldown this MarkBad sets.
  if (st == Status::kIOError) { health_.MarkBad(node, NowMs()); return false; }
  // Only a real reply (hit or logical miss) is "good"; kInvalid/kCacheFull are
  // client-side/oversize outcomes that must not clear a peer's cooldown.
  if (st == Status::kOk || st == Status::kNotFound) health_.MarkGood(node);
  return st == Status::kOk;
}

bool KVClient::Get(const std::string& key, void* out, size_t n) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;

  BlockKey bk = ToBlockKey(key, self_hdr_.model_hash);
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
  // Fresh clock: `now` was taken before a round trip that can burn the whole
  // io_timeout, which would shorten (or void) the cooldown this MarkBad sets.
  if (st == Status::kIOError) { health_.MarkBad(node, NowMs()); return false; }
  // Only a real reply (hit or logical miss) is "good"; kInvalid/kCacheFull are
  // client-side/oversize outcomes that must not clear a peer's cooldown.
  if (st == Status::kOk || st == Status::kNotFound) health_.MarkGood(node);
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
  BlockKey bk = ToBlockKey(key, self_hdr_.model_hash);

  if (t_->pipelined()) {
    // Zero-copy via RangeInto (a 1-element batch), matching Get(): the payload
    // scatters straight into `out`'s buffer. This also lifts the limit from the
    // single-Range control_cap (8 MiB) to the batch max_payload (64 MiB) — a
    // value in (8 MiB, 64 MiB] used to "miss on single read, hit on batch read".
    out->resize(max_bytes);
    std::vector<BlockKey> keys{bk};
    std::vector<RangeDst> dsts{RangeDst{max_bytes ? &(*out)[0] : nullptr, max_bytes}};
    std::vector<std::string> hdrs;
    Status st = t_->RangeInto(node, keys, dsts, ValueHeader::kSize, &hdrs)[0];
    if (st == Status::kIOError) { health_.MarkBad(node, NowMs()); out->clear(); return false; }
    if (st == Status::kOk || st == Status::kNotFound) health_.MarkGood(node);
    if (st != Status::kOk || hdrs[0].size() < ValueHeader::kSize) { out->clear(); return false; }
    ValueHeader h;
    if (!ValueHeader::Parse(hdrs[0].data(), hdrs[0].size(), &h) ||
        !HeaderMatches(self_hdr_, h) || h.payload_len > max_bytes) {
      out->clear();
      return false;
    }
    out->resize(h.payload_len);  // trim the receive buffer to the true stored length
    return true;
  }

  // TCP (non-pipelined): response into a string, then copy out.
  std::string raw;
  Status st = t_->Range(node, bk, 0, ValueHeader::kSize + max_bytes, &raw);
  // Fresh clock: `now` was taken before a round trip that can burn the whole
  // io_timeout, which would shorten (or void) the cooldown this MarkBad sets.
  if (st == Status::kIOError) { health_.MarkBad(node, NowMs()); return false; }
  // Only a real reply (hit or logical miss) is "good"; kInvalid/kCacheFull are
  // client-side/oversize outcomes that must not clear a peer's cooldown.
  if (st == Status::kOk || st == Status::kNotFound) health_.MarkGood(node);
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
  BlockKey bk = ToBlockKey(key, self_hdr_.model_hash);

  if (t_->pipelined()) {
    // Zero-copy: the payload scatters straight into the caller's buffer and the
    // header (true payload_len) comes back separately. Lifts the 8 MiB
    // single-Range cap to the 64 MiB batch max_payload, like Get()/GetAuto(str).
    std::vector<BlockKey> keys{bk};
    std::vector<RangeDst> dsts{RangeDst{cap ? out : nullptr, cap}};
    std::vector<std::string> hdrs;
    Status st = t_->RangeInto(node, keys, dsts, ValueHeader::kSize, &hdrs)[0];
    if (st == Status::kIOError) { health_.MarkBad(node, NowMs()); return false; }
    if (st == Status::kOk || st == Status::kNotFound) health_.MarkGood(node);
    if (st != Status::kOk || hdrs[0].size() < ValueHeader::kSize) return false;
    ValueHeader h;
    if (!ValueHeader::Parse(hdrs[0].data(), hdrs[0].size(), &h)) return false;
    if (!HeaderMatches(self_hdr_, h)) return false;        // geometry drift => miss
    if (h.payload_len > cap) return false;                 // won't fit caller buffer
    if (out_len) *out_len = h.payload_len;
    return true;
  }

  // TCP: read [header | up-to-cap payload]; the header tells us the true payload_len.
  std::string raw;
  Status st = t_->Range(node, bk, 0, ValueHeader::kSize + cap, &raw);
  // Fresh clock: `now` was taken before a round trip that can burn the whole
  // io_timeout, which would shorten (or void) the cooldown this MarkBad sets.
  if (st == Status::kIOError) { health_.MarkBad(node, NowMs()); return false; }
  // Only a real reply (hit or logical miss) is "good"; kInvalid/kCacheFull are
  // client-side/oversize outcomes that must not clear a peer's cooldown.
  if (st == Status::kOk || st == Status::kNotFound) health_.MarkGood(node);
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
  Status st = t_->Exist(node, ToBlockKey(key, self_hdr_.model_hash), &e);
  // Fresh clock: `now` was taken before a round trip that can burn the whole
  // io_timeout, which would shorten (or void) the cooldown this MarkBad sets.
  if (st == Status::kIOError) { health_.MarkBad(node, NowMs()); return false; }
  // Only a real reply (hit or logical miss) is "good"; kInvalid/kCacheFull are
  // client-side/oversize outcomes that must not clear a peer's cooldown.
  if (st == Status::kOk || st == Status::kNotFound) health_.MarkGood(node);
  return st == Status::kOk && e;
}

bool KVClient::Remove(const std::string& key) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  Status st = t_->Remove(node, ToBlockKey(key, self_hdr_.model_hash));
  // Fresh clock: `now` was taken before a round trip that can burn the whole
  // io_timeout, which would shorten (or void) the cooldown this MarkBad sets.
  if (st == Status::kIOError) { health_.MarkBad(node, NowMs()); return false; }
  // Only a real reply (hit or logical miss) is "good"; kInvalid/kCacheFull are
  // client-side/oversize outcomes that must not clear a peer's cooldown.
  if (st == Status::kOk || st == Status::kNotFound) health_.MarkGood(node);
  // kNotFound is a success for the caller: the goal (key not present) holds.
  return st == Status::kOk || st == Status::kNotFound;
}

std::vector<bool> KVClient::BatchPut(const std::vector<KvPutItem>& items) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> ok(N, 0);  // char (not vector<bool>) for thread-safe writes
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
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
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<CacheSrc> srcs;
    srcs.reserve(idx.size());
    for (size_t k : idx)
      srcs.push_back(CacheSrc{ToBlockKey(items[k].key, self_hdr_.model_hash), hdrs[k].data(),
                              ValueHeader::kSize, items[k].value, items[k].n});
    std::vector<Status> sts = t_->CacheFrom(node, srcs);
    for (size_t m = 0; m < idx.size(); ++m) ok[idx[m]] = (sts[m] == Status::kOk) ? 1 : 0;
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, NowMs());
  });
  // RDMA path records the batch once (TCP path above is counted per-key by Put()).
  uint64_t bytes = 0;
  for (size_t i = 0; i < N; ++i) if (ok[i]) bytes += items[i].n;
  return RecordBatch(OpMetrics::kPut, t0, ok, bytes);
}

std::vector<bool> KVClient::BatchGet(const std::vector<KvGetItem>& items) {
  if (!dedup_) return BatchGetDirect(items);
  // Same-host rendezvous (phase 5): first arrival fetches a key, peers copy
  // the published payload from shm — TP-replicated KV otherwise multiplies
  // every load by the rank count. Every dedup outcome degrades to the plain
  // remote path, so correctness never depends on the shm state.
  const size_t N = items.size();
  std::vector<bool> res(N, false);
  std::vector<BlockKey> bks(N);
  std::vector<KvGetItem> fetch_items;
  std::vector<size_t> fetch_map, wait_list;
  std::vector<char> fetch_claimed;  // 1 = we own a FETCHING slot
  fetch_items.reserve(N);
  // Host rendezvous can't serve device destinations (memcpy to a device VA);
  // route those straight to the fetch list unclaimed instead of crashing when
  // the env switch is set in a GPUDirect process. cu is null on CPU-only hosts.
  const CudaLib* cu = CudaLib::Get();
  for (size_t i = 0; i < N; ++i) {
    if (cu && cu->IsDevicePtr(items[i].out)) {
      fetch_map.push_back(i);
      fetch_items.push_back(items[i]);
      fetch_claimed.push_back(0);
      continue;
    }
    bks[i] = ToBlockKey(items[i].key, self_hdr_.model_hash);
    switch (dedup_->Claim(bks[i], items[i].n, static_cast<char*>(items[i].out))) {
      case NodeDedup::Role::kHit:
        res[i] = true;
        break;
      case NodeDedup::Role::kFetch:
        fetch_map.push_back(i);
        fetch_items.push_back(items[i]);
        fetch_claimed.push_back(1);
        break;
      case NodeDedup::Role::kWait:
        wait_list.push_back(i);
        break;
    }
  }
  if (!fetch_items.empty()) {
    auto r = BatchGetDirect(fetch_items);
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      if (!fetch_claimed[m]) continue;
      if (r[m])
        dedup_->Publish(bks[i], NodeDedup::Kind::kData,
                        static_cast<const char*>(items[i].out), items[i].n);
      else
        dedup_->Abort(bks[i], NodeDedup::Kind::kData);
    }
  }
  for (size_t i : wait_list) {
    if (dedup_->WaitCopy(bks[i], items[i].n, static_cast<char*>(items[i].out)))
      res[i] = true;
    else  // fetcher failed/slow: bounded fallback to a direct read
      res[i] = Get(items[i].key, items[i].out, items[i].n);
  }
  return res;
}

std::vector<bool> KVClient::BatchGetDirect(const std::vector<KvGetItem>& items) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
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
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
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
      keys.push_back(ToBlockKey(items[k].key, self_hdr_.model_hash));
      dsts.push_back(RangeDst{items[k].out, n});
    }
    // Zero-copy + zero-touch: the payload lands straight in items[].out (CPU never
    // reads it); hdrs[] gets the value header for geometry verification only.
    std::vector<std::string> hdrs;
    std::vector<Status> sts = t_->RangeInto(node, keys, dsts, ValueHeader::kSize, &hdrs);
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, NowMs());
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
  if (!dedup_) return BatchGetAutoDirect(items, out_lens);
  // Same-host rendezvous, variable-size flavor: identity is the key; a
  // published payload hits any reader whose cap fits it (see node_dedup.h).
  const size_t N = items.size();
  std::vector<bool> res(N, false);
  std::vector<size_t> lens(N, 0);
  std::vector<BlockKey> bks(N);
  std::vector<KvGetItem> fetch_items;
  std::vector<size_t> fetch_map, wait_list;
  std::vector<char> fetch_claimed;  // 1 = we own a FETCHING slot
  fetch_items.reserve(N);
  const CudaLib* cu = CudaLib::Get();  // device destinations bypass (see BatchGet)
  for (size_t i = 0; i < N; ++i) {
    if (cu && cu->IsDevicePtr(items[i].out)) {
      fetch_map.push_back(i);
      fetch_items.push_back(items[i]);
      fetch_claimed.push_back(0);
      continue;
    }
    bks[i] = ToBlockKey(items[i].key, self_hdr_.model_hash);
    size_t got = 0;
    switch (dedup_->ClaimAuto(bks[i], items[i].n, static_cast<char*>(items[i].out), &got)) {
      case NodeDedup::Role::kHit:
        res[i] = true;
        lens[i] = got;
        break;
      case NodeDedup::Role::kFetch:
        fetch_map.push_back(i);
        fetch_items.push_back(items[i]);
        fetch_claimed.push_back(1);
        break;
      case NodeDedup::Role::kWait:
        wait_list.push_back(i);
        break;
    }
  }
  if (!fetch_items.empty()) {
    std::vector<size_t> flens;
    auto r = BatchGetAutoDirect(fetch_items, &flens);
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      if (r[m]) lens[i] = flens[m];
      if (!fetch_claimed[m]) continue;
      if (r[m]) {
        dedup_->Publish(bks[i], NodeDedup::Kind::kData,
                        static_cast<const char*>(items[i].out), flens[m]);
      } else {
        dedup_->Abort(bks[i], NodeDedup::Kind::kData);
      }
    }
  }
  for (size_t i : wait_list) {
    size_t got = 0;
    if (dedup_->WaitCopyAuto(bks[i], items[i].n, static_cast<char*>(items[i].out), &got)) {
      res[i] = true;
      lens[i] = got;
    } else {  // fetcher failed/slow: bounded fallback to a direct read
      size_t got2 = 0;
      res[i] = GetAuto(items[i].key, items[i].out, items[i].n, &got2);
      if (res[i]) lens[i] = got2;
    }
  }
  if (out_lens) *out_lens = std::move(lens);
  return res;
}

std::vector<bool> KVClient::BatchGetAutoDirect(const std::vector<KvGetItem>& items,
                                         std::vector<size_t>* out_lens) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  std::vector<size_t> lens(N, 0);  // distinct indices => thread-safe writes
  if (!t_->pipelined()) {  // TCP: parallelize per item with our own threads.
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
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
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
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
      keys.push_back(ToBlockKey(items[k].key, self_hdr_.model_hash));
      dsts.push_back(RangeDst{items[k].out, cap});
    }
    std::vector<std::string> hdrs;
    std::vector<Status> sts = t_->RangeInto(node, keys, dsts, ValueHeader::kSize, &hdrs);
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, NowMs());
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
  if (!dedup_) return BatchExistDirect(keys);
  // Exist probes ride the rendezvous with a 1-byte answer. NEGATIVE answers
  // are published too (a valid result, unlike a failed GET); staleness is
  // bounded by the TTL and safe both ways (stale absent -> recompute, stale
  // present -> the GET misses and recomputes).
  const size_t N = keys.size();
  std::vector<bool> res(N, false);
  std::vector<BlockKey> bks(N);
  std::vector<std::string> fetch_keys;
  std::vector<size_t> fetch_map, wait_list;
  fetch_keys.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    bks[i] = ToBlockKey(keys[i], self_hdr_.model_hash);
    bool val = false;
    switch (dedup_->ClaimExist(bks[i], &val)) {
      case NodeDedup::Role::kHit:
        res[i] = val;
        break;
      case NodeDedup::Role::kFetch:
        fetch_map.push_back(i);
        fetch_keys.push_back(keys[i]);
        break;
      case NodeDedup::Role::kWait:
        wait_list.push_back(i);
        break;
    }
  }
  if (!fetch_keys.empty()) {
    auto r = BatchExistDirect(fetch_keys);
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      const char b = r[m] ? 1 : 0;
      dedup_->Publish(bks[i], NodeDedup::Kind::kExist, &b, 1);
    }
  }
  for (size_t i : wait_list) {
    bool val = false;
    if (dedup_->WaitExist(bks[i], &val)) res[i] = val;
    else res[i] = Exist(keys[i]);  // bounded fallback
  }
  return res;
}

std::vector<bool> KVClient::BatchExistDirect(const std::vector<std::string>& keys) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = keys.size();
  std::vector<char> e(N, 0);
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
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
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> bkeys;
    bkeys.reserve(idx.size());
    for (size_t k : idx) bkeys.push_back(ToBlockKey(keys[k], self_hdr_.model_hash));
    std::vector<char> ex;
    std::vector<Status> sts = t_->ExistMany(node, bkeys, &ex);
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, NowMs());
    for (size_t m = 0; m < idx.size(); ++m) e[idx[m]] = (m < ex.size() && ex[m]) ? 1 : 0;
  });
  return RecordBatch(OpMetrics::kExist, t0, e, 0);
}

std::vector<bool> KVClient::BatchRemove(const std::vector<std::string>& keys) {
  const size_t N = keys.size();
  std::vector<char> ok(N, 0);
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
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
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> bkeys;
    bkeys.reserve(idx.size());
    for (size_t k : idx) bkeys.push_back(ToBlockKey(keys[k], self_hdr_.model_hash));
    std::vector<Status> sts = t_->RemoveMany(node, bkeys);
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, NowMs());
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
        items[i].ptrs.size() > t_->MaxSgPayloadSegs())
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
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<CacheSrcMulti> srcs;
    srcs.reserve(idx.size());
    for (size_t k : idx) {
      CacheSrcMulti s;
      s.key = ToBlockKey(items[k].key, self_hdr_.model_hash);
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
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, NowMs());
  });
  uint64_t bytes = 0;
  for (size_t i = 0; i < N; ++i) if (ok[i]) for (size_t s : items[i].sizes) bytes += s;
  return RecordBatch(OpMetrics::kPut, t0, ok, bytes);
}

GpuNodeDedup* KVClient::GpuDedup(const void* device_dst_hint) {
  std::call_once(gpu_dedup_once_, [this, device_dst_hint] {
    gpu_dedup_ = GpuNodeDedup::FromEnv(self_hdr_.model_hash, device_dst_hint);
    gpu_dedup_raw_.store(gpu_dedup_.get(), std::memory_order_release);
  });
  return gpu_dedup_.get();
}

std::vector<bool> KVClient::BatchGetAutoSg(const std::vector<KvGetItemSg>& items,
                                           std::vector<size_t>* out_lens) {
  // Init the GPU rendezvous only once a DEVICE destination shows up: its
  // pointer picks the primary context to bind when this (transfer) thread
  // has none, and host-only callers never pay for an arena.
  GpuNodeDedup* gd = nullptr;
  if (const CudaLib* cu0 = CudaLib::Get()) {
    const void* hint = nullptr;
    for (const auto& it : items)
      if (!it.dsts.empty() && cu0->IsDevicePtr(it.dsts[0])) { hint = it.dsts[0]; break; }
    if (hint) gd = GpuDedup(hint);
  }
  if (!gd || items.empty()) return BatchGetAutoSgDirect(items, out_lens);
  // Same-host rendezvous for GPU destinations (phase 2b, see node_dedup_gpu.h):
  // the vLLM connector's SG gets land in device memory, where lockstep TP
  // ranks re-fetch identical TP-replicated KV. Payloads rendezvous through
  // per-process GPU staging arenas (CUDA IPC + NVLink D2D); every outcome
  // degrades to the plain remote path.
  const CudaLib* cu = CudaLib::Get();  // non-null: gd exists
  const size_t N = items.size();
  std::vector<bool> res(N, false);
  std::vector<size_t> lens(N, 0);
  std::vector<BlockKey> bks(N);
  std::vector<KvGetItemSg> fetch_items;
  std::vector<size_t> fetch_map, wait_list;
  std::vector<char> fetch_claimed;  // 1 = we own a FETCHING slot (must publish/abort)
  std::vector<GpuNodeDedup::Seg> segs;
  auto segs_of = [&segs](const KvGetItemSg& it) {
    segs.clear();
    segs.reserve(it.dsts.size());
    for (size_t j = 0; j < it.dsts.size(); ++j)
      segs.push_back(GpuNodeDedup::Seg{it.dsts[j], it.caps[j]});
  };
  auto total_cap = [](const KvGetItemSg& it) {
    size_t c = 0;
    for (size_t x : it.caps) c += x;
    return c;
  };
  for (size_t i = 0; i < N; ++i) {
    // Eligible = well-shaped AND device-memory destination. Host-destination
    // SG items (or malformed ones the Direct guard rejects) skip the
    // rendezvous entirely — an unclaimed fetch has no publish obligation.
    const bool eligible = !items[i].key.empty() && !items[i].dsts.empty() &&
                          items[i].dsts.size() == items[i].caps.size() &&
                          cu->IsDevicePtr(items[i].dsts[0]);
    if (!eligible) {
      fetch_map.push_back(i);
      fetch_items.push_back(items[i]);
      fetch_claimed.push_back(0);
      continue;
    }
    bks[i] = ToBlockKey(items[i].key, self_hdr_.model_hash);
    segs_of(items[i]);
    size_t got = 0;
    switch (gd->ClaimSg(bks[i], segs.data(), segs.size(), total_cap(items[i]), &got)) {
      case GpuNodeDedup::Role::kHit:
        res[i] = true;
        lens[i] = got;
        break;
      case GpuNodeDedup::Role::kFetch:
        fetch_map.push_back(i);
        fetch_items.push_back(items[i]);
        fetch_claimed.push_back(1);
        break;
      case GpuNodeDedup::Role::kWait:
        wait_list.push_back(i);
        break;
    }
  }
  if (!fetch_items.empty()) {
    std::vector<size_t> flens;
    auto r = BatchGetAutoSgDirect(fetch_items, &flens);
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      if (r[m]) lens[i] = flens[m];
      if (!fetch_claimed[m]) continue;
      if (r[m]) {
        segs_of(items[i]);
        gd->PublishSg(bks[i], segs.data(), segs.size(), flens[m]);
      } else {
        gd->Abort(bks[i]);
      }
    }
  }
  for (size_t i : wait_list) {
    segs_of(items[i]);
    size_t got = 0;
    if (gd->WaitSg(bks[i], segs.data(), segs.size(), total_cap(items[i]), &got)) {
      res[i] = true;
      lens[i] = got;
    } else {  // fetcher failed/slow: bounded fallback to a direct read
      std::vector<KvGetItemSg> one(1, items[i]);
      std::vector<size_t> ol;
      auto rr = BatchGetAutoSgDirect(one, &ol);
      res[i] = rr[0];
      if (rr[0]) lens[i] = ol[0];
    }
  }
  if (out_lens) *out_lens = std::move(lens);
  return res;
}

std::vector<bool> KVClient::BatchGetAutoSgDirect(const std::vector<KvGetItemSg>& items,
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
        items[i].dsts.size() > t_->MaxSgPayloadSegs())
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
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first.first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> keys;
    std::vector<RangeDstMulti> dsts;
    keys.reserve(idx.size());
    dsts.reserve(idx.size());
    for (size_t k : idx) {
      keys.push_back(ToBlockKey(items[k].key, self_hdr_.model_hash));
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
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node); else if (ioerr) health_.MarkBad(node, NowMs());
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
