#include "mds/mds_server.h"

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include "mds/mds_proto.h"
#include "utils/net_util.h"
#include "transport/wire.h"

namespace dfkv {

MdsServer::~MdsServer() { Stop(); }

std::string MdsServer::MemberKey(const std::string& group, const std::string& id) {
  return "/dfkv/v1/groups/" + group + "/members/" + id;
}

Status MdsServer::Start(int port) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return Status::kIOError;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_ANY);
  sa.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  if (::listen(listen_fd_, 128) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  socklen_t sl = sizeof(sa);
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&sa), &sl);
  port_ = ntohs(sa.sin_port);
  running_ = true;
  accept_thread_ = std::thread([this] { AcceptLoop(); });
  return Status::kOk;
}

void MdsServer::Stop() {
  if (!running_.exchange(false)) return;
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);
  if (accept_thread_.joinable()) accept_thread_.join();
  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  std::vector<int> fds;
  std::vector<Conn> conns;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    fds.assign(conn_fds_.begin(), conn_fds_.end());
    conns.swap(conns_);
  }
  for (int fd : fds) ::shutdown(fd, SHUT_RDWR);
  for (auto& c : conns) if (c.th.joinable()) c.th.join();
}

void MdsServer::ReapDoneLocked() {
  for (auto it = conns_.begin(); it != conns_.end();) {
    if (it->done->load(std::memory_order_acquire)) {
      if (it->th.joinable()) it->th.join();
      it = conns_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t MdsServer::live_conn_count() {
  std::lock_guard<std::mutex> lk(conn_mu_);
  return conns_.size();
}

void MdsServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { ::close(fd); break; }
    ReapDoneLocked();  // join handlers that finished since the last accept
    conn_fds_.push_back(fd);
    auto done = std::make_shared<std::atomic<bool>>(false);
    conns_.push_back({std::thread([this, fd, done] {
                        Handle(fd);
                        {
                          std::lock_guard<std::mutex> lk(conn_mu_);
                          for (auto it = conn_fds_.begin(); it != conn_fds_.end(); ++it)
                            if (*it == fd) { conn_fds_.erase(it); break; }
                        }
                        ::close(fd);
                        done->store(true, std::memory_order_release);  // last act
                      }),
                      done});
  }
}

Status MdsServer::Upsert(const std::string& group, const MemberInfo& m) {
  std::string key = MemberKey(group, m.id);
  std::string val = EncodeMembers({m}, 0);
  // Look up this member's known lease under the lock, then do the BLOCKING etcd
  // calls OUTSIDE it. Previously lease_mu_ was held across LeaseKeepAlive/Grant/Put,
  // so every member's heartbeat serialized on one mutex held across network I/O —
  // a slow etcd stalled all members' liveness. Now heartbeats for distinct members
  // run in parallel; the lock only guards the in-memory {key->leaseID} map.
  int64_t existing = 0;
  bool have = false;
  {
    std::lock_guard<std::mutex> lk(lease_mu_);
    auto it = leases_.find(key);
    if (it != leases_.end()) { existing = it->second; have = true; }
  }
  if (have && etcd_.LeaseKeepAlive(existing)) return Status::kOk;  // fast path: refresh
  // Slow path: (re)grant a lease and (re)write the member key. A same-key race
  // here (two heartbeats both missing the lease) can briefly orphan one lease,
  // which expires on its own via the TTL; same-key heartbeats are serial in
  // practice (one node, one connection), so this is rare and harmless.
  auto lid = etcd_.LeaseGrant(kTtlSeconds);
  if (!lid) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return Status::kIOError; }
  metrics_.lease_grants.fetch_add(1, std::memory_order_relaxed);
  if (!etcd_.Put(key, val, *lid)) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return Status::kIOError; }
  {
    std::lock_guard<std::mutex> lk(lease_mu_);
    leases_[key] = *lid;
  }
  return Status::kOk;
}

Status MdsServer::ListMembers(const std::string& group, std::string* out) {
  metrics_.list_requests.fetch_add(1, std::memory_order_relaxed);
  std::string prefix = "/dfkv/v1/groups/" + group + "/members/";
  auto r = etcd_.RangePrefix(prefix);
  if (!r) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return Status::kIOError; }
  std::vector<MemberInfo> members;
  for (const auto& kv : r->kvs) {
    std::vector<MemberInfo> one;
    uint64_t e = 0;
    if (DecodeMembers(kv.second.data(), kv.second.size(), &one, &e) && one.size() == 1)
      members.push_back(one[0]);
  }
  // Epoch = content hash of the member set, NOT etcd's global revision: the
  // revision bumps on every cluster write (including unrelated groups), which
  // would make clients rebuild their ring needlessly. The hash changes iff THIS
  // group's membership content changes.
  metrics_.members_last_list.store(members.size(), std::memory_order_relaxed);
  *out = EncodeMembers(members, MembersEpoch(members));
  return Status::kOk;
}

void MdsServer::Handle(int fd) {
  while (running_) {
    char prefix[kReqPrefix];
    if (!net::ReadAll(fd, prefix, kReqPrefix)) return;
    ReqFields rq;
    if (!DecodeReq(prefix, &rq)) return;
    std::vector<char> payload(rq.payload_len);
    if (rq.payload_len && !net::ReadAll(fd, payload.data(), rq.payload_len)) return;

    std::string data;
    Status st = Status::kInvalid;
    WireOp op = static_cast<WireOp>(rq.op);
    if (op == WireOp::kRegister || op == WireOp::kHeartbeat) {
      if (op == WireOp::kRegister)
        metrics_.register_requests.fetch_add(1, std::memory_order_relaxed);
      else
        metrics_.keepalives.fetch_add(1, std::memory_order_relaxed);
      std::string group;
      MemberInfo m;
      if (DecodeMemberReq(payload.data(), rq.payload_len, &group, &m))
        st = Upsert(group, m);
    } else if (op == WireOp::kListMembers) {
      std::string group(payload.data(), rq.payload_len);
      st = ListMembers(group, &data);
    }

    char rp[kRespPrefix];
    EncodeResp(rp, st, data.size());
    if (!net::WriteAll(fd, rp, kRespPrefix)) return;
    if (!data.empty() && !net::WriteAll(fd, data.data(), data.size())) return;
  }
}

}  // namespace dfkv
