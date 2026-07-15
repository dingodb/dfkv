#include "utils/metrics_http.h"

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>

#include "utils/net_util.h"

namespace dfkv {

MetricsHttpServer::~MetricsHttpServer() { Stop(); }

Status MetricsHttpServer::Start(int port, const std::string& bind_addr) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return Status::kIOError;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_ANY);
  if (!bind_addr.empty() &&
      ::inet_pton(AF_INET, bind_addr.c_str(), &sa.sin_addr) != 1) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kInvalid;  // bad bind addr
  }
  sa.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  if (::listen(listen_fd_, 16) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  socklen_t sl = sizeof(sa);
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&sa), &sl);
  port_ = ntohs(sa.sin_port);
  running_ = true;
  accept_thread_ = std::thread([this] { AcceptLoop(); });
  return Status::kOk;
}

void MetricsHttpServer::Stop() {
  if (!running_.exchange(false)) return;  // idempotent
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

void MetricsHttpServer::ReapDoneLocked() {
  for (auto it = conns_.begin(); it != conns_.end();) {
    if (it->done->load(std::memory_order_acquire)) {
      if (it->th.joinable()) it->th.join();
      it = conns_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t MetricsHttpServer::live_conn_count() {
  std::lock_guard<std::mutex> lk(conn_mu_);
  return conns_.size();
}

void MetricsHttpServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    // A scrape is one short request/response; a silent peer must not pin the
    // handler thread (Handle's recv loop otherwise blocks indefinitely).
    timeval tv{10, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { ::close(fd); break; }
    ReapDoneLocked();  // join finished scrape handlers (Connection: close => one per scrape)
    conn_fds_.insert(fd);
    auto done = std::make_shared<std::atomic<bool>>(false);
    conns_.push_back({std::thread([this, fd, done] {
                        Handle(fd);
                        { std::lock_guard<std::mutex> lk(conn_mu_); conn_fds_.erase(fd); }
                        ::close(fd);
                        done->store(true, std::memory_order_release);  // last act
                      }),
                      done});
  }
}

namespace {
// Build an HTTP/1.0 response (Connection: close — we recv to EOF on the client).
std::string Resp(const char* status_line, const char* ctype, const std::string& body) {
  return std::string("HTTP/1.0 ") + status_line + "\r\n" +
         "Content-Type: " + ctype + "\r\n" +
         "Content-Length: " + std::to_string(body.size()) + "\r\n" +
         "Connection: close\r\n\r\n" + body;
}
}  // namespace

void MetricsHttpServer::Handle(int fd) {
  // Read just the request line (up to CRLF). Headers/body are ignored — we only
  // need method + path. Bound the read so a silent peer can't pin the thread.
  std::string line;
  char c;
  size_t guard = 0;
  while (guard++ < 8192) {
    ssize_t r = ::recv(fd, &c, 1, 0);
    if (r <= 0) return;  // peer closed / error
    if (c == '\n') break;
    if (c != '\r') line.push_back(c);
  }
  // line == "GET /path HTTP/1.x"
  std::string path;
  {
    size_t s = line.find(' ');
    size_t e = (s == std::string::npos) ? std::string::npos : line.find(' ', s + 1);
    if (s != std::string::npos && e != std::string::npos) path = line.substr(s + 1, e - s - 1);
  }
  std::string out;
  if (path == "/metrics") {
    out = Resp("200 OK", "text/plain; version=0.0.4", render_ ? render_() : "");
  } else if (path == "/healthz") {
    // A registered predicate (e.g. MDS etcd reachability) gates the status; with
    // none set, keep the always-healthy 200 the endpoint had before.
    if (!health_ || health_())
      out = Resp("200 OK", "text/plain", "ok\n");
    else
      out = Resp("503 Service Unavailable", "text/plain", "unavailable\n");
  } else if (path == "/readyz") {
    // Readiness (serving-capable), not liveness: 503 while startup work
    // (arena pre-fault, MR anchor, listeners, MDS registration) is running.
    if (!ready_ || ready_())
      out = Resp("200 OK", "text/plain", "ready\n");
    else
      out = Resp("503 Service Unavailable", "text/plain", "starting\n");
  } else {
    out = Resp("404 Not Found", "text/plain", "not found\n");
  }
  net::WriteAll(fd, out.data(), out.size());
}

}  // namespace dfkv
