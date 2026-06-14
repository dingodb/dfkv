/* Minimal POSIX TCP + little-endian codec helpers for the portable test
 * transport. The real build uses brpc; this is only for the standalone,
 * GPU-free end-to-end harness. */
#ifndef DINGOFS_SRC_CACHE_KVCLIENT_NET_UTIL_H_
#define DINGOFS_SRC_CACHE_KVCLIENT_NET_UTIL_H_

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace dingofs {
namespace cache {
namespace kv {
namespace net {

inline bool WriteAll(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  size_t off = 0;
  while (off < n) {
    ssize_t w = ::send(fd, p + off, n - off, MSG_NOSIGNAL);
    if (w <= 0) {
      if (w < 0 && errno == EINTR) continue;
      return false;
    }
    off += static_cast<size_t>(w);
  }
  return true;
}

inline bool ReadAll(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  size_t off = 0;
  while (off < n) {
    ssize_t r = ::recv(fd, p + off, n - off, 0);
    if (r == 0) return false;  // peer closed
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    off += static_cast<size_t>(r);
  }
  return true;
}

inline void PutU64(char* p, uint64_t v) { std::memcpy(p, &v, 8); }
inline void PutU32(char* p, uint32_t v) { std::memcpy(p, &v, 4); }
inline uint64_t GetU64(const char* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t GetU32(const char* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }

// "ip:port" -> connected fd, or -1.
inline int Dial(const std::string& addr) {
  auto pos = addr.rfind(':');
  if (pos == std::string::npos) return -1;
  std::string ip = addr.substr(0, pos);
  int port = std::atoi(addr.substr(pos + 1).c_str());
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, ip.c_str(), &sa.sin_addr) != 1) { ::close(fd); return -1; }
  int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  if (::connect(fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) { ::close(fd); return -1; }
  return fd;
}

}  // namespace net
}  // namespace kv
}  // namespace cache
}  // namespace dingofs

#endif  // DINGOFS_SRC_CACHE_KVCLIENT_NET_UTIL_H_
