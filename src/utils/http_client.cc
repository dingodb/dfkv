#include "utils/http_client.h"

#include <unistd.h>

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "utils/net_util.h"

namespace dfkv {

std::string BuildPostRequest(const std::string& host_port, const std::string& path,
                             const std::string& body) {
  std::string r;
  r += "POST " + path + " HTTP/1.1\r\n";
  r += "Host: " + host_port + "\r\n";
  r += "Content-Type: application/json\r\n";
  r += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  r += "Connection: close\r\n";
  r += "\r\n";
  r += body;
  return r;
}

bool ParseResponseHead(const std::string& raw, int* status, long* content_length,
                       size_t* head_end) {
  size_t end = raw.find("\r\n\r\n");
  if (end == std::string::npos) return false;
  *head_end = end + 4;
  size_t sp = raw.find(' ');
  if (sp == std::string::npos) return false;
  *status = std::atoi(raw.c_str() + sp + 1);
  *content_length = -1;  // -1 = absent, -2 = chunked, >=0 = explicit length
  size_t pos = raw.find("\r\n");  // skip status line
  if (pos == std::string::npos) return true;
  pos += 2;
  while (pos < end) {
    size_t eol = raw.find("\r\n", pos);
    if (eol == std::string::npos || eol > end) eol = end;
    size_t colon = raw.find(':', pos);
    if (colon != std::string::npos && colon < eol) {
      std::string name = raw.substr(pos, colon - pos);
      for (char& ch : name) ch = char(tolower((unsigned char)ch));
      std::string value = raw.substr(colon + 1, eol - colon - 1);
      size_t b = value.find_first_not_of(" \t");
      size_t e = value.find_last_not_of(" \t");
      value = (b == std::string::npos) ? std::string() : value.substr(b, e - b + 1);
      if (name == "content-length") {
        if (*content_length != -2) *content_length = std::atol(value.c_str());
      } else if (name == "transfer-encoding") {
        std::string lv = value;
        for (char& ch : lv) ch = char(tolower((unsigned char)ch));
        if (lv.find("chunked") != std::string::npos) *content_length = -2;
      }
    }
    pos = eol + 2;
  }
  return true;
}

namespace {
// Dechunk a Transfer-Encoding: chunked body. `buf` already holds the header plus
// whatever body bytes arrived; `off` is the first body byte. Reads more from fd
// as needed. Returns false on premature close / malformed framing.
bool ReadChunkedBody(int fd, std::string& buf, size_t off, std::string* body) {
  char chunk[4096];
  body->clear();
  while (true) {
    size_t crlf;
    while ((crlf = buf.find("\r\n", off)) == std::string::npos) {
      ssize_t n = ::read(fd, chunk, sizeof(chunk));
      if (n <= 0) return false;
      buf.append(chunk, n);
    }
    std::string sizeline = buf.substr(off, crlf - off);
    size_t semi = sizeline.find(';');  // ignore chunk extensions
    if (semi != std::string::npos) sizeline.resize(semi);
    long sz = std::strtol(sizeline.c_str(), nullptr, 16);
    if (sz < 0) return false;
    off = crlf + 2;
    if (sz == 0) return true;  // final chunk; trailers (if any) ignored
    while (buf.size() < off + (size_t)sz + 2) {  // chunk data + trailing CRLF
      ssize_t n = ::read(fd, chunk, sizeof(chunk));
      if (n <= 0) return false;
      buf.append(chunk, n);
    }
    body->append(buf, off, (size_t)sz);
    off += (size_t)sz + 2;
  }
}
}  // namespace

bool TcpHttpTransport::Post(const std::string& path, const std::string& body,
                            HttpResponse* out) {
  int fd = net::Dial(addr_, timeout_ms_, timeout_ms_);
  if (fd < 0) return false;
  std::string req = BuildPostRequest(addr_, path, body);
  if (!net::WriteAll(fd, req.data(), req.size())) { ::close(fd); return false; }

  std::string buf;
  char chunk[4096];
  int status = 0; long clen = -1; size_t head_end = 0;
  while (!ParseResponseHead(buf, &status, &clen, &head_end)) {
    ssize_t n = ::read(fd, chunk, sizeof(chunk));
    if (n <= 0) { ::close(fd); return false; }
    buf.append(chunk, n);
  }
  if (clen == -2) {  // chunked
    if (!ReadChunkedBody(fd, buf, head_end, &out->body)) { ::close(fd); return false; }
  } else if (clen >= 0) {
    while (buf.size() - head_end < (size_t)clen) {
      ssize_t n = ::read(fd, chunk, sizeof(chunk));
      if (n <= 0) break;
      buf.append(chunk, n);
    }
    if (buf.size() - head_end < (size_t)clen) { ::close(fd); return false; }  // short body = fail loud
    out->body = buf.substr(head_end, clen);
  } else {  // no length, no chunked: read to EOF (Connection: close)
    ssize_t n;
    while ((n = ::read(fd, chunk, sizeof(chunk))) > 0) buf.append(chunk, n);
    out->body = buf.substr(head_end);
  }
  out->status = status;
  ::close(fd);
  return true;
}

}  // namespace dfkv
