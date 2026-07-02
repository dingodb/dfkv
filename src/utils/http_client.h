#ifndef DFKV_HTTP_CLIENT_H_
#define DFKV_HTTP_CLIENT_H_

#include <cstddef>
#include <string>
#include <utility>

namespace dfkv {

struct HttpResponse {
  int status = 0;
  std::string body;
};

std::string BuildPostRequest(const std::string& host_port, const std::string& path,
                             const std::string& body);

bool ParseResponseHead(const std::string& raw, int* status, long* content_length,
                       size_t* head_end);

class HttpTransport {
 public:
  virtual ~HttpTransport() = default;
  virtual bool Post(const std::string& path, const std::string& body,
                    HttpResponse* out) = 0;
};

class TcpHttpTransport : public HttpTransport {
 public:
  explicit TcpHttpTransport(std::string addr, int timeout_ms = 2000)
      : addr_(std::move(addr)), timeout_ms_(timeout_ms) {}
  bool Post(const std::string& path, const std::string& body,
            HttpResponse* out) override;

 private:
  std::string addr_;
  int timeout_ms_;
};

}  // namespace dfkv

#endif  // DFKV_HTTP_CLIENT_H_
