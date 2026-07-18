#include "utils/http_client.h"
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <thread>
using namespace dfkv;  // NOLINT

TEST(HttpClient, BuildPostRequestShape) {
  std::string req = BuildPostRequest("127.0.0.1:2379", "/v3/kv/put", "{\"a\":1}");
  EXPECT_NE(req.find("POST /v3/kv/put HTTP/1.1\r\n"), std::string::npos);
  EXPECT_NE(req.find("Host: 127.0.0.1:2379\r\n"), std::string::npos);
  EXPECT_NE(req.find("Content-Type: application/json\r\n"), std::string::npos);
  EXPECT_NE(req.find("Content-Length: 7\r\n"), std::string::npos);
  EXPECT_NE(req.find("\r\n\r\n{\"a\":1}"), std::string::npos);
}

TEST(HttpClient, ParseResponseHeadStatusAndLength) {
  int status = 0; long clen = -1; size_t head_end = 0;
  std::string raw = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nX-Foo: bar\r\n\r\nhello";
  ASSERT_TRUE(ParseResponseHead(raw, &status, &clen, &head_end));
  EXPECT_EQ(status, 200);
  EXPECT_EQ(clen, 5);
  EXPECT_EQ(raw.substr(head_end), "hello");
}

TEST(HttpClient, ParseResponseHeadNeedsMoreUntilBlankLine) {
  int status = 0; long clen = -1; size_t he = 0;
  EXPECT_FALSE(ParseResponseHead("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n", &status, &clen, &he));
}

TEST(HttpClient, TcpRoundTripAgainstLoopbackServer) {
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(srv, 0);
  int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
  ASSERT_EQ(bind(srv, (sockaddr*)&a, sizeof(a)), 0);
  socklen_t al = sizeof(a); ASSERT_EQ(getsockname(srv, (sockaddr*)&a, &al), 0);
  ASSERT_EQ(listen(srv, 1), 0);
  int port = ntohs(a.sin_port);

  std::thread th([srv] {
    int c = accept(srv, nullptr, nullptr);
    char buf[4096]; EXPECT_GT(::read(c, buf, sizeof(buf)), 0);
    const char* resp = "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{\"ok\":\"yes\"}\n";
    EXPECT_EQ(::write(c, resp, std::strlen(resp)), (ssize_t)std::strlen(resp));
    ::close(c);
  });

  TcpHttpTransport t("127.0.0.1:" + std::to_string(port), /*timeout_ms=*/2000);
  HttpResponse r;
  ASSERT_TRUE(t.Post("/v3/kv/range", "{\"key\":\"x\"}", &r));
  EXPECT_EQ(r.status, 200);
  EXPECT_EQ(r.body, "{\"ok\":\"yes\"}\n");
  th.join();
  ::close(srv);
}

TEST(HttpClient, ParseResponseHeadDetectsChunked) {
  int status = 0; long clen = -1; size_t he = 0;
  ASSERT_TRUE(ParseResponseHead(
      "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", &status, &clen, &he));
  EXPECT_EQ(status, 200);
  EXPECT_EQ(clen, -2);  // -2 = chunked sentinel
}

TEST(HttpClient, TcpRoundTripChunkedResponse) {
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(srv, 0);
  int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
  ASSERT_EQ(bind(srv, (sockaddr*)&a, sizeof(a)), 0);
  socklen_t al = sizeof(a); ASSERT_EQ(getsockname(srv, (sockaddr*)&a, &al), 0);
  ASSERT_EQ(listen(srv, 1), 0);
  int port = ntohs(a.sin_port);
  std::thread th([srv] {
    int c = accept(srv, nullptr, nullptr);
    char buf[4096]; EXPECT_GT(::read(c, buf, sizeof(buf)), 0);
    const char* resp =
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
        "6\r\n{\"ok\":\r\n6\r\n\"yes\"}\r\n0\r\n\r\n";  // 2 chunks of 6 bytes + terminator
    EXPECT_EQ(::write(c, resp, std::strlen(resp)), (ssize_t)std::strlen(resp));
    ::close(c);
  });
  TcpHttpTransport t("127.0.0.1:" + std::to_string(port), 2000);
  HttpResponse r;
  ASSERT_TRUE(t.Post("/v3/kv/range", "{}", &r));
  EXPECT_EQ(r.status, 200);
  EXPECT_EQ(r.body, "{\"ok\":\"yes\"}");  // dechunked, framing stripped
  th.join();
  ::close(srv);
}

TEST(HttpClient, ShortContentLengthBodyFailsLoud) {
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(srv, 0);
  int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
  ASSERT_EQ(bind(srv, (sockaddr*)&a, sizeof(a)), 0);
  socklen_t al = sizeof(a); ASSERT_EQ(getsockname(srv, (sockaddr*)&a, &al), 0);
  ASSERT_EQ(listen(srv, 1), 0);
  int port = ntohs(a.sin_port);
  std::thread th([srv] {
    int c = accept(srv, nullptr, nullptr);
    char buf[4096]; EXPECT_GT(::read(c, buf, sizeof(buf)), 0);
    const char* resp = "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nonly-9-by";  // promises 100, sends 9 then closes
    EXPECT_EQ(::write(c, resp, std::strlen(resp)), (ssize_t)std::strlen(resp));
    ::close(c);
  });
  TcpHttpTransport t("127.0.0.1:" + std::to_string(port), 2000);
  HttpResponse r;
  EXPECT_FALSE(t.Post("/x", "{}", &r));  // must NOT return a silently-truncated body
  th.join();
  ::close(srv);
}
