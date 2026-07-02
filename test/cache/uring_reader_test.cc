// UringReader unit tests. Compiled to an empty suite unless the tree is built
// with -DDFKV_WITH_URING=ON (which requires -DDFKV_WITH_RDMA=ON); the reader
// itself is env-gated at runtime, but here we drive it directly.
#ifdef DFKV_WITH_URING

#include "cache/uring_reader.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {

// Temp file filled with a deterministic byte pattern; returns {fd, path}.
std::pair<int, std::string> MakePatternFile(size_t n) {
  std::string path = "/tmp/dfkv_uring_test_XXXXXX";
  int fd = ::mkstemp(&path[0]);
  EXPECT_GE(fd, 0);
  std::vector<char> data(n);
  for (size_t i = 0; i < n; ++i) data[i] = static_cast<char>('a' + (i % 26));
  EXPECT_EQ(::pwrite(fd, data.data(), n, 0), static_cast<ssize_t>(n));
  return {fd, path};
}

bool PatternOk(const char* buf, size_t off, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (buf[i] != static_cast<char>('a' + ((off + i) % 26))) return false;
  return true;
}

bool RingAvailable() {
  UringReader probe(4);
  return probe.ok();  // io_uring may be unavailable/banned in some sandboxes
}

}  // namespace

TEST(UringReader, BatchReadFillsEveryDescAtItsOwnOffset) {
  if (!RingAvailable()) GTEST_SKIP() << "io_uring unavailable";
  auto [fd, path] = MakePatternFile(1 << 20);
  constexpr int kN = 7;
  std::vector<std::vector<char>> bufs(kN, std::vector<char>(4096));
  std::vector<UringReader::ReadDesc> descs(kN);
  for (int i = 0; i < kN; ++i) {
    descs[i].fd = fd;
    descs[i].buf = bufs[i].data();
    descs[i].len = 4096;
    descs[i].off = static_cast<uint64_t>(i) * 8192;
  }
  UringReader ring(4);  // depth < cnt: exercises the sub-batch loop too
  ASSERT_TRUE(ring.ok());
  ASSERT_TRUE(ring.BatchRead(descs.data(), kN));
  EXPECT_FALSE(ring.poisoned());
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(descs[i].result, 4096) << "desc " << i;
    EXPECT_TRUE(PatternOk(bufs[i].data(), descs[i].off, 4096)) << "desc " << i;
  }
  ::close(fd);
  ::unlink(path.c_str());
}

TEST(UringReader, PerReadErrorIsRecordedNotInfraFailure) {
  if (!RingAvailable()) GTEST_SKIP() << "io_uring unavailable";
  auto [fd, path] = MakePatternFile(8192);
  int bad = ::open(path.c_str(), O_WRONLY);  // read on O_WRONLY fd => -EBADF/-EACCES
  ASSERT_GE(bad, 0);
  std::vector<char> b0(4096), b1(4096);
  UringReader::ReadDesc descs[2];
  descs[0] = {fd, b0.data(), 4096, 0, 0};
  descs[1] = {bad, b1.data(), 4096, 0, 0};
  UringReader ring(4);
  ASSERT_TRUE(ring.ok());
  // A per-read failure is the caller's to inspect; the batch itself succeeds
  // and the ring stays clean (no poison, nothing left in flight).
  ASSERT_TRUE(ring.BatchRead(descs, 2));
  EXPECT_FALSE(ring.poisoned());
  EXPECT_EQ(descs[0].result, 4096);
  EXPECT_LT(descs[1].result, 0);
  EXPECT_TRUE(ring.Drain());  // no-op: everything reaped
  ::close(fd);
  ::close(bad);
  ::unlink(path.c_str());
}

TEST(UringReader, EofYieldsBytesUpToEof) {
  if (!RingAvailable()) GTEST_SKIP() << "io_uring unavailable";
  auto [fd, path] = MakePatternFile(1000);  // < one 4096 read
  std::vector<char> buf(4096);
  UringReader::ReadDesc d{fd, buf.data(), 4096, 0, 0};
  UringReader ring(4);
  ASSERT_TRUE(ring.ok());
  ASSERT_TRUE(ring.BatchRead(&d, 1));
  EXPECT_EQ(d.result, 1000);
  EXPECT_TRUE(PatternOk(buf.data(), 0, 1000));
  ::close(fd);
  ::unlink(path.c_str());
}

TEST(UringReader, DrainOnIdleRingIsNoop) {
  if (!RingAvailable()) GTEST_SKIP() << "io_uring unavailable";
  UringReader ring(4);
  ASSERT_TRUE(ring.ok());
  EXPECT_FALSE(ring.poisoned());
  EXPECT_TRUE(ring.Drain());
  EXPECT_TRUE(ring.Drain());  // idempotent
}

TEST(UringReader, RepeatedBatchesReuseTheRingCleanly) {
  if (!RingAvailable()) GTEST_SKIP() << "io_uring unavailable";
  auto [fd, path] = MakePatternFile(1 << 16);
  std::vector<char> buf(4096);
  UringReader ring(2);
  ASSERT_TRUE(ring.ok());
  for (int round = 0; round < 5; ++round) {
    UringReader::ReadDesc d{fd, buf.data(), 4096,
                            static_cast<uint64_t>(round) * 4096, 0};
    ASSERT_TRUE(ring.BatchRead(&d, 1)) << "round " << round;
    EXPECT_EQ(d.result, 4096);
    EXPECT_TRUE(PatternOk(buf.data(), d.off, 4096)) << "round " << round;
  }
  EXPECT_FALSE(ring.poisoned());
  ::close(fd);
  ::unlink(path.c_str());
}

#endif  // DFKV_WITH_URING
