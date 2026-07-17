// NodeDedup: same-host GET rendezvous over POSIX shm. Hermetic (no RDMA/etcd);
// cross-process behavior via fork. Every test uses its own uniquely-named
// segment and unlinks it, so runs never interfere.
#include "client/node_dedup.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using dfkv::BlockKey;
using dfkv::NodeDedup;
using namespace std::chrono_literals;

namespace {

struct ShmGuard {
  std::string name;
  explicit ShmGuard(const std::string& tag)
      : name("/dfkv-dedup-test-" + std::to_string(::getpid()) + "-" + tag) {
    ::shm_unlink(name.c_str());
  }
  ~ShmGuard() { ::shm_unlink(name.c_str()); }
};

NodeDedup::Options Opts(const std::string& name, uint64_t arena = 4 << 20) {
  NodeDedup::Options o;
  o.name = name;
  o.arena_bytes = arena;
  o.slots = 1024;
  o.wait_ms = 300;
  o.takeover_ms = 200;
  o.ttl_ms = 1000;
  return o;
}

BlockKey K(uint64_t id) { return BlockKey{id, 0, 1}; }

std::string Val(uint64_t id, size_t n) {
  std::string v(n, '\0');
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<char>((id * 131 + i) & 0xFF);
  return v;
}

}  // namespace

TEST(NodeDedup, FetchPublishThenPeersHit) {
  ShmGuard g("basic");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));  // second attach (peer)
  ASSERT_TRUE(a && b);
  const std::string v = Val(1, 8192);
  std::string dst(v.size(), '\0');

  ASSERT_EQ(a->Claim(K(1), v.size(), &dst[0]), NodeDedup::Role::kFetch);
  a->Publish(K(1), NodeDedup::Kind::kData, v.data(), v.size());
  ASSERT_EQ(b->Claim(K(1), v.size(), &dst[0]), NodeDedup::Role::kHit);
  EXPECT_EQ(dst, v);
  EXPECT_EQ(b->hits(), 1u);
}

TEST(NodeDedup, WaiterCopiesAfterPublish) {
  ShmGuard g("wait");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v = Val(2, 65536);
  std::string dst_a(v.size(), '\0'), dst_b(v.size(), '\0');

  ASSERT_EQ(a->Claim(K(2), v.size(), &dst_a[0]), NodeDedup::Role::kFetch);
  ASSERT_EQ(b->Claim(K(2), v.size(), &dst_b[0]), NodeDedup::Role::kWait);
  std::thread pub([&] {
    std::this_thread::sleep_for(50ms);
    a->Publish(K(2), NodeDedup::Kind::kData, v.data(), v.size());
  });
  EXPECT_TRUE(b->WaitCopy(K(2), v.size(), &dst_b[0]));
  pub.join();
  EXPECT_EQ(dst_b, v);
  EXPECT_EQ(b->wait_hits(), 1u);
}

TEST(NodeDedup, AbortLetsWaiterFallBack) {
  ShmGuard g("abort");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  std::string dst(4096, '\0');
  ASSERT_EQ(a->Claim(K(3), dst.size(), &dst[0]), NodeDedup::Role::kFetch);
  ASSERT_EQ(b->Claim(K(3), dst.size(), &dst[0]), NodeDedup::Role::kWait);
  a->Abort(K(3), NodeDedup::Kind::kData);
  // The waiter's bounded poll must return false (fall back to a direct fetch),
  // never hang.
  EXPECT_FALSE(b->WaitCopy(K(3), dst.size(), &dst[0]));
  EXPECT_EQ(b->wait_timeouts(), 1u);
}

TEST(NodeDedup, DeadFetcherIsTakenOver) {
  ShmGuard g("takeover");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  std::string dst(4096, '\0');
  ASSERT_EQ(a->Claim(K(4), dst.size(), &dst[0]), NodeDedup::Role::kFetch);
  // a never publishes (simulated crash). After takeover_ms a peer claims the
  // fetch instead of waiting forever.
  std::this_thread::sleep_for(250ms);  // > takeover_ms(200)
  EXPECT_EQ(b->Claim(K(4), dst.size(), &dst[0]), NodeDedup::Role::kFetch);
}

TEST(NodeDedup, TtlExpiryRecyclesEntry) {
  ShmGuard g("ttl");
  auto o = Opts(g.name);
  o.ttl_ms = 100;
  auto a = NodeDedup::Open(o);
  ASSERT_TRUE(a);
  const std::string v = Val(5, 4096);
  std::string dst(v.size(), '\0');
  ASSERT_EQ(a->Claim(K(5), v.size(), &dst[0]), NodeDedup::Role::kFetch);
  a->Publish(K(5), NodeDedup::Kind::kData, v.data(), v.size());
  ASSERT_EQ(a->Claim(K(5), v.size(), &dst[0]), NodeDedup::Role::kHit);
  std::this_thread::sleep_for(150ms);  // > ttl
  // Expired: a new arrival re-fetches instead of serving stale bytes.
  EXPECT_EQ(a->Claim(K(5), v.size(), &dst[0]), NodeDedup::Role::kFetch);
}

TEST(NodeDedup, LappedArenaNeverServesOverwrittenBytes) {
  ShmGuard g("lap");
  auto a = NodeDedup::Open(Opts(g.name, /*arena=*/1 << 20));  // 1 MiB ring
  ASSERT_TRUE(a);
  const std::string v = Val(6, 128 * 1024);
  std::string dst(v.size(), '\0');
  ASSERT_EQ(a->Claim(K(6), v.size(), &dst[0]), NodeDedup::Role::kFetch);
  a->Publish(K(6), NodeDedup::Kind::kData, v.data(), v.size());
  // Lap the ring: publish > arena_bytes of other payloads.
  for (uint64_t i = 100; i < 100 + 12; ++i) {
    const std::string w = Val(i, 128 * 1024);
    std::string tmp(w.size(), '\0');
    if (a->Claim(K(i), w.size(), &tmp[0]) == NodeDedup::Role::kFetch)
      a->Publish(K(i), NodeDedup::Kind::kData, w.data(), w.size());
  }
  // K(6)'s payload region has been overwritten: the entry must NOT hit with
  // stale bytes — any non-kHit outcome (re-fetch) is correct; a kHit MUST
  // still carry the exact original value (possible if the slot was recycled
  // and republished, which this workload doesn't do).
  std::string out(v.size(), '\1');
  if (a->Claim(K(6), v.size(), &out[0]) == NodeDedup::Role::kHit)
    EXPECT_EQ(out, v);
}

TEST(NodeDedup, CrossProcessRendezvous) {
  ShmGuard g("fork");
  const std::string v = Val(7, 32768);
  auto parent = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(parent);
  std::string dst(v.size(), '\0');
  ASSERT_EQ(parent->Claim(K(7), v.size(), &dst[0]), NodeDedup::Role::kFetch);
  parent->Publish(K(7), NodeDedup::Kind::kData, v.data(), v.size());

  pid_t pid = ::fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {  // child: attach and hit
    auto child = NodeDedup::Open(Opts(g.name));
    if (!child) ::_exit(2);
    std::string cdst(v.size(), '\0');
    if (child->Claim(K(7), v.size(), &cdst[0]) != NodeDedup::Role::kHit) ::_exit(3);
    if (cdst != v) ::_exit(4);
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0) << "child exit code";
}

TEST(NodeDedup, DisabledByDefaultFromEnv) {
  ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
  EXPECT_EQ(NodeDedup::FromEnv(0x51), nullptr);
}

TEST(NodeDedup, AutoAcceptsFittingPayloadStrictDoesNot) {
  ShmGuard g("auto");
  auto a = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a);
  const std::string v = Val(8, 5000);  // "unfull" payload
  std::string dst(8192, '\0');
  ASSERT_EQ(a->Claim(K(8), v.size(), &dst[0]), NodeDedup::Role::kFetch);
  a->Publish(K(8), NodeDedup::Kind::kData, v.data(), v.size());
  // Auto with a larger cap hits and reports the true length.
  size_t got = 0;
  ASSERT_EQ(a->ClaimAuto(K(8), 8192, &dst[0], &got), NodeDedup::Role::kHit);
  EXPECT_EQ(got, v.size());
  EXPECT_EQ(std::string(dst.data(), got), v);
  // Auto with a too-small cap must NOT hit (falls back to fetch/reserve path).
  std::string small(1024, '\0');
  EXPECT_NE(a->ClaimAuto(K(8), 1024, &small[0], &got), NodeDedup::Role::kHit);
  // Strict with a different n must NOT hit either.
  EXPECT_NE(a->Claim(K(8), 8192, &dst[0]), NodeDedup::Role::kHit);
}

TEST(NodeDedup, ExistRendezvousBothAnswers) {
  ShmGuard g("exist");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  bool val = false;
  ASSERT_EQ(a->ClaimExist(K(9), &val), NodeDedup::Role::kFetch);
  const char yes = 1;
  a->Publish(K(9), NodeDedup::Kind::kExist, &yes, 1);
  ASSERT_EQ(b->ClaimExist(K(9), &val), NodeDedup::Role::kHit);
  EXPECT_TRUE(val);
  // Negative answers are published too (valid result, unlike a failed GET).
  ASSERT_EQ(a->ClaimExist(K(10), &val), NodeDedup::Role::kFetch);
  const char no = 0;
  a->Publish(K(10), NodeDedup::Kind::kExist, &no, 1);
  ASSERT_EQ(b->ClaimExist(K(10), &val), NodeDedup::Role::kHit);
  EXPECT_FALSE(val);
  // The exist namespace never collides with the data namespace of the same key.
  std::string dst(4096, '\0');
  EXPECT_EQ(a->Claim(K(9), dst.size(), &dst[0]), NodeDedup::Role::kFetch);
}

TEST(NodeDedup, EnvSegmentNameCarriesLayoutVersion) {
  // A layout bump must land in a FRESH segment name: v1.23.0 bumped the header
  // magic behind the same name, the mismatch check refused the stale v1.22
  // segment, and the feature silently disabled itself fleet-wide on upgrade.
  const std::string nm = NodeDedup::EnvSegmentName(0xABCD);
  EXPECT_NE(nm.find("/dfkv-dedup-v2-"), std::string::npos) << nm;
  EXPECT_NE(nm.find("000000000000abcd"), std::string::npos) << nm;
}

TEST(NodeDedup, SegmentIsEagerlyBackedNotSparse) {
  // tmpfs allocates lazily on write: a full /dev/shm would SIGBUS a publisher
  // mid-copy instead of degrading. Open() posix_fallocates the whole segment
  // at creation, converting that into a clean attach-time ENOSPC (nullptr).
  // Verify the backing is real: allocated blocks cover the full length.
  ShmGuard g("backed");
  auto a = NodeDedup::Open(Opts(g.name, /*arena=*/4 << 20));
  ASSERT_TRUE(a);
  int fd = ::shm_open(g.name.c_str(), O_RDONLY, 0600);
  ASSERT_GE(fd, 0);
  struct stat st{};
  ASSERT_EQ(::fstat(fd, &st), 0);
  ::close(fd);
  EXPECT_GE(static_cast<uint64_t>(st.st_blocks) * 512, static_cast<uint64_t>(st.st_size))
      << "segment is sparse: a full tmpfs would SIGBUS at publish time";
}

// ---- cooperative partition (Options::coop) ----

NodeDedup::Options CoopOpts(const std::string& name, int rank, int world) {
  NodeDedup::Options o = Opts(name);
  o.coop = true;
  o.rank = rank;
  o.world = world;
  return o;
}

// Ownership is a pure function of the key: every rank computes the same
// owner, the partition covers all keys, and no key has two owners.
TEST(NodeDedupCoop, OwnershipPartitionsKeysDeterministically) {
  ShmGuard g("coop-own");
  auto r0 = NodeDedup::Open(CoopOpts(g.name, 0, 4));
  auto r1 = NodeDedup::Open(CoopOpts(g.name, 1, 4));
  auto r2 = NodeDedup::Open(CoopOpts(g.name, 2, 4));
  auto r3 = NodeDedup::Open(CoopOpts(g.name, 3, 4));
  ASSERT_TRUE(r0 && r1 && r2 && r3);
  NodeDedup* ranks[4] = {r0.get(), r1.get(), r2.get(), r3.get()};
  int owned[4] = {0, 0, 0, 0};
  for (uint64_t id = 1; id <= 400; ++id) {
    int owners = 0;
    for (int r = 0; r < 4; ++r)
      if (ranks[r]->Owns(K(id))) { ++owners; ++owned[r]; }
    EXPECT_EQ(owners, 1) << "key " << id;
  }
  for (int r = 0; r < 4; ++r)  // roughly balanced (no rank starved)
    EXPECT_GT(owned[r], 40) << "rank " << r;
}

// The passive side never reserves: arriving FIRST it still waits, the owner
// arriving later claims the fetch, publishes, and the waiter copies. This is
// the exact inversion of the classic race (first arrival claims everything)
// that serialized real inference batches.
TEST(NodeDedupCoop, NonOwnerNeverClaimsEvenWhenFirst) {
  ShmGuard g("coop-passive");
  auto a = NodeDedup::Open(CoopOpts(g.name, 0, 2));
  auto b = NodeDedup::Open(CoopOpts(g.name, 1, 2));
  ASSERT_TRUE(a && b);
  // Find a key owned by rank 1.
  uint64_t id = 1;
  while (!b->Owns(K(id))) ++id;
  const std::string v = Val(id, 4096);
  std::string dst(v.size(), '\0');

  // Non-owner (rank 0) arrives FIRST: must wait, not claim.
  EXPECT_EQ(a->ClaimPassive(K(id), v.size(), dst.data()), NodeDedup::Role::kWait);
  // Owner arrives: claims the fetch.
  std::string bdst(v.size(), '\0');
  ASSERT_EQ(b->Claim(K(id), v.size(), bdst.data()), NodeDedup::Role::kFetch);
  b->Publish(K(id), NodeDedup::Kind::kData, v.data(), v.size());
  // Waiter's wait path lands the payload.
  EXPECT_TRUE(a->WaitCopy(K(id), v.size(), dst.data()));
  EXPECT_EQ(dst, v);
  // A later passive claim is a straight hit.
  std::string dst2(v.size(), '\0');
  EXPECT_EQ(a->ClaimPassive(K(id), v.size(), dst2.data()), NodeDedup::Role::kHit);
  EXPECT_EQ(dst2, v);
}

// A crashed owner cannot park its partition: past takeover_ms the passive
// side takes the fetch over, same as the classic dead-fetcher path.
TEST(NodeDedupCoop, PassiveTakesOverDeadOwner) {
  ShmGuard g("coop-takeover");
  auto a = NodeDedup::Open(CoopOpts(g.name, 0, 2));
  auto b = NodeDedup::Open(CoopOpts(g.name, 1, 2));
  ASSERT_TRUE(a && b);
  uint64_t id = 1;
  while (!b->Owns(K(id))) ++id;
  const std::string v = Val(id, 1024);
  std::string dst(v.size(), '\0');
  // Owner claims then "dies" (never publishes).
  ASSERT_EQ(b->Claim(K(id), v.size(), dst.data()), NodeDedup::Role::kFetch);
  std::this_thread::sleep_for(250ms);  // > takeover_ms (200)
  // Passive side takes over the stale FETCHING slot.
  EXPECT_EQ(a->ClaimPassive(K(id), v.size(), dst.data()), NodeDedup::Role::kFetch);
}

// FromEnv: "coop" without usable rank/world degrades to the classic mode
// (dedup on, coop off) instead of silently owning nothing.
TEST(NodeDedupCoop, FromEnvDegradesWithoutRank) {
  ::setenv("DFKV_CLIENT_NODE_DEDUP", "coop", 1);
  auto d = NodeDedup::FromEnv(0xC0FFEE, /*rank=*/-1, /*world=*/0);
  ASSERT_TRUE(d);
  EXPECT_FALSE(d->coop());
  auto d2 = NodeDedup::FromEnv(0xC0FFEE, /*rank=*/3, /*world=*/8);
  ASSERT_TRUE(d2);
  EXPECT_TRUE(d2->coop());
  ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
  ::shm_unlink(NodeDedup::EnvSegmentName(0xC0FFEE).c_str());
}
