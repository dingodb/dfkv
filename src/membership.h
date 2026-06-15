#ifndef DFKV_MEMBERSHIP_H_
#define DFKV_MEMBERSHIP_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "net_util.h"  // net::PutU32/PutU64/GetU32/GetU64 (host-endian codec)

namespace dfkv {

struct MemberInfo {
  std::string id;
  std::string ip;
  uint32_t port = 0;
  uint32_t weight = 1;
  bool operator==(const MemberInfo& o) const {
    return id == o.id && ip == o.ip && port == o.port && weight == o.weight;
  }
};

// Wire format for a membership view. Register payload = 1 member; ListMembers
// response = N members + the epoch (etcd revision) clients compare to skip
// rebuilding an unchanged ring. Layout (host-endian via net::):
//   epoch u64 | count u32 | repeat{ idlen u32, id, iplen u32, ip, port u32, weight u32 }
// (host-endian via net::; correct between same-endianness peers — see net_util.h)
inline std::string EncodeMembers(const std::vector<MemberInfo>& ms,
                                 uint64_t epoch) {
  std::string out;
  char hdr[12];
  net::PutU64(hdr, epoch);
  net::PutU32(hdr + 8, static_cast<uint32_t>(ms.size()));
  out.append(hdr, 12);
  char num[4];
  for (const auto& m : ms) {
    net::PutU32(num, static_cast<uint32_t>(m.id.size())); out.append(num, 4);
    out += m.id;
    net::PutU32(num, static_cast<uint32_t>(m.ip.size())); out.append(num, 4);
    out += m.ip;
    net::PutU32(num, m.port);   out.append(num, 4);
    net::PutU32(num, m.weight); out.append(num, 4);
  }
  return out;
}

// Returns false on truncated / malformed input (caller treats as a failed RPC).
inline bool DecodeMembers(const char* p, size_t n,
                          std::vector<MemberInfo>* out, uint64_t* epoch) {
  if (n < 12) return false;
  *epoch = net::GetU64(p);
  uint32_t count = net::GetU32(p + 8);
  size_t off = 12;
  out->clear();
  for (uint32_t i = 0; i < count; ++i) {
    MemberInfo m;
    if (off + 4 > n) return false;
    uint32_t idlen = net::GetU32(p + off); off += 4;
    if (off + idlen > n) return false;
    m.id.assign(p + off, idlen); off += idlen;
    if (off + 4 > n) return false;
    uint32_t iplen = net::GetU32(p + off); off += 4;
    if (off + iplen > n) return false;
    m.ip.assign(p + off, iplen); off += iplen;
    if (off + 8 > n) return false;
    m.port = net::GetU32(p + off);   off += 4;
    m.weight = net::GetU32(p + off);  off += 4;
    out->push_back(std::move(m));
  }
  return true;
}

// Content epoch for a membership view: an order-independent 64-bit hash of the
// member SET. Clients compare it to decide whether to rebuild their ring. The
// MDS uses this instead of etcd's global revision, which bumps on ANY cluster
// write (other groups too) and would trigger needless full-ring rebuilds. Sorting
// by id makes the hash canonical regardless of etcd's return order; hashing every
// field means a content-only change (e.g. an ip/weight edit) still bumps the epoch.
inline uint64_t MembersEpoch(std::vector<MemberInfo> ms) {
  std::sort(ms.begin(), ms.end(), [](const MemberInfo& a, const MemberInfo& b) {
    return a.id < b.id;
  });
  uint64_t h = 1469598103934665603ull;  // FNV-1a 64-bit offset basis
  auto mix = [&h](const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  };
  for (const auto& m : ms) {
    uint32_t idn = static_cast<uint32_t>(m.id.size());
    uint32_t ipn = static_cast<uint32_t>(m.ip.size());
    mix(&idn, 4); mix(m.id.data(), m.id.size());  // length-prefixed: no field-boundary ambiguity
    mix(&ipn, 4); mix(m.ip.data(), m.ip.size());
    mix(&m.port, sizeof(m.port));
    mix(&m.weight, sizeof(m.weight));
  }
  return h;
}

}  // namespace dfkv

#endif  // DFKV_MEMBERSHIP_H_
