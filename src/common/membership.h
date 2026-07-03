#ifndef DFKV_MEMBERSHIP_H_
#define DFKV_MEMBERSHIP_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "utils/net_util.h"  // net::PutU32/PutU64/GetU32/GetU64 (host-endian codec)

namespace dfkv {

struct MemberInfo {
  std::string id;
  std::string ip;
  uint32_t port = 0;       // data-path port clients dial (the rdma-port in an RDMA deploy)
  uint32_t weight = 1;
  uint32_t tcp_port = 0;   // TCP wire/stat port that serves kStats; 0 = unknown / older peer.
  // Node self-description reported on register/heartbeat: "k=v,k=v,..." (version,
  // storage engine, capacity, RAM tier, RDMA dev ...). Empty = unknown / older peer.
  // Purely informational: surfaced by `dfkvctl ring` for fleet audit (version skew,
  // silently-skipped upgrades, engine mismatch) without per-node ssh.
  std::string info;
  // tcp_port and info ride OPTIONAL trailing extensions in Encode/DecodeMembers, so peers
  // that don't send them still interoperate. Both are intentionally EXCLUDED from
  // operator== and MembersEpoch: they are orthogonal metadata (not part of ring
  // placement), and their consumers (dfkvctl) re-fetch every run, so a change need not
  // trigger a client ring rebuild.
  bool operator==(const MemberInfo& o) const {
    return id == o.id && ip == o.ip && port == o.port && weight == o.weight;
  }
};

// A group or member id is embedded verbatim into the etcd key
//   /dfkv/v1/groups/<group>/members/<id>
// so a '/' (or an empty/overlong token) lets a registration escape its own
// key subtree -- e.g. group "a/members/ghost" injects a phantom member into
// group "a"'s RangePrefix, and any reachable client can point a real node id
// at an attacker IP. Restrict both to a conservative, path-safe alphabet.
inline bool IsValidGroupOrId(const std::string& s) {
  if (s.empty() || s.size() > 128) return false;
  for (unsigned char c : s) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

// Magic tags marking OPTIONAL extensions appended after the member list, in tag
// order (TCP1 then NFO1). Old encodings carry no trailing bytes and old decoders
// stop after the member list, so the tags are unambiguous when present and
// invisible to peers that predate them.
constexpr uint32_t kMemberExtTcpPort = 0x54435031u;  // "TCP1": one tcp_port u32 per member
constexpr uint32_t kMemberExtInfo = 0x4E464F31u;     // "NFO1": one length-prefixed info string per member

// Wire format for a membership view. Register payload = 1 member; ListMembers
// response = N members + the epoch (etcd revision) clients compare to skip
// rebuilding an unchanged ring. Layout (host-endian via net::):
//   epoch u64 | count u32 | repeat{ idlen u32, id, iplen u32, ip, port u32, weight u32 }
//   [ optional: kMemberExtTcpPort u32 | repeat count { tcp_port u32 } ]        <- old decoders ignore
//   [ optional: kMemberExtInfo u32 | repeat count { infolen u32, info bytes } ] <- old decoders ignore
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
  // Optional trailing extensions: a magic tag + per-member payload, same order.
  // Older decoders stop after the N members above and never read these bytes.
  net::PutU32(num, kMemberExtTcpPort); out.append(num, 4);
  for (const auto& m : ms) { net::PutU32(num, m.tcp_port); out.append(num, 4); }
  net::PutU32(num, kMemberExtInfo); out.append(num, 4);
  for (const auto& m : ms) {
    net::PutU32(num, static_cast<uint32_t>(m.info.size())); out.append(num, 4);
    out += m.info;
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
  // Optional trailing extensions (see EncodeMembers): read tagged blocks in
  // order; an unknown tag or a truncated block ends the (best-effort) ext scan
  // without failing the decode -- exts absent from older peers' encodings just
  // leave tcp_port==0 / info=="".
  while (off + 4 <= n) {
    const uint32_t tag = net::GetU32(p + off);
    if (tag == kMemberExtTcpPort) {
      off += 4;
      if (off + static_cast<size_t>(count) * 4 > n) break;
      for (uint32_t i = 0; i < count; ++i) { (*out)[i].tcp_port = net::GetU32(p + off); off += 4; }
    } else if (tag == kMemberExtInfo) {
      off += 4;
      bool ok = true;
      for (uint32_t i = 0; i < count; ++i) {
        if (off + 4 > n) { ok = false; break; }
        uint32_t ilen = net::GetU32(p + off); off += 4;
        if (off + ilen > n) { ok = false; break; }
        (*out)[i].info.assign(p + off, ilen); off += ilen;
      }
      if (!ok) break;
    } else {
      break;  // unknown/future tag: ignore the rest (forward compatible)
    }
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
