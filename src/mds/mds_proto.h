#ifndef DFKV_MDS_PROTO_H_
#define DFKV_MDS_PROTO_H_

#include <cstdint>
#include <string>
#include <vector>

#include "common/membership.h"  // MemberInfo + EncodeMembers/DecodeMembers
#include "utils/net_util.h"

namespace dfkv {

// kRegister / kHeartbeat request payload: a group name + ONE full MemberInfo.
// Full info (not just id) so an MDS that took over after failover can re-Put the
// member key with no prior state. Layout: glen u32 | group | EncodeMembers([m],0)
inline std::string EncodeMemberReq(const std::string& group, const MemberInfo& m) {
  std::string out;
  char n[4];
  net::PutU32(n, static_cast<uint32_t>(group.size()));
  out.append(n, 4);
  out += group;
  out += EncodeMembers({m}, 0);
  return out;
}

inline bool DecodeMemberReq(const char* p, size_t n, std::string* group,
                            MemberInfo* m) {
  if (n < 4) return false;
  uint32_t glen = net::GetU32(p);
  if (static_cast<size_t>(4) + glen > n) return false;
  group->assign(p + 4, glen);
  std::vector<MemberInfo> ms;
  uint64_t epoch = 0;
  if (!DecodeMembers(p + 4 + glen, n - 4 - glen, &ms, &epoch) || ms.size() != 1)
    return false;
  *m = ms[0];
  return true;
}

}  // namespace dfkv

#endif  // DFKV_MDS_PROTO_H_
