/* dfkv wire protocol — the single place that defines the request/response frame.
 * Both prefixes start with a 1-byte protocol version so a mixed-version deploy
 * fails fast (the server rejects an unknown version) instead of mis-parsing.
 * Centralizing encode/decode here keeps the byte offsets in one spot rather than
 * scattered across the TCP and RDMA transports + the server. */
#ifndef DFKV_WIRE_H_
#define DFKV_WIRE_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "common/status.h"
#include "common/kv_types.h"   // BlockKey
#include "utils/net_util.h"   // net::PutU32/PutU64/GetU32/GetU64 (host-endian codec)

namespace dfkv {

// Wire op codes (shared by TcpTransport, RdmaTransport and the server).
// kMembers is the legacy static-list discovery query, superseded by the MDS (kListMembers).
// kRegister/kHeartbeat/kListMembers are the MDS membership ops (M0+): the op byte
// reuses the existing request framing; variable content rides the payload/data blob.
enum class WireOp : uint8_t {
  kCache = 1, kRange = 2, kExist = 3, kStats = 4, kMembers = 5,
  kRegister = 6, kHeartbeat = 7, kListMembers = 8, kRemove = 9
};

constexpr uint8_t kProtoVersion = 1;
// Protocol v2 appends an 8-byte request seq (echoed in the response) at the TAIL
// of each prefix, so a v2 frame is a v1 frame + 8 bytes. The seq lets a client
// correlate a reply to its request explicitly instead of relying on the servers
// replying in strict arrival order (a cross-component FIFO assumption that any
// future out-of-order reply would silently break). Servers DUAL-ACCEPT: they
// decode either version and reply in the SAME version, so a v1 client and a v2
// client interoperate with the same server -- rolling upgrades need no flag day
// (upgrade servers first, then clients). Behavior is otherwise identical; the
// seq is validated but replies stay in arrival order until a later change.
constexpr uint8_t kProtoVersionV2 = 2;

// Hard ceiling on a single wire frame's variable payload. Decode rejects any
// frame whose declared length exceeds this, so a garbage/hostile 64-bit length
// (a version skew, corruption, or a hostile peer) can't drive a multi-exabyte
// std::vector/std::string allocation -> bad_alloc/OOM that kills the process.
// No real dfkv frame (one KV block value, or a stats/membership blob) comes
// anywhere near 16 GiB; callers that know a tighter bound pass it explicitly.
constexpr uint64_t kMaxFrameLen = 1ull << 34;  // 16 GiB

// Request prefix: ver(1) op(1) id(8) index(4) size(4) offset(8) length(8) payload_len(8)
constexpr size_t kReqPrefix = 1 + 1 + 8 + 4 + 4 + 8 + 8 + 8;  // = 42 (v1)
// Response prefix: ver(1) status(1) data_len(8)
constexpr size_t kRespPrefix = 1 + 1 + 8;  // = 10 (v1)
// v2 = v1 layout + an 8-byte trailing seq. A reader takes the v1 prefix first
// (the version byte is at [0] of both), and reads the extra 8 bytes only when
// [0] == kProtoVersionV2.
constexpr size_t kReqPrefixV2 = kReqPrefix + 8;    // = 50
constexpr size_t kRespPrefixV2 = kRespPrefix + 8;  // = 18

inline void EncodeReq(char* p, WireOp op, const BlockKey& k, uint64_t offset,
                      uint64_t length, uint64_t payload_len) {
  p[0] = static_cast<char>(kProtoVersion);
  p[1] = static_cast<char>(op);
  net::PutU64(p + 2, k.id);
  net::PutU32(p + 10, k.index);
  net::PutU32(p + 14, k.size);
  net::PutU64(p + 18, offset);
  net::PutU64(p + 26, length);
  net::PutU64(p + 34, payload_len);
}

// v2 request: the v1 layout plus an 8-byte seq at the tail (offset kReqPrefix).
// Writes kReqPrefixV2 bytes.
inline void EncodeReqV2(char* p, WireOp op, const BlockKey& k, uint64_t offset,
                        uint64_t length, uint64_t payload_len, uint64_t seq) {
  EncodeReq(p, op, k, offset, length, payload_len);
  p[0] = static_cast<char>(kProtoVersionV2);  // overwrite the version byte
  net::PutU64(p + kReqPrefix, seq);
}

struct ReqFields {
  uint8_t op;
  uint64_t id;
  uint32_t index;
  uint32_t size;
  uint64_t offset;
  uint64_t length;
  uint64_t payload_len;
  uint64_t seq = 0;  // v2 only (0 for a v1 frame)
};

// Returns the decoded frame version (kProtoVersion or kProtoVersionV2), or 0 on
// a version mismatch / oversized declared payload (> max_payload) — the caller
// drops the connection when 0. `if (!DecodeReq(...))` therefore still means "bad
// frame", and a caller that needs the version keeps the return value.
// For a v2 frame `p` MUST hold kReqPrefixV2 bytes (the seq is read from
// p+kReqPrefix); for a v1 frame only kReqPrefix bytes are touched.
inline uint8_t DecodeReq(const char* p, ReqFields* o,
                         uint64_t max_payload = kMaxFrameLen) {
  const uint8_t ver = static_cast<uint8_t>(p[0]);
  if (ver != kProtoVersion && ver != kProtoVersionV2) return 0;
  o->op = static_cast<uint8_t>(p[1]);
  o->id = net::GetU64(p + 2);
  o->index = net::GetU32(p + 10);
  o->size = net::GetU32(p + 14);
  o->offset = net::GetU64(p + 18);
  o->length = net::GetU64(p + 26);
  o->payload_len = net::GetU64(p + 34);
  o->seq = (ver == kProtoVersionV2) ? net::GetU64(p + kReqPrefix) : 0;
  if (o->payload_len > max_payload) return 0;  // reject oversized frame
  return ver;
}

inline void EncodeResp(char* p, Status st, uint64_t data_len) {
  p[0] = static_cast<char>(kProtoVersion);
  p[1] = static_cast<char>(st);
  net::PutU64(p + 2, data_len);
}

// v2 response: v1 layout plus an 8-byte echoed seq. Writes kRespPrefixV2 bytes.
inline void EncodeRespV2(char* p, Status st, uint64_t data_len, uint64_t seq) {
  EncodeResp(p, st, data_len);
  p[0] = static_cast<char>(kProtoVersionV2);
  net::PutU64(p + kRespPrefix, seq);
}

// Returns the response frame version (1 or 2), or 0 on version mismatch /
// oversize. For a v2 frame `p` must hold kRespPrefixV2 bytes. `seq` (nullable)
// receives the echoed seq (0 for a v1 frame).
inline uint8_t DecodeResp(const char* p, Status* st, uint64_t* data_len,
                          uint64_t max_data = kMaxFrameLen, uint64_t* seq = nullptr) {
  const uint8_t ver = static_cast<uint8_t>(p[0]);
  if (ver != kProtoVersion && ver != kProtoVersionV2) return 0;
  *st = static_cast<Status>(static_cast<uint8_t>(p[1]));
  *data_len = net::GetU64(p + 2);
  if (seq) *seq = (ver == kProtoVersionV2) ? net::GetU64(p + kRespPrefix) : 0;
  if (*data_len > max_data) return 0;  // reject oversized frame
  return ver;
}

// Client-selected wire version, from DFKV_WIRE_VERSION (default 1). Cached once.
// A v2 client sends v2 frames with a seq and validates the echoed seq; servers
// dual-accept, so raising this on the client is safe only AFTER every server it
// talks to is on a dual-accept build (roll servers first, then clients).
inline uint8_t ClientWireVersion() {
  static const uint8_t v = [] {
    const char* e = std::getenv("DFKV_WIRE_VERSION");
    return (e && e[0] == '2' && e[1] == '\0') ? kProtoVersionV2 : kProtoVersion;
  }();
  return v;
}

}  // namespace dfkv

#endif  // DFKV_WIRE_H_
