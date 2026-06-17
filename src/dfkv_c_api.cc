#include "dfkv_c_api.h"

#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "kv_client.h"
#include "log.h"
#include "value_header.h"

using dfkv::KVClient;
using dfkv::ValueHeader;

namespace {
std::vector<std::pair<std::string, std::string>> ParseMembers(const char* s) {
  std::vector<std::pair<std::string, std::string>> out;
  std::string in(s ? s : "");
  size_t i = 0;
  while (i < in.size()) {
    size_t comma = in.find(',', i);
    std::string tok = in.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
    size_t eq = tok.find('=');
    if (eq != std::string::npos) out.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
    if (comma == std::string::npos) break;
    i = comma + 1;
  }
  return out;
}
}  // namespace

extern "C" {

dfkv_client_t dfkv_open(const char* members, uint64_t model_hash,
                        uint32_t page_size, uint32_t dtype_tag, uint32_t flags,
                        uint32_t tp_size, uint32_t tp_rank, uint32_t layer_num,
                        uint32_t head_num, uint32_t head_dim) {
  try {
    auto mem = ParseMembers(members);
    // Empty member list is valid: discovery-only client; ring gets populated by
    // dfkv_start_mds_discovery. KVClient handles an empty member vector.
    ValueHeader self = ValueHeader::Make(
        model_hash, page_size, dtype_tag, static_cast<uint16_t>(flags),
        static_cast<uint16_t>(tp_size), static_cast<uint16_t>(tp_rank),
        static_cast<uint16_t>(layer_num), static_cast<uint16_t>(head_num),
        static_cast<uint16_t>(head_dim));
    return new KVClient(std::move(mem), self);
  } catch (const std::exception& e) {
    DFKV_LOG_ERROR(std::string("dfkv_open failed: ") + e.what());
    return nullptr;
  } catch (...) {
    DFKV_LOG_ERROR("dfkv_open failed: unknown exception");
    return nullptr;
  }
}

int dfkv_put(dfkv_client_t c, const char* key, const void* ptr, uint64_t n) {
  if (!c || !key) return -1;  // key is wrapped in std::string -> must be non-null
  return static_cast<KVClient*>(c)->Put(key, ptr, static_cast<size_t>(n)) ? 0 : -1;
}

int dfkv_get(dfkv_client_t c, const char* key, void* ptr, uint64_t n) {
  if (!c || !key) return 0;  // 0 == miss; null key can't construct std::string
  return static_cast<KVClient*>(c)->Get(key, ptr, static_cast<size_t>(n)) ? 1 : 0;
}

int dfkv_get_auto(dfkv_client_t c, const char* key, void* ptr, uint64_t cap,
                  uint64_t* out_len) {
  if (out_len) *out_len = 0;
  if (!c || !key) return 0;  // 0 == miss
  size_t got = 0;
  bool hit = static_cast<KVClient*>(c)->GetAuto(key, ptr, static_cast<size_t>(cap), &got);
  if (out_len) *out_len = static_cast<uint64_t>(got);
  return hit ? 1 : 0;
}

int dfkv_exist(dfkv_client_t c, const char* key) {
  if (!c || !key) return 0;
  return static_cast<KVClient*>(c)->Exist(key) ? 1 : 0;
}

int dfkv_register_memory(dfkv_client_t c, const void* base, uint64_t size) {
  if (!c) return -1;
  static_cast<KVClient*>(c)->RegisterMemory(const_cast<void*>(base),
                                            static_cast<size_t>(size));
  return 0;
}

// The batch entrypoints take caller-owned C arrays. We validate the array
// pointers up front (null with n>0 => bad call) and tolerate a null element:
// a null keys[i] is substituted with an empty item so BatchPut/Get stays index-
// aligned, and that slot is reported as failed (out[i]=0) instead of feeding a
// null into std::string() (undefined behavior across the FFI boundary).
int dfkv_batch_put(dfkv_client_t c, const char** keys, const void** ptrs,
                   const uint64_t* sizes, int n, int* out_ok) {
  if (!c || n < 0) return -1;
  if (n > 0 && (!keys || !ptrs || !sizes || !out_ok)) return -1;
  std::vector<dfkv::KvPutItem> items(n);
  for (int i = 0; i < n; ++i) {
    const char* k = keys[i];
    items[i] = {k ? std::string(k) : std::string(), k ? ptrs[i] : nullptr,
                k ? static_cast<size_t>(sizes[i]) : 0};
  }
  auto r = static_cast<KVClient*>(c)->BatchPut(items);
  for (int i = 0; i < n; ++i) out_ok[i] = (keys[i] && r[i]) ? 1 : 0;
  return 0;
}

int dfkv_batch_get(dfkv_client_t c, const char** keys, void** ptrs,
                   const uint64_t* sizes, int n, int* out_hit) {
  if (!c || n < 0) return -1;
  if (n > 0 && (!keys || !ptrs || !sizes || !out_hit)) return -1;
  std::vector<dfkv::KvGetItem> items(n);
  for (int i = 0; i < n; ++i) {
    const char* k = keys[i];
    items[i] = {k ? std::string(k) : std::string(), k ? ptrs[i] : nullptr,
                k ? static_cast<size_t>(sizes[i]) : 0};
  }
  auto r = static_cast<KVClient*>(c)->BatchGet(items);
  for (int i = 0; i < n; ++i) out_hit[i] = (keys[i] && r[i]) ? 1 : 0;
  return 0;
}

int dfkv_batch_get_auto(dfkv_client_t c, const char** keys, void** ptrs,
                        const uint64_t* caps, int n, int* out_hit,
                        uint64_t* out_len) {
  if (!c || n < 0) return -1;
  if (n > 0 && (!keys || !ptrs || !caps || !out_hit || !out_len)) return -1;
  std::vector<dfkv::KvGetItem> items(n);
  for (int i = 0; i < n; ++i) {
    const char* k = keys[i];
    // items[i].n carries the buffer CAPACITY for the variable-size read.
    items[i] = {k ? std::string(k) : std::string(), k ? ptrs[i] : nullptr,
                k ? static_cast<size_t>(caps[i]) : 0};
  }
  std::vector<size_t> lens;
  auto r = static_cast<KVClient*>(c)->BatchGetAuto(items, &lens);
  for (int i = 0; i < n; ++i) {
    bool hit = keys[i] && r[i];
    out_hit[i] = hit ? 1 : 0;
    out_len[i] = hit ? static_cast<uint64_t>(lens[i]) : 0;
  }
  return 0;
}

int dfkv_batch_exist(dfkv_client_t c, const char** keys, int n, int* out_exist) {
  if (!c || n < 0) return -1;
  if (n > 0 && (!keys || !out_exist)) return -1;
  std::vector<std::string> ks(n);
  for (int i = 0; i < n; ++i) ks[i] = keys[i] ? std::string(keys[i]) : std::string();
  auto r = static_cast<KVClient*>(c)->BatchExist(ks);
  for (int i = 0; i < n; ++i) out_exist[i] = (keys[i] && r[i]) ? 1 : 0;
  return 0;
}

int dfkv_set_members(dfkv_client_t c, const char* members) {
  if (!c) return -1;
  static_cast<KVClient*>(c)->SetMembers(ParseMembers(members));
  return 0;
}

int dfkv_refresh_members(dfkv_client_t c, const char* seed) {
  if (!c || !seed) return -1;
  return static_cast<KVClient*>(c)->RefreshMembers(seed) ? 0 : 1;
}

int dfkv_start_mds_discovery(dfkv_client_t c, const char* mds_endpoints,
                             const char* group, int poll_ms) {
  if (!c || !mds_endpoints || !group) return -1;
  std::vector<std::string> eps;
  std::string s(mds_endpoints);
  for (size_t i = 0, j; i <= s.size(); i = j + 1) {
    j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    if (j > i) eps.push_back(s.substr(i, j - i));
  }
  static_cast<KVClient*>(c)->StartMdsDiscovery(std::move(eps), group,
                                               poll_ms > 0 ? poll_ms : 3000);
  return 0;
}

const char* dfkv_transport_mode(dfkv_client_t c) {
  if (!c) return "";
  return static_cast<KVClient*>(c)->TransportMode().c_str();
}

int dfkv_set_batch_concurrency(dfkv_client_t c, uint64_t n) {
  if (!c) return -1;
  static_cast<KVClient*>(c)->set_batch_concurrency(static_cast<size_t>(n));
  return 0;
}

uint64_t dfkv_stats_snapshot(dfkv_client_t c, char* buf, uint64_t cap) {
  if (!c) return 0;
  std::string text = static_cast<KVClient*>(c)->MetricsSnapshot();
  uint64_t full = static_cast<uint64_t>(text.size());
  if (buf && cap > 0) {
    uint64_t n = full < cap ? full : cap - 1;  // leave room for NUL
    std::memcpy(buf, text.data(), static_cast<size_t>(n));
    buf[n] = '\0';
  }
  return full;
}

void dfkv_close(dfkv_client_t c) { delete static_cast<KVClient*>(c); }

}  // extern "C"
