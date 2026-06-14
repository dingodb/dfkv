#include "dfkv_c_api.h"

#include <string>
#include <utility>
#include <vector>

#include "kv_client.h"
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
  auto mem = ParseMembers(members);
  if (mem.empty()) return nullptr;
  ValueHeader self = ValueHeader::Make(
      model_hash, page_size, dtype_tag, static_cast<uint16_t>(flags),
      static_cast<uint16_t>(tp_size), static_cast<uint16_t>(tp_rank),
      static_cast<uint16_t>(layer_num), static_cast<uint16_t>(head_num),
      static_cast<uint16_t>(head_dim));
  return new KVClient(std::move(mem), self);
}

int dfkv_put(dfkv_client_t c, const char* key, const void* ptr, uint64_t n) {
  if (!c) return -1;
  return static_cast<KVClient*>(c)->Put(key, ptr, static_cast<size_t>(n)) ? 0 : -1;
}

int dfkv_get(dfkv_client_t c, const char* key, void* ptr, uint64_t n) {
  if (!c) return 0;
  return static_cast<KVClient*>(c)->Get(key, ptr, static_cast<size_t>(n)) ? 1 : 0;
}

int dfkv_exist(dfkv_client_t c, const char* key) {
  if (!c) return 0;
  return static_cast<KVClient*>(c)->Exist(key) ? 1 : 0;
}

int dfkv_batch_put(dfkv_client_t c, const char** keys, const void** ptrs,
                   const uint64_t* sizes, int n, int* out_ok) {
  if (!c || n < 0) return -1;
  std::vector<dfkv::KvPutItem> items(n);
  for (int i = 0; i < n; ++i)
    items[i] = {keys[i], ptrs[i], static_cast<size_t>(sizes[i])};
  auto r = static_cast<KVClient*>(c)->BatchPut(items);
  for (int i = 0; i < n; ++i) out_ok[i] = r[i] ? 1 : 0;
  return 0;
}

int dfkv_batch_get(dfkv_client_t c, const char** keys, void** ptrs,
                   const uint64_t* sizes, int n, int* out_hit) {
  if (!c || n < 0) return -1;
  std::vector<dfkv::KvGetItem> items(n);
  for (int i = 0; i < n; ++i)
    items[i] = {keys[i], ptrs[i], static_cast<size_t>(sizes[i])};
  auto r = static_cast<KVClient*>(c)->BatchGet(items);
  for (int i = 0; i < n; ++i) out_hit[i] = r[i] ? 1 : 0;
  return 0;
}

int dfkv_batch_exist(dfkv_client_t c, const char** keys, int n, int* out_exist) {
  if (!c || n < 0) return -1;
  std::vector<std::string> ks(n);
  for (int i = 0; i < n; ++i) ks[i] = keys[i];
  auto r = static_cast<KVClient*>(c)->BatchExist(ks);
  for (int i = 0; i < n; ++i) out_exist[i] = r[i] ? 1 : 0;
  return 0;
}

int dfkv_set_members(dfkv_client_t c, const char* members) {
  if (!c) return -1;
  static_cast<KVClient*>(c)->SetMembers(ParseMembers(members));
  return 0;
}

void dfkv_close(dfkv_client_t c) { delete static_cast<KVClient*>(c); }

}  // extern "C"
