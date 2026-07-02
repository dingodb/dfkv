#include "mds/etcd_client.h"

#include <cstdlib>

#include "utils/base64.h"
#include "utils/json_min.h"

namespace dfkv {

namespace {
// etcd range over a prefix uses range_end = key with last byte incremented.
std::string PrefixRangeEnd(const std::string& prefix) {
  std::string end = prefix;
  for (size_t i = end.size(); i-- > 0;) {
    if (uint8_t(end[i]) < 0xFF) { end[i] = char(uint8_t(end[i]) + 1); end.resize(i + 1); return end; }
  }
  return std::string("\0", 1);  // all 0xFF => use "\0" for scan-to-end (empty would mean single-key get)
}
}  // namespace

std::optional<int64_t> EtcdClient::LeaseGrant(int ttl_seconds) {
  HttpResponse r;
  std::string body = "{\"TTL\":\"" + std::to_string(ttl_seconds) + "\",\"ID\":\"0\"}";
  if (!http_->Post("/v3/lease/grant", body, &r) || r.status != 200) return std::nullopt;
  JsonValue root;
  if (!JsonParser::Parse(r.body, &root)) return std::nullopt;
  std::string id = root.get_str("ID");
  if (id.empty()) return std::nullopt;
  return std::optional<int64_t>(std::strtoll(id.c_str(), nullptr, 10));
}

bool EtcdClient::LeaseKeepAlive(int64_t lease_id) {
  HttpResponse r;
  std::string body = "{\"ID\":\"" + std::to_string(lease_id) + "\"}";
  if (!http_->Post("/v3/lease/keepalive", body, &r) || r.status != 200) return false;
  JsonValue root;
  if (!JsonParser::Parse(r.body, &root)) return false;
  const JsonValue* res = root.find("result");
  std::string ttl = res ? res->get_str("TTL") : root.get_str("TTL");
  return !ttl.empty() && ttl != "0";
}

bool EtcdClient::Put(const std::string& key, const std::string& value, int64_t lease_id) {
  HttpResponse r;
  std::string body = "{\"key\":\"" + Base64Encode(key) + "\",\"value\":\"" +
                     Base64Encode(value) + "\",\"lease\":\"" + std::to_string(lease_id) + "\"}";
  return http_->Post("/v3/kv/put", body, &r) && r.status == 200;
}

std::optional<RangeResult> EtcdClient::RangePrefix(const std::string& prefix) {
  HttpResponse r;
  std::string end = PrefixRangeEnd(prefix);
  std::string body = "{\"key\":\"" + Base64Encode(prefix) + "\",\"range_end\":\"" +
                     Base64Encode(end) + "\"}";
  if (!http_->Post("/v3/kv/range", body, &r) || r.status != 200) return std::nullopt;
  JsonValue root;
  if (!JsonParser::Parse(r.body, &root)) return std::nullopt;
  RangeResult out;
  const JsonValue* hdr = root.find("header");
  if (hdr) out.revision = std::strtoll(hdr->get_str("revision", "0").c_str(), nullptr, 10);
  const JsonValue* kvs = root.find("kvs");
  if (kvs && kvs->type == JsonValue::kArray) {
    for (const auto& kv : kvs->arr) {
      std::string k, v;
      if (!Base64Decode(kv.get_str("key"), &k)) return std::nullopt;
      if (!Base64Decode(kv.get_str("value"), &v)) return std::nullopt;
      out.kvs.emplace_back(std::move(k), std::move(v));
    }
  }
  return out;
}

}  // namespace dfkv
