#ifndef DFKV_JSON_MIN_H_
#define DFKV_JSON_MIN_H_

#include <string>
#include <utility>
#include <vector>

namespace dfkv {

// Minimal JSON reader, just enough to read etcd v3 gateway responses. Scalars
// (string/number/bool/null) are stored as their string form; objects/arrays
// nest. Not a general serializer.
struct JsonValue {
  enum Type { kNull, kBool, kNumber, kString, kArray, kObject } type = kNull;
  std::string scalar;  // string contents (unescaped) / number token / "true"/"false"
  std::vector<JsonValue> arr;
  std::vector<std::pair<std::string, JsonValue>> obj;

  const JsonValue* find(const std::string& key) const {
    for (const auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  std::string get_str(const std::string& key, const std::string& dflt = "") const {
    const JsonValue* v = find(key);
    return v ? v->scalar : dflt;
  }
};

class JsonParser {
 public:
  static bool Parse(const std::string& s, JsonValue* out) {
    JsonParser p(s);
    p.ws();
    if (!p.value(out)) return false;
    p.ws();
    return p.i_ == p.s_.size();  // trailing garbage => error
  }

 private:
  explicit JsonParser(const std::string& s) : s_(s) {}
  const std::string& s_;
  size_t i_ = 0;

  void ws() {
    while (i_ < s_.size() &&
           (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r'))
      ++i_;
  }
  bool value(JsonValue* o) {
    if (i_ >= s_.size()) return false;
    char c = s_[i_];
    if (c == '"') { o->type = JsonValue::kString; return str(&o->scalar); }
    if (c == '{') return object(o);
    if (c == '[') return array(o);
    if (c == 't' || c == 'f') return boolean(o);
    if (c == 'n') return null(o);
    return number(o);
  }
  bool str(std::string* out) {
    if (i_ >= s_.size() || s_[i_] != '"') return false;
    ++i_;
    out->clear();
    while (i_ < s_.size()) {
      char c = s_[i_++];
      if (c == '"') return true;
      if (c == '\\') {
        if (i_ >= s_.size()) return false;
        char e = s_[i_++];
        switch (e) {
          case '"': *out += '"'; break;   case '\\': *out += '\\'; break;
          case '/': *out += '/'; break;   case 'n': *out += '\n'; break;
          case 't': *out += '\t'; break;  case 'r': *out += '\r'; break;
          case 'b': *out += '\b'; break;  case 'f': *out += '\f'; break;
          case 'u': {
            if (i_ + 4 > s_.size()) return false;
            int cp = 0;
            for (int k = 0; k < 4; ++k) {
              char h = s_[i_++]; cp <<= 4;
              if (h >= '0' && h <= '9') cp |= h - '0';
              else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
              else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
              else return false;
            }
            if (cp < 0x80) { *out += char(cp); }
            else if (cp < 0x800) { *out += char(0xC0 | (cp >> 6)); *out += char(0x80 | (cp & 0x3F)); }
            else { *out += char(0xE0 | (cp >> 12)); *out += char(0x80 | ((cp >> 6) & 0x3F)); *out += char(0x80 | (cp & 0x3F)); }
            break;
          }
          default: return false;
        }
      } else {
        *out += c;
      }
    }
    return false;  // unterminated
  }
  bool number(JsonValue* o) {
    size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
    while (i_ < s_.size()) {
      char c = s_[i_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') ++i_;
      else break;
    }
    if (i_ == start) return false;
    o->type = JsonValue::kNumber;
    o->scalar = s_.substr(start, i_ - start);
    return true;
  }
  bool boolean(JsonValue* o) {
    if (s_.compare(i_, 4, "true") == 0) { i_ += 4; o->type = JsonValue::kBool; o->scalar = "true"; return true; }
    if (s_.compare(i_, 5, "false") == 0) { i_ += 5; o->type = JsonValue::kBool; o->scalar = "false"; return true; }
    return false;
  }
  bool null(JsonValue* o) {
    if (s_.compare(i_, 4, "null") == 0) { i_ += 4; o->type = JsonValue::kNull; return true; }
    return false;
  }
  bool array(JsonValue* o) {
    ++i_; o->type = JsonValue::kArray; ws();
    if (i_ < s_.size() && s_[i_] == ']') { ++i_; return true; }
    while (true) {
      JsonValue v; ws();
      if (!value(&v)) return false;
      o->arr.push_back(std::move(v)); ws();
      if (i_ >= s_.size()) return false;
      if (s_[i_] == ',') { ++i_; continue; }
      if (s_[i_] == ']') { ++i_; return true; }
      return false;
    }
  }
  bool object(JsonValue* o) {
    ++i_; o->type = JsonValue::kObject; ws();
    if (i_ < s_.size() && s_[i_] == '}') { ++i_; return true; }
    while (true) {
      ws();
      std::string key;
      if (!str(&key)) return false;
      ws();
      if (i_ >= s_.size() || s_[i_] != ':') return false;
      ++i_; ws();
      JsonValue v;
      if (!value(&v)) return false;
      o->obj.emplace_back(std::move(key), std::move(v)); ws();
      if (i_ >= s_.size()) return false;
      if (s_[i_] == ',') { ++i_; continue; }
      if (s_[i_] == '}') { ++i_; return true; }
      return false;
    }
  }
};

}  // namespace dfkv

#endif  // DFKV_JSON_MIN_H_
