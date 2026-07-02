/* Prometheus label-value escaping. The exposition format requires `\`, `"`, and
 * newline in a label value to be escaped, else the whole scrape target is
 * rejected. dfkv puts operator/config-controlled strings in labels (node/group
 * identity, and especially disk paths from --dir, which may legally contain a
 * quote or backslash on Linux), so escape them. Off the datapath. */
#ifndef DFKV_PROM_ESCAPE_H_
#define DFKV_PROM_ESCAPE_H_

#include <string>

namespace dfkv {

inline std::string PromLabelEscape(const std::string& v) {
  std::string out;
  out.reserve(v.size());
  for (char c : v) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      default: out += c;
    }
  }
  return out;
}

}  // namespace dfkv

#endif  // DFKV_PROM_ESCAPE_H_
