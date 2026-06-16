/* Minimal Prometheus exposition value extractor for client tooling (dfkvctl
 * stat --all). Finds the sample line for an exact metric name — with or without
 * a {label} set — and returns its trailing uint64 value. Skips # HELP/# TYPE.
 * Not on any datapath. */
#ifndef DFKV_PROM_PARSE_H_
#define DFKV_PROM_PARSE_H_

#include <cstdint>
#include <cstdlib>
#include <string>

namespace dfkv {

// Returns the value of the first sample line whose metric name == `name`
// (followed by ' ' or '{'); 0 if not present.
inline uint64_t PromMetricValue(const std::string& text, const std::string& name) {
  size_t pos = 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    size_t len = (eol == std::string::npos) ? text.size() - pos : eol - pos;
    if (len > name.size() && text.compare(pos, name.size(), name) == 0) {
      char after = text[pos + name.size()];
      if ((after == ' ' || after == '{') && text[pos] != '#') {
        // value is the token after the last space on the line
        size_t line_end = pos + len;
        size_t sp = text.rfind(' ', line_end - 1);
        if (sp != std::string::npos && sp + 1 < line_end)
          return std::strtoull(text.c_str() + sp + 1, nullptr, 10);
      }
    }
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }
  return 0;
}

}  // namespace dfkv

#endif  // DFKV_PROM_PARSE_H_
