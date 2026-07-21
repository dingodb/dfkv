#include "common/config_dump.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "utils/log.h"

extern char** environ;

namespace dfkv {
namespace config_dump {
namespace {

struct Entry {
  std::string value;
  Source src;
};

std::mutex g_mu;
// Keyed + sorted by name; last write wins so a re-resolved knob does not
// duplicate. Ordered map => stable, alphabetical Emit output.
std::map<std::string, Entry> g_entries;

const char* SourceLabel(Source s) {
  switch (s) {
    case Source::kFlag: return "flag";
    case Source::kEnv: return "env";
    case Source::kArg: return "arg";
    case Source::kDefault: return "default";
  }
  return "?";
}

// Only DFKV_-namespaced vars are dfkv knobs; other env (PATH, etc.) is noise.
bool IsDfkvEnv(const char* kv) { return std::strncmp(kv, "DFKV_", 5) == 0; }

}  // namespace

void Record(const std::string& name, const std::string& value, Source src) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_entries[name] = Entry{value, src};
}

void RecordResolved(const std::string& name, const std::string& value) {
  Record(name, value,
         std::getenv(name.c_str()) ? Source::kEnv : Source::kDefault);
}

std::string EnvStr(const char* name, const std::string& def) {
  const char* e = std::getenv(name);
  bool set = (e != nullptr && *e != '\0');
  std::string val = set ? std::string(e) : def;
  Record(name, val, set ? Source::kEnv : Source::kDefault);
  return val;
}

int64_t EnvI64(const char* name, int64_t def) {
  const char* e = std::getenv(name);
  int64_t val = def;
  Source src = Source::kDefault;
  if (e != nullptr && *e != '\0') {
    char* end = nullptr;
    long long parsed = std::strtoll(e, &end, 10);
    if (end != e) { val = static_cast<int64_t>(parsed); src = Source::kEnv; }
  }
  Record(name, std::to_string(val), src);
  return val;
}

uint64_t EnvU64(const char* name, uint64_t def) {
  const char* e = std::getenv(name);
  uint64_t val = def;
  Source src = Source::kDefault;
  if (e != nullptr && *e != '\0') {
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(e, &end, 10);
    if (end != e) { val = static_cast<uint64_t>(parsed); src = Source::kEnv; }
  }
  Record(name, std::to_string(val), src);
  return val;
}

bool EnvBool(const char* name, bool def) {
  const char* e = std::getenv(name);
  bool val = def;
  Source src = Source::kDefault;
  if (e != nullptr && *e != '\0') {
    std::string s(e);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    val = !(s == "0" || s == "false" || s == "no" || s == "off");
    src = Source::kEnv;
  }
  Record(name, val ? "1" : "0", src);
  return val;
}

void Emit(const std::string& section) {
  std::lock_guard<std::mutex> lk(g_mu);
  // Fold in DFKV_* env vars that no component recorded, so a set-but-unwired
  // knob is still visible (marked env). Recorded entries are authoritative.
  for (char** ep = environ; ep && *ep; ++ep) {
    if (!IsDfkvEnv(*ep)) continue;
    const char* eq = std::strchr(*ep, '=');
    if (!eq) continue;
    std::string name(*ep, eq - *ep);
    if (name == "DFKV_LOG") continue;  // logger level, echoed by the logger
    if (g_entries.find(name) == g_entries.end())
      g_entries[name] = Entry{std::string(eq + 1), Source::kEnv};
  }
  if (g_entries.empty()) return;
  DFKV_LOG_INFO("effective config (" + section + "): " +
                std::to_string(g_entries.size()) + " parameter(s)");
  size_t width = 0;
  for (const auto& kv : g_entries) width = std::max(width, kv.first.size());
  for (const auto& kv : g_entries) {
    std::string pad(width - kv.first.size(), ' ');
    DFKV_LOG_INFO("  " + kv.first + pad + " = " + kv.second.value +
                  "  (" + SourceLabel(kv.second.src) + ")");
  }
  g_entries.clear();
}

void ResetForTest() {
  std::lock_guard<std::mutex> lk(g_mu);
  g_entries.clear();
}

}  // namespace config_dump
}  // namespace dfkv
