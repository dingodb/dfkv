#include "common/version.h"

#include <cstring>

#ifndef DFKV_VERSION
#define DFKV_VERSION "dev"
#endif

namespace dfkv {

const char* Version() { return DFKV_VERSION; }

bool WantsVersion(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--version") || !std::strcmp(argv[i], "-V"))
      return true;
  return false;
}

bool WantsHelp(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h"))
      return true;
  return false;
}

}  // namespace dfkv
