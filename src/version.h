/* dfkv version string, baked in at build time (DFKV_VERSION from CMake
 * PROJECT_VERSION). Defined in version.cc which is compiled into dfkv_core — the
 * only target that carries the DFKV_VERSION compile definition — so every binary
 * that links the core gets the same value. */
#ifndef DFKV_VERSION_H_
#define DFKV_VERSION_H_

namespace dfkv {

// e.g. "1.4.0" (or "dev" for an un-stamped build).
const char* Version();

// True if argv contains "--version" or "-V". Helper so each *_main can early-exit
// before its (flag-pair) argument parser, which would otherwise ignore a lone
// --version and fall through to running the daemon.
bool WantsVersion(int argc, char** argv);

// True if argv contains "--help" or "-h". Same rationale: an unknown flag would
// otherwise be ignored and the daemon would start with default config.
bool WantsHelp(int argc, char** argv);

}  // namespace dfkv

#endif  // DFKV_VERSION_H_
