/* NameThisThread — role-name the calling thread for observability.
 *
 * Every background thread used to show up as "dfkv_server"/"python3" in
 * /proc, perf, eBPF and gdb: the phase-7 write-path investigation could not
 * even attribute CPU between serve/flush/reclaim roles. Called first thing
 * inside each thread body. Kernel limit is 15 chars + NUL; longer names are
 * truncated by pthread_setname_np, so keep them short and stable — they are
 * an observability ABI of sorts (dashboards/bpftrace scripts match on them). */
#ifndef DFKV_THREAD_NAME_H_
#define DFKV_THREAD_NAME_H_

#include <pthread.h>

#include <cstdio>

namespace dfkv {

inline void NameThisThread(const char* name) {
#ifdef __linux__
  ::pthread_setname_np(::pthread_self(), name);
#else
  (void)name;
#endif
}

// Indexed flavor for worker pools: "<prefix><idx>", truncated to the kernel's
// 15-char limit ("rt-flush-12", "kv-fan-7").
inline void NameThisThread(const char* prefix, size_t idx) {
  // pthread_setname_np fails (ERANGE) on names >15 chars rather than
  // truncating, so the name must be clamped. Format into a buffer wide enough
  // for any real prefix + 20-digit idx (a 16-byte buffer makes snprintf do the
  // clamp, but trips -Wformat-truncation at every inlined call site), then
  // clamp to the kernel limit explicitly.
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%s%zu", prefix, idx);
  buf[15] = '\0';
  NameThisThread(buf);
}

}  // namespace dfkv

#endif  // DFKV_THREAD_NAME_H_
