/* C ABI for the DingoFS KV client, consumed by the SGLang HiCache plugin via
 * Python ctypes (Python-version-agnostic; the real build also ships a nanobind
 * module). Pointers are raw host addresses (zero-copy: put reads n bytes from
 * ptr; get writes n bytes to ptr). */
#ifndef DFKV_DFKV_C_API_H_
#define DFKV_DFKV_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* dfkv_client_t;

// members: "name1=ip:port,name2=ip:port". Remaining args = this engine's
// geometry identity baked into the value header.
dfkv_client_t dfkv_open(const char* members, uint64_t model_hash,
                        uint32_t page_size, uint32_t dtype_tag, uint32_t flags,
                        uint32_t tp_size, uint32_t tp_rank, uint32_t layer_num,
                        uint32_t head_num, uint32_t head_dim);

int dfkv_put(dfkv_client_t c, const char* key, const void* ptr, uint64_t n);  // 0=ok
int dfkv_get(dfkv_client_t c, const char* key, void* ptr, uint64_t n);        // 1=hit,0=miss
// Variable-size get: writes the stored payload (whatever length it was put with)
// into ptr (which has capacity `cap`) and reports the actual byte length via
// *out_len. Unlike dfkv_get, the caller does NOT need to know the exact stored
// size; a stored payload larger than cap is a miss. 1=hit, 0=miss.
int dfkv_get_auto(dfkv_client_t c, const char* key, void* ptr, uint64_t cap,
                  uint64_t* out_len);
int dfkv_exist(dfkv_client_t c, const char* key);                            // 1/0
// Register a large host memory region (e.g. the whole SGLang host KV pool) so
// put/get into any buffer inside it never do a per-op RDMA MR registration — the
// region is registered once per connection. No-op on the TCP transport. Call once
// at startup after the pool is allocated, before traffic. Returns 0 on success.
int dfkv_register_memory(dfkv_client_t c, const void* base, uint64_t size);
// Hot-swap cluster membership ("n1=ip:port,n2=ip:port"). Returns 0 on success.
int dfkv_set_members(dfkv_client_t c, const char* members);
// Discovery: query seed ("ip:port") for the cluster member list and apply it.
// Returns 0 on success, non-zero on failure (seed unreachable / empty list).
int dfkv_refresh_members(dfkv_client_t c, const char* seed);
// Start background MDS-based discovery. mds_endpoints: comma-separated "ip:port" list.
// poll_ms: poll interval (default 3000 if <= 0). Returns 0 on success.
int dfkv_start_mds_discovery(dfkv_client_t c, const char* mds_endpoints,
                             const char* group, int poll_ms);

// Actual client transport selected at dfkv_open(), e.g. "rdma",
// "tcp(rdma-not-requested)", or "injected". Returns "" for null clients.
const char* dfkv_transport_mode(dfkv_client_t c);
int dfkv_set_batch_concurrency(dfkv_client_t c, uint64_t n);

// Batched, concurrently fanned out. n items; out_* arrays (len n) receive
// per-item results (1/0). Return 0 on call success.
int dfkv_batch_put(dfkv_client_t c, const char** keys, const void** ptrs,
                   const uint64_t* sizes, int n, int* out_ok);
int dfkv_batch_get(dfkv_client_t c, const char** keys, void** ptrs,
                   const uint64_t* sizes, int n, int* out_hit);
// Variable-size batched get. caps[i] is the buffer capacity of ptrs[i]; the
// stored payload (any size <= caps[i]) is written into ptrs[i]. out_hit[i] is
// 1/0; out_len[i] (caller array, len n) receives the actual payload byte length
// per item (0 on miss). Keeps the RDMA zero-copy datapath for full-size items.
// Return 0 on call success.
int dfkv_batch_get_auto(dfkv_client_t c, const char** keys, void** ptrs,
                        const uint64_t* caps, int n, int* out_hit,
                        uint64_t* out_len);
int dfkv_batch_exist(dfkv_client_t c, const char** keys, int n, int* out_exist);

// Client-side Prometheus metrics snapshot (ops served, IO errors, peer health
// transitions, per-peer errors). Writes up to `cap` bytes (NUL-terminated when
// it fits) into buf and returns the FULL text length (excluding NUL). Pass cap=0
// (buf may be NULL) to size the buffer first. The plugin polls this and mirrors
// the deltas onto its Prometheus counters.
uint64_t dfkv_stats_snapshot(dfkv_client_t c, char* buf, uint64_t cap);

void dfkv_close(dfkv_client_t c);

#ifdef __cplusplus
}
#endif

#endif  // DFKV_DFKV_C_API_H_
