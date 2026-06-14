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
int dfkv_exist(dfkv_client_t c, const char* key);                            // 1/0
// Hot-swap cluster membership ("n1=ip:port,n2=ip:port"). Returns 0 on success.
int dfkv_set_members(dfkv_client_t c, const char* members);

// Batched, concurrently fanned out. n items; out_* arrays (len n) receive
// per-item results (1/0). Return 0 on call success.
int dfkv_batch_put(dfkv_client_t c, const char** keys, const void** ptrs,
                   const uint64_t* sizes, int n, int* out_ok);
int dfkv_batch_get(dfkv_client_t c, const char** keys, void** ptrs,
                   const uint64_t* sizes, int n, int* out_hit);
int dfkv_batch_exist(dfkv_client_t c, const char** keys, int n, int* out_exist);

void dfkv_close(dfkv_client_t c);

#ifdef __cplusplus
}
#endif

#endif  // DFKV_DFKV_C_API_H_
