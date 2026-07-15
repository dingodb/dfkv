/* CudaLib — dlopen'd CUDA *driver* API surface for the GPU rendezvous path.
 *
 * libdfkv.so must not link CUDA: SGLang hosts and CPU-only tools load the
 * same library. Everything here resolves from libcuda.so.1 at first use; if
 * the driver (or any required symbol) is absent, Get() returns nullptr and
 * callers run without GPU dedup — the same "nothing here is load-bearing"
 * rule as NodeDedup. The declarations below intentionally mirror cuda.h so
 * no CUDA toolkit is needed at build time; they are ABI, not API (the driver
 * exports them with C linkage and fixed layouts).
 */
#ifndef DFKV_CUDA_IPC_H_
#define DFKV_CUDA_IPC_H_

#include <cstddef>
#include <cstdint>

namespace dfkv {

// Minimal cuda.h mirror (ABI-stable driver types).
using CUresult = int;
using CUdeviceptr = unsigned long long;
using CUcontext = void*;
struct CUipcMemHandle {
  char reserved[64];
};
constexpr CUresult kCudaSuccess = 0;
constexpr unsigned kCuIpcMemLazyEnablePeerAccess = 0x1;
constexpr int kCuPointerAttributeMemoryType = 2;    // CU_POINTER_ATTRIBUTE_MEMORY_TYPE
constexpr int kCuPointerAttributeDeviceOrdinal = 9; // CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
constexpr unsigned kCuMemoryTypeDevice = 2;         // CU_MEMORYTYPE_DEVICE

class CudaLib {
 public:
  // dlopens libcuda.so.1 and resolves the surface exactly once per process.
  // nullptr => no usable driver; callers must degrade silently.
  static const CudaLib* Get();

  // True iff p is a CUDA device pointer (UVA query; host/unknown => false).
  bool IsDevicePtr(const void* p) const;
  // Device ordinal owning device pointer p; -1 on failure.
  int DeviceOf(const void* p) const;
  // True iff the calling thread has a current CUDA context. Framework compute
  // threads always do; the connectors' pure-Python transfer threads (ctypes ->
  // C ABI, never touched CUDA) do NOT — those must BindPrimaryCtx first.
  bool HasCurrentCtx() const;
  int CurrentDevice() const;  // -1 without a context
  // Make device dev's PRIMARY context current on the calling thread (retains
  // it once per process — the framework already holds it; this only bumps a
  // refcount, it never creates a fresh context behind the framework's back).
  bool BindPrimaryCtx(int dev) const;

  CUresult (*MemAlloc)(CUdeviceptr*, size_t) = nullptr;
  CUresult (*MemFree)(CUdeviceptr) = nullptr;
  // Unified-addressing copy: any host/device src/dst combination.
  CUresult (*Memcpy)(CUdeviceptr, CUdeviceptr, size_t) = nullptr;
  CUresult (*IpcGetMemHandle)(CUipcMemHandle*, CUdeviceptr) = nullptr;
  CUresult (*IpcOpenMemHandle)(CUdeviceptr*, CUipcMemHandle, unsigned) = nullptr;
  CUresult (*IpcCloseMemHandle)(CUdeviceptr) = nullptr;

 private:
  CudaLib() = default;
  bool Resolve();

  CUresult (*ctx_get_current_)(CUcontext*) = nullptr;
  CUresult (*ctx_set_current_)(CUcontext) = nullptr;
  CUresult (*ctx_get_device_)(int*) = nullptr;
  CUresult (*primary_ctx_retain_)(CUcontext*, int) = nullptr;
  CUresult (*pointer_get_attribute_)(void*, int, CUdeviceptr) = nullptr;
};

}  // namespace dfkv

#endif  // DFKV_CUDA_IPC_H_
