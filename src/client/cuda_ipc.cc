#include "client/cuda_ipc.h"

#include <dlfcn.h>

#include <mutex>

#include "utils/log.h"

namespace dfkv {

namespace {
// Prefer the versioned symbol (what cuda.h binds since CUDA 3.2/11); fall
// back to the legacy name so ancient drivers still resolve.
void* SymV2(void* h, const char* v2, const char* legacy) {
  if (void* p = ::dlsym(h, v2)) return p;
  return ::dlsym(h, legacy);
}
}  // namespace

bool CudaLib::Resolve() {
  void* h = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!h) return false;
  using F = void*;
  F init = ::dlsym(h, "cuInit");
  MemAlloc = reinterpret_cast<CUresult (*)(CUdeviceptr*, size_t)>(
      SymV2(h, "cuMemAlloc_v2", "cuMemAlloc"));
  MemFree = reinterpret_cast<CUresult (*)(CUdeviceptr)>(
      SymV2(h, "cuMemFree_v2", "cuMemFree"));
  Memcpy = reinterpret_cast<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t)>(
      ::dlsym(h, "cuMemcpy"));
  IpcGetMemHandle = reinterpret_cast<CUresult (*)(CUipcMemHandle*, CUdeviceptr)>(
      ::dlsym(h, "cuIpcGetMemHandle"));
  IpcOpenMemHandle = reinterpret_cast<CUresult (*)(CUdeviceptr*, CUipcMemHandle,
                                                   unsigned)>(
      SymV2(h, "cuIpcOpenMemHandle_v2", "cuIpcOpenMemHandle"));
  IpcCloseMemHandle = reinterpret_cast<CUresult (*)(CUdeviceptr)>(
      ::dlsym(h, "cuIpcCloseMemHandle"));
  ctx_get_current_ = reinterpret_cast<CUresult (*)(CUcontext*)>(
      SymV2(h, "cuCtxGetCurrent_v2", "cuCtxGetCurrent"));
  ctx_get_device_ = reinterpret_cast<CUresult (*)(int*)>(
      SymV2(h, "cuCtxGetDevice_v2", "cuCtxGetDevice"));
  pointer_get_attribute_ = reinterpret_cast<CUresult (*)(void*, int, CUdeviceptr)>(
      ::dlsym(h, "cuPointerGetAttribute"));
  if (!init || !MemAlloc || !MemFree || !Memcpy || !IpcGetMemHandle ||
      !IpcOpenMemHandle || !IpcCloseMemHandle || !ctx_get_current_ ||
      !ctx_get_device_ || !pointer_get_attribute_)
    return false;
  // cuInit is idempotent; the host framework normally beat us to it. A
  // failure here (no device, driver/library mismatch) disables the surface.
  return reinterpret_cast<CUresult (*)(unsigned)>(init)(0) == kCudaSuccess;
}

const CudaLib* CudaLib::Get() {
  static CudaLib* lib = [] {
    auto* l = new CudaLib();
    if (l->Resolve()) return l;
    delete l;
    return static_cast<CudaLib*>(nullptr);
  }();
  return lib;
}

bool CudaLib::IsDevicePtr(const void* p) const {
  unsigned type = 0;
  if (pointer_get_attribute_(&type, kCuPointerAttributeMemoryType,
                             reinterpret_cast<CUdeviceptr>(p)) != kCudaSuccess)
    return false;  // unregistered host memory: attribute query fails
  return type == kCuMemoryTypeDevice;
}

bool CudaLib::HasCurrentCtx() const {
  CUcontext c = nullptr;
  return ctx_get_current_(&c) == kCudaSuccess && c != nullptr;
}

int CudaLib::CurrentDevice() const {
  int dev = -1;
  if (ctx_get_device_(&dev) != kCudaSuccess) return -1;
  return dev;
}

}  // namespace dfkv
