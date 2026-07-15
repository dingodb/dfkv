#include "client/cuda_ipc.h"

#include <dlfcn.h>

#include <mutex>

#include "utils/log.h"

namespace dfkv {

namespace {
// Prefer the versioned symbol ONLY where cuda.h itself binds it (the classic
// 32->64-bit _v2 set: cuMemAlloc/cuMemFree/cuIpcOpenMemHandle) — there the
// _v2 signature is the one we declare. NEVER guess _v2 for other entry
// points: drivers export e.g. cuCtxGetDevice_v2 with a DIFFERENT signature
// (an extra CUcontext parameter), and calling it through the plain prototype
// reads a garbage register — a segfault that only fires on some threads
// (found live: one of four vLLM workers died in exactly that call).
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
  MemcpyAsync = reinterpret_cast<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t,
                                              CUstream)>(
      ::dlsym(h, "cuMemcpyAsync"));
  StreamCreate = reinterpret_cast<CUresult (*)(CUstream*, unsigned)>(
      ::dlsym(h, "cuStreamCreate"));
  StreamSynchronize = reinterpret_cast<CUresult (*)(CUstream)>(
      ::dlsym(h, "cuStreamSynchronize"));
  StreamDestroy = reinterpret_cast<CUresult (*)(CUstream)>(
      SymV2(h, "cuStreamDestroy_v2", "cuStreamDestroy"));
  IpcGetMemHandle = reinterpret_cast<CUresult (*)(CUipcMemHandle*, CUdeviceptr)>(
      ::dlsym(h, "cuIpcGetMemHandle"));
  IpcOpenMemHandle = reinterpret_cast<CUresult (*)(CUdeviceptr*, CUipcMemHandle,
                                                   unsigned)>(
      SymV2(h, "cuIpcOpenMemHandle_v2", "cuIpcOpenMemHandle"));
  IpcCloseMemHandle = reinterpret_cast<CUresult (*)(CUdeviceptr)>(
      ::dlsym(h, "cuIpcCloseMemHandle"));
  ctx_get_current_ = reinterpret_cast<CUresult (*)(CUcontext*)>(
      ::dlsym(h, "cuCtxGetCurrent"));
  ctx_set_current_ = reinterpret_cast<CUresult (*)(CUcontext)>(
      ::dlsym(h, "cuCtxSetCurrent"));
  ctx_get_device_ = reinterpret_cast<CUresult (*)(int*)>(
      ::dlsym(h, "cuCtxGetDevice"));
  primary_ctx_retain_ = reinterpret_cast<CUresult (*)(CUcontext*, int)>(
      ::dlsym(h, "cuDevicePrimaryCtxRetain"));
  pointer_get_attribute_ = reinterpret_cast<CUresult (*)(void*, int, CUdeviceptr)>(
      ::dlsym(h, "cuPointerGetAttribute"));
  if (!init || !MemAlloc || !MemFree || !Memcpy || !MemcpyAsync ||
      !StreamCreate || !StreamSynchronize || !StreamDestroy ||
      !IpcGetMemHandle || !IpcOpenMemHandle || !IpcCloseMemHandle ||
      !ctx_get_current_ || !ctx_set_current_ || !ctx_get_device_ ||
      !primary_ctx_retain_ || !pointer_get_attribute_)
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

int CudaLib::DeviceOf(const void* p) const {
  int dev = -1;
  if (pointer_get_attribute_(&dev, kCuPointerAttributeDeviceOrdinal,
                             reinterpret_cast<CUdeviceptr>(p)) != kCudaSuccess)
    return -1;
  return dev;
}

bool CudaLib::BindPrimaryCtx(int dev) const {
  if (dev < 0) return false;
  CUcontext ctx = nullptr;
  if (primary_ctx_retain_(&ctx, dev) != kCudaSuccess || !ctx) return false;
  return ctx_set_current_(ctx) == kCudaSuccess;
}

}  // namespace dfkv
