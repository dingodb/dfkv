"""DingoFS HiCache storage backend for SGLang (loaded via --hicache-storage-backend
dynamic). Zero-copy v1 path: hands raw host-buffer pointers from
mem_pool_host.get_page_buffer_meta() straight to the DingoFS KV client (C ABI).

MLA (GLM-5.1): one packed-latent object per page, key has NO tp_rank suffix, and
only tp_rank 0 writes (backup_skip) since the latent is replicated across TP.

This file is the production plugin. On a GPU host it imports the real SGLang
HiCacheStorage; the test harness supplies a no-torch shim with the same surface.
"""
from __future__ import annotations

import ctypes
import os
from typing import List, Optional

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

_FLAG_IS_MLA = 0x1


def _load_lib(path: Optional[str] = None) -> ctypes.CDLL:
    lib_path = (path or os.environ.get("DFKV_LIB")
                or os.path.join(os.environ.get("DFKV_BUILD", "/home/ketor/dfkv-dev/build"),
                                "libdfkv.so"))
    lib = ctypes.CDLL(lib_path)
    lib.dfkv_open.restype = ctypes.c_void_p
    lib.dfkv_open.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint32,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                              ctypes.c_uint32]
    lib.dfkv_put.restype = ctypes.c_int
    lib.dfkv_put.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_get.restype = ctypes.c_int
    lib.dfkv_get.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_exist.restype = ctypes.c_int
    lib.dfkv_exist.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.dfkv_close.restype = None
    lib.dfkv_close.argtypes = [ctypes.c_void_p]
    return lib


class DingoFSHiCache(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, kwargs: Optional[dict] = None):
        cfg = (kwargs or {}) or (getattr(storage_config, "extra_config", None) or {})
        self.cfg = cfg
        self.model = (storage_config.model_name or "").replace("/", "-")
        self.tp_rank = int(storage_config.tp_rank)
        self.tp_size = int(storage_config.tp_size)
        self.is_mla = bool(storage_config.is_mla_model)
        if "members" not in cfg:
            raise ValueError("dingofs hicache: extra_config.members required")
        self._lib = _load_lib(cfg.get("lib_path"))
        flags = _FLAG_IS_MLA if self.is_mla else 0
        model_hash = int(cfg.get("model_hash", 0)) & 0xFFFFFFFFFFFFFFFF
        self._h = self._lib.dfkv_open(
            cfg["members"].encode(), model_hash,
            int(cfg.get("page_size", 64)), int(cfg.get("dtype_tag", 0)), flags,
            self.tp_size, self.tp_rank,
            int(cfg.get("layer_num", 0)), int(cfg.get("head_num", 0)),
            int(cfg.get("head_dim", 0)))
        if not self._h:
            raise RuntimeError("dfkv_open failed (bad members?)")
        self.mem_pool_host = None

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                self._lib.dfkv_close(self._h)
                self._h = None
        except Exception:
            pass

    def register_mem_pool_host(self, mem_pool_host):
        self.mem_pool_host = mem_pool_host

    # --- key scheme: MLA single object (no rank suffix); MHA two objects ---
    def _keys(self, page_hash: str) -> List[str]:
        if self.is_mla:
            return [f"{self.model}/{page_hash}_k"]
        base = f"{self.model}/{page_hash}_{self.tp_size}_{self.tp_rank}"
        return [base + "_k", base + "_v"]

    def _sub(self) -> int:
        return 1 if self.is_mla else 2

    # --- zero-copy v1 batch path (the one the controller calls) ---
    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        # MLA backup_skip: latent is replicated across TP, only rank 0 writes.
        if self.is_mla and self.tp_rank != 0:
            return [True] * len(keys)
        ptrs, sizes = self.mem_pool_host.get_page_buffer_meta(host_indices)
        sub = self._sub()
        assert len(ptrs) == len(keys) * sub, (len(ptrs), len(keys), sub)
        out = []
        for i, k in enumerate(keys):
            good = True
            for j, sk in enumerate(self._keys(k)):
                rc = self._lib.dfkv_put(self._h, sk.encode(),
                                        ctypes.c_void_p(int(ptrs[i * sub + j])),
                                        ctypes.c_uint64(int(sizes[i * sub + j])))
                good = good and rc == 0
            out.append(good)
        return out

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        ptrs, sizes = self.mem_pool_host.get_page_buffer_meta(host_indices)
        sub = self._sub()
        assert len(ptrs) == len(keys) * sub, (len(ptrs), len(keys), sub)
        out = []
        for i, k in enumerate(keys):
            hit = True
            for j, sk in enumerate(self._keys(k)):
                rc = self._lib.dfkv_get(self._h, sk.encode(),
                                        ctypes.c_void_p(int(ptrs[i * sub + j])),
                                        ctypes.c_uint64(int(sizes[i * sub + j])))
                hit = hit and rc == 1
            out.append(hit)
        return out

    def batch_exists(self, keys, extra_info=None) -> int:
        n = 0
        for k in keys:
            if all(self._lib.dfkv_exist(self._h, sk.encode()) == 1 for sk in self._keys(k)):
                n += 1
            else:
                break
        return n

    # --- required abstract methods (non zero-copy / introspection) ---
    def exists(self, key) -> bool:
        return all(self._lib.dfkv_exist(self._h, sk.encode()) == 1 for sk in self._keys(key))

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        if value is None:
            return False
        mv = memoryview(value).cast("B")
        sk = self._keys(key)[0]
        buf = (ctypes.c_char * len(mv)).from_buffer_copy(mv)
        return self._lib.dfkv_put(self._h, sk.encode(), ctypes.cast(buf, ctypes.c_void_p),
                                  ctypes.c_uint64(len(mv))) == 0

    def get(self, key, target_location=None, target_sizes=None):
        return None  # non zero-copy reads go through batch_get_v1

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        if values is None:
            return False
        return all(self.set(k, v) for k, v in zip(keys, values))

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [None] * len(keys)
