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
    lib.dfkv_batch_put.restype = ctypes.c_int
    lib.dfkv_batch_put.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
                                   ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int)]
    lib.dfkv_batch_get.restype = ctypes.c_int
    lib.dfkv_batch_get.argtypes = lib.dfkv_batch_put.argtypes
    lib.dfkv_batch_exist.restype = ctypes.c_int
    lib.dfkv_batch_exist.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.dfkv_set_members.restype = ctypes.c_int
    lib.dfkv_set_members.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.dfkv_close.restype = None
    lib.dfkv_close.argtypes = [ctypes.c_void_p]
    return lib


def _arrays(subkeys, ptrs, sizes):
    """Build parallel C arrays (keys, ptrs, sizes) for a batch call."""
    n = len(subkeys)
    kbuf = [k.encode() for k in subkeys]
    karr = (ctypes.c_char_p * n)(*kbuf)
    parr = (ctypes.c_void_p * n)(*[ctypes.c_void_p(int(p)) for p in ptrs])
    sarr = (ctypes.c_uint64 * n)(*[int(s) for s in sizes])
    out = (ctypes.c_int * n)()
    return karr, parr, sarr, out, kbuf


class DfkvHiCache(HiCacheStorage):
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

    def set_members(self, members: str):
        """Hot-swap cluster membership, e.g. 'n1=ip:12000,n2=ip:12000'."""
        self._lib.dfkv_set_members(self._h, members.encode())

    # --- key scheme: MLA single object (no rank suffix); MHA two objects ---
    def _keys(self, page_hash: str) -> List[str]:
        if self.is_mla:
            return [f"{self.model}/{page_hash}_k"]
        base = f"{self.model}/{page_hash}_{self.tp_size}_{self.tp_rank}"
        return [base + "_k", base + "_v"]

    def _sub(self) -> int:
        return 1 if self.is_mla else 2

    def _flatten(self, keys, ptrs, sizes):
        """Expand per-page keys into per-object (sub) flat arrays."""
        sub = self._sub()
        assert len(ptrs) == len(keys) * sub, (len(ptrs), len(keys), sub)
        sks, sp, ss = [], [], []
        for i, k in enumerate(keys):
            for j, sk in enumerate(self._keys(k)):
                sks.append(sk); sp.append(int(ptrs[i * sub + j])); ss.append(int(sizes[i * sub + j]))
        return sub, sks, sp, ss

    def _fold(self, flat_results, npages, sub):
        """A page succeeds iff all its sub-objects succeeded."""
        return [all(flat_results[i * sub + j] for j in range(sub)) for i in range(npages)]

    # --- zero-copy v1 batch path (the one the controller calls) ---
    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        # MLA backup_skip: latent is replicated across TP, only rank 0 writes.
        if self.is_mla and self.tp_rank != 0:
            return [True] * len(keys)
        ptrs, sizes = self.mem_pool_host.get_page_buffer_meta(host_indices)
        sub, sks, sp, ss = self._flatten(keys, ptrs, sizes)
        karr, parr, sarr, out, _kb = _arrays(sks, sp, ss)
        self._lib.dfkv_batch_put(self._h, karr, parr, sarr, len(sks), out)
        return self._fold([out[i] == 1 for i in range(len(sks))], len(keys), sub)

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        ptrs, sizes = self.mem_pool_host.get_page_buffer_meta(host_indices)
        sub, sks, sp, ss = self._flatten(keys, ptrs, sizes)
        karr, parr, sarr, out, _kb = _arrays(sks, sp, ss)
        self._lib.dfkv_batch_get(self._h, karr, parr, sarr, len(sks), out)
        return self._fold([out[i] == 1 for i in range(len(sks))], len(keys), sub)

    def batch_exists(self, keys, extra_info=None) -> int:
        # longest contiguous prefix of pages whose every sub-object exists
        sub = self._sub()
        sks = [sk for k in keys for sk in self._keys(k)]
        karr = (ctypes.c_char_p * len(sks))(*[s.encode() for s in sks])
        out = (ctypes.c_int * len(sks))()
        self._lib.dfkv_batch_exist(self._h, karr, len(sks), out)
        page_ok = self._fold([out[i] == 1 for i in range(len(sks))], len(keys), sub)
        n = 0
        for ok in page_ok:
            if not ok:
                break
            n += 1
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
