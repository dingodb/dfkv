# SPDX-License-Identifier: Apache-2.0
"""Asyncio wrapper around the dfkv C ABI (libdfkv.so) loaded via ctypes.

This replaces the dingofs connector's pybind11 ``_dingofs_native`` module +
eventfd completion queue. dfkv's C ABI is synchronous and internally
thread-safe: ``dfkv_batch_*`` block and fan out across owning nodes with their
own thread pool (KVClient::RunParallel), and the membership ring is mutex
guarded — so one ``dfkv_open`` handle is safe to share across many worker
threads. ``ctypes.CDLL`` releases the GIL for the duration of each foreign call,
so dispatching the blocking calls to a ``ThreadPoolExecutor`` gives real
concurrency without a native demux thread or a cross-thread Future bridge.

Reads use ``dfkv_batch_get_auto`` (variable-size): each buffer is full-chunk
sized, and the call reports the true stored payload length per key so the
connector can ``reshape_partial_chunk`` an unfull chunk. Full chunks keep the
RDMA zero-copy datapath.
"""

from __future__ import annotations

import ctypes
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Sequence, Tuple

from .access_log import access_log

__all__ = ["DfkvNativeClient", "load_lib"]

logger = logging.getLogger(__name__)

c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_uint64 = ctypes.c_uint64
c_uint32 = ctypes.c_uint32
c_int = ctypes.c_int
POINTER = ctypes.POINTER


def load_lib(path: Optional[str] = None) -> ctypes.CDLL:
    """Load libdfkv.so and declare the C ABI signatures.

    Path precedence: explicit ``path`` arg → env ``DFKV_LIB`` →
    ``$DFKV_BUILD/libdfkv.so`` → ``./build/libdfkv.so`` relative to cwd.
    """
    lib_path = (
        path
        or os.environ.get("DFKV_LIB")
        or os.path.join(os.environ.get("DFKV_BUILD", "build"), "libdfkv.so")
    )
    lib = ctypes.CDLL(lib_path)
    lib.dfkv_open.restype = c_void_p
    lib.dfkv_open.argtypes = [c_char_p, c_uint64, c_uint32, c_uint32, c_uint32,
                              c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]
    lib.dfkv_put.restype = c_int
    lib.dfkv_put.argtypes = [c_void_p, c_char_p, c_void_p, c_uint64]
    lib.dfkv_get.restype = c_int
    lib.dfkv_get.argtypes = [c_void_p, c_char_p, c_void_p, c_uint64]
    lib.dfkv_get_auto.restype = c_int
    lib.dfkv_get_auto.argtypes = [c_void_p, c_char_p, c_void_p, c_uint64,
                                  POINTER(c_uint64)]
    lib.dfkv_exist.restype = c_int
    lib.dfkv_exist.argtypes = [c_void_p, c_char_p]
    lib.dfkv_register_memory.restype = c_int
    lib.dfkv_register_memory.argtypes = [c_void_p, c_void_p, c_uint64]
    lib.dfkv_batch_put.restype = c_int
    lib.dfkv_batch_put.argtypes = [c_void_p, POINTER(c_char_p), POINTER(c_void_p),
                                   POINTER(c_uint64), c_int, POINTER(c_int)]
    lib.dfkv_batch_get.restype = c_int
    lib.dfkv_batch_get.argtypes = lib.dfkv_batch_put.argtypes
    lib.dfkv_batch_get_auto.restype = c_int
    lib.dfkv_batch_get_auto.argtypes = [c_void_p, POINTER(c_char_p),
                                        POINTER(c_void_p), POINTER(c_uint64),
                                        c_int, POINTER(c_int), POINTER(c_uint64)]
    lib.dfkv_batch_exist.restype = c_int
    lib.dfkv_batch_exist.argtypes = [c_void_p, POINTER(c_char_p), c_int,
                                     POINTER(c_int)]
    lib.dfkv_set_members.restype = c_int
    lib.dfkv_set_members.argtypes = [c_void_p, c_char_p]
    lib.dfkv_start_mds_discovery.restype = c_int
    lib.dfkv_start_mds_discovery.argtypes = [c_void_p, c_char_p, c_char_p, c_int]
    lib.dfkv_close.restype = None
    lib.dfkv_close.argtypes = [c_void_p]
    return lib


def _mv_ptr(mv: memoryview) -> Tuple[int, Any]:
    """Return ``(address, keepalive)`` for a memoryview buffer.

    Zero-copy for a writable, C-contiguous buffer (the common case — every
    MemoryObj byte_array is a slice of LMCache's pinned arena). A read-only
    buffer can't be aliased by ctypes, so we copy into a ctypes array (rare;
    correctness over zero-copy). The keepalive must outlive the C call.
    """
    try:
        arr = (ctypes.c_char * mv.nbytes).from_buffer(mv)
    except (TypeError, ValueError):
        arr = (ctypes.c_char * mv.nbytes).from_buffer_copy(mv)
    return ctypes.addressof(arr), arr


def _total_size(bufs: Sequence[memoryview]) -> int:
    return sum(mv.nbytes for mv in bufs)


class DfkvNativeClient:
    """Awaitable interface over libdfkv.so (ctypes + thread-pool executor)."""

    def __init__(
        self,
        raw_endpoint: str,
        group: str,
        membership: str,
        geometry: dict,
        lib_path: Optional[str] = None,
        mds_poll_ms: int = 3000,
        rdma_pools: Optional[Sequence[Tuple[int, int]]] = None,
        loop=None,
        get_parallelism: int = 1,
    ) -> None:
        import asyncio

        with access_log(
            "native.__init__",
            lambda: f"membership={membership} endpoint={raw_endpoint} "
                    f"group={group} rdma_pools={len(rdma_pools or [])} "
                    f"parallelism={get_parallelism}",
        ) as r:
            self._lib = load_lib(lib_path)
            self._loop = loop or asyncio.get_running_loop()
            self._closed = False
            g = geometry

            members = raw_endpoint if membership == "static" else ""
            self._h = self._lib.dfkv_open(
                members.encode(),
                c_uint64(int(g["model_hash"]) & 0xFFFFFFFFFFFFFFFF),
                int(g["page_size"]) & 0xFFFFFFFF,
                int(g["dtype_tag"]) & 0xFFFFFFFF,
                int(g["flags"]) & 0xFFFFFFFF,
                int(g["tp_size"]) & 0xFFFFFFFF,
                int(g["tp_rank"]) & 0xFFFFFFFF,
                int(g["layer_num"]) & 0xFFFFFFFF,
                int(g["head_num"]) & 0xFFFFFFFF,
                int(g["head_dim"]) & 0xFFFFFFFF,
            )
            if not self._h:
                raise RuntimeError("dfkv_open failed")

            if membership == "mds":
                rc = self._lib.dfkv_start_mds_discovery(
                    self._h, raw_endpoint.encode(), group.encode(),
                    int(mds_poll_ms),
                )
                if rc != 0:
                    self._lib.dfkv_close(self._h)
                    self._h = None
                    raise RuntimeError(
                        f"dfkv_start_mds_discovery failed (rc={rc})"
                    )

            # Register the host arena(s) once so RDMA Put/Get into any slice of
            # them skips per-op MR registration. No-op on TCP.
            regs = 0
            for base, size in (rdma_pools or []):
                if base and size and self._lib.dfkv_register_memory(
                    self._h, c_void_p(int(base)), c_uint64(int(size))
                ) == 0:
                    regs += 1

            self._executor = ThreadPoolExecutor(
                max_workers=max(1, int(get_parallelism)),
                thread_name_prefix="dfkv-io",
            )
            r.result = f"ok regs={regs}"

    # ------------------------------------------------------------------
    # blocking ctypes helpers (run in the executor)
    # ------------------------------------------------------------------

    def _batch_set_blocking(
        self, keys: List[str], bufs: List[memoryview]
    ) -> Tuple[bool, List[bool]]:
        n = len(keys)
        kbuf = [k.encode() for k in keys]
        karr = (c_char_p * n)(*kbuf)
        ptrs: List[int] = []
        keepalive: List[Any] = []
        for mv in bufs:
            p, ka = _mv_ptr(mv)
            ptrs.append(p)
            keepalive.append(ka)
        parr = (c_void_p * n)(*[c_void_p(p) for p in ptrs])
        sarr = (c_uint64 * n)(*[mv.nbytes for mv in bufs])
        out = (c_int * n)()
        rc = self._lib.dfkv_batch_put(self._h, karr, parr, sarr, n, out)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_put rc={rc}")
        per_key = [out[i] == 1 for i in range(n)]
        del keepalive  # C is done reading the source buffers
        return all(per_key), per_key

    def _batch_get_blocking(
        self, keys: List[str], bufs: List[memoryview]
    ) -> Tuple[bool, List[bool], List[int]]:
        n = len(keys)
        kbuf = [k.encode() for k in keys]
        karr = (c_char_p * n)(*kbuf)
        ptrs: List[int] = []
        keepalive: List[Any] = []
        for mv in bufs:
            p, ka = _mv_ptr(mv)
            ptrs.append(p)
            keepalive.append(ka)
        parr = (c_void_p * n)(*[c_void_p(p) for p in ptrs])
        caps = (c_uint64 * n)(*[mv.nbytes for mv in bufs])
        out_hit = (c_int * n)()
        out_len = (c_uint64 * n)()
        rc = self._lib.dfkv_batch_get_auto(
            self._h, karr, parr, caps, n, out_hit, out_len
        )
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_get_auto rc={rc}")
        per_key = [out_hit[i] == 1 for i in range(n)]
        lengths = [int(out_len[i]) for i in range(n)]
        del keepalive  # C has finished writing the destination buffers
        return all(per_key), per_key, lengths

    def _batch_exists_blocking(self, keys: List[str]) -> List[bool]:
        n = len(keys)
        karr = (c_char_p * n)(*[k.encode() for k in keys])
        out = (c_int * n)()
        rc = self._lib.dfkv_batch_exist(self._h, karr, n, out)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_exist rc={rc}")
        return [out[i] == 1 for i in range(n)]

    # ------------------------------------------------------------------
    # public async API (returns the same tuple shapes the connector expects)
    # ------------------------------------------------------------------

    async def batch_set(
        self, keys: Sequence[str], bufs: Sequence[memoryview]
    ) -> Tuple[bool, Optional[List[bool]]]:
        n = len(keys)
        with access_log("native.batch_set",
                        lambda: f"{n} keys, {_total_size(bufs)} bytes") as r:
            ok, per_key = await self._loop.run_in_executor(
                self._executor, self._batch_set_blocking, list(keys), list(bufs)
            )
            r.result = "ok" if ok else "partial_fail"
            return ok, per_key

    async def batch_get(
        self, keys: Sequence[str], bufs: Sequence[memoryview]
    ) -> Tuple[bool, Optional[List[bool]], List[int]]:
        n = len(keys)
        with access_log("native.batch_get",
                        lambda: f"{n} keys, {_total_size(bufs)} bytes") as r:
            ok, per_key, lengths = await self._loop.run_in_executor(
                self._executor, self._batch_get_blocking, list(keys), list(bufs)
            )
            hits = sum(1 for b in per_key if b)
            r.result = f"hits={hits}/{n}"
            return ok, per_key, lengths

    async def batch_exists(
        self, keys: Sequence[str]
    ) -> Optional[List[bool]]:
        n = len(keys)
        with access_log("native.batch_exists", lambda: f"{n} keys") as r:
            per_key = await self._loop.run_in_executor(
                self._executor, self._batch_exists_blocking, list(keys)
            )
            hits = sum(1 for b in per_key if b)
            r.result = f"hits={hits}/{n}"
            return per_key

    def exists_sync(self, key: str) -> bool:
        with access_log("native.exists_sync", lambda: key) as r:
            found = self._lib.dfkv_exist(self._h, key.encode()) == 1
            r.result = "found" if found else "not_found"
            return found

    def ping_sync(self) -> None:
        """Cheap liveness check — connectivity is established at construction
        (dfkv_open + MDS discovery). Just verify the handle is live."""
        with access_log("native.ping_sync", lambda: ""):
            if self._closed or not self._h:
                raise RuntimeError("dfkv client is closed")

    def close(self) -> None:
        if self._closed:
            return
        with access_log("native.close", lambda: ""):
            self._closed = True
            try:
                self._executor.shutdown(wait=False)
            except Exception:  # pragma: no cover
                pass
            try:
                if self._h:
                    self._lib.dfkv_close(self._h)
                    self._h = None
            except Exception as exc:  # pragma: no cover
                logger.warning("dfkv_close failed: %s", exc)
