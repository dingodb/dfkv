"""Device-pointer dfkv client for the vLLM connector.

Every buffer here is a raw integer pointer -- a slice of the GPU paged KV cache
that was pre-registered via :meth:`register_memory`. This differs from the
LMCache integration's ``native_client`` (which operates on host ``memoryview``s):
the vLLM connector hands GPU device pointers, so we call the C ABI with
``c_void_p(int)`` directly.

Validated facts (hd04 H100 + IB, nvidia-peermem loaded):
  * ``register_memory`` accepts a CUDA device pointer -> GPUDirect MR.
  * GPU buffers MUST go through the batch path (``batch_get``); the single
    ``dfkv_get_auto`` computes a CRC over the destination on the CPU and
    segfaults on device memory.
  * The member address MUST point at the server's RDMA bootstrap port
    (``--rdma-port``), not the main ``--port``, when ``DFKV_RDMA=1``.
"""
import ctypes
from typing import Optional, Sequence

from ._cabi import load_lib

c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_uint64 = ctypes.c_uint64
c_uint32 = ctypes.c_uint32
c_int = ctypes.c_int

# dfkv geometry guard fields (src/value_header.h). Default 0 = no geometry guard;
# isolation is by model_hash + key namespace. A KV pool sharing one model_hash
# must share kv-cache-dtype / page-size / layout (see dfkv sharing-safety notes).
_GEOMETRY_ZEROS = (0,) * 8


class DfkvDeviceClient:
    """Thin device-pointer wrapper over libdfkv.so for the vLLM connector."""

    def __init__(
        self,
        members: str = "",
        model_hash: int = 0,
        lib_path: Optional[str] = None,
        batch_concurrency: int = 8,
        geometry: Sequence[int] = _GEOMETRY_ZEROS,
        mds_endpoints: str = "",
        mds_group: str = "default",
        mds_poll_ms: int = 3000,
    ):
        if len(geometry) != 8:
            raise ValueError("geometry must have exactly 8 fields")
        if not members and not mds_endpoints:
            raise ValueError(
                "DfkvDeviceClient needs 'mds_endpoints' (MDS discovery) or "
                "'members' (static list)"
            )
        self._lib = load_lib(lib_path)
        # MDS path opens with an empty/seed member list and lets discovery build
        # the ring; the static path opens directly with the given members.
        self._h = self._lib.dfkv_open(
            members.encode(),
            c_uint64(model_hash & 0xFFFFFFFFFFFFFFFF),
            *(c_uint32(int(g) & 0xFFFFFFFF) for g in geometry),
        )
        if not self._h:
            raise RuntimeError(
                f"dfkv_open failed (members={members!r}, mds={mds_endpoints!r})"
            )
        self._lib.dfkv_set_batch_concurrency(self._h, c_uint64(batch_concurrency))
        if mds_endpoints:
            rc = self._lib.dfkv_start_mds_discovery(
                self._h, mds_endpoints.encode(), mds_group.encode(), int(mds_poll_ms)
            )
            if rc != 0:
                raise RuntimeError(
                    f"dfkv_start_mds_discovery failed (rc={rc}, "
                    f"mds={mds_endpoints!r}, group={mds_group!r})"
                )

    @property
    def transport_mode(self) -> str:
        return self._lib.dfkv_transport_mode(self._h).decode()

    def register_memory(self, base: int, size: int) -> None:
        """Register a (host or GPU device) region as an RDMA MR. One call per
        contiguous KV-cache storage region; later put/get reference offsets."""
        rc = self._lib.dfkv_register_memory(self._h, c_void_p(base), c_uint64(size))
        if rc != 0:
            raise RuntimeError(
                f"dfkv_register_memory(base={base:#x}, size={size}) rc={rc}"
            )

    def batch_put(
        self, keys: Sequence[str], ptrs: Sequence[int], sizes: Sequence[int]
    ) -> list:
        """Store ``keys[i]`` from buffer ``ptrs[i]`` (device or host) of
        ``sizes[i]`` bytes. Returns per-key status (``0`` = ok, ``1`` = failed).

        NOTE: the dfkv C ABI's ``dfkv_batch_put`` sets ``out[i]=1`` for SUCCESS
        and ``0`` for failure (matching batch_get/batch_exist, but the OPPOSITE
        of the single ``dfkv_put`` which returns 0=ok). We normalize here to the
        connector's "0 = ok" convention so callers don't have to know this."""
        n = len(keys)
        karr = (c_char_p * n)(*[k.encode() for k in keys])
        parr = (c_void_p * n)(*[c_void_p(p) for p in ptrs])
        sarr = (c_uint64 * n)(*sizes)
        out = (c_int * n)()
        rc = self._lib.dfkv_batch_put(self._h, karr, parr, sarr, n, out)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_put rc={rc}")
        return [0 if ok == 1 else 1 for ok in out]

    def batch_get(
        self, keys: Sequence[str], ptrs: Sequence[int], caps: Sequence[int]
    ):
        """Load ``keys[i]`` into buffer ``ptrs[i]`` (device or host, capacity
        ``caps[i]``). Returns ``(hits, lengths)``: ``hits[i]`` is 1 on hit, 0 on
        miss; ``lengths[i]`` is the true stored byte length on hit."""
        n = len(keys)
        karr = (c_char_p * n)(*[k.encode() for k in keys])
        parr = (c_void_p * n)(*[c_void_p(p) for p in ptrs])
        carr = (c_uint64 * n)(*caps)
        out_hit = (c_int * n)()
        out_len = (c_uint64 * n)()
        rc = self._lib.dfkv_batch_get_auto(
            self._h, karr, parr, carr, n, out_hit, out_len
        )
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_get_auto rc={rc}")
        return list(out_hit), list(out_len)

    def batch_put_sg(
        self,
        keys: Sequence[str],
        seg_ptrs: Sequence[Sequence[int]],
        seg_sizes: Sequence[Sequence[int]],
    ) -> list:
        """Scatter-gather put: ``keys[i]`` stores the in-order concatenation of
        the buffers ``seg_ptrs[i][..]`` (device pointers) of ``seg_sizes[i][..]``
        bytes, as ONE dfkv key + one RDMA multi-SGE op. Each key takes <=29 segs
        (max_sge-1); the C ABI reports a >29 key failed. Returns per-key status
        (``0`` = ok, ``1`` = failed), same "0=ok" normalization as ``batch_put``."""
        n = len(keys)
        karr = (c_char_p * n)(*[k.encode() for k in keys])
        # Keep the per-key inner arrays alive for the duration of the call.
        inner_p = [(c_void_p * len(p))(*[c_void_p(x) for x in p]) for p in seg_ptrs]
        inner_s = [(c_uint64 * len(s))(*s) for s in seg_sizes]
        parr = (ctypes.POINTER(c_void_p) * n)(
            *[ctypes.cast(a, ctypes.POINTER(c_void_p)) for a in inner_p]
        )
        sarr = (ctypes.POINTER(c_uint64) * n)(
            *[ctypes.cast(a, ctypes.POINTER(c_uint64)) for a in inner_s]
        )
        narr = (c_int * n)(*[len(p) for p in seg_ptrs])
        out = (c_int * n)()
        rc = self._lib.dfkv_batch_put_sg(self._h, karr, parr, sarr, narr, n, out)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_put_sg rc={rc}")
        return [0 if ok == 1 else 1 for ok in out]

    def batch_get_auto_sg(
        self,
        keys: Sequence[str],
        seg_ptrs: Sequence[Sequence[int]],
        seg_caps: Sequence[Sequence[int]],
    ):
        """Scatter-gather get: ``keys[i]``'s stored blob is scattered in order
        across destination buffers ``seg_ptrs[i][..]`` of capacity
        ``seg_caps[i][..]`` (the segment sizes define the split). Returns
        ``(hits, lengths)`` where ``lengths[i]`` is the total stored bytes."""
        n = len(keys)
        karr = (c_char_p * n)(*[k.encode() for k in keys])
        inner_p = [(c_void_p * len(p))(*[c_void_p(x) for x in p]) for p in seg_ptrs]
        inner_c = [(c_uint64 * len(c))(*c) for c in seg_caps]
        parr = (ctypes.POINTER(c_void_p) * n)(
            *[ctypes.cast(a, ctypes.POINTER(c_void_p)) for a in inner_p]
        )
        carr = (ctypes.POINTER(c_uint64) * n)(
            *[ctypes.cast(a, ctypes.POINTER(c_uint64)) for a in inner_c]
        )
        narr = (c_int * n)(*[len(p) for p in seg_ptrs])
        out_hit = (c_int * n)()
        out_len = (c_uint64 * n)()
        rc = self._lib.dfkv_batch_get_auto_sg(
            self._h, karr, parr, carr, narr, n, out_hit, out_len
        )
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_get_auto_sg rc={rc}")
        return list(out_hit), list(out_len)

    def batch_exist(self, keys: Sequence[str]) -> list:
        """Return ``[1|0]`` per key (1 = present in the cache)."""
        n = len(keys)
        karr = (c_char_p * n)(*[k.encode() for k in keys])
        out = (c_int * n)()
        rc = self._lib.dfkv_batch_exist(self._h, karr, n, out)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_exist rc={rc}")
        return list(out)

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.dfkv_close(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
