"""GPU/torch-free unit tests for the HiCache L2-bypass (device-direct write) path
in dfkv_hicache.py.

Covers the pieces that do NOT need a live cache node or libdfkv: sub-key
expansion of per-layer device segments (_flatten_device), the exist-gate +
transient-retry semantics of the scatter-gather put (_put_sg_flat, mirroring the
contiguous _put_flat), and the supports_device_transfer() capability gate
(SG symbols present AND max_sg_segs >= layer_num). Uses object.__new__ +
monkeypatched primitives so no client is opened.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
# shim 'sglang' (no torch) first, then the real plugin source dir.
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "integration", "hicache"))

os.environ.setdefault("DFKV_CLIENT_NODE_DEDUP", "0")

import ctypes  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig  # noqa: E402
import dfkv_hicache  # noqa: E402

BUILD = os.environ.get("DFKV_BUILD", os.path.join(HERE, "..", "..", "build"))
SERVER_BIN = os.path.join(BUILD, "dfkv_server")


def _bare(**attrs):
    """A DfkvHiCache with only the attributes a unit needs (no client opened)."""
    obj = object.__new__(dfkv_hicache.DfkvHiCache)
    obj.model = "m"
    obj.tp_rank = 0
    obj.tp_size = 1
    obj.is_mla = True
    obj.pp_rank = 0
    obj.pp_size = 1
    obj.enable_pp = False
    obj._put_retry_recovered = 0
    obj._backup_exist_gate = True
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


class FakeLib:
    def __init__(self, has_sg=True, has_width=True, max_segs=32):
        self._max_segs = max_segs
        if has_sg:
            self.dfkv_batch_put_sg = lambda *a: 0
        if has_width:
            self.dfkv_max_sg_segs = lambda h: self._max_segs


class TestFlattenDevice(unittest.TestCase):
    def test_mla_single_subkey_carries_layer_segments(self):
        obj = _bare(is_mla=True)
        keys = ["h0", "h1"]
        # sub=1; per-page one entry, each a 3-layer segment list.
        seg_ptrs = [[10, 11, 12], [20, 21, 22]]
        seg_sizes = [[100, 100, 100], [100, 100, 100]]
        sub, sks, sp, ss = obj._flatten_device(keys, seg_ptrs, seg_sizes)
        self.assertEqual(sub, 1)
        self.assertEqual(sks, ["m/h0_k", "m/h1_k"])
        self.assertEqual(sp, [[10, 11, 12], [20, 21, 22]])
        self.assertEqual(ss, [[100, 100, 100], [100, 100, 100]])

    def test_mha_expands_k_and_v_subkeys(self):
        obj = _bare(is_mla=False, tp_size=2, tp_rank=1)
        keys = ["h0"]
        seg_ptrs = [[1, 2], [3, 4]]  # entry0=k, entry1=v (sub=2)
        seg_sizes = [[8, 8], [8, 8]]
        sub, sks, sp, ss = obj._flatten_device(keys, seg_ptrs, seg_sizes)
        self.assertEqual(sub, 2)
        self.assertEqual(sks, ["m/h0_2_1_k", "m/h0_2_1_v"])
        self.assertEqual(sp, [[1, 2], [3, 4]])

    def test_shape_mismatch_asserts(self):
        obj = _bare(is_mla=False)  # sub=2
        with self.assertRaises(AssertionError):
            obj._flatten_device(["h0"], [[1, 2]], [[8, 8]])  # only 1 entry, needs 2


class TestPutSgFlat(unittest.TestCase):
    def test_exist_gate_skips_present_subkeys(self):
        obj = _bare()
        obj._batch_exist_flat = lambda sks: [True, True]  # both already in L3
        calls = []
        obj._batch_put_sg = lambda sks, sp, ss: calls.append(sks) or [True] * len(sks)
        res = obj._put_sg_flat(["a", "b"], [[1]], [[8]])
        self.assertEqual(res, [True, True])
        self.assertEqual(calls, [])  # nothing written — gate short-circuited

    def test_transient_failure_recovered_on_retry(self):
        obj = _bare()
        obj._batch_exist_flat = lambda sks: [False]  # not present -> must write
        seq = [[False], [True]]  # first put fails, retry succeeds
        obj._batch_put_sg = lambda sks, sp, ss: seq.pop(0)
        res = obj._put_sg_flat(["a"], [[1]], [[8]])
        self.assertEqual(res, [True])
        self.assertEqual(obj._put_retry_recovered, 1)

    def test_persistent_failure_stays_false(self):
        obj = _bare()
        obj._batch_exist_flat = lambda sks: [False]
        obj._batch_put_sg = lambda sks, sp, ss: [False]  # never succeeds
        res = obj._put_sg_flat(["a"], [[1]], [[8]])
        self.assertEqual(res, [False])
        self.assertEqual(obj._put_retry_recovered, 0)

    def test_gate_disabled_writes_all(self):
        obj = _bare()
        obj._backup_exist_gate = False
        obj._batch_exist_flat = lambda sks: (_ for _ in ()).throw(
            AssertionError("exist must not be called when gate disabled"))
        obj._batch_put_sg = lambda sks, sp, ss: [True] * len(sks)
        self.assertEqual(obj._put_sg_flat(["a", "b"], [[1], [2]], [[8], [8]]),
                         [True, True])


class TestSupportsDeviceTransfer(unittest.TestCase):
    def test_enabled_when_symbols_present_and_width_ok(self):
        obj = _bare(cfg={"layer_num": 32}, _lib=FakeLib(max_segs=32), _h=1)
        self.assertTrue(obj.supports_device_transfer())

    def test_declined_when_width_too_small(self):
        obj = _bare(cfg={"layer_num": 61}, _lib=FakeLib(max_segs=29), _h=1)
        self.assertFalse(obj.supports_device_transfer())

    def test_declined_without_sg_symbol(self):
        obj = _bare(cfg={"layer_num": 4}, _lib=FakeLib(has_sg=False), _h=1)
        self.assertFalse(obj.supports_device_transfer())

    def test_declined_without_layer_num(self):
        obj = _bare(cfg={}, _lib=FakeLib(), _h=1)
        self.assertFalse(obj.supports_device_transfer())


class _NpTensor:
    """numpy-backed stand-in for a torch device tensor (data_ptr/numel/elt size)."""

    def __init__(self, arr):
        self._arr = arr

    def data_ptr(self):
        return self._arr.ctypes.data

    def numel(self):
        return int(self._arr.size)

    def element_size(self):
        return int(self._arr.itemsize)


class FakeLayerFirstMlaDevicePool:
    """Layer-first MLA GPU pool emulated with host numpy per-layer buffers, so the
    device-direct SG write path can be exercised end-to-end without a GPU."""

    def __init__(self, layer_num, size, page_size, kv_cache_dim):
        self.page_size = page_size
        self.kv_cache_dim = kv_cache_dim
        self.use_dsa = False
        slots = size + page_size
        self._np = []
        self.kv_buffer = []
        for L in range(layer_num):
            arr = np.zeros(slots * kv_cache_dim, dtype=np.uint8)
            # Recognizable per-(layer, slot, dim) byte pattern for ordering checks.
            for s in range(slots):
                for d in range(kv_cache_dim):
                    arr[s * kv_cache_dim + d] = (L * 100 + s * 10 + d) % 256
            self._np.append(arr)
            self.kv_buffer.append(_NpTensor(arr))

    def layer_page_bytes(self, layer, slot0, npages_slots):
        off = slot0 * self.kv_cache_dim
        return bytes(self._np[layer][off:off + npages_slots * self.kv_cache_dim])


@unittest.skipUnless(
    os.path.exists(SERVER_BIN)
    and hasattr(dfkv_hicache._load_lib(os.environ.get("DFKV_LIB")), "dfkv_batch_put_sg"),
    "needs dfkv_server + a libdfkv.so with dfkv_batch_put_sg",
)
class TestDeviceDirectEndToEnd(unittest.TestCase):
    """Drives batch_set_v1_device against a real cache node and reads the stored
    blob back, proving (a) the write path works and (b) the stored blob is
    LAYER-major — the transpose of a page-first host read (the load-bearing
    correctness limitation for increment 1)."""

    LAYER_NUM = 3
    PAGE_SIZE = 4
    KV_DIM = 5  # bytes/token/layer (dtype itemsize 1)

    @classmethod
    def setUpClass(cls):
        cls.procs = []

    @classmethod
    def tearDownClass(cls):
        for p in cls.procs:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()

    def _node(self, tag):
        d = tempfile.mkdtemp(prefix=f"dfkv_dd_{tag}_")
        p = subprocess.Popen(
            [SERVER_BIN, "--dir", d, "--port", "0", "--cap", str(1 << 30)],
            stdout=subprocess.PIPE, text=True)
        line = p.stdout.readline().strip()
        assert line.startswith("PORT "), f"bad server greeting: {line!r}"
        self.procs.append(p)
        return f"{tag}=127.0.0.1:{int(line.split()[1])}"

    def _plugin(self, members):
        cfg = HiCacheStorageConfig(
            tp_rank=0, tp_size=1, is_mla_model=True, is_page_first_layout=False,
            model_name="dd-mla", pp_rank=0, pp_size=1,
            extra_config={
                "members": members, "model_hash": 0x51, "dtype_tag": 0,
                "page_size": self.PAGE_SIZE, "layer_num": self.LAYER_NUM,
                "head_num": 1, "head_dim": self.KV_DIM, "interface_v1": 1,
            })
        return dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)

    def test_sg_write_stores_layer_major_blob(self):
        st = self._plugin(self._node("dd"))
        pool = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        st.register_mem_pool_device(pool)
        self.assertTrue(st.supports_device_transfer())

        page_hash = "deadbeef"
        device_indices = list(range(0, self.PAGE_SIZE))  # one page at slot 0
        res = st.batch_set_v1_device([page_hash], device_indices)
        self.assertEqual(res, [True])

        # Read the stored blob back contiguously (what the stock host get does).
        sk = st._keys(page_hash)[0]
        cap = self.LAYER_NUM * self.PAGE_SIZE * self.KV_DIM
        buf = ctypes.create_string_buffer(cap)
        rc = st._lib.dfkv_get(st._h, sk.encode(), ctypes.cast(buf, ctypes.c_void_p),
                              ctypes.c_uint64(cap))
        self.assertEqual(rc, 1, "device-direct page must be retrievable")
        blob = bytes(buf.raw[:cap])

        # The SG-stored blob is the in-order concat of per-layer segments:
        # layer0(page), layer1(page), layer2(page) — LAYER-major.
        layer_major = b"".join(
            pool.layer_page_bytes(L, 0, self.PAGE_SIZE)
            for L in range(self.LAYER_NUM))
        self.assertEqual(blob, layer_major)

        # A page-first host pool would lay the same page out TOKEN-major
        # (tok0[L0..L2], tok1[...], ...). It differs — so an unchanged page-first
        # host read of this device-written page would transpose the bytes. This
        # is the documented increment-1 limitation.
        token_major = bytearray()
        for tok in range(self.PAGE_SIZE):
            for L in range(self.LAYER_NUM):
                base = tok * self.KV_DIM
                token_major += pool._np[L][base:base + self.KV_DIM].tobytes()
        self.assertNotEqual(blob, bytes(token_major),
                            "layer-major vs page-first must differ for layer_num>1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
