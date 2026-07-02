"""TDD R5 — DingoFS SGLang HiCache plugin, validated GPU/torch-free.

Spawns real dfkv_server cache nodes, drives the plugin's zero-copy v1 path with
numpy host buffers (CPU), and asserts MLA single-object keying, backup_skip,
batch_exists longest-prefix, and header-mismatch => miss.
"""
import ctypes
import logging
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# shim 'sglang' (no torch) first on path, then the real plugin source dir.
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "integration", "hicache"))  # <repo>/integration/hicache

BUILD = os.environ.get("DFKV_BUILD", os.path.join(HERE, "..", "..", "build"))
SERVER_BIN = os.path.join(BUILD, "dfkv_server")

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig  # noqa: E402
import dfkv_hicache  # noqa: E402  (RED until implemented)
import dfkv_access_log as alog  # noqa: E402


class FakeMlaPool:
    """Stand-in for MLATokenToKVPoolHost: one packed latent object per page."""

    def __init__(self, num_pages, page_bytes, page_size=64):
        self.page_size = page_size
        self.page_bytes = page_bytes
        self._num_pages = num_pages
        self.buf = np.zeros(num_pages * page_bytes, dtype=np.uint8)
        self._base = self.buf.ctypes.data

    def get_ksize_per_token(self):
        return self.page_bytes // self.page_size

    def get_page_buffer_meta(self, host_indices):
        # MLA: one pointer + size per page; host_indices stride = page_size.
        ptrs, sizes = [], []
        for i in range(0, len(host_indices), self.page_size):
            page_idx = host_indices[i] // self.page_size
            ptrs.append(self._base + page_idx * self.page_bytes)
            sizes.append(self.page_bytes)
        return ptrs, sizes

    def fill_page(self, page_idx, byte):
        s = page_idx * self.page_bytes
        self.buf[s:s + self.page_bytes] = byte

    def page_bytes_at(self, page_idx):
        s = page_idx * self.page_bytes
        return bytes(self.buf[s:s + self.page_bytes])

    def zero(self):
        self.buf[:] = 0


class FlatBuf:
    """Mimics a torch flat host page tensor (data_ptr/numel/element_size) over a
    numpy buffer, for exercising the generic get/batch_get path without torch."""

    def __init__(self, nbytes):
        self.arr = np.zeros(nbytes, dtype=np.uint8)

    def data_ptr(self):
        return self.arr.ctypes.data

    def numel(self):
        return int(self.arr.size)

    def element_size(self):
        return int(self.arr.itemsize)

    def tobytes(self):
        return bytes(self.arr)


class RegistrablePool(FakeMlaPool):
    """Fake v2 pool with a tensor-like backing buffer attribute."""

    def __init__(self, num_pages, page_bytes, page_size=64):
        super().__init__(num_pages, page_bytes, page_size)
        self.data_buffer = FlatBuf(num_pages * page_bytes)


class FakeLogicalAnchorPool:
    """Stand-in for SGLang's LogicalHostPool: the primary "kv" pool of a
    V4/DSA-compressed model (e.g. GLM-5.2). Holds NO KV buffer, so its
    get_page_buffer_meta() returns None — the real KV lives in side-pools moved
    over the v2 PoolTransfer path. The v1 anchor call must no-op on this."""

    def __init__(self, page_size=64):
        self.page_size = page_size

    def get_page_buffer_meta(self, host_indices):
        return None


def _spawn_node(tag):
    d = tempfile.mkdtemp(prefix=f"dfkv_py_{tag}_")
    p = subprocess.Popen([SERVER_BIN, "--dir", d, "--port", "0", "--cap", str(1 << 30)],
                         stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline().strip()
    assert line.startswith("PORT "), f"bad server greeting: {line!r}"
    port = int(line.split()[1])
    return p, d, port


def _count_objects(node_dir):
    n = 0
    for root, _dirs, files in os.walk(os.path.join(node_dir, "blocks")):
        n += sum(1 for f in files if not f.endswith(".tmp"))
    return n


class DingoFSHiCacheTest(unittest.TestCase):
    PAGE_SIZE = 64
    PAGE_BYTES = 4096  # small page for fast tests

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
        p, d, port = _spawn_node(tag)
        self.procs.append(p)
        return f"{tag}=127.0.0.1:{port}", port, d

    def _cfg(self, members, tp_rank=0, tp_size=8, page_size=64, model="glm-5.1"):
        return HiCacheStorageConfig(
            tp_rank=tp_rank, tp_size=tp_size, is_mla_model=True,
            is_page_first_layout=False, model_name=model,
            extra_config={
                "members": members, "model_hash": 0x51,
                "dtype_tag": 0x46384534, "page_size": page_size,
                "layer_num": 78, "head_num": 1, "head_dim": 576,
                "interface_v1": 1,
            })

    def _plugin(self, cfg, pool):
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        st.register_mem_pool_host(pool)
        return st

    def test_instantiable_all_abstract_methods_present(self):
        members, _, _ = self._node("inst")
        pool = FakeMlaPool(4, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        self.assertTrue(hasattr(st, "get") and hasattr(st, "set"))

    def test_requires_interface_v1(self):
        # Missing interface_v1 must fail fast (enforces the zero-copy deploy
        # contract; otherwise SGLang silently uses the slower generic copy path).
        cfg = self._cfg("n=127.0.0.1:1")
        cfg.extra_config.pop("interface_v1")
        with self.assertRaises(ValueError):
            dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)

    def test_rdma_depth_extra_config_sets_env(self):
        # extra_config rdma_depth must propagate to DFKV_RDMA_DEPTH so the C client
        # builds its transport with write pipelining (#1). Set before dfkv_open.
        members, _, _ = self._node("rdepth")
        os.environ.pop("DFKV_RDMA_DEPTH", None)
        try:
            cfg = self._cfg(members)
            cfg.extra_config["rdma_depth"] = 8
            dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
            self.assertEqual(os.environ.get("DFKV_RDMA_DEPTH"), "8")
        finally:
            os.environ.pop("DFKV_RDMA_DEPTH", None)

    def _rail_for(self, members, rails_csv, tp_rank, tp_size):
        saved = os.environ.get("DFKV_RDMA_DEV")
        os.environ["DFKV_RDMA_DEV"] = rails_csv
        os.environ.pop("DFKV_RDMA_NUMA", None)
        try:
            cfg = self._cfg(members, tp_rank=tp_rank, tp_size=tp_size)
            cfg.extra_config["rail_affinity"] = True
            dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
            return os.environ.get("DFKV_RDMA_DEV"), os.environ.get("DFKV_RDMA_NUMA")
        finally:
            os.environ.pop("DFKV_RDMA_NUMA", None)
            if saved is None:
                os.environ.pop("DFKV_RDMA_DEV", None)
            else:
                os.environ["DFKV_RDMA_DEV"] = saved

    def test_rail_affinity_deprecated_is_noop_8rails(self):
        # rail_affinity is DEPRECATED: it must NOT narrow DFKV_RDMA_DEV anymore.
        # The full multi-rail list is preserved (the C++ client now selects a
        # NUMA-local rail per connection). It also no longer forces DFKV_RDMA_NUMA.
        members, _, _ = self._node("rail8")
        rails = "ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
        dev, numa = self._rail_for(members, rails, tp_rank=3, tp_size=8)
        self.assertEqual(dev, rails)   # unchanged: no per-rank narrowing
        self.assertIsNone(numa)        # rail_affinity no longer sets DFKV_RDMA_NUMA

    def test_rail_affinity_deprecated_is_noop_2rails(self):
        # Regardless of tp_rank, the full rail list survives (no DP-attention collapse).
        members, _, _ = self._node("rail2")
        rails = "ibA,ibB"
        self.assertEqual(self._rail_for(members, rails, tp_rank=2, tp_size=8)[0], rails)
        self.assertEqual(self._rail_for(members, rails, tp_rank=5, tp_size=8)[0], rails)
        self.assertEqual(self._rail_for(members, rails, tp_rank=7, tp_size=8)[0], rails)

    def test_rdma_numa_extra_config_sets_env(self):
        # The new rdma_numa knob opts into client-side NUMA-aware rail selection
        # by setting DFKV_RDMA_NUMA=1 (replaces rail_affinity's old side effect).
        members, _, _ = self._node("rnuma")
        os.environ.pop("DFKV_RDMA_NUMA", None)
        try:
            cfg = self._cfg(members)
            cfg.extra_config["rdma_numa"] = True
            dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
            self.assertEqual(os.environ.get("DFKV_RDMA_NUMA"), "1")
        finally:
            os.environ.pop("DFKV_RDMA_NUMA", None)

    def test_transport_mode_recorded(self):
        members, _, _ = self._node("tmode")
        saved_rdma = os.environ.get("DFKV_RDMA")
        saved_req = os.environ.get("DFKV_REQUIRE_RDMA")
        os.environ.pop("DFKV_RDMA", None)
        os.environ.pop("DFKV_REQUIRE_RDMA", None)
        try:
            cfg = self._cfg(members)
            st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
            self.assertTrue(st.transport_mode.startswith("tcp"), st.transport_mode)
        finally:
            if saved_rdma is None:
                os.environ.pop("DFKV_RDMA", None)
            else:
                os.environ["DFKV_RDMA"] = saved_rdma
            if saved_req is None:
                os.environ.pop("DFKV_REQUIRE_RDMA", None)
            else:
                os.environ["DFKV_REQUIRE_RDMA"] = saved_req

    def test_generic_get_roundtrip(self):
        # Generic (non zero-copy) set/get round-trips a page through dfkv.
        members, _, _ = self._node("gget")
        cfg = self._cfg(members)
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        payload = bytes((i * 7) & 0xFF for i in range(self.PAGE_BYTES))
        self.assertTrue(st.set("g0", payload))
        tgt = FlatBuf(self.PAGE_BYTES)
        self.assertIs(st.get("g0", tgt), tgt)        # hit returns the buffer
        self.assertEqual(tgt.tobytes(), payload)     # bytes round-trip
        self.assertIsNone(st.get("g_missing", FlatBuf(self.PAGE_BYTES)))  # miss

    def test_generic_batch_get(self):
        members, _, _ = self._node("gbget")
        cfg = self._cfg(members)
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        p0, p1 = bytes([1]) * self.PAGE_BYTES, bytes([2]) * self.PAGE_BYTES
        self.assertTrue(st.set("b0", p0))
        self.assertTrue(st.set("b1", p1))
        bufs = [FlatBuf(self.PAGE_BYTES) for _ in range(3)]
        res = st.batch_get(["b0", "b1", "b_miss"], bufs)
        self.assertIs(res[0], bufs[0]); self.assertEqual(bufs[0].tobytes(), p0)
        self.assertIs(res[1], bufs[1]); self.assertEqual(bufs[1].tobytes(), p1)
        self.assertIsNone(res[2])

    def test_batch_set_get_v1_roundtrip(self):
        members, _, _ = self._node("rt")
        pool = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        keys = ["p0", "p1", "p2"]
        for i in range(3):
            pool.fill_page(i, 10 + i)
        host_indices = list(range(3 * self.PAGE_SIZE))  # 3 pages
        self.assertEqual(st.batch_set_v1(keys, host_indices), [True, True, True])
        expected = [pool.page_bytes_at(i) for i in range(3)]
        pool.zero()
        self.assertEqual(st.batch_get_v1(keys, host_indices), [True, True, True])
        for i in range(3):
            self.assertEqual(pool.page_bytes_at(i), expected[i])

    def test_client_metrics_count_set_and_get(self):
        # #5: plugin counts client-side read/write volume (pages/bytes/hits).
        members, _, _ = self._node("metrics")
        pool = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        keys = ["m0", "m1", "m2"]
        host_indices = list(range(3 * self.PAGE_SIZE))
        st.batch_set_v1(keys, host_indices)
        st.batch_get_v1(keys, host_indices)
        m = st._metrics.snapshot()
        self.assertEqual(m["set_calls"], 1)
        self.assertEqual(m["set_pages"], 3)
        self.assertEqual(m["set_ok_pages"], 3)
        self.assertEqual(m["set_bytes"], 3 * self.PAGE_BYTES)
        self.assertEqual(m["get_calls"], 1)
        self.assertEqual(m["get_pages"], 3)
        self.assertEqual(m["get_hit_pages"], 3)
        self.assertEqual(m["get_bytes"], 3 * self.PAGE_BYTES)
        # latency histograms observed one duration per batch call
        self.assertEqual(m["set_observations"], 1)
        self.assertEqual(m["get_observations"], 1)

    def test_mla_writes_single_object_per_page(self):
        members, port, ndir = self._node("mla")
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        keys = ["q0", "q1"]
        st.batch_set_v1(keys, list(range(2 * self.PAGE_SIZE)))
        # one object per page (MLA), not two (no _k/_v split)
        self.assertEqual(_count_objects(ndir), 2)

    def test_backup_skip_nonzero_tp_rank_does_not_write(self):
        members, port, ndir = self._node("skip")
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members, tp_rank=3), pool)  # MLA + rank!=0
        keys = ["r0", "r1"]
        self.assertEqual(st.batch_set_v1(keys, list(range(2 * self.PAGE_SIZE))),
                         [True, True])  # reported success...
        self.assertEqual(_count_objects(ndir), 0)  # ...but nothing written (backup_skip)

    def test_batch_exists_longest_prefix(self):
        members, _, _ = self._node("exist")
        pool = FakeMlaPool(4, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        st.batch_set_v1(["e0", "e1"], list(range(2 * self.PAGE_SIZE)))
        # first 2 exist, 3rd missing -> prefix length 2
        n = st.batch_exists(["e0", "e1", "e2"])
        self.assertEqual(n, 2)

    def test_header_mismatch_is_miss(self):
        members, _, _ = self._node("hdr")
        pool_w = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        pool_w.fill_page(0, 7)
        writer = self._plugin(self._cfg(members, page_size=64), pool_w)
        writer.batch_set_v1(["h0"], list(range(self.PAGE_SIZE)))
        # reader with different geometry (page_size 32) must MISS, not read garbage
        pool_r = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        reader = self._plugin(self._cfg(members, page_size=32), pool_r)
        self.assertEqual(reader.batch_get_v1(["h0"], list(range(self.PAGE_SIZE))),
                         [False])

    def test_batch_v2_multi_pool_roundtrip(self):
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer
        members, _, _ = self._node("v2")
        cfg = self._cfg(members)
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        kv = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        ex = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        st.register_mem_host_pool_v2(kv, "kv")
        st.register_mem_host_pool_v2(ex, "extra")
        for i in range(3):
            kv.fill_page(i, 20 + i)
            ex.fill_page(i, 50 + i)
        keys = ["a0", "a1", "a2"]
        hi = list(range(3 * self.PAGE_SIZE))
        trs = [PoolTransfer(name="kv", host_indices=hi, keys=keys),
               PoolTransfer(name="extra", host_indices=hi, keys=keys)]
        res = st.batch_set_v2(trs)
        self.assertEqual(res["kv"], [True, True, True])
        self.assertEqual(res["extra"], [True, True, True])
        expk = [kv.page_bytes_at(i) for i in range(3)]
        expe = [ex.page_bytes_at(i) for i in range(3)]
        kv.zero(); ex.zero()
        g = st.batch_get_v2(trs)
        self.assertEqual(g["kv"], [True, True, True])
        self.assertEqual(g["extra"], [True, True, True])
        for i in range(3):
            self.assertEqual(kv.page_bytes_at(i), expk[i])
            self.assertEqual(ex.page_bytes_at(i), expe[i])
        r = st.batch_exists_v2(keys, [PoolTransfer(name="extra", host_indices=hi, keys=keys)])
        self.assertEqual(r.kv_hit_pages, 3)

    def test_v1_logical_anchor_writes_markers_no_crash(self):
        # V4/DSA models (e.g. GLM-5.2): the primary "kv" pool is a logical anchor
        # whose get_page_buffer_meta() returns None. batch_set_v1 must not crash
        # (was: TypeError unpacking None); it writes an empty "kv" marker per page
        # so batch_exists can find the primary prefix, and batch_get_v1 no-ops
        # (all pages "present" so the hybrid controller loads the real side pools).
        members, _, ndir = self._node("anchor")
        st = self._plugin(self._cfg(members, model="glm-5.2"),
                          FakeLogicalAnchorPool(self.PAGE_SIZE))
        keys = ["kv0", "kv1", "kv2"]
        hi = list(range(3 * self.PAGE_SIZE))
        self.assertEqual(st.batch_set_v1(keys, hi), [True, True, True])
        # markers were written (MLA: one object per page) and are discoverable
        self.assertEqual(_count_objects(ndir), 3)
        self.assertEqual(st.batch_exists(keys), 3)
        self.assertEqual(st.batch_exists(keys + ["kv3"]), 3)  # absent page stops prefix
        # anchor load is a no-op but reports all pages complete
        self.assertEqual(st.batch_get_v1(keys, hi), [True, True, True])
        self.assertTrue(st._anchor_noop_warned)

    def test_v2_exists_with_logical_anchor_full_and_shrunk(self):
        # End-to-end multi-pool existence for a logical-anchor (DSA) model: the
        # "kv" markers (v1) + a compressed side pool (v2) together define the hit
        # prefix, and a missing side-pool page shrinks it.
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer
        members, _, _ = self._node("anchorv2")
        st = dfkv_hicache.DfkvHiCache(self._cfg(members, model="glm-5.2"),
                                      self._cfg(members, model="glm-5.2").extra_config)
        st.register_mem_pool_host(FakeLogicalAnchorPool(self.PAGE_SIZE))
        side = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        st.register_mem_host_pool_v2(side, "deepseek_v4_c4")
        keys = ["a0", "a1", "a2"]
        hi = list(range(3 * self.PAGE_SIZE))
        # backup: v1 anchor markers (all 3) + v2 side pool for only the first 2
        st.batch_set_v1(keys, hi)
        st.batch_set_v2([PoolTransfer(name="deepseek_v4_c4",
                                      host_indices=hi[:2 * self.PAGE_SIZE],
                                      keys=keys[:2])])
        trs = [PoolTransfer(name="deepseek_v4_c4", host_indices=hi, keys=keys)]
        # side pool present for 2/3 -> usable prefix shrinks to 2
        self.assertEqual(st.batch_exists_v2(keys, trs).kv_hit_pages, 2)
        # fill the 3rd side-pool page -> full 3-page hit
        st.batch_set_v2([PoolTransfer(name="deepseek_v4_c4",
                                      host_indices=hi[2 * self.PAGE_SIZE:],
                                      keys=keys[2:])])
        self.assertEqual(st.batch_exists_v2(keys, trs).kv_hit_pages, 3)

    def test_v2_exists_trailing_pool_sliding_window(self):
        # TRAILING_PAGES (SWA / Mamba state): only the last N pages of a prefix
        # need this pool. A hit must NOT collapse just because earlier window
        # pages were evicted (the old `all(present)` logic wrongly did).
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer, PoolHitPolicy
        members, _, _ = self._node("swa")
        cfg = self._cfg(members, model="glm-5.2")
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        st.register_mem_pool_host(FakeLogicalAnchorPool(self.PAGE_SIZE))
        swa = FakeMlaPool(4, self.PAGE_BYTES, self.PAGE_SIZE)
        st.register_mem_host_pool_v2(swa, "swa")
        keys = ["s0", "s1", "s2", "s3"]
        hi = list(range(4 * self.PAGE_SIZE))
        st.batch_set_v1(keys, hi)  # 4 "kv" markers -> kv prefix = 4
        # SWA present for only the trailing 2 pages (sliding window = 2)
        st.batch_set_v2([PoolTransfer(name="swa",
                                      host_indices=hi[2 * self.PAGE_SIZE:],
                                      keys=keys[2:])])
        trs = [PoolTransfer(name="swa", host_indices=hi, keys=keys[2:],
                            hit_policy=PoolHitPolicy.TRAILING_PAGES)]
        # trailing window (last 2) present -> full 4-page prefix stays usable
        self.assertEqual(st.batch_exists_v2(keys, trs).kv_hit_pages, 4)
        # but if the trailing window is broken (last page missing), it shrinks
        swa2 = FakeMlaPool(4, self.PAGE_BYTES, self.PAGE_SIZE)
        members2, _, _ = self._node("swa2")
        cfg2 = self._cfg(members2, model="glm-5.2")
        st2 = dfkv_hicache.DfkvHiCache(cfg2, cfg2.extra_config)
        st2.register_mem_pool_host(FakeLogicalAnchorPool(self.PAGE_SIZE))
        st2.register_mem_host_pool_v2(swa2, "swa")
        st2.batch_set_v1(keys, hi)
        # SWA present only for pages 0,1 (window covers a stale tail) -> for a
        # window of 2 the best prefix whose last 2 pages are present is len 2.
        st2.batch_set_v2([PoolTransfer(name="swa",
                                       host_indices=hi[:2 * self.PAGE_SIZE],
                                       keys=keys[:2])])
        self.assertEqual(st2.batch_exists_v2(keys, trs).kv_hit_pages, 2)

    def test_register_mem_host_pool_v2_registers_backing_buffer(self):
        members, _, _ = self._node("v2reg")
        cfg = self._cfg(members)
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        pool = RegistrablePool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        calls = []
        st.register_memory = lambda base, size: calls.append((base, size)) or True
        st.register_mem_host_pool_v2(pool, "extra")
        self.assertIs(st.registered_pools["extra"], pool)
        self.assertEqual(calls, [
            (pool.data_buffer.data_ptr(),
             pool.data_buffer.numel() * pool.data_buffer.element_size())
        ])


def _reset_access_log():
    """Reset the process-global access-log state for test isolation.

    configure() is idempotent per process, so without this every test after the
    first would inherit the first config. Tests legitimately reach into the
    module internals here (there is no production reset API by design)."""
    alog._stop_listener(alog._listener)
    logging.getLogger("dfkv.access").handlers.clear()
    alog._ENABLED = False
    alog._THRESHOLD_US = 0
    alog._logger = None
    alog._listener = None
    alog._configured = False


class DfkvAccessLogTest(unittest.TestCase):
    """Access log on the inherited interface methods: toggle, path, format."""

    PAGE_SIZE = 64
    PAGE_BYTES = 4096

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

    def setUp(self):
        _reset_access_log()
        self.tmp = tempfile.mkdtemp(prefix="dfkv_alog_")

    def tearDown(self):
        _reset_access_log()

    def _node(self, tag):
        p, d, port = _spawn_node(tag)
        self.procs.append(p)
        return f"{tag}=127.0.0.1:{port}"

    def _cfg(self, members, alog_extra, tp_rank=0):
        ec = {
            "members": members, "model_hash": 0x51,
            "dtype_tag": 0x46384534, "page_size": self.PAGE_SIZE,
            "layer_num": 78, "head_num": 1, "head_dim": 576,
            "interface_v1": 1,
        }
        ec.update(alog_extra)
        return HiCacheStorageConfig(
            tp_rank=tp_rank, tp_size=8, is_mla_model=True,
            is_page_first_layout=False, model_name="glm-5.1", extra_config=ec)

    def _plugin(self, cfg, pool=None):
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        if pool is not None:
            st.register_mem_pool_host(pool)
        return st

    def _flush(self):
        # Drain the async queue so emitted lines hit the file before we read it.
        if alog._listener is not None:
            alog._listener.stop()
            alog._listener = None

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def test_disabled_by_default_writes_no_file(self):
        members = self._node("ad")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        # access_log absent -> disabled even though a path is supplied.
        cfg = self._cfg(members, {"access_log_path": path})
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(cfg, pool)
        st.batch_set_v1(["d0", "d1"], list(range(2 * self.PAGE_SIZE)))
        self._flush()
        self.assertFalse(alog.is_enabled())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "acc.0.log")))

    def test_enabled_writes_one_line_per_op(self):
        members = self._node("aw")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path})
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(cfg, pool)
        self.assertTrue(alog.is_enabled())
        hi = list(range(2 * self.PAGE_SIZE))
        st.batch_set_v1(["w0", "w1"], hi)
        st.batch_get_v1(["w0", "w1"], hi)
        st.batch_exists(["w0", "w1", "w_miss"])  # 2 present, 1 missing
        self._flush()
        txt = self._read(os.path.join(self.tmp, "acc.0.log"))
        self.assertIn("init(r0 glm-5.1 tp=0/8 mla=1) : ok static", txt)
        self.assertIn("batch_set_v1(r0 2 keys) : ok 2/2", txt)
        self.assertIn("batch_get_v1(r0 2 keys) : hits=2/2", txt)
        self.assertIn("batch_exists(r0 3 keys) : prefix=2/3", txt)
        self.assertIn("<0.", txt)  # duration token present

    def test_generic_get_set_exists_logged(self):
        members = self._node("ag")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path})
        st = self._plugin(cfg)
        payload = bytes((i * 3) & 0xFF for i in range(self.PAGE_BYTES))
        self.assertTrue(st.set("g0", payload))
        self.assertIs(st.get("g0", FlatBuf(self.PAGE_BYTES)).__class__, FlatBuf)
        self.assertTrue(st.exists("g0"))
        self.assertFalse(st.exists("g_missing"))
        self.assertIsNone(st.get("g_missing", FlatBuf(self.PAGE_BYTES)))
        self._flush()
        txt = self._read(os.path.join(self.tmp, "acc.0.log"))
        self.assertIn("set(r0 g0, 4.00KiB) : ok", txt)
        self.assertIn(": hit", txt)        # get hit
        self.assertIn(": miss", txt)       # get miss
        self.assertIn("exists(r0 g0) : found", txt)
        self.assertIn("exists(r0 g_missing) : not_found", txt)

    def test_auto_rank_suffix_and_backup_skip(self):
        members = self._node("as")
        # no {rank} placeholder -> path is auto-suffixed .r{rank}
        path = os.path.join(self.tmp, "acc.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path},
                        tp_rank=3)  # MLA + rank!=0 -> backup_skip
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(cfg, pool)
        self.assertEqual(st.batch_set_v1(["s0", "s1"], list(range(2 * self.PAGE_SIZE))),
                         [True, True])  # value unchanged by logging
        self._flush()
        suffixed = os.path.join(self.tmp, "acc.log.r3")
        self.assertTrue(os.path.exists(suffixed))
        self.assertIn("batch_set_v1(r3 2 keys) : backup_skip", self._read(suffixed))

    def test_path_placeholder_substituted(self):
        members = self._node("ap")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path},
                        tp_rank=5)
        pool = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(cfg, pool)
        st.batch_set_v1(["x0"], list(range(self.PAGE_SIZE)))
        self._flush()
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "acc.5.log")))

    def test_threshold_filters_fast_ops(self):
        members = self._node("at")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path,
                                  "access_log_threshold_us": 10_000_000})  # 10s
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(cfg, pool)
        hi = list(range(2 * self.PAGE_SIZE))
        st.batch_set_v1(["t0", "t1"], hi)
        st.batch_get_v1(["t0", "t1"], hi)
        self._flush()
        p = os.path.join(self.tmp, "acc.0.log")
        txt = self._read(p) if os.path.exists(p) else ""
        self.assertNotIn("batch_set_v1", txt)
        self.assertNotIn("batch_get_v1", txt)

    def test_env_var_fallback_enables(self):
        members = self._node("ae")
        envp = os.path.join(self.tmp, "env.{rank}.log")
        keys = ("DFKV_ACCESS_LOG_ENABLED", "DFKV_ACCESS_LOG_PATH")
        old = {k: os.environ.get(k) for k in keys}
        os.environ["DFKV_ACCESS_LOG_ENABLED"] = "1"
        os.environ["DFKV_ACCESS_LOG_PATH"] = envp
        try:
            cfg = self._cfg(members, {})  # no extra_config access_log keys
            pool = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
            st = self._plugin(cfg, pool)
            self.assertTrue(alog.is_enabled())
            st.batch_get_v1(["z0"], list(range(self.PAGE_SIZE)))
            self._flush()
            self.assertTrue(os.path.exists(os.path.join(self.tmp, "env.0.log")))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_failures_are_logged_and_reraised(self):
        members = self._node("af")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path})
        st = self._plugin(cfg)
        self.assertFalse(st.set("f0", None))  # value None -> "fail none"

        def _boom(*a, **k):
            raise RuntimeError("boom")
        st._lib.dfkv_put = _boom  # fresh per-plugin CDLL, isolated to this test
        with self.assertRaises(RuntimeError):
            st.set("f1", b"x" * 16)  # exception must propagate, not be swallowed
        self._flush()
        txt = self._read(os.path.join(self.tmp, "acc.0.log"))
        self.assertIn("set(r0 f0, 0B) : fail none", txt)
        self.assertIn("FAIL RuntimeError: boom", txt)


class DfkvAccessLogRotationTest(unittest.TestCase):
    """Size-based rotation of the access-log file (RotatingFileHandler).

    Pure module-level: drives dfkv_access_log.configure + the logger directly, so
    it needs no cache node / libdfkv.so (unlike DfkvAccessLogTest). Codifies that
    an enabled access log no longer grows a single file unbounded."""

    def setUp(self):
        _reset_access_log()
        self.tmp = tempfile.mkdtemp(prefix="dfkv_alog_rot_")

    def tearDown(self):
        _reset_access_log()

    def _flush(self):
        # stop() drains the queue (writing + rolling over remaining records) and
        # joins the listener thread before we inspect the files.
        if alog._listener is not None:
            alog._listener.stop()
            alog._listener = None

    def _emit(self, n, nbytes=200):
        line = "x" * nbytes
        for _ in range(n):
            alog._logger.info("%s", line)

    def test_rotates_by_size_and_caps_backup_count(self):
        path = os.path.join(self.tmp, "acc.{rank}.log")
        alog.configure({"access_log": 1, "access_log_path": path,
                        "access_log_max_bytes": 4096,
                        "access_log_backup_count": 2}, tp_rank=0)
        self.assertTrue(alog.is_enabled())
        base = os.path.join(self.tmp, "acc.0.log")
        self._emit(400)  # ~400 * ~225B = ~90KB >> 4KiB*3 cap -> many rollovers
        self._flush()
        # base + .1 + .2 only; .3 must never appear (backup_count=2 caps it)
        self.assertTrue(os.path.exists(base))
        self.assertTrue(os.path.exists(base + ".1"))
        self.assertTrue(os.path.exists(base + ".2"))
        self.assertFalse(os.path.exists(base + ".3"))
        # disk is bounded: each retained file <= maxBytes (+ one line of slack)
        for suffix in ("", ".1", ".2"):
            self.assertLessEqual(os.path.getsize(base + suffix), 4096 + 512)

    def test_max_bytes_zero_keeps_single_unbounded_file(self):
        # Escape hatch: max_bytes=0 restores the legacy plain-FileHandler behavior
        # (one file, no rotation) for anyone who wants it.
        path = os.path.join(self.tmp, "acc.{rank}.log")
        alog.configure({"access_log": 1, "access_log_path": path,
                        "access_log_max_bytes": 0}, tp_rank=0)
        base = os.path.join(self.tmp, "acc.0.log")
        self._emit(200)
        self._flush()
        self.assertFalse(os.path.exists(base + ".1"))   # rotation disabled
        self.assertGreater(os.path.getsize(base), 4096)  # grew past a rotation size

    def test_defaults_enable_rotation(self):
        # No explicit knobs -> rotation ON by default so disk is always bounded.
        # The 128MiB default isn't reached here, so only the base file exists.
        path = os.path.join(self.tmp, "acc.{rank}.log")
        alog.configure({"access_log": 1, "access_log_path": path}, tp_rank=0)
        import logging.handlers as _h
        sink = alog._listener.handlers[0]  # capture before _flush() nulls listener
        base = os.path.join(self.tmp, "acc.0.log")
        self._emit(50)
        self._flush()
        self.assertIsInstance(sink, _h.RotatingFileHandler)
        self.assertTrue(os.path.exists(base))
        self.assertFalse(os.path.exists(base + ".1"))  # 128MiB default not reached


class ClientStatsPollerTest(unittest.TestCase):
    """Mirrors the C client's Prometheus snapshot onto delta counters. Pure
    module-level: feeds a fake snapshot provider, no node / libdfkv.so."""

    def _poller(self, texts):
        from dfkv_metrics import ClientStatsPoller
        # get_text returns successive snapshots from `texts`, repeating the last
        seq = {"i": 0}

        def get_text():
            i = min(seq["i"], len(texts) - 1)
            seq["i"] += 1
            return texts[i]

        return ClientStatsPoller(get_text, tp_rank=0, interval_s=0.01)

    def test_parse_handles_bare_and_labeled(self):
        from dfkv_metrics import ClientStatsPoller
        text = ("# TYPE dfkv_client_ops_served_total counter\n"
                "dfkv_client_ops_served_total 7\n"
                "dfkv_client_peer_errors_total{peer=\"1.2.3.4:1\"} 3\n")
        vals = ClientStatsPoller._parse(text)
        self.assertEqual(vals.get("dfkv_client_ops_served_total"), 7)
        # per-peer series is not in the mirrored aggregate set
        self.assertNotIn("dfkv_client_peer_errors_total", vals)

    def test_poll_once_accumulates_deltas(self):
        p = self._poller([
            "dfkv_client_ops_served_total 5\ndfkv_client_io_errors_total 1\n",
            "dfkv_client_ops_served_total 8\ndfkv_client_io_errors_total 1\n",
        ])
        p.poll_once()  # first read: delta 5 (from 0)
        t1 = p.totals()
        self.assertEqual(t1["dfkv_client_ops_served_total"], 5)
        self.assertEqual(t1["dfkv_client_io_errors_total"], 1)
        p.poll_once()  # cumulative 8 -> delta +3
        t2 = p.totals()
        self.assertEqual(t2["dfkv_client_ops_served_total"], 8)
        self.assertEqual(t2["dfkv_client_io_errors_total"], 1)  # unchanged

    def test_disabled_interval_starts_no_thread(self):
        from dfkv_metrics import ClientStatsPoller
        p = ClientStatsPoller(lambda: "", tp_rank=0, interval_s=0)
        p.start()
        self.assertIsNone(p._thread)
        p.stop()  # must not crash

    def test_start_stop_lifecycle(self):
        p = self._poller(["dfkv_client_ops_served_total 1\n"])
        p.start()
        self.assertIsNotNone(p._thread)
        p.stop()
        self.assertIsNone(p._thread)


if __name__ == "__main__":
    unittest.main(verbosity=2)
