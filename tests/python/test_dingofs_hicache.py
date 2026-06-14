"""TDD R5 — DingoFS SGLang HiCache plugin, validated GPU/torch-free.

Spawns real dfkv_server cache nodes, drives the plugin's zero-copy v1 path with
numpy host buffers (CPU), and asserts MLA single-object keying, backup_skip,
batch_exists longest-prefix, and header-mismatch => miss.
"""
import ctypes
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
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))  # <repo>/python

BUILD = os.environ.get("DFKV_BUILD", os.path.join(HERE, "..", "..", "build"))
SERVER_BIN = os.path.join(BUILD, "dfkv_server")

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig  # noqa: E402
import dingofs_hicache  # noqa: E402  (RED until implemented)


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
            })

    def _plugin(self, cfg, pool):
        st = dingofs_hicache.DingoFSHiCache(cfg, cfg.extra_config)
        st.register_mem_pool_host(pool)
        return st

    def test_instantiable_all_abstract_methods_present(self):
        members, _, _ = self._node("inst")
        pool = FakeMlaPool(4, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        self.assertTrue(hasattr(st, "get") and hasattr(st, "set"))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
