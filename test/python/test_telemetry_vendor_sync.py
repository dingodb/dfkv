"""Drift guard: the vendored telemetry copies inside the vLLM / LMCache pip
connectors must stay byte-identical to the canonical integration/hicache/dfkv_telemetry/.

If this fails, run ``deploy/sync_telemetry.sh`` (you edited the canonical files
but not the vendored copies, or vice-versa).
"""
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

CANON = os.path.join(ROOT, "integration", "hicache", "dfkv_telemetry")
VENDORED = [
    os.path.join(ROOT, "integration", "vllm", "src", "dfkv_vllm", "_telemetry"),
    os.path.join(ROOT, "integration", "lmcache", "src", "dfkv_connector", "_telemetry"),
]
FILES = ["__init__.py", "config.py", "metrics_push.py", "otlp_json.py",
         "tracing.py", "otlp_traces.py"]


class VendorSyncTest(unittest.TestCase):
    def test_vendored_copies_are_byte_identical(self):
        for f in FILES:
            with open(os.path.join(CANON, f), "rb") as fh:
                canon = fh.read()
            for vdir in VENDORED:
                path = os.path.join(vdir, f)
                self.assertTrue(os.path.exists(path), "missing vendored file: " + path)
                with open(path, "rb") as fh:
                    got = fh.read()
                self.assertEqual(
                    got, canon,
                    "{} drifted from canonical; run deploy/sync_telemetry.sh".format(path))


if __name__ == "__main__":
    unittest.main()
