# SPDX-License-Identifier: Apache-2.0
"""Runtime hot-reload of the access-log knob (dfkv_hot_config + dfkv_access_log).

Pure-Python: no native lib, no cache nodes. Drives dfkv_access_log.apply_hot()
directly and the dfkv_hot_config file watcher end-to-end.
"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "integration", "hicache"))

import dfkv_access_log as alog  # noqa: E402
import dfkv_hot_config as hot   # noqa: E402


def _reset():
    """Reset both modules' process-global state for test isolation."""
    alog._stop_listener(alog._listener)
    import logging
    logging.getLogger("dfkv.access").handlers.clear()
    alog._ENABLED = False
    alog._THRESHOLD_US = 0
    alog._logger = None
    alog._listener = None
    alog._configured = False
    alog._sink_want = None
    alog._launch_cfg = {}
    hot.stop()
    with hot._lock:
        hot._appliers.clear()
        hot._watcher = None
    for k in list(os.environ):
        if k.startswith("DFKV_"):
            del os.environ[k]


class AccessLogHotToggleTest(unittest.TestCase):
    def setUp(self):
        _reset()
        self.tmp = tempfile.mkdtemp(prefix="dfkv_hot_")
        self.logpath = os.path.join(self.tmp, "acc.log")

    def tearDown(self):
        _reset()

    def _emit(self, op):
        with alog.access_log(op, lambda: "x"):
            pass

    def _flush(self):
        if alog._listener is not None:
            alog._listener.stop()
            alog._listener = None

    def _read(self):
        p = self.logpath + ".r0"
        if not os.path.exists(p):
            return ""
        with open(p) as f:
            return f.read()

    def test_hot_enable_then_revert(self):
        # launch disabled
        alog.configure({"access_log": False}, tp_rank=0, model="m")
        self.assertFalse(alog.is_enabled())
        self._emit("get")
        self.assertEqual(self._read(), "")

        # hot enable
        alog.apply_hot({"access_log": True, "access_log_path": self.logpath})
        self.assertTrue(alog.is_enabled())
        self._emit("batch_get")
        self._flush()
        self.assertIn("batch_get", self._read())

        # empty overrides -> revert to launch (disabled)
        alog.apply_hot({})
        self.assertFalse(alog.is_enabled())

    def test_threshold_hot_change(self):
        alog.configure({"access_log": True, "access_log_path": self.logpath},
                       tp_rank=0, model="m")
        # 10s threshold suppresses a sub-ms op
        alog.apply_hot({"access_log": True, "access_log_path": self.logpath,
                        "access_log_threshold_us": 10_000_000})
        n0 = self._read().count("\n")
        self._emit("fast")
        self.assertEqual(self._read().count("\n"), n0)
        # threshold 0 logs again
        alog.apply_hot({"access_log": True, "access_log_path": self.logpath,
                        "access_log_threshold_us": 0})
        self._emit("logged")
        self._flush()
        self.assertIn("logged", self._read())

    def test_apply_hot_before_configure_is_noop(self):
        # no configure() yet -> apply_hot must not enable
        alog.apply_hot({"access_log": True, "access_log_path": self.logpath})
        self.assertFalse(alog.is_enabled())


class WatcherOptInTest(unittest.TestCase):
    def setUp(self):
        _reset()
        self.tmp = tempfile.mkdtemp(prefix="dfkv_hotw_")
        self.logpath = os.path.join(self.tmp, "acc.log")
        self.ctl = os.path.join(self.tmp, "hot.json")

    def tearDown(self):
        _reset()

    def test_watcher_disabled_without_explicit_path(self):
        hot.register("access_log", alog.apply_hot)
        self.assertIsNone(hot.start({}, tp_rank=0))

    def test_watcher_file_lifecycle(self):
        alog.configure({"access_log": False}, tp_rank=0, model="m")
        hot.register("access_log", alog.apply_hot)
        os.environ["DFKV_HOT_CONFIG"] = self.ctl
        os.environ["DFKV_HOT_CONFIG_POLL_S"] = "0.2"
        w = hot.start({}, tp_rank=0)
        self.assertIsNotNone(w)
        time.sleep(0.4)  # first tick, file absent -> disabled
        self.assertFalse(alog.is_enabled())

        with open(self.ctl, "w") as f:
            json.dump({"access_log": True, "access_log_path": self.logpath}, f)
        self._wait(lambda: alog.is_enabled(), 3.0)
        self.assertTrue(alog.is_enabled())

        os.remove(self.ctl)
        self._wait(lambda: not alog.is_enabled(), 3.0)
        self.assertFalse(alog.is_enabled())

    def _wait(self, pred, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return
            time.sleep(0.1)


if __name__ == "__main__":
    unittest.main()
