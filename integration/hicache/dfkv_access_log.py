# SPDX-License-Identifier: Apache-2.0
"""Per-operation access log for the dfkv SGLang HiCache plugin.

Mirrors the format used by dingofs's vfs / lmcache access log, one line per
operation::

    <asctime>.<ms> <op>(<args>) : <result> <duration_seconds>

e.g.::

    2026-06-15 10:00:01.234 batch_get_v1(r0 3 keys) : hits=3/3 <0.007234>
    2026-06-15 10:00:01.250 set(r0 model/abc_k, 4.00KiB) : ok <0.000821>
    2026-06-15 10:00:01.260 get(r0 model/missing_k, 0B) : miss <0.001012>
    2026-06-15 10:00:01.270 batch_set_v1(r0 1 keys) : FAIL OSError: ... <0.001230>

Off by default. Unlike dingofs (which reads env once at import), dfkv's config
arrives at runtime via the SGLang ``extra_config`` dict, so the logger is built
in :func:`configure`, called from ``DfkvHiCache.__init__``. For each knob the
``extra_config`` value wins, then an env-var fallback, then the built-in
default:

    extra_config key          env fallback                   default
    access_log (bool)         DFKV_ACCESS_LOG_ENABLED         off
    access_log_path (str)     DFKV_ACCESS_LOG_PATH            "" -> stderr
    access_log_threshold_us   DFKV_ACCESS_LOG_THRESHOLD_US    0 (log every call)
    access_log_max_bytes      DFKV_ACCESS_LOG_MAX_BYTES       128*1024*1024 (128MiB)
    access_log_backup_count   DFKV_ACCESS_LOG_BACKUP_COUNT    5

``access_log_path`` supports ``{rank}``/``{model}`` placeholders; if neither is
present the path is auto-suffixed ``.r{rank}`` so the one-process-per-TP-rank
SGLang layout never has two ranks corrupting a shared file.

When a path is set the file is **size-rotated** via a ``RotatingFileHandler``:
at ``max_bytes`` it rolls ``acc.log`` -> ``acc.log.1`` -> ... up to
``backup_count`` backups, then overwrites the oldest. Disk per rank is therefore
bounded by ``max_bytes * (backup_count + 1)`` (default ~768MiB), so an access log
left on forever can never fill the disk. Set ``max_bytes=0`` to disable rotation
and keep one unbounded file (the legacy behavior). Each TP rank owns its own file
(and its own rotation), so there is no cross-process rollover race.

Performance:
  - Disabled: ~tens of ns/call. A frozen _NoopLog singleton is returned, so the
    context-manager protocol skips the timer / arg formatting entirely.
  - Enabled: file writes go through a logging.handlers.QueueHandler with a
    background QueueListener thread, so foreground emit cost is ~3 us (enqueue
    only) — no synchronous write/flush in the hot path.
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Optional

# Module state. configure() (first call in the process) snapshots the launch
# config; apply_hot() (from the hot-config watcher thread, dfkv_hot_config) may
# re-apply it later, so all mutating paths take _lock. access_log() reads
# _ENABLED / _THRESHOLD_US WITHOUT the lock — plain bool/int reads under the GIL;
# a stale read at worst mislabels one op, acceptable for a diagnostic log.
_ENABLED: bool = False
_THRESHOLD_US: int = 0
_logger: Optional[logging.Logger] = None
_listener: Optional[logging.handlers.QueueListener] = None
_configured: bool = False
_lock = threading.Lock()
_sink_want: Optional[tuple] = None   # (path, max_bytes, backup_count) the live sink was built for
_launch_cfg: dict = {}               # snapshot from configure(); apply_hot() layers file overrides on top
_tp_rank: int = 0
_model: str = ""


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _resolve(cfg: dict, key: str, env: str, default: Any) -> Any:
    """extra_config key wins; then env var; then default."""
    if cfg.get(key) is not None:
        return cfg[key]
    if env in os.environ:
        return os.environ[env]
    return default


def _stop_listener(listener) -> None:
    """Stop a QueueListener at most once. Guards against double-stop, which on
    Python 3.9 raises (stop() nulls _thread but doesn't re-check it)."""
    if listener is not None and getattr(listener, "_thread", None) is not None:
        listener.stop()


def _int(cfg: dict, key: str, env: str, default: int) -> int:
    try:
        return int(_resolve(cfg, key, env, default))
    except (TypeError, ValueError):
        return default


def _build_sink(path: str, max_bytes: int, backup_count: int,
                tp_rank: int, model: str) -> None:
    """(Re)build the logger + async QueueListener for the given sink params.

    Tears down any previous listener/handlers first so a hot path change never
    double-writes. Sets _logger / _listener. Caller holds _lock.

    Size-based rotation keeps an enabled log from growing a single file
    unbounded: at max_bytes it rolls acc.log -> acc.log.1 -> ... up to
    backup_count, then overwrites the oldest (disk per rank bounded by
    max_bytes * (backup_count + 1)). max_bytes=0 disables rotation (legacy
    single unbounded file, an escape hatch).
    """
    global _logger, _listener
    if path:
        if "{rank}" in path or "{model}" in path:
            path = path.format(rank=tp_rank, model=model)
        else:
            # auto-suffix so per-rank processes never share one file:
            # access.log -> access.log.r0, access.log.r1, ...
            path = f"{path}.r{tp_rank}"

    if path:
        try:
            if max_bytes > 0:
                sink: logging.Handler = logging.handlers.RotatingFileHandler(
                    path, mode="a", maxBytes=max_bytes, backupCount=backup_count)
            else:
                sink = logging.FileHandler(path, mode="a")  # rotation disabled
        except OSError as exc:
            sys.stderr.write(
                f"[dfkv.access] cannot open {path!r}: {exc}; "
                f"falling back to stderr\n"
            )
            sink = logging.StreamHandler(sys.stderr)
    else:
        sink = logging.StreamHandler(sys.stderr)
    sink.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )

    # Tear down a prior listener/handlers (a hot path/rotation change rebuilds).
    _stop_listener(_listener)
    _listener = None
    log = logging.getLogger("dfkv.access")
    log.setLevel(logging.INFO)
    log.propagate = False  # keep out of root logger / SGLang's stack
    for h in list(log.handlers):
        log.removeHandler(h)

    # Unbounded queue: enqueueing never blocks; if the listener can't keep up
    # we'd rather grow memory than stall the hot path. In practice the queue
    # stays empty because emit is ~10 us and ops are ~ms.
    q: "queue.Queue[logging.LogRecord]" = queue.Queue(-1)
    _listener = logging.handlers.QueueListener(q, sink, respect_handler_level=False)
    _listener.start()
    atexit.register(_stop_listener, _listener)
    log.addHandler(logging.handlers.QueueHandler(q))
    _logger = log


def _apply(cfg: Optional[dict], tp_rank: int, model: str) -> None:
    """Resolve the access-log knobs from cfg and (re)apply them. Caller holds
    _lock. Enables/disables live; builds the sink lazily on first enable and
    rebuilds only when path/rotation params change."""
    global _ENABLED, _THRESHOLD_US, _sink_want
    cfg = cfg or {}
    _THRESHOLD_US = max(0, _int(cfg, "access_log_threshold_us",
                                "DFKV_ACCESS_LOG_THRESHOLD_US", 0))
    if not _truthy(_resolve(cfg, "access_log", "DFKV_ACCESS_LOG_ENABLED", False)):
        _ENABLED = False   # keep the sink around for a cheap re-enable
        return
    path = str(_resolve(cfg, "access_log_path", "DFKV_ACCESS_LOG_PATH", "")).strip()
    max_bytes = _int(cfg, "access_log_max_bytes", "DFKV_ACCESS_LOG_MAX_BYTES",
                     128 * 1024 * 1024)
    backup_count = _int(cfg, "access_log_backup_count",
                        "DFKV_ACCESS_LOG_BACKUP_COUNT", 5)
    if max_bytes < 0:
        max_bytes = 0
    if backup_count < 0:
        backup_count = 0
    want = (path, max_bytes, backup_count)
    if _logger is None or _sink_want != want:
        _build_sink(path, max_bytes, backup_count, tp_rank, model)
        _sink_want = want
    _ENABLED = True


def configure(cfg: Optional[dict], tp_rank: int = 0, model: str = "") -> None:
    """Initialize the access logger from an extra_config dict (runtime).

    Idempotent: the first call in the process snapshots the launch config and
    applies it; later calls are no-ops, so it is safe to call from every
    ``DfkvHiCache.__init__``. The snapshot lets :func:`apply_hot` (driven by the
    dfkv_hot_config file watcher) layer live file overrides on top and revert
    cleanly to launch behavior when the control file is removed.
    """
    global _configured, _launch_cfg, _tp_rank, _model
    with _lock:
        if _configured:
            return
        _configured = True
        _launch_cfg = dict(cfg or {})
        _tp_rank = tp_rank
        _model = model
        _apply(_launch_cfg, tp_rank, model)


def apply_hot(overrides: Optional[dict]) -> None:
    """Re-apply access-log knobs from a dict of live control-file overrides,
    layered over the launch config (absent key -> launch default, so deleting
    the control file reverts everything). Thread-safe; called by the
    dfkv_hot_config watcher. No-op before :func:`configure` has run."""
    with _lock:
        if not _configured:
            return
        merged = dict(_launch_cfg)
        if overrides:
            merged.update(overrides)
        _apply(merged, _tp_rank, _model)


def is_enabled() -> bool:
    return _ENABLED


# ---------------------------------------------------------------------------
# Arg formatting helpers (kept here so callers' lambdas stay one-liners).
# ---------------------------------------------------------------------------

def fmt_bytes(n: int) -> str:
    """Pretty-print a byte count for access log args."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.2f}KiB"
    return f"{n / 1024 / 1024:.2f}MiB"


def fmt_pools(transfers) -> str:
    """Summarize v2 pool transfers, e.g. 'kv:3,extra:3'."""
    return ",".join(f"{tr.name}:{len(tr.keys or [])}" for tr in (transfers or []))


def fmt_pool_results(res: dict) -> str:
    """Summarize a v2 result dict, e.g. 'kv ok=3/3, extra ok=3/3'."""
    return ", ".join(f"{name} ok={sum(rs)}/{len(rs)}" for name, rs in res.items())


# ---------------------------------------------------------------------------
# Real (enabled) context manager — keeps a timer and emits on exit.
# ---------------------------------------------------------------------------

class _RealLog:
    __slots__ = ("_op", "_args_fn", "_start", "result")

    def __init__(self, op: str, args_fn: Callable[[], str]) -> None:
        self._op = op
        self._args_fn = args_fn
        self._start = 0.0
        self.result: str = "OK"

    def __enter__(self) -> "_RealLog":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.perf_counter() - self._start
        if exc_type is not None:
            self.result = f"FAIL {exc_type.__name__}: {exc_val}"
        if duration * 1_000_000 >= _THRESHOLD_US and _logger is not None:
            _logger.info("%s(%s) : %s <%.6f>",
                         self._op, self._args_fn(), self.result, duration)
        return False  # don't swallow exceptions


# ---------------------------------------------------------------------------
# Noop singleton — what disabled access_log returns. ~tens of ns / call.
# ---------------------------------------------------------------------------

class _NoopLog:
    """Frozen singleton used when the access log is disabled.

    Implements just enough of the context-manager protocol to be a drop-in for
    _RealLog. Has a writable `result` attribute that nobody reads.
    """
    __slots__ = ()

    result: Any = "OK"  # class attribute; .result = ... below is a no-op

    def __enter__(self) -> "_NoopLog":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __setattr__(self, name: str, value: Any) -> None:
        # Allow `r.result = "..."` from callers without raising; just drop it.
        pass


_NOOP = _NoopLog()


# ---------------------------------------------------------------------------
# Public entry: cheap singleton when disabled, real timer when enabled.
# ---------------------------------------------------------------------------

def access_log(op: str = "", args_fn: Callable[[], str] = lambda: ""):
    """Cheap when disabled (returns frozen singleton); times when enabled."""
    if _ENABLED:
        return _RealLog(op, args_fn)
    return _NOOP
