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

``access_log_path`` supports ``{rank}``/``{model}`` placeholders; if neither is
present the path is auto-suffixed ``.r{rank}`` so the one-process-per-TP-rank
SGLang layout never has two ranks corrupting a shared file.

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
import time
from typing import Any, Callable, Optional

# Module state, populated by configure() (first call in the process wins).
_ENABLED: bool = False
_THRESHOLD_US: int = 0
_logger: Optional[logging.Logger] = None
_listener: Optional[logging.handlers.QueueListener] = None
_configured: bool = False


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


def configure(cfg: Optional[dict], tp_rank: int = 0, model: str = "") -> None:
    """Initialize the access logger from an extra_config dict (runtime).

    Idempotent: the first call in the process builds (or skips) the logger and
    later calls are no-ops, so it is safe to call from every
    ``DfkvHiCache.__init__``. In SGLang each process owns one backend instance,
    so divergent configs in one process don't occur in practice.
    """
    global _ENABLED, _THRESHOLD_US, _logger, _listener, _configured
    if _configured:
        return
    _configured = True

    cfg = cfg or {}
    enabled = _truthy(_resolve(cfg, "access_log", "DFKV_ACCESS_LOG_ENABLED", False))
    path = str(_resolve(cfg, "access_log_path", "DFKV_ACCESS_LOG_PATH", "")).strip()
    try:
        _THRESHOLD_US = int(_resolve(cfg, "access_log_threshold_us",
                                     "DFKV_ACCESS_LOG_THRESHOLD_US", 0))
    except (TypeError, ValueError):
        _THRESHOLD_US = 0

    if not enabled:
        _ENABLED = False
        return

    if path:
        if "{rank}" in path or "{model}" in path:
            path = path.format(rank=tp_rank, model=model)
        else:
            # auto-suffix so per-rank processes never share one file:
            # access.log -> access.log.r0, access.log.r1, ...
            path = f"{path}.r{tp_rank}"

    log = logging.getLogger("dfkv.access")
    log.setLevel(logging.INFO)
    log.propagate = False  # keep out of root logger / SGLang's stack
    if log.handlers:  # already configured (defensive, e.g. on re-import)
        _logger = log
        _ENABLED = True
        return

    if path:
        try:
            sink: logging.Handler = logging.FileHandler(path, mode="a")
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

    # Unbounded queue: enqueueing never blocks; if the listener can't keep up
    # we'd rather grow memory than stall the hot path. In practice the queue
    # stays empty because emit is ~10 us and ops are ~ms.
    q: "queue.Queue[logging.LogRecord]" = queue.Queue(-1)
    _listener = logging.handlers.QueueListener(q, sink, respect_handler_level=False)
    _listener.start()
    atexit.register(_stop_listener, _listener)

    log.addHandler(logging.handlers.QueueHandler(q))
    _logger = log
    _ENABLED = True


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
