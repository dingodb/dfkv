# SPDX-License-Identifier: Apache-2.0
"""dfkv connector telemetry — shared, single-source-of-truth package.

Canonical source lives in ``python/dfkv_telemetry/`` and is shipped with the
SGLang plugin via CMake ``install(FILES ...)``; the vLLM and LMCache pip
connectors vendor a byte-identical copy (guarded by a drift test). Keep it
pure-python and 3.9-compatible.

Subpackages:
  config        env/extra_config resolution + connector identity
  metrics_push  unified per-connector fleet metrics, pushed over OTLP

Usage in a connector __init__:
    from dfkv_telemetry import metrics, config
    metrics.configure(extra_cfg, connector_type=config.TYPE_VLLM,
                      tp_rank=rank, model=model_name)

Usage at an op chokepoint:
    with metrics.op("get", num_keys=n, num_bytes=nbytes) as m:
        ... do the C call ...
        m.keys = hits          # optional: refine keys/bytes after the call
        # m.status defaults to 'fail' if the block raised, else 'ok'
"""

from __future__ import annotations

from . import config, metrics_push
from . import metrics_push as metrics  # friendly alias: dfkv_telemetry.metrics

__all__ = ["config", "metrics", "metrics_push"]
