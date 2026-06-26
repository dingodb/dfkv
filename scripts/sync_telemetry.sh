#!/usr/bin/env bash
# Single source of truth for the dfkv telemetry package is python/dfkv_telemetry/.
# The vLLM and LMCache connectors are independent pip wheels that can't import a
# repo-level module at runtime, so they vendor a byte-identical copy as an internal
# _telemetry sub-package. Run this after editing the canonical files; CI guards
# against drift via tests/python/test_telemetry_vendor_sync.py.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="python/dfkv_telemetry"
FILES="__init__.py config.py metrics_push.py otlp_json.py"
DSTS=(
  "integration/vllm/src/dfkv_vllm/_telemetry"
  "integration/lmcache/src/dfkv_connector/_telemetry"
)

for dst in "${DSTS[@]}"; do
  mkdir -p "$dst"
  for f in $FILES; do
    cp "$SRC/$f" "$dst/$f"
  done
  echo "synced $SRC -> $dst"
done
