#!/usr/bin/env bash
# Integration smoke for dfkv_server + dfkv_smoke + dfkvctl. arg1 = build dir.
set -e
BUILD="${1:?build dir}"
D=$(mktemp -d)
: > "$D/srv.out"   # pre-create so the awk below can't fail on a missing file under `set -e`
"$BUILD/dfkv_server" --dir "$D" --port 0 --cap 1073741824 >>"$D/srv.out" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; rm -rf "$D"' EXIT
P=""
# `|| true`: a transient awk read must never trip `set -e` before the server prints PORT.
for i in $(seq 1 100); do P=$(awk '/PORT/{print $2}' "$D/srv.out" 2>/dev/null || true); [ -n "$P" ] && break; sleep 0.05; done
[ -n "$P" ] || { echo "server did not report PORT"; cat "$D/srv.out"; exit 1; }
"$BUILD/dfkv_smoke" --members "n=127.0.0.1:$P" --size 4096
"$BUILD/dfkvctl" --members "n=127.0.0.1:$P" put k v12345
got=$("$BUILD/dfkvctl" --members "n=127.0.0.1:$P" get k)
[ "$got" = "v12345" ] || { echo "get mismatch: '$got'"; exit 1; }
"$BUILD/dfkvctl" --members "n=127.0.0.1:$P" exist k | grep -q true
"$BUILD/dfkvctl" stat "127.0.0.1:$P" | grep -q dfkv_cache_put_total
echo "tool_smoke OK (port $P)"
