# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dfkv LMCache key_mapper (cache_engine_key_to_dfkv_str).

Exercises the LMCache ``CacheEngineKey`` → dfkv key string serialization with
fake key objects (``types.SimpleNamespace``), so it needs NEITHER lmcache/torch
NOR a real dfkv ring / libdfkv.so. The ``key_mapper`` module's only external
dependency is the ``CacheEngineKey`` *type* used in its annotation, which we
substitute via ``sys.modules`` so the import succeeds without lmcache installed.

Covers:
  - Non-layerwise (base CacheEngineKey): legacy format, byte-identical to before.
  - Layerwise (LayerCacheEngineKey): ``@{layer_id}`` appended; N layers of the
    same chunk_hash + worker produce N distinct dfkv keys (no collision).

Run:
    python3 integration/lmcache/tests/test_key_mapper.py
or with pytest:
    python3 -m pytest integration/lmcache/tests/test_key_mapper.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

# --- Stub lmcache.utils so key_mapper imports without lmcache/torch installed.
# The real CacheEngineKey lives in lmcache.utils; key_mapper references it only
# in an annotation (a bare name lookup at import time), so a module object with
# the attribute set is enough. The tests use SimpleNamespace fake keys, not the
# real class, so no torch is needed.
if "lmcache" not in sys.modules:
    sys.modules["lmcache"] = types.ModuleType("lmcache")
if "lmcache.utils" not in sys.modules:
    _stub = types.ModuleType("lmcache.utils")

    class CacheEngineKey:  # minimal stand-in for the annotation target
        pass

    _stub.CacheEngineKey = CacheEngineKey
    sys.modules["lmcache.utils"] = _stub

# Load key_mapper.py directly by path (bypassing the dfkv_connector package
# __init__, which pulls in adapter → lmcache.logging and would need the full
# lmcache install). key_mapper has no other dfkv_connector-internal deps.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "_dfkv_key_mapper_under_test",
    os.path.join(_HERE, "..", "src", "dfkv_connector", "key_mapper.py"),
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
cache_engine_key_to_dfkv_str = _MOD.cache_engine_key_to_dfkv_str


def _base_key(model="glm-5.1", world_size=8, worker_id=0, chunk_hash=0xABCDEF):
    """A base CacheEngineKey stand-in (no layer_id attribute)."""
    return types.SimpleNamespace(
        model_name=model,
        world_size=world_size,
        worker_id=worker_id,
        chunk_hash=chunk_hash,
    )


def _layer_key(model="glm-5.1", world_size=8, worker_id=0, chunk_hash=0xABCDEF,
               layer_id=0):
    """A LayerCacheEngineKey stand-in (has layer_id)."""
    k = _base_key(model, world_size, worker_id, chunk_hash)
    k.layer_id = layer_id
    return k


def test_base_key_legacy_format():
    # Non-layerwise: format must be unchanged (back-compat with deployments
    # running use_layerwise=False, the default).
    k = _base_key(worker_id=2, chunk_hash=0x1234)
    assert cache_engine_key_to_dfkv_str(k) == "glm-5.1@8@2@1234"


def test_base_key_full_hash_preserved():
    # chunk_hash is rendered as full hex (not truncated).
    k = _base_key(chunk_hash=0xDEADBEEFCAFE)
    assert cache_engine_key_to_dfkv_str(k) == "glm-5.1@8@0@deadbeefcafe"


def test_layer_key_appends_layer_id():
    # Layerwise: @layer_id appended, mirroring LMCache's LayerCacheEngineKey
    # .to_string() which emits "...@{dtype}@{layer_id}".
    k = _layer_key(chunk_hash=0x99, layer_id=7)
    assert cache_engine_key_to_dfkv_str(k) == "glm-5.1@8@0@99@7"


def test_layer_keys_same_chunk_different_layers_distinct():
    # THE BUG THIS FIXES: under use_layerwise=True, LMCache splits one chunk
    # into num_layers per-layer keys (same chunk_hash + worker, distinct
    # layer_id). Before the fix, these all collapsed to one dfkv key and
    # overwrote each other. They must now be distinct.
    keys = [_layer_key(chunk_hash=0x100, layer_id=L) for L in range(8)]
    strs = [cache_engine_key_to_dfkv_str(k) for k in keys]
    assert len(set(strs)) == 8, f"layer keys collided: {strs}"
    # And each carries its own layer_id.
    for L, s in enumerate(keys):
        assert cache_engine_key_to_dfkv_str(s).endswith(f"@{L}")


def test_base_and_layer_zero_distinct():
    # A base key (no layer_id) and a layer key with layer_id=0 must NOT be equal:
    # the base key represents the whole-chunk blob (non-layerwise), the layer=0
    # key represents just layer 0's blob (layerwise). Encoding layer_id=0 makes
    # them distinct, which is correct — they are different objects.
    base = _base_key(chunk_hash=0x200)
    layer0 = _layer_key(chunk_hash=0x200, layer_id=0)
    assert cache_engine_key_to_dfkv_str(base) == "glm-5.1@8@0@200"
    assert cache_engine_key_to_dfkv_str(layer0) == "glm-5.1@8@0@200@0"
    assert cache_engine_key_to_dfkv_str(base) != cache_engine_key_to_dfkv_str(layer0)


def test_layer_id_zero_is_encoded():
    # layer_id=0 is a real layer, not "absent" — must be encoded (otherwise
    # layer 0 collides with the non-layerwise base key, and with the chunk
    # itself). getattr(..., None) returns 0 (not None) so the @0 suffix is added.
    k = _layer_key(layer_id=0)
    assert cache_engine_key_to_dfkv_str(k).endswith("@0")



def test_canonical_worker_zeroes_rank():
    # Phase 9: MLA shared keyspace — worker_id renders as 0.
    k = _base_key(worker_id=5, chunk_hash=0xABC)
    assert cache_engine_key_to_dfkv_str(k, canonicalize_worker=True) == \
        "glm-5.1@8@0@abc"


def test_canonical_default_off_keeps_legacy():
    k = _base_key(worker_id=5, chunk_hash=0xABC)
    assert cache_engine_key_to_dfkv_str(k) == "glm-5.1@8@5@abc"


def test_canonical_all_ranks_converge():
    rendered = {cache_engine_key_to_dfkv_str(
        _base_key(worker_id=w, chunk_hash=0xDEAD), canonicalize_worker=True)
        for w in range(8)}
    assert len(rendered) == 1


def test_canonical_layerwise_keeps_layer_id():
    k = _layer_key(worker_id=3, chunk_hash=0x1, layer_id=7)
    assert cache_engine_key_to_dfkv_str(k, canonicalize_worker=True) == \
        "glm-5.1@8@0@1@7"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

