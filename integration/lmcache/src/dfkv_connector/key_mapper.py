# SPDX-License-Identifier: Apache-2.0
"""Serialize an LMCache CacheEngineKey to a dfkv key string.

dfkv keys are opaque strings — the server shards on a hash of the whole string
(con_hash), so any stable, collision-free rendering works. We use

    "{model_name}@{world_size}@{worker_id}@{chunk_hash_hex}"

plus, when the key is a LMCache ``LayerCacheEngineKey`` (the per-layer key
emitted under ``use_layerwise=True``), a trailing ``@{layer_id}``:

    "{model_name}@{world_size}@{worker_id}@{chunk_hash_hex}@{layer_id}"

Differences from the dingofs connector's key_mapper:
  - The full ``chunk_hash`` is used (not truncated to 32 bits). The dingofs
    connector truncated to match the upstream NativeConnectorL2Adapter's 4-byte
    ObjectKey layout; dfkv's RemoteConnector path has no such constraint, so the
    full hash avoids needless collisions.
  - No (world_size << 24 | worker_id << 16) bit-packing — world_size/worker_id
    are written verbatim, which is just as unique and far easier to read in logs.

dtype is intentionally NOT encoded: chunk_hash is already a content hash, so two
keys with the same model+rank+hash but different dtype are effectively the same
chunk. (dfkv's value header still carries the geometry as a self-consistency tag,
so a read with a different geometry safely misses.)

layer_id IS encoded when present. LMCache's layerwise mode
(``config.use_layerwise=True``, forced on for HPU) splits each chunk into
``num_layers`` per-layer keys (``LayerCacheEngineKey``, same chunk_hash + worker
but distinct ``layer_id``) and stores each one independently through the
RemoteConnector. Without the layer_id in the dfkv key those N per-layer objects
collapse to one key and overwrite each other → cross-layer corruption. The base
``CacheEngineKey`` has no layer_id attribute, so ``getattr(..., None)`` keeps
this path byte-identical to the legacy format when layerwise is off.
"""

from __future__ import annotations

from lmcache.utils import CacheEngineKey

__all__ = ["cache_engine_key_to_dfkv_str"]


def cache_engine_key_to_dfkv_str(key: CacheEngineKey) -> str:
    """Render a CacheEngineKey as a dfkv key string.

    Appends ``@{layer_id}`` when ``key`` is a ``LayerCacheEngineKey`` (i.e. LMCache
    layerwise mode) so per-layer objects do not collide. Omit it for the base
    ``CacheEngineKey`` (non-layerwise mode) — byte-identical to the legacy format.
    """
    base = f"{key.model_name}@{key.world_size}@{key.worker_id}@{key.chunk_hash:x}"
    layer_id = getattr(key, "layer_id", None)
    if layer_id is not None:
        return f"{base}@{layer_id}"
    return base
