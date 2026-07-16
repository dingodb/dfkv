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


def cache_engine_key_to_dfkv_str(key: CacheEngineKey,
                                 canonicalize_worker: bool = False) -> str:
    """Render a CacheEngineKey as a dfkv key string.

    Appends ``@{layer_id}`` when ``key`` is a ``LayerCacheEngineKey`` (i.e. LMCache
    layerwise mode) so per-layer objects do not collide. Omit it for the base
    ``CacheEngineKey`` (non-layerwise mode) — byte-identical to the legacy format.

    ``canonicalize_worker=True`` renders worker_id as 0 regardless of the
    calling rank. For MLA models the KV is REPLICATED across TP ranks, yet
    LMCache's key scheme embeds worker_id — so the same bytes were stored,
    written and fetched once PER RANK (8x everything at TP8), and the dfkv
    same-host rendezvous could never collapse the reads (different keys).
    Canonicalizing gives all ranks one shared keyspace: storage and writes
    dedup to 1x, and lockstep GETs become rendezvous-able. Only valid when
    the per-rank KV really is identical (MLA); the caller gates it.
    NOTE: flipping this changes the keyspace — existing cached data written
    with per-rank keys is not reachable under canonical keys (cold start).
    """
    worker = 0 if canonicalize_worker else key.worker_id
    base = f"{key.model_name}@{key.world_size}@{worker}@{key.chunk_hash:x}"
    layer_id = getattr(key, "layer_id", None)
    if layer_id is not None:
        return f"{base}@{layer_id}"
    return base
