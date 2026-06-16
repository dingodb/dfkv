# SPDX-License-Identifier: Apache-2.0
"""Serialize an LMCache CacheEngineKey to a dfkv key string.

dfkv keys are opaque strings — the server shards on a hash of the whole string
(con_hash), so any stable, collision-free rendering works. We use

    "{model_name}@{world_size}@{worker_id}@{chunk_hash_hex}"

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
"""

from __future__ import annotations

from lmcache.utils import CacheEngineKey

__all__ = ["cache_engine_key_to_dfkv_str"]


def cache_engine_key_to_dfkv_str(key: CacheEngineKey) -> str:
    """Render a CacheEngineKey as a dfkv key string."""
    return f"{key.model_name}@{key.world_size}@{key.worker_id}@{key.chunk_hash:x}"
