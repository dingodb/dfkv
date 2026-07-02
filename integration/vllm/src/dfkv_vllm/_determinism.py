"""Fail-fast guard: dfkv store keys require cross-process block hashes.

Kept free of vllm/torch imports (duck-typed on cache_config) so it can be
unit-tested anywhere; both the scheduler- and worker-side components call it
at construction time.
"""

import os


def ensure_deterministic_block_hashing(cache_config) -> None:
    """Raise unless vLLM's prefix-cache block hashes survive a process change.

    dfkv keys embed ``request.block_hashes`` verbatim. With the default
    ``builtin`` hash algorithm Python randomizes ``hash()`` per process unless
    ``PYTHONHASHSEED`` is pinned, so the keys computed by a restarted engine --
    or by the decode instance of a PD pair -- never match the stored ones.
    The failure mode is the worst kind: **silent zero hit rate while puts keep
    landing** (P keeps writing, D never loads, storage fills with unreachable
    keys). This bit production before and was hand-fixed with
    ``PYTHONHASHSEED=0``; guard it structurally instead of by folklore.

    Accepted as deterministic:
    - a sha256-family ``prefix_caching_hash_algo`` (content-defined,
      process-independent), or
    - ``PYTHONHASHSEED`` pinned to a fixed value in the environment. It must
      be the SAME value on every engine sharing the store (both PD sides,
      every restart); that cross-instance agreement is the operator's part,
      this guard can only see its own process.
    """
    algo = str(getattr(cache_config, "prefix_caching_hash_algo", "builtin"))
    if "sha256" in algo.lower():
        return
    seed = os.environ.get("PYTHONHASHSEED", "")
    if seed and seed.lower() != "random":
        return
    raise RuntimeError(
        "DfkvStoreConnector: vLLM block hashes are not deterministic across "
        f"processes (prefix_caching_hash_algo={algo!r} and PYTHONHASHSEED is "
        "not pinned). Cross-restart and PD cross-instance lookups would "
        "silently never hit while stores keep writing. Fix: set the SAME "
        "fixed PYTHONHASHSEED on every engine sharing this store (e.g. "
        "PYTHONHASHSEED=0), or use --prefix-caching-hash-algo sha256."
    )
