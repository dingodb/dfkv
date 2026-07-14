"""CLIENT_RANKS resolution (issue #111): how many TP ranks act as dfkv
clients for STORE striping. Pure logic, no torch/vllm imports, unit-testable.

Never rejects: an incompatible layout CLAMPS to full participation (today's
behavior) with a reason string -- one fleet-wide env template must be safe to
push to every instance regardless of model layout (user requirement).

  DFKV_CONNECTOR_CLIENT_RANKS = "<N>" | "auto" | unset
    unset          -> tp_size (today's behavior, no surprises on upgrade)
    auto           -> 1 when the KV object is TP-replicated, else tp_size
    N (int)        -> min(N, tp_size) when replicated, else tp_size

Replicated == MLA latent with a full-TP dedup stride and no DCP/PCP sharding:
each rank holds identical KV bytes, so any subset of ranks can store on
behalf of all (SGLang HiCache has shipped the N=1 mode as backup_skip)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ENV_NAME = "DFKV_CONNECTOR_CLIENT_RANKS"


def resolve_client_ranks(
    requested: str | None,
    tp_size: int,
    replicated: bool,
) -> tuple[int, str]:
    """Returns (effective_ranks, reason). effective == tp_size means legacy
    full participation; the caller must leave the store path untouched then."""
    if not requested:
        return tp_size, "default(tp)"
    req = requested.strip().lower()
    if not replicated:
        # DCP/PCP-sharded or GQA-grouped KV: every rank's bytes are unique,
        # convergence would drop data. Clamp to full participation, loudly.
        return tp_size, f"clamped(kv-not-replicated, requested={requested})"
    if req == "auto":
        return 1, "auto(mla-replicated)"
    try:
        n = int(req)
    except ValueError:
        return tp_size, f"clamped(unparseable={requested!r})"
    if n < 1 or n > tp_size:
        n = min(max(n, 1), tp_size)
        return n, f"clamped(range, requested={requested})"
    return n, f"requested({n})"


def participant(tp_rank: int, tp_size: int, effective: int) -> int | None:
    """Store-stripe index of this rank, or None if it does not store.
    Participants are spread evenly (spacing tp_size//effective) so they land
    on different NICs/NUMA nodes rather than all crowding rank 0..N-1."""
    spacing = tp_size // effective
    if tp_rank % spacing != 0:
        return None
    idx = tp_rank // spacing
    return idx if idx < effective else None


ELIDE_ENV = "DFKV_CONNECTOR_CLIENT_ELIDE"


def should_create_client(
    role: str,
    tp_rank: int,
    tp_size: int,
    effective: int,
    elide_requested: bool,
) -> tuple[bool, str]:
    """Phase 2a of issue #111 (producer-side client elision): a kv_producer's
    non-participant ranks never touch dfkv -- their store path early-exits
    (Phase 1 striping) and a producer has no load path -- so they can skip
    creating the client entirely: no RDMA connections, no MDS registration,
    no MR pin. instances x TP -> instances x N connections on the P side.

    Deliberately NOT applied to kv_consumer / kv_both: their load path runs on
    every rank (each rank fills its own GPU KV), so eliding would turn loads
    into misses. Load-side convergence (Phase 2b: participant-proxied GET +
    same-host page sharing) is a separate, measurement-gated design.

    Opt-in (DFKV_CONNECTOR_CLIENT_ELIDE=1) and layout-clamped like Phase 1:
    unset, or any non-eligible combination, keeps today's behavior exactly."""
    if not elide_requested:
        return True, "default(create)"
    if role != "kv_producer":
        return True, f"clamped(role={role})"
    if effective >= tp_size:
        return True, "clamped(no-convergence)"
    if participant(tp_rank, tp_size, effective) is not None:
        return True, "participant"
    return False, f"elided(producer non-participant, ranks={effective}/{tp_size})"
