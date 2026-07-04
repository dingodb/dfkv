"""CLIENT_RANKS clamp semantics (issue #111): must never reject, must never
converge non-replicated layouts, and participant spread must cover exactly
the effective count."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfkv_vllm.client_ranks import participant, resolve_client_ranks


def test_default_is_full_participation():
    assert resolve_client_ranks(None, 8, True)[0] == 8
    assert resolve_client_ranks("", 8, True)[0] == 8


def test_auto_converges_only_replicated():
    assert resolve_client_ranks("auto", 8, True)[0] == 1
    eff, reason = resolve_client_ranks("auto", 8, False)
    assert eff == 8 and "clamped" in reason


def test_explicit_n_clamps_but_never_errors():
    assert resolve_client_ranks("2", 8, True)[0] == 2
    eff, reason = resolve_client_ranks("2", 8, False)   # sharded layout
    assert eff == 8 and "kv-not-replicated" in reason
    assert resolve_client_ranks("99", 8, True)[0] == 8  # over TP
    assert resolve_client_ranks("0", 8, True)[0] == 1   # under 1
    eff, reason = resolve_client_ranks("garbage", 8, True)
    assert eff == 8 and "unparseable" in reason


def test_participant_spread_is_even_and_exact():
    # N=2 of TP8: ranks 0 and 4 (different NUMA/NIC halves), indices 0,1.
    got = {r: participant(r, 8, 2) for r in range(8)}
    assert got == {0: 0, 1: None, 2: None, 3: None, 4: 1, 5: None, 6: None, 7: None}
    # N=TP: everyone participates with idx == rank (legacy identity).
    assert [participant(r, 4, 4) for r in range(4)] == [0, 1, 2, 3]
    # N=1: only rank 0.
    assert [participant(r, 4, 1) for r in range(4)] == [0, None, None, None]
