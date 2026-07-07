"""resolve_model_hash: the connector's model_hash config is a free-form label
folded with the served model name into the uint64 the dfkv client isolates by
(md5(model_name || label)). Must accept any string (never crash on a
non-numeric value), self-disambiguate across models, and stay deterministic so
cross-instance / cross-restart reuse lines up."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfkv_vllm._telemetry.config import resolve_model_hash


def test_deterministic():
    assert resolve_model_hash("GLM-5.2-Int4", "v2") == resolve_model_hash(
        "GLM-5.2-Int4", "v2"
    )


def test_stable_golden_value():
    # Pin the algorithm (md5(model||\0||label)[:8], little-endian). A change here
    # silently invalidates every deployment's cache, so it must be intentional.
    assert resolve_model_hash("GLM", "glm-5.2-int4-fp8-v2") == 145915887401238509


def test_different_label_different_id():
    assert resolve_model_hash("GLM-5.2-Int4", "v1") != resolve_model_hash(
        "GLM-5.2-Int4", "v2"
    )


def test_same_label_different_model_is_isolated():
    # Two different models sharing a label still get distinct ids.
    assert resolve_model_hash("model-a", "shared") != resolve_model_hash(
        "model-b", "shared"
    )


def test_accepts_arbitrary_string_no_crash():
    # The old int(...) crashed the engine on a non-numeric value; now any string
    # resolves to a valid uint64.
    v = resolve_model_hash("GLM", "glm-5.2/int4::fp8 v2!")
    assert 0 <= v < 2**64


def test_empty_and_none_isolate_by_model_name():
    assert resolve_model_hash("GLM", "") == resolve_model_hash("GLM", None)
    # still deterministic and model-scoped
    assert resolve_model_hash("A", "") != resolve_model_hash("B", "")


def test_whitespace_is_stripped():
    assert resolve_model_hash("GLM", " v2 ") == resolve_model_hash("GLM", "v2")


def test_numeric_label_is_hashed_like_any_string():
    # No special-casing of numeric strings (legacy numeric model_hashes are not
    # preserved -- old cache is intentionally dropped).
    v = resolve_model_hash("GLM", "20260708100")
    assert 0 <= v < 2**64
