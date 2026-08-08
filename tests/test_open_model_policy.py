"""Open-model policy + VRAM estimator unit tests (CPU-only, no network)."""

from __future__ import annotations

import pytest

from flash.catalog import resolve_model
from flash.engine.vram import VramEstimate, check_fit, estimate_vram_gb
from flash.schema import ConfigError, spec_from_dict
from tests._helpers.specs import raw_spec as _raw


def test_catalog_policy_rejects_unlisted_with_hint():
    with pytest.raises(ValueError, match="model_policy"):
        resolve_model("some-org/some-model", "sft", policy="catalog")


def test_catalog_model_resolves_normally():
    info = resolve_model("Qwen/Qwen3.5-4B", "grpo", policy="catalog")
    assert info.id == "Qwen/Qwen3.5-4B"


def test_allow_policy_accepts_fitting_model(monkeypatch):
    monkeypatch.setattr(
        "flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: 1.2, raising=True
    )
    info = resolve_model("acme/tiny-1b", "grpo", policy="allow", gpu="RTX 4090")
    assert info.thinking == "unknown"
    assert "open-model policy" in info.notes


def test_allow_policy_blocks_provably_too_big(monkeypatch):
    monkeypatch.setattr(
        "flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: 70.0, raising=True
    )
    with pytest.raises(ValueError, match="does not fit"):
        resolve_model("acme/huge-70b", "sft", policy="allow", gpu="RTX 4090")


def test_allow_policy_unknown_size_warns_but_allows(monkeypatch, capsys):
    monkeypatch.setattr(
        "flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: None, raising=True
    )
    info = resolve_model("acme/mystery", "sft", policy="allow", gpu="RTX 5090")
    assert info.id == "acme/mystery"
    assert "warning" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Estimator sanity: calibrated against catalog anchors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("params_b", "algo", "quant", "gpu", "expected"),
    [
        (4.0, "grpo", "bf16", "RTX 5090", "fits"),  # Qwen3-4B colocate on 32 GB (measured)
        (4.0, "sft", "bf16", "RTX 4090", "fits"),
        (9.65, "sft", "bf16", "RTX 5090", "tight"),  # Qwen3.5-9B SFT real logits peak
        (36.0, "sft", "bf16", "RTX 5090", "too_big"),  # 72 GB of weights
        (36.0, "grpo", "bf16", "RTX 5090", "too_big"),  # 2 bf16 copies + KV >> 32 GB
    ],
)
def test_estimator_anchors(monkeypatch, params_b, algo, quant, gpu, expected):
    est = check_fit("anchor/model", algo, gpu, quant=quant, params_b=params_b)
    assert isinstance(est, VramEstimate)
    assert est.verdict == expected, est.describe()


def test_grpo_needs_more_than_sft():
    assert estimate_vram_gb(4.0, "grpo") > estimate_vram_gb(4.0, "sft")


# ---------------------------------------------------------------------------
# Config schema plumbing
# ---------------------------------------------------------------------------
def test_spec_model_policy_default_catalog():
    spec = spec_from_dict(_raw())
    assert spec.model_policy == "catalog"


def test_spec_accepts_an_authored_model_policy():
    # Parsing is not authorizing. The parser runs client-side too, where FLASH_STANDALONE is
    # invisible, so it accepts the key; the control plane's _authorize_model_policy is what
    # rejects it on a managed plane (see tests/test_server_standalone.py).
    spec = spec_from_dict(_raw(model="Qwen/Qwen3.5-4B", model_policy="allow"))
    assert spec.model_policy == "allow"


@pytest.mark.parametrize("value", ["", "ALLOW ", "permissive", 1, True, None])
def test_spec_rejects_a_model_policy_outside_the_known_set(value):
    # An unknown policy must not silently degrade to "catalog": a self-hoster who typoed it would
    # get the curated-model rejection and no hint that their policy key was the problem.
    if value == "ALLOW ":
        assert spec_from_dict(_raw(model="Qwen/Qwen3.5-4B", model_policy=value)).model_policy == (
            "allow"
        )
        return
    with pytest.raises(ConfigError, match="model_policy must be one of"):
        spec_from_dict(_raw(model="Qwen/Qwen3.5-4B", model_policy=value))


def test_an_authored_model_policy_survives_the_submit_round_trip():
    # to_dict() is what the client SENDS. If it stripped model_policy the self-hosted config could
    # never reach the plane that authorizes it, and "allow" would be unreachable in practice.
    spec = spec_from_dict(_raw(model="Qwen/Qwen3.5-4B", model_policy="allow"))
    assert spec.to_dict()["model_policy"] == "allow"
    assert spec_from_dict(spec.to_dict()).model_policy == "allow"


def test_the_default_policy_is_still_absent_from_the_public_payload():
    # Only a non-default policy is emitted, so a managed submit payload is byte-identical to before.
    assert "model_policy" not in spec_from_dict(_raw()).to_dict()


def test_spec_unlisted_model_under_catalog_policy_fails():
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(model="acme/unlisted"))
    assert "model_policy" in str(ei.value)
