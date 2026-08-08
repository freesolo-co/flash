"""Only curated catalog models are trainable + VRAM estimator unit tests (CPU-only, no network).

Adding a model means forking Flash and adding a ModelInfo entry to flash/catalog.py. There is no
config key that accepts an uncataloged model: the rejection is catalog membership itself, so it
holds identically on a managed and a self-hosted plane.
"""

from __future__ import annotations

import pytest

from flash.catalog import resolve_model
from flash.engine.vram import GPU_VRAM_GB, estimate_vram_gb
from flash.schema import ConfigError, spec_from_dict
from tests._helpers.specs import raw_spec as _raw


def test_an_uncataloged_model_is_rejected_with_the_fork_instruction():
    # The error is the primary documentation for the new workflow -- it is what every rejected
    # user sees, so it must name the concrete next step, not just refuse.
    with pytest.raises(ValueError, match="unsupported model") as ei:
        resolve_model("some-org/some-model", "sft")
    message = str(ei.value)
    assert "fork" in message.lower()
    assert "flash/catalog.py" in message


def test_catalog_model_resolves_normally():
    info = resolve_model("Qwen/Qwen3.5-4B", "grpo")
    assert info.id == "Qwen/Qwen3.5-4B"


def test_the_parser_rejects_an_uncataloged_model_before_submit():
    # Rejecting client-side means a fork with a new catalog entry works without touching the
    # plane, and a typo fails locally instead of after a network round trip.
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(model="acme/unlisted"))
    assert "unsupported model" in str(ei.value)


def test_model_policy_is_no_longer_an_accepted_config_key():
    # The key is gone, not merely ignored: a config carrying it must fail loudly rather than
    # silently training something other than what it asked for.
    with pytest.raises(ConfigError, match="unknown config key"):
        spec_from_dict(_raw(model="Qwen/Qwen3.5-4B", model_policy="allow"))


def test_a_catalog_spec_round_trips_without_a_policy_field():
    spec = spec_from_dict(_raw(model="Qwen/Qwen3.5-4B"))
    payload = spec.to_dict()
    assert "model_policy" not in payload
    assert spec_from_dict(payload).model == "Qwen/Qwen3.5-4B"


# ---------------------------------------------------------------------------
# Estimator sanity: calibrated against catalog anchors
# ---------------------------------------------------------------------------
def _headroom(params_b: float, algo: str, quant: str, gpu: str) -> str:
    """The advisory fits/tight/too_big banding, applied straight to the sizing equations.

    This used to be `check_fit`, an estimator wrapper that fetched the model's size from HF. Only the
    open-model path consulted its verdict, so the wrapper went with it -- but the BANDS are what the
    anchors below calibrate, and those still gate real sizing decisions.
    """
    est = estimate_vram_gb(params_b, algo, quant)
    gpu_gb = GPU_VRAM_GB[gpu]
    if est > gpu_gb * 1.15:
        return "too_big"
    return "tight" if est > gpu_gb * 0.85 else "fits"


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
def test_estimator_anchors(params_b, algo, quant, gpu, expected):
    # params_b is supplied directly, so this calibrates the SIZING EQUATIONS against known anchors
    # and never resolves the id through the catalog.
    assert _headroom(params_b, algo, quant, gpu) == expected


def test_grpo_needs_more_than_sft():
    assert estimate_vram_gb(4.0, "grpo") > estimate_vram_gb(4.0, "sft")
