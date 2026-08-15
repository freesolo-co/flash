"""The SFT startup block: billed training wall the FLOPs term does not account for.

`setup_seconds` and `train_wall` are DISJOINT intervals in the worker -- setup is stamped before
the training subprocess launches, and `train_wall` wraps `run_verl_training` plus the upload drain.
So verl startup, model load, LoRA wrap and FSDP init sit INSIDE the wall the customer is billed for,
and the FLOPs term priced them at zero: 17 measured RunPod arms scored 28.6x UNDER their realized
train wall, 0/17 inside the 0.70-1.43x acceptance band.

The block scales with checkpoint size (measured median implied block 82s at 0.8B against 248s at
4B), which is why it is a base plus a per-param term rather than one constant: a size-blind constant
at the same in-band count leaves a 3.75x worst arm against 2.13x for this form.

This mirrors what `test_step_floor` pins for GRPO/OPD. The two must not drift into one algorithm
being priced for its startup while the other is not.
"""

from __future__ import annotations

import pytest

from flash.cost.analytical import (
    SFT_STARTUP_BASE_SECONDS,
    SFT_STARTUP_SECONDS_PER_PARAM_B,
    SFT_STEP_FLOOR_SECONDS,
    estimate_cost,
    sft_overhead_seconds,
)
from flash.cost.types import RunConfig


def _sft(**overrides):
    base = {
        "model_id": "Qwen/Qwen3.5-2B",
        "method": "sft",
        "steps": 32,
        "seq_len": 1024,
        "batch_size": 8,
        "gpu_type": "H100",
        "provider": "runpod",
    }
    base.update(overrides)
    return RunConfig(**base)


def test_the_block_has_both_a_fixed_and_a_size_scaled_part():
    """Both halves are load-bearing: the measured block roughly triples from 0.8B to 4B, so a
    size-blind constant cannot cover both ends of the catalog."""
    small = sft_overhead_seconds(_sft(model_id="Qwen/Qwen3.5-0.8B"), 0)
    large = sft_overhead_seconds(_sft(model_id="Qwen/Qwen3.5-4B"), 0)
    assert large > small, "a bigger checkpoint must pay a longer startup block"
    assert small >= SFT_STARTUP_BASE_SECONDS, "the fixed part is paid at every size"
    assert SFT_STARTUP_SECONDS_PER_PARAM_B > 0.0


def test_more_steps_pay_a_bigger_floor():
    """The per-step half is the publish/sync work that recurs, so it must grow with step count."""
    config = _sft()
    assert sft_overhead_seconds(config, 128) - sft_overhead_seconds(config, 32) == pytest.approx(
        SFT_STEP_FLOOR_SECONDS * 96
    )


def test_the_overhead_is_never_negative():
    """A zero/negative step count is a degenerate config, not a credit against the block."""
    assert sft_overhead_seconds(_sft(), 0) > 0.0
    assert sft_overhead_seconds(_sft(), -5) == sft_overhead_seconds(_sft(), 0)


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_only_sft_pays_this_block(method, monkeypatch):
    """GRPO and OPD already carry their startup in the rollout step floor. Charging this on top
    would bill the same phases twice, so their quotes must be insensitive to these constants."""
    import flash.cost.analytical as analytical

    config = RunConfig(
        model_id="Qwen/Qwen3.5-2B",
        method=method,
        steps=8,
        seq_len=1024,
        completion_len=512,
        batch_size=8,
        group_size=4,
        gpu_type="H100",
        provider="runpod",
    )
    before = estimate_cost(config).train_seconds
    monkeypatch.setattr(analytical, "SFT_STARTUP_BASE_SECONDS", 9_999.0)
    monkeypatch.setattr(analytical, "SFT_STARTUP_SECONDS_PER_PARAM_B", 9_999.0)
    monkeypatch.setattr(analytical, "SFT_STEP_FLOOR_SECONDS", 9_999.0)
    assert estimate_cost(config).train_seconds == pytest.approx(before), (
        f"{method} must not pay the sft startup block on top of its rollout floor"
    )


def test_a_short_sft_run_is_not_quoted_at_almost_zero():
    """The regression that motivated this: a 2-step SFT run was quoted 0.47s of training against a
    measured 533s, because a short run is almost entirely startup. The quote must be dominated by
    the block at that shape, not by the FLOPs term."""
    quote = estimate_cost(_sft(model_id="Qwen/Qwen3.5-4B", steps=2))
    assert quote.train_seconds > 100.0, (
        "a 2-step SFT run is startup-dominated; quoting only its FLOPs bills a fraction of the pod"
    )


def test_the_block_reaches_the_quote_through_the_real_path(monkeypatch):
    """Guard the WIRING, not just the helper. A correct constant that `estimate_cost` never reads
    would leave every quote exactly as under-priced as before, and a helper-only test cannot see
    that -- so drop the block to zero and require the billed quote to move by its full amount."""
    import flash.cost.analytical as analytical

    config = _sft()
    with_block = estimate_cost(config).train_seconds
    block = sft_overhead_seconds(config, config.steps)

    monkeypatch.setattr(analytical, "SFT_STARTUP_BASE_SECONDS", 0.0)
    monkeypatch.setattr(analytical, "SFT_STARTUP_SECONDS_PER_PARAM_B", 0.0)
    monkeypatch.setattr(analytical, "SFT_STEP_FLOOR_SECONDS", 0.0)
    without_block = estimate_cost(config).train_seconds

    assert with_block - without_block == pytest.approx(block), (
        "the quote must carry the whole block; a constant the quote never reads fixes nothing"
    )


def test_a_token_priced_sft_run_also_pays_the_block():
    """SFT has two pricing paths -- step-count and profile-measured tokens -- and a run that took
    the token path is the SAME pod paying the same startup. Only the step path carrying the block
    would make an exactly-profiled run cheaper than an estimated one for identical work."""
    stepwise = estimate_cost(_sft()).train_seconds
    tokenised = estimate_cost(_sft(train_tokens=32197.0)).train_seconds
    assert tokenised > sft_overhead_seconds(_sft(), 32)
    # both paths price the same shape, so neither may collapse toward the bare FLOPs term
    assert min(stepwise, tokenised) > 50.0
