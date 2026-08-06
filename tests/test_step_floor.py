"""The per-step floor: unmodelled GPU work that the fictitious reward term was standing in for.

A GRPO step has three phases with no term in the cost model. Measured by verl's own `timing_s/*`
on real hardware (Qwen3.5-0.8B, RTX 4090, 8 steps, 100.0% of the step accounted):

    update_actor      62.1%   -> update_s
    old_log_prob      17.9%   -> NOTHING
    gen               15.5%   -> gen_s
    update_weights     3.0%   -> NOTHING
    save_checkpoint    1.5%   -> NOTHING
    reward             0.0%   -> reward_s (which billed 1.0s/completion)

~22% of every step was missing, and the fake reward wall was numerically impersonating it -- both
scale with completion count, which is why the model looked calibrated while both halves were
wrong.

Those three phases are the smaller half of the problem. On that arm the residual over the modelled
gpu seconds is 44.74s: 11.62s is the unmodelled phases, and 33.12s is update_actor + gen running
40.19s against a modelled 7.06s. Most of the floor is fixed per-step overhead the peak-FLOPs model
misses at 0.8B-4B, which is why a CONSTANT fits and every scaled form loses. These tests pin the
properties that make that constant the right shape.
"""

from __future__ import annotations

import pytest

from flash.cost.analytical import STEP_FLOOR_SECONDS, seconds_per_step, step_floor_seconds
from flash.cost.types import RunConfig

CARDS = ["A100 PCIe", "B200", "H100", "H200", "RTX 4090", "RTX 5090"]


def _config(method="grpo", **overrides):
    base = {
        "model_id": "Qwen/Qwen3.5-2B",
        "method": method,
        "steps": 20,
        "seq_len": 2048,
        "completion_len": 1024,
        "batch_size": 8,
        "group_size": 4,
        "gpu_type": "H100",
        "provider": "runpod",
    }
    base.update(overrides)
    return RunConfig(**base)


@pytest.mark.parametrize("gpu", [*CARDS, "RTX Pro 6000", "some-unreleased-card"])
def test_the_floor_is_one_constant_for_every_card(gpu):
    """A per-card table was measured and does not beat this constant out of sample (8/11 either
    way), so the extra six parameters buy nothing -- and the fitted table quotes B200 71s per step
    FASTER than H200, which is backwards. One number cannot express a card ranking at all."""
    assert step_floor_seconds(gpu) == STEP_FLOOR_SECONDS


def test_b200_and_h200_get_the_same_floor():
    """The invariant the per-card fit broke. B200 training is H100/H200-class on portable kernels
    at a higher $/hr, so B200 must never be quoted faster or cheaper for the same run. H200's
    fitted 152.5s came from 4 arms of a particular model/completion mix, not from hardware."""
    assert step_floor_seconds("B200") == step_floor_seconds("H200")


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_the_floor_is_charged_once_per_step_not_per_completion(method):
    """It stands in for old_log_prob, weight sync and checkpointing: work that happens once per
    step. Charging it per completion would rebuild the exact fiction being removed -- the old
    reward term was 1.0s x completions, which reached 97% of a quote."""
    few = seconds_per_step(_config(method, batch_size=2, group_size=2), "H100")
    many = seconds_per_step(_config(method, batch_size=8, group_size=8), "H100")
    # more completions still costs more (generation and teacher scoring scale), but the gap must
    # be far below the 15x the completion count grew by, or the floor is being charged per rollout.
    assert many > few
    assert many - few < STEP_FLOOR_SECONDS


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_the_floor_is_gpu_time_not_overhead(method):
    """old_log_prob and update_weights run ON the card, so the floor belongs to the gpu-bound half
    of the split. Filing it as fixed overhead would under-report gpu utilisation and mislead the
    allocator about how much of a step sharding can actually shorten."""
    from flash.cost.analytical import step_seconds_split

    gpu_bound, fixed = step_seconds_split(_config(method), "H100")
    no_floor_gpu = gpu_bound - STEP_FLOOR_SECONDS
    assert no_floor_gpu >= 0
    assert fixed < STEP_FLOOR_SECONDS


def test_removing_the_floor_shortens_every_step(monkeypatch):
    """Mutation guard: if the floor stops being added, this test must fail. Without it the whole
    correction can be deleted and the suite stays green, which is how the original 22% gap
    survived in the first place."""
    import flash.cost.analytical as analytical

    with_floor = seconds_per_step(_config(), "H100")
    monkeypatch.setattr(analytical, "step_floor_seconds", lambda _gpu: 0.0)
    without = seconds_per_step(_config(), "H100")
    assert with_floor - without == pytest.approx(STEP_FLOOR_SECONDS)
