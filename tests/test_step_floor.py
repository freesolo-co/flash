"""The per-step floor: work the FLOPs terms do not account for.

A GRPO step has three phases with no term in the cost model. Measured by verl's own `timing_s/*`
on real hardware (Qwen3.5-0.8B, RTX 4090, 8 steps, 100.0% of the step accounted):

    update_actor      62.1%   -> update_s
    old_log_prob      17.9%   -> NOTHING
    gen               15.5%   -> gen_s
    update_weights     3.0%   -> NOTHING
    save_checkpoint    1.5%   -> NOTHING
    reward             0.0%   -> reward_s (which billed 1.0s/completion)

The fake reward wall was numerically impersonating the gap -- both scale with completion count,
which is why the model looked calibrated while both halves were wrong.

Those three phases are the smaller half of it. On that arm the residual over the modelled gpu
seconds is 44.74s: 11.62s is the unmodelled phases, and 33.12s is update_actor + gen running
40.19s against a modelled 7.06s. So the floor has two real parts -- a fixed per-step overhead the
peak-FLOPs model misses at 0.8B-4B, and work that grows with the rollout batch -- which is why it
is an intercept PLUS a slope rather than either alone.
"""

from __future__ import annotations

import pytest

from flash.cost.analytical import (
    STEP_FLOOR_BASE_SECONDS,
    STEP_FLOOR_SECONDS_PER_COMPLETION,
    seconds_per_step,
    step_floor_seconds,
    step_seconds_split,
)
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
def test_the_floor_does_not_depend_on_the_card(gpu):
    """A per-card table was measured and does not beat this out of sample -- and its fitted values
    quote B200 71s per step FASTER than H200, which is backwards. One form for every card cannot
    express a hardware ranking at all, which is the point."""
    assert step_floor_seconds(gpu, 32) == step_floor_seconds("H100", 32)


def test_b200_and_h200_get_the_same_floor():
    """The invariant a per-card fit broke. B200 training is H100/H200-class on portable kernels at
    a higher $/hr, so B200 must never be quoted faster or cheaper for the same run."""
    assert step_floor_seconds("B200", 64) == step_floor_seconds("H200", 64)


def test_the_floor_has_both_a_fixed_and_a_per_completion_part():
    """Either half alone fails a whole class of runs. A pure constant scores 0/6 on gens=256 arms
    (worst 4.23x) and 1/43 on gens=32 when it cannot fit that class; a pure per-completion slope
    with no intercept drops the fixed overhead and scores 32/56 against 38/56."""
    assert step_floor_seconds("H100", 0) == pytest.approx(STEP_FLOOR_BASE_SECONDS)
    grew = step_floor_seconds("H100", 100) - step_floor_seconds("H100", 0)
    assert grew == pytest.approx(100 * STEP_FLOOR_SECONDS_PER_COMPLETION)


@pytest.mark.parametrize("completions", [-5, 0, 1, 32, 256, 4096])
def test_the_floor_is_never_negative_and_never_shrinks(completions):
    """A negative or shrinking floor would credit a run for generating more, which is backwards
    and would let a large rollout batch quote cheaper than a small one."""
    value = step_floor_seconds("H100", completions)
    assert value >= STEP_FLOOR_BASE_SECONDS
    assert value >= step_floor_seconds("H100", max(0, completions - 1))


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_a_bigger_rollout_batch_pays_a_bigger_floor(method):
    """Measured: the floor is 77.2s at 32 completions and 230.5s at 256. It tracks the rollout
    batch because old_log_prob is a forward pass over exactly that batch."""
    small = seconds_per_step(_config(method, batch_size=2, group_size=2), "H100")
    large = seconds_per_step(_config(method, batch_size=8, group_size=8), "H100")
    assert large > small


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_the_floor_is_gpu_time_not_overhead(method):
    """old_log_prob and update_weights run ON the card, so the floor belongs to the gpu-bound half
    of the split. Filing it as fixed overhead would under-report gpu utilisation and mislead the
    allocator about how much of a step sharding can actually shorten."""
    config = _config(method)
    completions = (config.batch_size or 1) * (config.group_size or 1)
    gpu_bound, fixed = step_seconds_split(config, "H100")
    assert gpu_bound >= step_floor_seconds("H100", completions)
    assert fixed < step_floor_seconds("H100", completions)


def test_removing_the_floor_shortens_every_step(monkeypatch):
    """Mutation guard: if the floor stops being added, this test must fail. Without it the whole
    correction can be deleted and the suite stays green, which is how the original gap survived."""
    import flash.cost.analytical as analytical

    config = _config()
    completions = (config.batch_size or 1) * (config.group_size or 1)
    with_floor = seconds_per_step(config, "H100")
    monkeypatch.setattr(analytical, "step_floor_seconds", lambda _gpu, _n: 0.0)
    without = seconds_per_step(config, "H100")
    assert with_floor - without == pytest.approx(step_floor_seconds("H100", completions))
