"""What a measured rollout profile is allowed to change in a quote, and what it must not."""

from __future__ import annotations

import pytest

from flash.cost.analytical import estimate_cost, seconds_per_step
from flash.cost.types import RunConfig

GPU = "H200"
# the reference sample: mean 661 completion tokens against a 2048 cap.
MEASURED_COMPLETION = 660.6
MEASURED_PROMPT = 25.4


def _config(method, **overrides):
    base = dict(
        model_id="Qwen/Qwen3.5-9B",
        method=method,
        steps=100,
        seq_len=2048,
        completion_len=2048,
        batch_size=8,
        group_size=4,
        gpu_type=GPU,
        provider="runpod",
    )
    base.update(overrides)
    return RunConfig(**base)


def _measured(method, **overrides):
    return _config(
        method,
        measured_completion_tokens=MEASURED_COMPLETION,
        measured_prompt_tokens=MEASURED_PROMPT,
        **overrides,
    )


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_a_measured_profile_lowers_the_quote(method):
    """Realized generation ran 0.32x of the cap, so billing the cap overstates the work."""
    capped = seconds_per_step(_config(method), GPU)
    measured = seconds_per_step(_measured(method), GPU)
    assert measured < capped
    assert capped / measured > 1.5


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_sizing_is_immune_to_the_measured_mean(method):
    """THE dangerous failure mode: completion_len and seq_len also size the gpu. Pricing the mean
    is correct; SIZING the mean would provision vram for an average rollout and OOM on the tail.
    A cheaper quote that kills the run is worse than an expensive one."""
    assert _config(method).train_knobs() == _measured(method).train_knobs()


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_no_profile_reproduces_todays_quote_exactly(method):
    """The fallback path must be bit-identical to current behaviour, or every unhosted model
    silently reprices the day this lands."""
    both_none = _config(method)
    assert both_none.measured_completion_tokens is None
    assert seconds_per_step(both_none, GPU) == seconds_per_step(_config(method), GPU)


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_a_measured_mean_above_the_cap_is_clamped(method):
    """The engine's cap still binds. A profile claiming more than the run permits means the two
    disagree about the configuration, and the configuration wins."""
    absurd = _config(
        method,
        measured_completion_tokens=999_999.0,
        measured_prompt_tokens=999_999.0,
    )
    assert seconds_per_step(absurd, GPU) == pytest.approx(seconds_per_step(_config(method), GPU))


def test_opd_teacher_cost_falls_with_measured_tokens():
    """OPD bills the teacher per token scored, so the cap error compounds there rather than only
    on gpu time -- on this shape the teacher is the dominant line item."""
    capped = estimate_cost(_config("opd"))
    measured = estimate_cost(_measured("opd"))
    assert measured.teacher_api_usd < capped.teacher_api_usd
    assert capped.teacher_api_usd / measured.teacher_api_usd > 2.0


def test_half_a_profile_is_not_used():
    """Substituting a measured completion onto a full-context prompt prices a shape no rollout
    has, so opd's sequence billing requires BOTH halves before it trusts either."""
    half = _config("opd", measured_completion_tokens=MEASURED_COMPLETION)
    assert seconds_per_step(half, GPU) == pytest.approx(seconds_per_step(_config("opd"), GPU))


def test_the_note_says_the_length_was_measured():
    """A measured quote and a capped quote differ ~1.7x. The note has to name which one it is or
    the cheaper number reads as an unexplained price change."""
    notes = estimate_cost(_measured("grpo")).notes
    step_note = next(n for n in notes if "GRPO step =" in n)
    assert "measured" in step_note
    assert "cap 2048" in step_note

    capped_note = next(n for n in estimate_cost(_config("grpo")).notes if "GRPO step =" in n)
    assert "measured" not in capped_note
