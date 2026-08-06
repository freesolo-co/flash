"""Sampling-pass behaviour: what the profiler draws, and what it refuses to infer."""

from __future__ import annotations

import pytest

from flash.engine.rollout_profile import (
    RolloutProfileUnavailable,
    RolloutSample,
    SampledRollout,
    sample_one_rollout,
    sample_rollouts,
)

PROMPTS = [
    [{"role": "user", "content": "a"}],
    [{"role": "user", "content": "b"}],
    [{"role": "user", "content": "c"}],
    [{"role": "user", "content": "d"}],
]


def _rollout(completion_tokens=100, prompt_tokens=20, truncated=False, seconds=1.0):
    return SampledRollout(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        truncated=truncated,
        seconds=seconds,
    )


def _sampler(script):
    """A fake endpoint that returns scripted rollouts and records what it was asked."""
    calls = []

    def sample_one(*, messages, served_model, max_completion_tokens, temperature):
        calls.append(
            {
                "content": messages[0]["content"],
                "model": served_model,
                "cap": max_completion_tokens,
                "temperature": temperature,
            }
        )
        item = script[min(len(calls) - 1, len(script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    return sample_one, calls


def _run(script, *, target=8, group_size=4, prompts=PROMPTS):
    sample_one, calls = _sampler(script)
    result = sample_rollouts(
        prompts=prompts,
        served_model="qwen/qwen3.5-9b",
        max_completion_tokens=2048,
        temperature=1.0,
        target_rollouts=target,
        group_size=group_size,
        sample_one=sample_one,
    )
    return result, calls


# --- the sampling shape, which is the whole reason this is not a simple loop -------------------


def test_draws_spread_across_distinct_prompts_before_repeating_one():
    """Between-prompt variance measured 5x within-prompt, so 8 draws from one prompt describe
    that prompt rather than the environment. Round-robin is the correctness property here."""
    _, calls = _run([_rollout()] * 8, target=8)

    first_four = [c["content"] for c in calls[:4]]
    assert sorted(first_four) == ["a", "b", "c", "d"], "first pass must touch every prompt once"
    assert sorted(c["content"] for c in calls) == ["a", "a", "b", "b", "c", "c", "d", "d"]


def test_a_truncated_budget_still_leaves_a_spread_sample():
    """Stopping early must not concentrate the sample in the first example."""
    result, calls = _run([_rollout()] * 3, target=3)
    assert [c["content"] for c in calls] == ["a", "b", "c"]
    assert result.sampled_prompts == 3


def test_the_run_sampling_settings_reach_the_endpoint():
    """A profile taken at a different cap or temperature measures a different distribution."""
    _, calls = _run([_rollout()], target=1)
    assert calls[0]["cap"] == 2048
    assert calls[0]["temperature"] == 1.0
    assert calls[0]["model"] == "qwen/qwen3.5-9b"


def test_the_environments_own_messages_are_sent_unmodified():
    """The profile must ask what training asks; reformatting would measure the reformatting."""
    _, calls = _run([_rollout()], target=1)
    assert calls[0]["content"] == "a"


# --- failures are counted, never retried into silence -----------------------------------------


def test_failed_draws_are_counted_not_hidden():
    result, _ = _run([_rollout(), RuntimeError("boom"), _rollout()], target=3)
    assert result.completed == 2
    assert result.failed == 1


def test_a_sample_where_everything_failed_reports_no_measurement():
    result, _ = _run([RuntimeError("boom")] * 4, target=4)
    assert result.completed == 0
    assert result.failed == 4
    assert result.completion_tokens_max == 0
    assert result.completion_tokens_mean == 0.0


def test_no_prompts_is_unavailable_not_an_empty_sample():
    with pytest.raises(RolloutProfileUnavailable, match="no prompts"):
        sample_rollouts(
            prompts=[],
            served_model="m",
            max_completion_tokens=128,
            temperature=None,
            target_rollouts=4,
        )


# --- the aggregate -----------------------------------------------------------------------------


def test_aggregate_reports_the_measured_distribution():
    rollouts = [_rollout(completion_tokens=n) for n in (200, 300, 400, 1000)]
    sample = RolloutSample.from_rollouts(rollouts, sampled_prompts=4, failed=0)
    assert sample.completion_tokens_mean == 475.0
    assert sample.completion_tokens_max == 1000
    assert sample.completion_tokens_p50 == 300
    # nearest-rank on 4 samples: index int(0.9*3) == 2. the tail itself is completion_tokens_max,
    # which is reported separately precisely so p90 is not asked to stand in for it.
    assert sample.completion_tokens_p90 == 400
    assert sample.completed == 4


def test_percentiles_stay_ordered_on_a_wide_sample():
    """The reference sample spanned 200-2048 tokens, so ordering has to hold on real spreads."""
    rollouts = [
        _rollout(completion_tokens=n)
        for n in (
            200,
            234,
            249,
            265,
            289,
            292,
            296,
            344,
            349,
            351,
            470,
            789,
            1109,
            1236,
            2048,
            2048,
        )
    ]
    sample = RolloutSample.from_rollouts(rollouts, sampled_prompts=4, failed=0)
    assert (
        sample.completion_tokens_p50 <= sample.completion_tokens_p90 <= sample.completion_tokens_max
    )
    assert sample.completion_tokens_max == 2048
    assert round(sample.completion_tokens_mean) == 661


def test_truncated_and_eos_partition_the_sample():
    rollouts = [
        _rollout(truncated=True),
        _rollout(truncated=True),
        _rollout(truncated=False),
    ]
    sample = RolloutSample.from_rollouts(rollouts, sampled_prompts=3, failed=0)
    assert sample.truncated == 2
    assert sample.eos == 1
    assert sample.truncated + sample.eos == sample.completed


def test_seconds_uses_the_median_not_the_mean():
    """One slow response from a busy shared endpoint must not define the reading."""
    rollouts = [_rollout(seconds=s) for s in (1.0, 1.0, 1.0, 100.0)]
    sample = RolloutSample.from_rollouts(rollouts, sampled_prompts=4, failed=0)
    assert sample.seconds_per_completion == 1.0


# --- what the endpoint parser refuses to infer -------------------------------------------------


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        import json

        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _endpoint_returning(payload, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response(payload))


def test_a_response_without_usage_counts_is_refused(monkeypatch):
    """Re-deriving a count from the returned text would drop the reasoning tokens a hybrid model
    emits but does not return in content -- 61% of measured output on the reference sample."""
    _endpoint_returning(
        {"choices": [{"finish_reason": "stop", "message": {"content": "42"}}]}, monkeypatch
    )
    with pytest.raises(RolloutProfileUnavailable, match="usage token counts"):
        sample_one_rollout(
            messages=[{"role": "user", "content": "q"}],
            served_model="m",
            max_completion_tokens=128,
            temperature=1.0,
        )


def test_length_finish_reason_marks_the_rollout_truncated(monkeypatch):
    _endpoint_returning(
        {
            "choices": [{"finish_reason": "length", "message": {"content": "x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 128},
        },
        monkeypatch,
    )
    got = sample_one_rollout(
        messages=[{"role": "user", "content": "q"}],
        served_model="m",
        max_completion_tokens=128,
        temperature=1.0,
    )
    assert got.truncated is True
    assert got.completion_tokens == 128


def test_stop_finish_reason_is_not_truncated(monkeypatch):
    _endpoint_returning(
        {
            "choices": [{"finish_reason": "stop", "message": {"content": "x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 55},
        },
        monkeypatch,
    )
    got = sample_one_rollout(
        messages=[{"role": "user", "content": "q"}],
        served_model="m",
        max_completion_tokens=128,
        temperature=1.0,
    )
    assert got.truncated is False
    assert got.completion_tokens == 55


def test_no_configured_endpoint_is_unavailable(monkeypatch):
    """3 of 6 catalog models are too small for anyone to host. That is a fallback, not a failure."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RolloutProfileUnavailable, match="no rollout sampling endpoint"):
        sample_one_rollout(
            messages=[{"role": "user", "content": "q"}],
            served_model="m",
            max_completion_tokens=128,
            temperature=1.0,
        )
