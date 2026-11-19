"""GRPO fails fast when it has provably no learning signal.

Regression for two 2026-11-18 runs (flash-1795017945-c7b42b52, flash-1795013975-350c3343) whose
reward function raised on every call. Each failure was scored 0.0, so every group's rewards were
identical, every advantage was 0 and the actor gradient was 0 -- yet the runs held an H200 for
130 and 24 steps before the publish-time refusal fired.
"""

from __future__ import annotations

import json

import pytest

import flash.engine.worker.train.rl.rollout.single_turn as rl_single
from flash.core.spec import JobSpec
from flash.engine.worker.train.entry import rl_train_runner
from flash.engine.worker.train.entry.rl_train_runner import _ingest_step_metrics, _StepMetricState
from flash.engine.worker.train.rl.launch import checkpoints as rl_checkpoints
from flash.engine.worker.train.rl.launch.signal_guard import (
    DEFAULT_ZERO_SIGNAL_ABORT_STEPS,
    RewardFaultTally,
    ZeroLearningSignalError,
    ZeroLearningSignalGuard,
    resolve_zero_signal_abort_steps,
)
from flash.schema import ConfigError, spec_from_dict

# the worker's error heartbeat keeps sanitize_diagnostic(e, limit=500), class name included.
_HEARTBEAT_ERROR_LIMIT = 500


def _zero_step(guard: ZeroLearningSignalGuard, step: int, *, reward: float = 0.0) -> None:
    guard.observe(step, grad_norm=0.0, reward_mean=reward, advantage_min=0.0, advantage_max=0.0)


def _assert_fits_heartbeat(exc: BaseException) -> None:
    assert len(f"{type(exc).__name__}: {exc}") <= _HEARTBEAT_ERROR_LIMIT


def test_constant_reward_aborts_after_threshold_with_actionable_message() -> None:
    guard = ZeroLearningSignalGuard(5)
    for step in range(1, 5):
        _zero_step(guard, step)
    with pytest.raises(ZeroLearningSignalError) as raised:
        _zero_step(guard, 5)
    message = str(raised.value)
    assert "no learning signal" in message
    assert "steps 1-5" in message
    assert "all rollouts in every group received the same reward" in message
    assert "mean reward 0 on every step" in message
    assert "group_size/temperature" in message
    assert "zero_signal_abort_steps = 0" in message
    # a deterministic job failure, never a retriable infrastructure fault.
    assert isinstance(raised.value, RuntimeError)
    _assert_fits_heartbeat(raised.value)


def test_reward_that_raises_on_every_call_is_named_as_the_cause() -> None:
    faults = RewardFaultTally()
    faults.record_scored(24)
    for _ in range(24):
        faults.record_fault(KeyError("reward_url"))
    guard = ZeroLearningSignalGuard(2, faults=faults)
    _zero_step(guard, 1)
    with pytest.raises(ZeroLearningSignalError) as raised:
        _zero_step(guard, 2)
    message = str(raised.value)
    assert "every reward call raised (24/24)" in message
    assert "KeyError: 'reward_url'" in message
    assert "fix the environment's reward/score function" in message
    _assert_fits_heartbeat(raised.value)


def test_long_fault_text_still_fits_the_error_heartbeat() -> None:
    faults = RewardFaultTally()
    faults.record_scored(10_000)
    faults.record_fault(RuntimeError("HTML judge service unavailable; " + "x" * 2000))
    guard = ZeroLearningSignalGuard(1000, faults=faults)
    for step in range(1, 1000):
        _zero_step(guard, step)
    with pytest.raises(ZeroLearningSignalError) as raised:
        _zero_step(guard, 1000)
    assert "1/10000 reward calls raised" in str(raised.value)
    _assert_fits_heartbeat(raised.value)


def test_nonzero_advantage_with_zero_gradient_points_at_masking() -> None:
    guard = ZeroLearningSignalGuard(1)
    with pytest.raises(ZeroLearningSignalError, match="mask_truncated_completions"):
        guard.observe(1, grad_norm=0.0, reward_mean=0.4, advantage_min=-0.5, advantage_max=0.5)


def test_one_positive_gradient_disarms_the_guard_for_the_rest_of_the_run() -> None:
    # a run that learned and then saturated (every group solved) is healthy; the publish check
    # accepts it, so the early stop must not kill it.
    guard = ZeroLearningSignalGuard(2)
    _zero_step(guard, 1)
    guard.observe(2, grad_norm=0.01, reward_mean=0.5, advantage_min=-1.0, advantage_max=1.0)
    for step in range(3, 50):
        _zero_step(guard, step, reward=1.0)
    assert not guard.armed


def test_zero_disables_and_prior_positive_evidence_disarms() -> None:
    disabled = ZeroLearningSignalGuard(0)
    resumed = ZeroLearningSignalGuard(2, resume_step=10, prior_positive_step=4)
    for step in range(11, 40):
        _zero_step(disabled, step)
        _zero_step(resumed, step)


def test_replayed_repeated_and_unparsed_steps_are_not_counted() -> None:
    guard = ZeroLearningSignalGuard(3, resume_step=10)
    # verl replays its resume step's line; it belongs to the previous attempt.
    _zero_step(guard, 10)
    _zero_step(guard, 11)
    _zero_step(guard, 11)  # a reprinted step line
    guard.observe(12, grad_norm=None, reward_mean=0.0)  # no parseable grad norm proves nothing
    _zero_step(guard, 12)
    with pytest.raises(ZeroLearningSignalError, match="steps 11-13"):
        _zero_step(guard, 13)


def test_threshold_resolution() -> None:
    assert DEFAULT_ZERO_SIGNAL_ABORT_STEPS == 8
    assert resolve_zero_signal_abort_steps(None) == DEFAULT_ZERO_SIGNAL_ABORT_STEPS

    class _Train:
        zero_signal_abort_steps = None

    assert resolve_zero_signal_abort_steps(_Train()) == DEFAULT_ZERO_SIGNAL_ABORT_STEPS
    _Train.zero_signal_abort_steps = 0
    assert resolve_zero_signal_abort_steps(_Train()) == 0
    _Train.zero_signal_abort_steps = 3
    assert resolve_zero_signal_abort_steps(_Train()) == 3


def _verl_line(step: int, *, grad_norm: float, reward: float, adv: float) -> str:
    return (
        f"step:{step} - actor/grad_norm:{grad_norm} - critic/rewards/mean:{reward} - "
        f"critic/advantages/min:{-adv} - critic/advantages/max:{adv} - timing_s/step:60.0"
    )


@pytest.fixture
def _quiet_heartbeat(monkeypatch, worker_context):
    monkeypatch.setattr(rl_train_runner._worker_heartbeat, "heartbeat", lambda *_a, **_k: True)
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    monkeypatch.setattr(worker_context, "remaining_wall_seconds", lambda: 20000.0)


@pytest.mark.usefixtures("_quiet_heartbeat")
def test_verl_step_lines_trip_the_guard_mid_run(worker_context) -> None:
    """the real metric path: verl's stdout step lines reach the guard through ingestion."""
    state = _StepMetricState()
    state.signal_guard = ZeroLearningSignalGuard(4)
    inp = {"max_completion": 2048, "steps": 130}
    for step in range(1, 4):
        _ingest_step_metrics(
            _verl_line(step, grad_norm=0.0, reward=0.0, adv=0.0),
            inp,
            state,
            dict,
            context=worker_context,
        )
    with pytest.raises(ZeroLearningSignalError, match="steps 1-4"):
        _ingest_step_metrics(
            _verl_line(4, grad_norm=0.0, reward=0.0, adv=0.0),
            inp,
            state,
            dict,
            context=worker_context,
        )
    # the evidence the failure heartbeat reports is already recorded for every step it saw.
    assert sorted(state.grad_norms) == [1, 2, 3, 4]
    assert [row["step"] for row in state.metrics_last] == [1, 2, 3, 4]


@pytest.mark.usefixtures("_quiet_heartbeat")
def test_learning_run_streams_to_completion_and_passes_publish_check(worker_context) -> None:
    state = _StepMetricState()
    state.signal_guard = ZeroLearningSignalGuard(2)
    inp = {"max_completion": 2048, "steps": 6}
    for step in range(1, 7):
        grad = 0.0 if step in (1, 4, 5, 6) else 0.012
        _ingest_step_metrics(
            _verl_line(step, grad_norm=grad, reward=0.5, adv=0.0 if grad == 0.0 else 0.8),
            inp,
            state,
            dict,
            context=worker_context,
        )
    rl_checkpoints._check_grpo_had_a_gradient(
        state.reward_history,
        state.adv_spread_history,
        state.grad_norms,
        expected_steps=range(1, 7),
    )


class _MetadataRewardEnv:
    """the html-kimi environment's failure: reward_url read from metadata, where it is absent."""

    def reward(self, completion, example, state=None):
        return float(example["metadata"]["reward_url"] == "never")


@pytest.fixture
def _identity_graded(monkeypatch):
    import flash.engine.worker.model.decoding as decoding

    monkeypatch.setattr(
        decoding, "graded_text", lambda text, prompt_opened_thinking=False, **_context: text
    )
    monkeypatch.setattr(decoding, "thinking_text", lambda text, prompt_opened_thinking=False: "")


@pytest.mark.usefixtures("_identity_graded")
def test_scoring_faults_are_still_scored_zero_but_reported(worker_context) -> None:
    faults = RewardFaultTally()
    example = {"input": "<p>x</p>", "metadata": {}, "reward_url": "https://judge"}
    requests = [("<p>x</p>", example)] * 4
    faults.record_scored(len(requests))
    rows = rl_single.score_single_turn_batch(
        _MetadataRewardEnv(),
        requests,
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        context=worker_context,
        on_fault=faults.record_fault,
    )
    assert [score for score, _ in rows] == [0.0] * 4
    assert faults.snapshot() == (4, 4, "KeyError: 'reward_url'")


@pytest.mark.usefixtures("_identity_graded")
def test_non_finite_reward_is_reported_as_a_fault(worker_context) -> None:
    class _NanEnv:
        def reward(self, completion, example, state=None):
            return float("nan")

    seen: list[BaseException] = []
    score = rl_single.score_single_turn(
        _NanEnv(),
        "x",
        {"input": "x"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        context=worker_context,
        on_fault=seen.append,
    )
    assert score == 0.0
    assert len(seen) == 1
    assert "non-finite" in str(seen[0])


_GRPO_RAW = {
    "model": "Qwen/Qwen3.5-9B",
    "algorithm": "grpo",
    "project": "11111111-1111-4111-8111-111111111111",
    "environment": {"id": "freesolo/math-agent/gsm8k"},
    "train": {"max_steps": 10},
    "gpu": {},
}


def _raw(algorithm: str = "grpo", **train) -> dict:
    raw = json.loads(json.dumps(_GRPO_RAW))
    raw["algorithm"] = algorithm
    raw["train"].update(train)
    return raw


def test_config_knob_parses_round_trips_and_stays_out_of_unset_digests() -> None:
    unset = spec_from_dict(_raw())
    assert unset.train.zero_signal_abort_steps is None
    # an unauthored knob is omitted, so existing preparation digests stay byte-identical.
    assert "zero_signal_abort_steps" not in unset.to_internal_dict()["train"]
    assert "zero_signal_abort_steps" not in unset.to_dict()["train"]

    for value in (0, 3):
        spec = spec_from_dict(_raw(zero_signal_abort_steps=value))
        assert spec.train.zero_signal_abort_steps == value
        assert JobSpec.from_json(spec.to_json()).train.zero_signal_abort_steps == value
        assert spec_from_dict(spec.to_dict()).train.zero_signal_abort_steps == value


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (_raw(zero_signal_abort_steps=-1), "zero_signal_abort_steps"),
        (_raw(zero_signal_abort_steps=True), "zero_signal_abort_steps"),
        (_raw(zero_signal_abort_steps="5"), "zero_signal_abort_steps"),
        (_raw("sft", zero_signal_abort_steps=5), "only applies to grpo"),
    ],
)
def test_config_knob_rejections(raw, match) -> None:
    with pytest.raises(ConfigError, match=match):
        spec_from_dict(raw)
