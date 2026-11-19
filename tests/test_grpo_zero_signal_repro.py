"""Reproduce the html-kimi / html-astra zero-gradient runs through the production GRPO parent path.

See tests/grpo_zero_signal_harness.py for exactly what is real and what stands in for verl.
"""

from __future__ import annotations

import pytest

from tests.grpo_zero_signal_harness import run_scenario


def test_html_kimi_reward_aborts_after_k_steps_naming_the_keyerror(tmp_path) -> None:
    result = run_scenario("html-kimi", tmp_path, steps=12, abort_steps=3)
    assert result.outcome == "failed", result.summary()
    assert result.steps_seen == 3
    assert result.grad_norms == {1: 0.0, 2: 0.0, 3: 0.0}
    assert set(result.rewards) == {0.0}
    assert result.error.startswith("ZeroLearningSignalError: grpo has no learning signal")
    assert "every reward call raised (96/96)" in result.error
    assert "KeyError: 'reward_url'" in result.error
    # the verl child was torn down rather than left holding the GPU.
    assert not result.child_alive_after


def test_html_astra_reward_aborts_after_k_steps_naming_the_judge_refusal(tmp_path) -> None:
    result = run_scenario("html-astra", tmp_path, steps=12, abort_steps=3)
    assert result.outcome == "failed", result.summary()
    assert result.steps_seen == 3
    assert set(result.rewards) == {0.0}
    assert "every reward call raised (36/36)" in result.error
    assert "HTML judge service unavailable" in result.error
    assert not result.child_alive_after


def test_guard_disabled_reproduces_the_original_publish_time_refusal(tmp_path) -> None:
    # zero_signal_abort_steps = 0 is the pre-fix behaviour: every step runs, then publish refuses.
    result = run_scenario("html-astra", tmp_path, steps=4, abort_steps=0)
    assert result.outcome == "failed", result.summary()
    assert result.steps_seen == 4
    assert result.error == (
        "RuntimeError: grpo reported zero actor gradient norm on all 4 newly executed steps "
        "and no prior positive-gradient evidence; refusing to publish an unchanged adapter"
    )


@pytest.mark.parametrize("abort_steps", [None, 1])
def test_working_reward_trains_every_step_without_tripping_the_guard(tmp_path, abort_steps) -> None:
    # same judge, data, policy and sampling; only the astra env's case-id bug is repaired. even the
    # most aggressive threshold (1) must not fire on a reward that varies within groups.
    result = run_scenario("healthy", tmp_path, steps=10, abort_steps=abort_steps)
    assert result.outcome == "published", result.summary()
    assert result.steps_seen == 10
    assert all(value > 0.0 for value in result.grad_norms.values())
    assert len(set(result.rewards)) > 1
