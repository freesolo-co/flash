from __future__ import annotations

import ast
import importlib
import inspect
import math
import sys
import textwrap
import types

import pytest

from flash.engine.worker.rollout_samples import build_rollout_sample, select_rollout_samples
from flash.providers._poll import _format_heartbeat


def test_build_rollout_sample_truncates_and_sanitizes_without_redacting_placeholder(
    monkeypatch,
) -> None:
    secret = "sk-test-secret-1234567890"
    monkeypatch.setenv("ROLLOUT_SAMPLE_API_KEY", secret)
    prompt = "p" * 520 + f" api_key={secret}"
    completion = "ANSWER: <value> " + secret + " " + "c" * 1100

    sample = build_rollout_sample(prompt, completion, 0.75, 2)

    assert len(sample["prompt_tail"]) <= 500
    assert secret not in sample["prompt_tail"]
    assert secret not in sample["completion"]
    assert sample["completion"].startswith("ANSWER: <value> <redacted>")
    assert sample["completion"].endswith("\n[truncated]")
    assert len(sample["completion"]) <= 1000 + len("\n[truncated]")
    assert sample["reward"] == 0.75
    assert sample["generated_at_step"] == 2


def test_select_rollout_samples_prefers_distinct_prompts_then_fills_repeats() -> None:
    triples = [
        ("prompt-a", "a-first", 0.1),
        ("prompt-a", "a-second", 0.2),
        ("prompt-b", "b-first", 0.3),
        ("prompt-c", "c-first", 0.4),
        ("prompt-d", "d-first", 0.5),
    ]

    selected = select_rollout_samples(triples, generated_at_step=7)

    assert [sample["completion"] for sample in selected] == ["a-first", "b-first", "c-first"]
    assert all(sample["generated_at_step"] == 7 for sample in selected)


def test_select_rollout_samples_hard_caps_at_four() -> None:
    triples = [(f"prompt-{index}", f"completion-{index}", index) for index in range(8)]

    selected = select_rollout_samples(triples, generated_at_step=1, limit=99)

    assert len(selected) == 4


def test_reward_heartbeat_carries_bounded_samples_and_forces_only_the_first(
    monkeypatch,
) -> None:
    import flash.engine.worker as worker

    heartbeat_module = importlib.import_module("flash.engine.worker.heartbeat")
    emitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        worker, "heartbeat", lambda stage, **payload: emitted.append((stage, payload))
    )
    monkeypatch.setattr(
        heartbeat_module,
        "_maybe_attach_gpu_diag",
        lambda payload, last, now: last,
    )
    transformers = types.ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    samples = [
        {
            "prompt_tail": f"prompt-{index}",
            "completion": f"completion-{index}",
            "reward": index / 10,
            "generated_at_step": 1,
        }
        for index in range(6)
    ]
    callback = heartbeat_module.make_reward_heartbeat_callback(
        lambda: {"success": 0.8}, lambda: samples
    )
    state = types.SimpleNamespace(global_step=1)

    callback.on_log(None, state, None, logs={"reward": 0.65})
    callback.on_log(None, state, None, logs={"reward": 0.70})

    first_payload = emitted[0][1]
    assert emitted[0][0] == "rl_step"
    assert first_payload["reward"] == 0.65
    assert first_payload["reward_metrics"] == {"success": 0.8}
    assert len(first_payload["sampled_completions"]) == 4
    assert first_payload["force"] is True
    assert "force" not in emitted[1][1]


def test_format_heartbeat_renders_samples_after_reward_metrics() -> None:
    heartbeat = {
        "stage": "rl_step",
        "step": 2,
        "reward": 0.6,
        "reward_metrics": {"success": 0.75},
        "sampled_completions": [
            {
                "prompt_tail": "question tail",
                "completion": "ANSWER: <value>",
                "reward": 0.75,
                "generated_at_step": 1,
            },
            {"prompt_tail": "missing fields"},
        ],
    }

    assert _format_heartbeat(heartbeat) == (
        "worker: stage=rl_step step=2 reward=0.600 success=0.750\n"
        "  sample 1 reward=0.750 step=1\n"
        "    prompt: question tail\n"
        "    completion: ANSWER: <value>"
    )
    assert _format_heartbeat({"stage": "rl_step", "reward": 0.6}) == (
        "worker: stage=rl_step reward=0.600"
    )


def _load_nested_reward_fn(monkeypatch):
    import flash.engine.worker.rl as rl

    source = textwrap.dedent(inspect.getsource(rl.run_rl))
    tree = ast.parse(source)
    reward_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reward_fn"
    )
    module = ast.Module(body=[reward_node], type_ignores=[])
    ast.fix_missing_locations(module)

    latest_named_metrics: dict[str, float] = {}
    latest_samples: list[dict] = []

    def forbidden_upload(*args, **kwargs):
        pytest.fail("reward_fn must not upload reward_debug.jsonl")

    fake_worker = types.SimpleNamespace(
        THINKING=False,
        graded_text=lambda completion, **kwargs: completion,
        upload_debug_jsonl=forbidden_upload,
    )

    class FakeEnv:
        def scores_breakdown(self, completion, example, state):
            return {"quality": example["reward"] + 0.1, "total": example["reward"]}

    namespace = {
        "_w": fake_worker,
        "_prompt_opens_thinking": False,
        "_think_penalty": 0.0,
        "tok": object(),
        "env": FakeEnv(),
        "rollout_examples": [
            {"input": "fallback-a", "reward": 0.1},
            {"input": "fallback-b", "reward": 0.2},
            {"input": "fallback-a", "reward": 0.3},
        ],
        "latest_named_metrics": latest_named_metrics,
        "latest_samples": latest_samples,
        "select_rollout_samples": select_rollout_samples,
        "_mean_named_reward_metrics": rl._mean_named_reward_metrics,
    }
    exec(compile(module, rl.__file__, "exec"), namespace)
    return namespace["reward_fn"], latest_named_metrics, latest_samples


def test_reward_fn_captures_aligned_samples_without_debug_upload(monkeypatch) -> None:
    reward_fn, latest_named_metrics, latest_samples = _load_nested_reward_fn(monkeypatch)

    rewards = reward_fn(
        ["completion-a1", "completion-b", "completion-a2"],
        example_idx=[0, 1, 2],
        prompts=["prompt-a", "prompt-b", "prompt-a"],
        trainer_state=types.SimpleNamespace(global_step=7),
    )

    assert rewards == [0.1, 0.2, 0.3]
    assert math.isclose(latest_named_metrics["quality"], 0.3)
    assert [sample["completion"] for sample in latest_samples] == [
        "completion-a1",
        "completion-b",
        "completion-a2",
    ]
    assert [sample["reward"] for sample in latest_samples] == rewards
    assert [sample["prompt_tail"] for sample in latest_samples] == [
        "prompt-a",
        "prompt-b",
        "prompt-a",
    ]
    assert all(sample["generated_at_step"] == 7 for sample in latest_samples)

    reward_fn(
        ["completion-a1", "completion-b", "completion-a2"],
        example_idx=[0, 1, 2],
        trainer_state=types.SimpleNamespace(global_step=8),
    )

    assert [sample["prompt_tail"] for sample in latest_samples] == [
        "fallback-a",
        "fallback-b",
        "fallback-a",
    ]
