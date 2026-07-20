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


def test_build_rollout_sample_shows_full_text_and_sanitizes_without_redacting_placeholder(
    monkeypatch,
) -> None:
    secret = "sk-test-secret-1234567890"
    monkeypatch.setenv("ROLLOUT_SAMPLE_API_KEY", secret)
    prompt = "p" * 520 + f" api_key={secret}"
    completion = "ANSWER: <value> " + secret + " " + "c" * 1100

    sample = build_rollout_sample(prompt, completion, reward=0.75, generated_at_step=2)

    # Full prompt + completion are preserved (no length truncation), only the secret is redacted.
    assert len(sample["prompt_tail"]) > 500
    assert "p" * 520 in sample["prompt_tail"]
    assert secret not in sample["prompt_tail"]
    assert secret not in sample["completion"]
    assert sample["completion"].startswith("ANSWER: <value> <redacted>")
    assert sample["completion"].endswith("c" * 1100)
    assert "[truncated]" not in sample["completion"]
    assert sample["reward"] == 0.75
    assert "loss" not in sample
    assert sample["generated_at_step"] == 2


def test_build_rollout_sample_redacts_prompt_secret_without_truncating(monkeypatch) -> None:
    secret = "prompt-boundary-secret-xyz"
    monkeypatch.setenv("ROLLOUT_SAMPLE_API_KEY", secret)
    prompt = f"prefix-{secret}{'p' * 490}"

    sample = build_rollout_sample(prompt, "completion", reward=1.0, generated_at_step=1)

    assert "secret-xyz" not in sample["prompt_tail"]
    assert secret not in sample["prompt_tail"]
    assert "<redacted>" in sample["prompt_tail"]
    assert sample["prompt_tail"].startswith("prefix-<redacted>")
    assert sample["prompt_tail"].endswith("p" * 490)


def test_build_rollout_sample_redacts_completion_secret_without_truncating(monkeypatch) -> None:
    secret = "completion-boundary-secret-xyz"
    monkeypatch.setenv("ROLLOUT_SAMPLE_API_KEY", secret)
    completion = f"{'c' * 988}{secret}-suffix"

    sample = build_rollout_sample("prompt", completion, reward=1.0, generated_at_step=1)

    assert secret not in sample["completion"]
    assert "<redacted>" in sample["completion"]
    assert sample["completion"].endswith("-suffix")
    assert "[truncated]" not in sample["completion"]


def test_build_rollout_sample_neutralizes_terminal_control_characters() -> None:
    sample = build_rollout_sample(
        "line one\rrewrite\nline two\tindented",
        "answer\x1b[2J\nnext\x00done\x7f",
        reward=1.0,
        generated_at_step=1,
    )

    assert sample["prompt_tail"] == "line one\\x0drewrite\nline two\\x09indented"
    assert sample["completion"] == "answer\\x1b[2J\nnext\\x00done\\x7f"
    for field in ("prompt_tail", "completion"):
        assert all(char == "\n" or ord(char) >= 0x20 for char in sample[field])
        assert "\x7f" not in sample[field]


def test_build_rollout_sample_carries_loss_scalar_for_opd() -> None:
    sample = build_rollout_sample("prompt", "student answer", loss=0.4213, generated_at_step=3)

    assert sample["loss"] == 0.4213
    assert "reward" not in sample
    assert sample["completion"] == "student answer"
    assert sample["generated_at_step"] == 3


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
    assert [sample["reward"] for sample in selected] == [0.1, 0.3, 0.4]
    assert all(sample["generated_at_step"] == 7 for sample in selected)


def test_select_rollout_samples_hard_caps_at_three() -> None:
    triples = [(f"prompt-{index}", f"completion-{index}", index) for index in range(8)]

    selected = select_rollout_samples(triples, generated_at_step=1)

    assert len(selected) == 3


def test_select_rollout_samples_loss_scalar_for_opd() -> None:
    triples = [
        ([{"role": "user", "content": "q-a"}], "answer-a", 0.11),
        ([{"role": "user", "content": "q-b"}], "answer-b", 0.22),
    ]

    selected = select_rollout_samples(triples, generated_at_step=5, scalar="loss")

    assert [sample["loss"] for sample in selected] == [0.11, 0.22]
    assert all("reward" not in sample for sample in selected)
    assert [sample["prompt_tail"] for sample in selected] == ["user: q-a", "user: q-b"]
    assert all(sample["generated_at_step"] == 5 for sample in selected)


def test_select_rollout_samples_rejects_unknown_scalar() -> None:
    with pytest.raises(ValueError, match="scalar"):
        select_rollout_samples([("p", "c", 1.0)], generated_at_step=1, scalar="entropy")


def test_reward_heartbeat_carries_bounded_samples_and_forces_only_the_first(
    monkeypatch,
) -> None:
    import flash.engine.worker as worker

    heartbeat_module = importlib.import_module("flash.engine.worker.heartbeat")
    emitted: list[tuple[str, dict]] = []

    def committed_heartbeat(stage, **payload):
        emitted.append((stage, payload))
        return True

    monkeypatch.setattr(worker, "heartbeat", committed_heartbeat)
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
    assert len(first_payload["sampled_completions"]) == 3
    assert first_payload["force"] is True
    assert "force" not in emitted[1][1]


def test_reward_heartbeat_retries_force_until_sample_payload_commits(monkeypatch) -> None:
    import flash.engine.worker as worker

    heartbeat_module = importlib.import_module("flash.engine.worker.heartbeat")
    emitted: list[tuple[str, dict]] = []
    outcomes = iter([False, True, True])

    def heartbeat(stage, **payload):
        emitted.append((stage, payload))
        return next(outcomes)

    monkeypatch.setattr(worker, "heartbeat", heartbeat)
    monkeypatch.setattr(
        heartbeat_module,
        "_maybe_attach_gpu_diag",
        lambda payload, last, now: last,
    )
    transformers = types.ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    callback = heartbeat_module.make_reward_heartbeat_callback(
        samples=lambda: [
            {
                "prompt_tail": "prompt",
                "completion": "completion",
                "reward": 1.0,
                "generated_at_step": 1,
            }
        ]
    )

    for step in (1, 2, 3):
        callback.on_log(
            None,
            types.SimpleNamespace(global_step=step),
            None,
            logs={"reward": 1.0},
        )

    assert emitted[0][1]["force"] is True
    assert emitted[1][1]["force"] is True
    assert "force" not in emitted[2][1]


def test_heartbeat_reports_failed_then_successful_forced_delivery(monkeypatch) -> None:
    import flash.engine.worker as worker

    outcomes = iter([False, True])
    monkeypatch.setattr(worker, "hf_upload_file", lambda *args, **kwargs: next(outcomes))
    monkeypatch.setattr(worker, "_HB_TERMINAL_ONLY", False)
    monkeypatch.setattr(worker, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(worker, "_HB_FORCE_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(worker, "_HB_LAST_UPLOAD", 0.0)
    monkeypatch.setattr(worker, "_HB_LAST_FORCED_UPLOAD", 0.0)
    monkeypatch.setattr(worker, "_HB_LAST_COMMITTED_STEP", 0)
    monkeypatch.setattr(worker, "_HB_PROGRESS_SEQ", 0)
    monkeypatch.setattr(worker, "_HB_PROGRESS_UPLOADED_SEQ", 0)
    monkeypatch.setattr(worker, "_HB_LAST_PROGRESS_TS", 0.0)

    assert worker.heartbeat("rl_step", step=1, force=True) is False
    assert worker.heartbeat("rl_step", step=1, force=True) is True


def test_format_heartbeat_renders_reward_samples_after_reward_metrics() -> None:
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


def test_format_heartbeat_renders_opd_loss_samples() -> None:
    heartbeat = {
        "stage": "opd_step",
        "step": 3,
        "loss": 0.4213,
        "sampled_completions": [
            {
                "prompt_tail": "distil prompt",
                "completion": "student answer",
                "loss": 0.4213,
                "generated_at_step": 3,
            }
        ],
    }

    assert _format_heartbeat(heartbeat) == (
        "worker: stage=opd_step step=3 loss=0.4213\n"
        "  sample 1 loss=0.4213 step=3\n"
        "    prompt: distil prompt\n"
        "    completion: student answer"
    )


def test_format_heartbeat_renders_full_untruncated_completion() -> None:
    long_completion = "z" * 4000
    rendered = _format_heartbeat(
        {
            "stage": "rl_step",
            "sampled_completions": [
                {
                    "prompt_tail": "p",
                    "completion": long_completion,
                    "reward": 0.5,
                    "generated_at_step": 1,
                }
            ],
        }
    )

    assert long_completion in rendered  # shown in full, never truncated at render


def test_format_heartbeat_caps_rendered_samples_at_three() -> None:
    rendered = _format_heartbeat(
        {
            "stage": "rl_step",
            "sampled_completions": [
                {
                    "prompt_tail": f"p{index}",
                    "completion": f"c{index}",
                    "reward": index / 10,
                    "generated_at_step": 1,
                }
                for index in range(5)
            ],
        }
    )

    assert rendered.count("  sample ") == 3
    assert "sample 3" in rendered
    assert "sample 4" not in rendered


def test_bounded_sampled_completions_resanitizes_and_neutralizes_without_truncating(
    monkeypatch,
) -> None:
    heartbeat_module = importlib.import_module("flash.engine.worker.heartbeat")
    secret = "heartbeat-secret-boundary-xyz"
    monkeypatch.setenv("HEARTBEAT_API_KEY", secret)

    bounded = heartbeat_module._bounded_sampled_completions(
        [
            {
                "prompt_tail": f"{'p' * 600}\r api_key={secret}",
                "completion": f"bad\x1b[2J{'c' * 980}{secret}{'z' * 100}",
                "reward": 0.5,
                "generated_at_step": "4",
            }
        ]
    )

    assert len(bounded) == 1
    sample = bounded[0]
    # Full text preserved (no length bound), only redaction + control-char neutralization applied.
    assert len(sample["prompt_tail"]) > 500
    assert sample["completion"].endswith("z" * 100)
    assert "[truncated]" not in sample["completion"]
    assert secret not in sample["prompt_tail"]
    assert secret not in sample["completion"]
    assert "api_key=<redacted>" in sample["prompt_tail"]
    assert "\\x0d" in sample["prompt_tail"]
    assert "\\x1b" in sample["completion"]
    assert "\x1b" not in sample["completion"]
    assert sample["reward"] == 0.5
    assert sample["generated_at_step"] == 4


def test_bounded_sampled_completions_accepts_opd_loss_samples() -> None:
    heartbeat_module = importlib.import_module("flash.engine.worker.heartbeat")

    bounded = heartbeat_module._bounded_sampled_completions(
        [
            {
                "prompt_tail": "prompt",
                "completion": "student answer",
                "loss": 0.375,
                "generated_at_step": 2,
            }
        ]
    )

    assert len(bounded) == 1
    assert bounded[0]["loss"] == 0.375
    assert "reward" not in bounded[0]


def test_bounded_sampled_completions_caps_at_three() -> None:
    heartbeat_module = importlib.import_module("flash.engine.worker.heartbeat")

    bounded = heartbeat_module._bounded_sampled_completions(
        [
            {
                "prompt_tail": f"p{index}",
                "completion": f"c{index}",
                "reward": index / 10,
                "generated_at_step": 1,
            }
            for index in range(6)
        ]
    )

    assert len(bounded) == 3


def test_format_heartbeat_defensively_neutralizes_control_characters() -> None:
    rendered = _format_heartbeat(
        {
            "stage": "rl_step",
            "sampled_completions": [
                {
                    "prompt_tail": "prompt\roverwrite\nnext\tfield",
                    "completion": "answer\x1b[2J\nsecond",
                    "reward": 1.0,
                    "generated_at_step": 1,
                }
            ],
        }
    )

    assert "prompt\\x0doverwrite\n      next\\x09field" in rendered
    assert "answer\\x1b[2J\n      second" in rendered
    assert "\r" not in rendered
    assert "\x1b" not in rendered
    assert "\t" not in rendered


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


def test_reward_fn_forwarded_rewards_capture_aligned_multi_turn_samples(monkeypatch) -> None:
    reward_fn, latest_named_metrics, latest_samples = _load_nested_reward_fn(monkeypatch)
    latest_named_metrics["stale"] = 1.0
    latest_samples.append({"stale": True})
    prompts = [
        [{"role": "user", "content": "question-a"}],
        [{"role": "user", "content": "question-b"}],
    ]
    completions = [
        [{"role": "assistant", "content": "answer-a"}],
        [{"role": "assistant", "content": "answer-b"}],
    ]

    rewards = reward_fn(
        completions,
        reward=[0.25, 0.75],
        prompts=prompts,
        trainer_state=types.SimpleNamespace(global_step=9),
    )

    assert rewards == [0.25, 0.75]
    assert latest_named_metrics == {}
    assert [sample["prompt_tail"] for sample in latest_samples] == [
        "user: question-a",
        "user: question-b",
    ]
    assert [sample["completion"] for sample in latest_samples] == [
        "assistant: answer-a",
        "assistant: answer-b",
    ]
    assert [sample["reward"] for sample in latest_samples] == rewards
    assert all(sample["generated_at_step"] == 9 for sample in latest_samples)
