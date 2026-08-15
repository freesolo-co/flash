"""Tests for local offline environment contract validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import pytest

import flash.cli as cli
from flash.cli.commands.env.test import cmd_env_test
from flash.envs.base import RolloutReward


class _SingleTurnEnv:
    multi_turn = False
    max_turns = 8

    def __init__(self, *, rows=None, reward=1.0):
        self.rows = [{"input": "what is 2 + 2?", "output": "4"}] if rows is None else rows
        self.reward_value = reward
        self.completions: list[str] = []

    def dataset(self):
        return self.rows

    def prompt_messages(self, example):
        return [{"role": "user", "content": example["input"]}]

    def sft_completion(self, example):
        return [{"role": "assistant", "content": example.get("output", "")}]

    def reward(self, completion, example, state=None):
        self.completions.append(completion)
        return self.reward_value


class _MultiTurnEnv:
    multi_turn = True
    max_turns = 8

    def __init__(self):
        self.scored_state = None

    def dataset(self):
        return [
            {
                "input": "finish the exchange",
                "output": [
                    {"role": "assistant", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ],
            }
        ]

    def sft_completion(self, example):
        return example["output"]

    def new_rollout_state(self, example):
        prompt = [{"role": "user", "content": example["input"]}]
        return {"prompt": prompt, "messages": list(prompt), "done": False, "turn": 0}

    def record_model_turn(self, state, content):
        message = {"role": "assistant", "content": content}
        state["messages"].append(message)
        state["response_text"] = content
        return message

    def env_reply(self, messages, state):
        state["turn"] += 1
        state["done"] = state["turn"] >= 2
        reply = {"role": "user", "content": "continue"}
        messages.append(reply)
        return [reply]

    def rollout_done(self, state, max_turns=None):
        return state["done"] or (max_turns is not None and state["turn"] >= max_turns)

    def reward(self, completion, example, state=None):
        self.scored_state = state
        return 0.5


class _TextFreeMultiTurnEnv(_MultiTurnEnv):
    # a native tool-call assistant turn (content=null) sits between two text turns; the
    # offline driver must replay the gold turns positionally so the null turn maps to model
    # turn 2 instead of collapsing the sequence and shifting "third" up a slot.
    def __init__(self):
        super().__init__()
        self.recorded = []

    def dataset(self):
        return [
            {
                "input": "finish the exchange",
                "output": [
                    {"role": "assistant", "content": "first"},
                    {"role": "assistant", "content": None},
                    {"role": "assistant", "content": "third"},
                ],
            }
        ]

    def record_model_turn(self, state, content):
        self.recorded.append(content)
        return super().record_model_turn(state, content)

    def env_reply(self, messages, state):
        state["turn"] += 1
        state["done"] = state["turn"] >= 3
        reply = {"role": "user", "content": "continue"}
        messages.append(reply)
        return [reply]


class _PerExampleCapMultiTurnEnv(_MultiTurnEnv):
    max_turns = 8

    def new_rollout_state(self, example):
        prompt = [{"role": "user", "content": example["input"]}]
        return {
            "prompt": prompt,
            "messages": list(prompt),
            "done": False,
            "turn": 0,
            "max_episode_turns": 12,
        }

    def env_reply(self, messages, state):
        state["turn"] += 1
        reply = {"role": "user", "content": "continue"}
        messages.append(reply)
        return [reply]

    def rollout_done(self, state, max_turns=None):
        per_example_cap = state.get("max_episode_turns")
        cap = per_example_cap if per_example_cap is not None else max_turns
        return state.get("done") or (cap is not None and state["turn"] >= cap)


class _StatefulBoardMultiTurnEnv(_MultiTurnEnv):
    # a stateful env that only counts a model turn once env_reply applies it, and whose reward is
    # that count. it never declares itself done, so the driver's own turn cap ends the episode --
    # which is exactly the exit that skipped the last env step.
    max_turns = 3

    def new_rollout_state(self, example):
        prompt = [{"role": "user", "content": example["input"]}]
        return {"prompt": prompt, "messages": list(prompt), "done": False, "turn": 0, "applied": 0}

    def env_reply(self, messages, state):
        state["turn"] += 1
        state["applied"] += 1
        reply = {"role": "user", "content": "continue"}
        messages.append(reply)
        return [reply]

    def rollout_done(self, state, max_turns=None):
        return False

    def reward(self, completion, example, state=None):
        self.scored_state = state
        return float((state or {}).get("applied", 0))


class _BadPromptEnv(_SingleTurnEnv):
    def prompt_messages(self, example):
        # content must be a string, content-block list, or null; an int is malformed
        return [{"role": "user", "content": 123}]


class _SystemExitRewardEnv(_SingleTurnEnv):
    def reward(self, completion, example, state=None):
        raise SystemExit(0)


class _NonTextSftEnv(_SingleTurnEnv):
    def sft_completion(self, example):
        # an image-only content block carries no replay text, so the driver falls to echo
        return [{"role": "assistant", "content": [{"type": "image", "image": "x.png"}]}]


class _TextBlockSftEnv(_SingleTurnEnv):
    def sft_completion(self, example):
        # a valid gold answer in openai text-block form, the same shape the real reward
        # path flattens before grading; it must be replayed, not echoed
        return [{"role": "assistant", "content": [{"type": "text", "text": "4"}]}]

    def reward(self, completion, example, state=None):
        self.completions.append(completion)
        return 1.0 if completion == "4" else 0.0


class _ScalarSftEnv(_SingleTurnEnv):
    def sft_completion(self, example):
        # scalar content is a malformed message envelope, not a non-text replay target
        return [{"role": "assistant", "content": 123}]


class _EmptyReplyMultiTurnEnv(_MultiTurnEnv):
    def env_reply(self, messages, state):
        # env yields no further messages, which the worker treats as terminal
        state["turn"] += 1
        return []


class _MalformedReplyMultiTurnEnv(_MultiTurnEnv):
    def env_reply(self, messages, state):
        # a non-empty but malformed reply (scalar content) would break the real chat
        # template on the next turn, so the driver must fail the episode, not pass on it
        state["turn"] += 1
        reply = {"role": "user", "content": 123}
        messages.append(reply)
        return [reply]


def _environment_dir(tmp_path):
    env_dir = tmp_path / "local-env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**kwargs): pass\n")
    return env_dir


def _patch_loader(monkeypatch, env, seen=None):
    seen = {} if seen is None else seen

    def load(reference, **kwargs):
        seen["reference"] = reference
        seen["kwargs"] = kwargs
        return env

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", load)
    return seen


def _args(path, **overrides):
    namespace = argparse.Namespace(path=str(path), split=None, param=[])
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


def test_env_test_single_turn_replays_reference_and_passes(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv()
    seen = _patch_loader(monkeypatch, env)
    args = cli._build_parser().parse_args(["env", "test", str(env_dir)])

    assert args.func is cmd_env_test
    assert args.func(args) == 0
    assert seen["reference"] == str((env_dir / "environment.py").resolve())
    assert env.completions == ["4"]
    out = capsys.readouterr().out
    assert "episode 1: policy=replay turns=1 reward=1.000000" in out
    assert "1/1 episodes passed contract checks" in out
    assert "overall: PASS" in out


def test_env_test_without_evaluations_keeps_output_byte_identical(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert captured.out == (
        "episode 1: policy=replay turns=1 reward=1.000000\n"
        "  prompt: user: what is 2 + 2?\n"
        "  response: 4\n"
        "1/1 episodes passed contract checks\n"
        "overall: PASS\n"
    )
    assert captured.err == ""


def test_env_test_validates_evaluation_sidecar_offline(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'held-out'\n"
        "    def cases(self): return [EvalCase(input='what is 2 + 2?', expected='4')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    output = capsys.readouterr().out
    assert (
        "evaluation suite held-out: 1/1 cases passed contract checks mean_score=1.000000" in output
    )
    assert "overall: PASS" in output


def test_env_test_rejects_an_evaluation_case_whose_image_cannot_be_resolved(
    monkeypatch, tmp_path, capsys
):
    # `prompt_messages()` is only half of the prompt: env eval and every training worker then run
    # `normalize_prompt_images`. checking only the message envelope approved a case whose
    # package-relative image does not exist, and the online command then recorded a
    # prompt-construction failure for a suite this gate had reported `overall: PASS`.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'held-out'\n"
        "    def cases(self): return [EvalCase(input='what is this?', expected='a cat',\n"
        "        metadata={'image': 'missing-cat.png'})]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class _ImageEnv(_SingleTurnEnv):
        package_root = env_dir

    _patch_loader(monkeypatch, _ImageEnv())

    assert cmd_env_test(_args(env_dir)) != 0
    output = capsys.readouterr().out
    assert "overall: PASS" not in output


def test_env_test_rejects_an_empty_evaluation_suite(monkeypatch, tmp_path, capsys):
    # approving 0/0 contract checks makes the offline gate pass a sidecar that env eval refuses
    # to run, so the first online use fails after setup already declared the package valid.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'empty'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "evaluation suite empty failed contract checks: suite produced no cases" in captured.err
    assert "0/0 cases passed contract checks" not in captured.out
    assert "overall: FAIL" in captured.err


def test_env_test_fails_when_a_scorer_reports_an_error(monkeypatch, tmp_path, capsys):
    # a scorer that returned an error graded nothing. `flash env eval` counts those as errors
    # and fails the suite, so approving them offline would greenlight exactly the sidecar the
    # online command refuses -- the gate passing is what hides it.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase, EvalResult\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'judged'\n"
        "    def cases(self): return [EvalCase(id='a', input='what is 2 + 2?', expected='4')]\n"
        "    def score(self, case, response_text):\n"
        "        return EvalResult(case_id='a', passed=False, score=0.0, response=response_text,\n"
        "                          error='judge unavailable')\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "1/1 case(s) reported a scoring error" in captured.err
    assert "judge unavailable" in captured.err
    assert "cases passed contract checks" not in captured.out
    assert "overall: FAIL" in captured.err


def test_env_test_fails_when_a_held_out_case_breaks_prompt_construction(
    monkeypatch, tmp_path, capsys
):
    """Build each held-out prompt exactly as ``flash env eval`` will.

    Validating only dataset rows can pass while every held-out case fails prompt construction.
    """
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'held-out'\n"
        "    def cases(self): return [EvalCase(id='a', input='held-out only', expected='4')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class _PickyPrompt(_SingleTurnEnv):
        def prompt_messages(self, example):
            if example["input"] != "what is 2 + 2?":
                raise KeyError("no template for this input")
            return super().prompt_messages(example)

    _patch_loader(monkeypatch, _PickyPrompt())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "evaluation suite held-out failed contract checks" in captured.err
    assert "no template for this input" in captured.err
    assert "cases passed contract checks" not in captured.out
    assert "overall: FAIL" in captured.err


def test_env_test_malformed_evaluation_sidecar_fails(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    sidecar = env_dir / "evaluations.py"
    sidecar.write_text("EVALUATIONS = [object()]\n")
    _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert str(sidecar) in captured.err
    assert "non-empty string name" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_auto_falls_back_to_echo_for_empty_reference(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "say anything", "output": ""}], reward=0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    assert env.completions == ["test"]
    captured = capsys.readouterr()
    assert "episode 1: policy=echo turns=1 reward=0.000000" in captured.out
    assert "1/1 episodes passed contract checks" in captured.out
    assert "warning:" not in captured.err


def test_env_test_fails_when_every_replayed_gold_answer_scores_zero(monkeypatch, tmp_path, capsys):
    # LS-005: the command warned per episode and passed anyway, so an environment whose reward
    # function cannot recognize its own reference answers reached a gpu where it could only ever
    # see flat-zero reward.
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(
        rows=[{"input": "what is 2 + 2?", "output": "4"}, {"input": "2 + 3?", "output": "5"}],
        reward=0.0,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    # the episodes themselves are contract-clean; it is the run-wide signature that fails.
    assert "2/2 episodes passed contract checks" in captured.out
    # this grader returns 0.0 for anything, so the wrong answer it is checked against scores 0.0
    # too: the zero really is flat, not the low end of a scale.
    assert "all 2 replayed gold answer(s) scored zero, no better than a deliberately wrong" in (
        captured.err
    )
    assert "overall: FAIL" in captured.err
    assert "overall: PASS" not in captured.out


def test_env_test_partial_zero_gold_answers_warn_but_pass(monkeypatch, tmp_path, capsys):
    # deliberately narrow: a strict reward function with some hard rows is legitimate and must not
    # be blocked. only every replayed answer scoring zero is the broken-grader signature.
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(
        rows=[{"input": "what is 2 + 2?", "output": "4"}, {"input": "2 + 3?", "output": "5"}]
    )
    rewards = iter([0.0, 1.0])
    monkeypatch.setattr(env, "reward", lambda completion, example, state=None: next(rewards))
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "replay gold answer scored low" in captured.err
    assert "scored zero" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_does_not_blame_the_grader_for_a_reference_it_cannot_replay(
    monkeypatch, tmp_path, capsys
):
    # a gold answer written in reasoning markup is graded on what survives the `<think>` strip in a
    # thinking run, but this command has no run config to read `thinking` from and replays the
    # tagged reference verbatim. against a strict answer-only grader every reference then scores
    # zero, and the gate reported a working environment as unable to recognize its own gold answers.
    # the evidence for that conclusion cannot be produced from here.
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(
        rows=[
            {"input": "what is 2 + 2?", "output": "<think>2 plus 2</think>4"},
            {"input": "2 + 3?", "output": "<think>2 plus 3</think>5"},
        ],
        reward=0.0,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "2/2 episodes passed contract checks" in captured.out
    # the zero is still surfaced -- it is worth seeing -- it just cannot be the reason to fail.
    assert "replay gold answer scored low" in captured.err
    assert "cannot recognize its own reference answers" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_control_answer_differs_from_a_gold_answer_that_is_the_control_text(
    monkeypatch, tmp_path, capsys
):
    """Ensure the negative control differs from the gold answer.

    A fixed ``test`` control can equal the reference and falsely make a separating centered scorer
    look flat.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "echo the word", "output": "test"}])
    graded: list[str] = []

    def reward(completion, example, state=None):
        graded.append(completion)
        # centered: the reference earns 0, anything else earns less.
        return 0.0 if completion.strip() == example["output"] else -1.0

    monkeypatch.setattr(env, "reward", reward)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    # the probe must have been handed something the grader scores as wrong, not the gold answer.
    assert graded[-1].strip() != "test"
    assert "cannot recognize its own reference answers" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_does_not_blame_the_grader_for_a_replay_shorter_than_the_episode(
    monkeypatch, tmp_path, capsys
):
    """Do not treat a partial multi-turn replay as a graded reference trajectory.

    Padding later turns with the junk control makes the episode part reference and part control.
    """
    env_dir = _environment_dir(tmp_path)
    env = _MultiTurnEnv()
    # the episode runs two model turns; the reference supplies only the first.
    monkeypatch.setattr(
        env,
        "dataset",
        lambda: [
            {
                "input": "finish the exchange",
                "output": [{"role": "assistant", "content": "first"}],
            }
        ],
    )
    monkeypatch.setattr(env, "reward", lambda completion, example, state=None: 0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    # the zero is still surfaced -- it is worth seeing -- it just cannot be the reason to fail.
    assert "replay gold answer scored low" in captured.err
    assert "cannot recognize its own reference answers" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_still_blames_the_grader_for_a_replay_that_covers_the_episode(
    monkeypatch, tmp_path, capsys
):
    """The exclusion above is scoped to references that ran out, not to multi-turn as such.

    A gold answer covering every turn of the episode is replayed faithfully, so its zero is real
    evidence about the grader and must still block.
    """
    env_dir = _environment_dir(tmp_path)
    env = _MultiTurnEnv()
    monkeypatch.setattr(env, "reward", lambda completion, example, state=None: 0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "all 1 replayed gold answer(s) scored zero, no better than a deliberately wrong" in (
        captured.err
    )
    assert "overall: FAIL" in captured.err


def test_env_test_does_not_apply_the_reward_gate_to_an_sft_environment(
    monkeypatch, tmp_path, capsys
):
    """Skip the reward gate for SFT and OPD.

    SFT uses supervised loss and OPD uses teacher-token loss, so a no-op scorer is valid.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "what is 2 + 2?", "output": "4"}], reward=0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "cannot recognize its own reference answers" not in captured.err
    # the zero is still surfaced; it just cannot be the reason to fail an algorithm that ignores it.
    assert "replay gold answer scored low" in captured.err
    assert "overall: PASS" in captured.out


class _ImagePromptEnv(_SingleTurnEnv):
    def prompt_messages(self, example):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": example["input"]}, {"type": "image"}],
            }
        ]


class _ImageCompletionEnv(_SingleTurnEnv):
    def sft_completion(self, example):
        return [{"role": "assistant", "content": [{"type": "image"}]}]


def test_env_test_rejects_an_image_environment_for_sft(monkeypatch, tmp_path, capsys):
    """This command exists so a contract failure costs a local run instead of a submission.

    An image-bearing sft environment is rejected at submit, so `overall: PASS` here would send the
    user to the one surface guaranteed to turn them away.
    """
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _ImagePromptEnv())

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 1
    captured = capsys.readouterr()
    assert "image-bearing SFT is not supported" in captured.err
    assert "grpo or opd" in captured.err
    assert "overall: PASS" not in captured.out


def test_env_test_rejects_an_image_sft_completion(monkeypatch, tmp_path, capsys):
    """The completion is checked separately because submit rejects it separately.

    An image in `sft_completion` clears the prompt-side gate, so checking only the prompt would
    pass the one shape that survives every other offline check and fails after submission.
    """
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _ImageCompletionEnv())

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 1
    captured = capsys.readouterr()
    assert "image-bearing SFT completions are not supported" in captured.err
    assert "overall: PASS" not in captured.out


class _LateImageEnv(_SingleTurnEnv):
    """Text rows, then a row whose prompt raises, then an image row."""

    def __init__(self):
        super().__init__(
            rows=[
                {"input": "a", "output": "1"},
                {"input": "b", "output": "2"},
                {"input": "c", "output": "3"},
                {"input": "boom", "output": "4"},
                {"input": "d", "output": "5", "image": "dataset/red.png"},
            ]
        )

    def prompt_messages(self, example):
        if example["input"] == "boom":
            raise ValueError("prompt construction exploded")
        return super().prompt_messages(example)


def test_env_test_finds_an_image_row_past_a_row_whose_prompt_raises(monkeypatch, tmp_path, capsys):
    """A raising hook is not an image verdict, and must not end the scan.

    The episode loop drives only the first few rows, so abandoning the scan at the first broken
    row would let an image row further down the dataset reach `overall: PASS` -- the exact silent
    approval this check exists to prevent.
    """
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _LateImageEnv())

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 1
    captured = capsys.readouterr()
    assert "image-bearing SFT is not supported" in captured.err
    assert "overall: PASS" not in captured.out


def test_env_test_still_passes_an_image_environment_for_grpo(monkeypatch, tmp_path, capsys):
    """The limit is sft's, not the model's. grpo and opd tokenize on the worker and are unaffected.

    Without this control the rejection could be widened to every algorithm and the suite would
    still be green.
    """
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _ImagePromptEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert "overall: PASS" in capsys.readouterr().out


def test_env_test_keeps_the_reward_gate_when_no_algorithm_is_given(monkeypatch, tmp_path, capsys):
    """An absent algorithm must not be the way to switch a blocking check off.

    The flag defaults to grpo -- the one algorithm that trains from the reward -- and a caller that
    passes no algorithm at all still gets the gate.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "what is 2 + 2?", "output": "4"}], reward=0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "cannot recognize its own reference answers" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_reads_per_turn_rewards_before_calling_the_grader_flat(
    monkeypatch, tmp_path, capsys
):
    """Read per-turn rewards before judging a grader by its episode scalar.

    ``credit_assignment = "per_turn"`` trains through ``rollout_rewards_many`` in
    flash/envs/adapter.py, so a flat ``env.reward`` can still hide separating turn rewards.
    """
    env_dir = _environment_dir(tmp_path)
    env = _MultiTurnEnv()
    monkeypatch.setattr(env, "reward", lambda completion, example, state=None: 0.0)
    # the turns separate even though the episode scalar is flat.
    monkeypatch.setattr(
        env,
        "rollout_rewards_many",
        lambda items: [RolloutReward(episode=0.0, turns=(1.0, 0.0)) for _ in items],
        raising=False,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "cannot recognize its own reference answers" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_still_blames_the_grader_for_a_flat_per_turn_vector(monkeypatch, tmp_path, capsys):
    """The exclusion is separation, not the mere presence of a per-turn vector.

    A vector whose turns are all identical distinguishes nothing the scalar did not, so it is no
    evidence that the grader recognizes its references and the gate must still block.
    """
    env_dir = _environment_dir(tmp_path)
    env = _MultiTurnEnv()
    monkeypatch.setattr(env, "reward", lambda completion, example, state=None: 0.0)
    monkeypatch.setattr(
        env,
        "rollout_rewards_many",
        lambda items: [RolloutReward(episode=0.0, turns=(0.0, 0.0)) for _ in items],
        raising=False,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "cannot recognize its own reference answers" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_still_blames_the_grader_for_plain_gold_answers(monkeypatch, tmp_path, capsys):
    # the exclusion above is scoped to the markup it cannot reproduce: a plain reference alongside a
    # tagged one is replayed faithfully, so its zero is real evidence and must still block.
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(
        rows=[
            {"input": "what is 2 + 2?", "output": "<think>2 plus 2</think>4"},
            {"input": "2 + 3?", "output": "5"},
        ],
        reward=0.0,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    # 1, not 2: only the replayable reference is counted, and it is enough to fail the run.
    assert "all 1 replayed gold answer(s) scored zero, no better than a deliberately wrong" in (
        captured.err
    )
    assert "overall: FAIL" in captured.err


def test_env_test_low_reward_warning_names_the_gold_completion_too(monkeypatch, tmp_path, capsys):
    """The warning must not blame the grader alone for a zero it did not cause.

    A dataset whose `output` is a bare value replays that value as the gold turn, so a grader that
    requires a wrapper is right to score it zero. Naming only the reward function sends the reader
    to edit a working scorer; the message names both candidates and prints the text it scored.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "what is 2 + 2?", "output": "4"}], reward=0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "check the reward function or the gold completion it scored" in captured.err
    # the exact text that scored zero, so a bare `4` is visibly the wrong gold turn.
    assert "scored text: '4'" in captured.err


def test_env_test_surfaces_the_scorer_error_behind_a_zero_reward(monkeypatch, tmp_path, capsys):
    """A crashed scorer and a judged-wrong answer both reach the CLI as 0.0.

    `FreesoloEnvironment.reward` keeps only `RewardResult.score`, so a missing runtime dependency is
    reported as a bare zero. The error string is what names it, so the warning must print it.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "what is 2 + 2?", "output": "4"}], reward=0.0)
    env.reward_with_error = lambda completion, example, state=None: (
        0.0,
        "ModuleNotFoundError: No module named 'pymongo'",
        completion,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "scorer error: ModuleNotFoundError: No module named 'pymongo'" in captured.err


def test_env_test_omits_the_scorer_error_line_when_the_scorer_reported_none(
    monkeypatch, tmp_path, capsys
):
    # an env that reports no error must not grow an empty `scorer error:` line, which would read as
    # a crash on every deliberately-zero reward.
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "what is 2 + 2?", "output": "4"}], reward=0.0)
    env.reward_with_error = lambda completion, example, state=None: (0.0, "", completion)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "scorer error:" not in captured.err


def test_env_test_reports_the_text_the_scorer_actually_received(monkeypatch, tmp_path, capsys):
    """A multi-turn env may override the answer, and the diagnostic must follow it.

    When `step_episode` returns a `final_response_text`, the adapter replaces the episode's
    `response_text` with it and the scorer grades the replacement, not the replayed turns. Printing
    the replayed turns labelled that text as the one that scored zero, sending the reader to edit a
    dataset row the grader never saw. Driven through the real adapter, since the override lives
    there.
    """
    from freesolo.datasets.types import TaskExample
    from freesolo.environments import (
        EnvironmentEpisode,
        EnvironmentMultiTurn,
        EnvironmentStepResult,
        RewardResult,
    )

    from flash.envs.adapter import FreesoloEnvironment

    graded: list[str] = []

    class _Env(EnvironmentMultiTurn):
        dataset: ClassVar[list] = [
            {"input": "guess", "output": [{"role": "assistant", "content": "RAW_TURN"}]}
        ]

        def build_prompt_messages(self, example: TaskExample, prompt_text: str):
            return [{"role": "user", "content": example.input}]

        def start_episode(self, example: TaskExample, prompt_text: str):
            return [{"role": "user", "content": example.input}]

        def sft_completion(self, example: TaskExample):
            return list(example.output or [])

        def max_episode_turns(self, example: TaskExample) -> int:
            return 2

        def step_episode(self, example, messages, assistant_response):
            return EnvironmentStepResult(done=True, final_response_text="ENV_OVERRODE")

        def score_episode(self, example, episode: EnvironmentEpisode) -> RewardResult:
            graded.append(str(episode.response_text))
            return RewardResult(score=0.0, threshold=1.0)

    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, FreesoloEnvironment(_Env(), "env", source=None))

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    # the scorer really did receive the override, so that is the only honest thing to print.
    assert graded == ["ENV_OVERRODE"]
    assert "scored text: 'ENV_OVERRODE'" in captured.err
    assert "scored text: 'RAW_TURN'" not in captured.err


def test_env_test_reports_an_empty_override_as_the_scored_text(monkeypatch, tmp_path, capsys):
    """An env that overrode the answer to nothing graded an empty string, and must say so.

    `step_episode` propagates its override on `is not None` (flash/envs/adapter.py), so `""` really
    does reach the grader. Treating a captured `""` as "never captured" fell back to the replayed
    turns and named text the scorer never saw -- and an empty answer is the very fault a reader
    most needs pointed at, since it explains the zero on its own.
    """
    from freesolo.datasets.types import TaskExample
    from freesolo.environments import (
        EnvironmentEpisode,
        EnvironmentMultiTurn,
        EnvironmentStepResult,
        RewardResult,
    )

    from flash.envs.adapter import FreesoloEnvironment

    graded: list[str] = []

    class _Env(EnvironmentMultiTurn):
        dataset: ClassVar[list] = [
            {"input": "guess", "output": [{"role": "assistant", "content": "RAW_TURN"}]}
        ]

        def build_prompt_messages(self, example: TaskExample, prompt_text: str):
            return [{"role": "user", "content": example.input}]

        def start_episode(self, example: TaskExample, prompt_text: str):
            return [{"role": "user", "content": example.input}]

        def sft_completion(self, example: TaskExample):
            return list(example.output or [])

        def max_episode_turns(self, example: TaskExample) -> int:
            return 2

        def step_episode(self, example, messages, assistant_response):
            # the env discarded the model's turn entirely -- a real override, to empty.
            return EnvironmentStepResult(done=True, final_response_text="")

        def score_episode(self, example, episode: EnvironmentEpisode) -> RewardResult:
            graded.append(str(episode.response_text))
            return RewardResult(score=0.0, threshold=1.0)

    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, FreesoloEnvironment(_Env(), "env", source=None))

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    # the grader really did receive the empty override, so that is what must be reported.
    assert graded == [""]
    assert "scored text: ''" in captured.err
    # the replayed turn was never scored; naming it would send the reader to the wrong place.
    assert "scored text: 'RAW_TURN'" not in captured.err


def test_env_test_surfaces_a_scorer_error_on_an_echo_episode(monkeypatch, tmp_path, capsys):
    """An echo episode has no gold answer, which is exactly when a crash goes unreported.

    With no reference to replay the policy is `echo`, so the replay-only warning never fires and
    `replayed` stays zero, which also disables the grpo gate. A scorer crashing behind the SDK's
    guard then reached `overall: PASS` with nothing on stderr naming the cause.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "solve", "output": ""}], reward=0.0)
    env.reward_with_error = lambda completion, example, state=None: (
        0.0,
        "ModuleNotFoundError: No module named 'pymongo'",
        completion,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    # the policy really is echo: this is the path where nothing else would have spoken up.
    assert "policy=echo" in captured.out
    assert "scorer error: ModuleNotFoundError: No module named 'pymongo'" in captured.err


def test_env_test_negative_reward_scale_is_not_a_zero_reward_grader(monkeypatch, tmp_path, capsys):
    # a grader scaled -1 for a correct reference and -2 for an incorrect completion still separates
    # them, which is all GRPO's relative advantage needs. counting every non-positive reward as a
    # zero made the gate reject those environments outright.
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(
        rows=[{"input": "what is 2 + 2?", "output": "4"}, {"input": "2 + 3?", "output": "5"}],
        reward=-1.0,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    # the advisory warning still fires -- it only advises -- but the run-wide gate does not.
    assert "replay gold answer scored low (reward=-1.000000)" in captured.err
    assert "scored zero" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_centered_reward_scale_passes_on_what_the_grader_pays_junk(
    monkeypatch, tmp_path, capsys
):
    """Accept a centered grader that pays zero for correct and less for wrong.

    Gold zero alone is ambiguous, so the gate must score a deliberately wrong answer.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(
        rows=[{"input": "what is 2 + 2?", "output": "4"}, {"input": "2 + 3?", "output": "5"}]
    )
    scored = []

    def _centered(completion, example, state=None):
        scored.append(completion)
        return 0.0 if completion == example["output"] else -1.0

    monkeypatch.setattr(env, "reward", _centered)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "cannot recognize its own reference answers" not in captured.err
    assert "overall: PASS" in captured.out
    # both gold answers scored 0.0 and the run still passed, so the verdict came from the extra
    # wrong-answer call -- and it took exactly one, after the two episodes, not one per episode.
    assert scored == ["4", "5", "test"]


def test_env_test_non_text_sft_completion_uses_echo(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    env = _NonTextSftEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    assert env.completions == ["test"]
    out = capsys.readouterr().out
    assert "episode 1: policy=echo turns=1" in out
    assert "overall: PASS" in out


def test_env_test_text_block_sft_completion_replays(monkeypatch, tmp_path, capsys):
    # a gold answer in openai text-block form must be replayed and graded as that text,
    # matching the real reward path, instead of falling back to the canned echo
    env_dir = _environment_dir(tmp_path)
    env = _TextBlockSftEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    assert env.completions == ["4"]
    out = capsys.readouterr().out
    assert "episode 1: policy=replay turns=1 reward=1.000000" in out
    assert "overall: PASS" in out


class _UserOnlySftEnv(_SingleTurnEnv):
    def sft_completion(self, example):
        # gold completion with NO assistant turn: user/system text must never stand in for the
        # model response -- the driver has to fall back to echo instead of replaying it.
        return [
            {"role": "system", "content": "system preamble"},
            {"role": "user", "content": "user text that must not be replayed"},
        ]

    def reward(self, completion, example, state=None):
        self.completions.append(completion)
        return 1.0 if completion == "4" else 0.0


def test_env_test_non_assistant_sft_completion_uses_echo(monkeypatch, tmp_path, capsys):
    # regression: a gold completion whose only messages are user/system (no assistant) must echo,
    # not replay the user/system text as the model response.
    env_dir = _environment_dir(tmp_path)
    env = _UserOnlySftEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    out = capsys.readouterr().out
    assert "episode 1: policy=echo turns=1" in out
    assert "user text that must not be replayed" not in "".join(env.completions)


def test_env_test_scalar_sft_completion_fails_contract(monkeypatch, tmp_path, capsys):
    # a malformed gold answer (scalar content) must fail the episode, not echo-pass; the
    # adapter passes scalars through untouched, so the driver has to reject them itself
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _ScalarSftEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "0/1 episodes passed contract checks" in captured.out
    assert "sft_completion is not well-formed" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_multi_turn_terminates_and_scores(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    env = _MultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    assert env.scored_state is not None
    assert env.scored_state["turn"] == 2
    out = capsys.readouterr().out
    assert "episode 1: policy=replay turns=2 reward=0.500000" in out
    assert "1/1 episodes passed contract checks" in out


def test_env_test_multi_turn_replays_text_free_turn_positionally(monkeypatch, tmp_path, capsys):
    # dropping the null tool-call turn would shift "third" into its slot and misgrade; the
    # driver must replay ["first", "", "third"] positionally.
    env_dir = _environment_dir(tmp_path)
    env = _TextFreeMultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    assert env.recorded == ["first", "", "third"]
    out = capsys.readouterr().out
    assert "overall: PASS" in out


def test_env_test_steps_the_env_on_the_final_turn_before_scoring(monkeypatch, tmp_path, capsys):
    # the worker applies the last assistant turn before scoring (see the turn loops in
    # opd_train / rl_train). without the same close-out here the command scores a
    # board missing the model's last move, so it disagrees with the paid run it exists to predict:
    # a passing env can look unrankable, and a final step that raises goes unseen until training.
    env_dir = _environment_dir(tmp_path)
    env = _StatefulBoardMultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    assert env.scored_state["applied"] == 3, "the final model turn was scored without being applied"
    assert "reward=3.000000" in capsys.readouterr().out


def test_env_test_multi_turn_bounds_turns_to_hard_cap(monkeypatch, tmp_path, capsys):
    # per-example cap (12) exceeds the adapter hard ceiling (max_turns=8); the worker
    # stops at the hard cap before asking for more turns, so the offline driver must too
    env_dir = _environment_dir(tmp_path)
    env = _PerExampleCapMultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    out = capsys.readouterr().out
    assert "episode 1: policy=replay turns=8 reward=0.500000" in out
    assert "overall: PASS" in out


def test_env_test_drives_three_episodes_by_default(monkeypatch, tmp_path, capsys):
    # the command always drives a fixed number of episodes (no --episodes flag); a larger
    # dataset is capped at that default, a smaller one runs every available record
    env_dir = _environment_dir(tmp_path)
    rows = [{"input": f"q{i}", "output": str(i)} for i in range(5)]
    env = _SingleTurnEnv(rows=rows)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    assert len(env.completions) == 3
    out = capsys.readouterr().out
    assert "3/3 episodes passed contract checks" in out
    assert "overall: PASS" in out


def test_env_test_nan_reward_fails_contract(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=float("nan")))

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "0/1 episodes passed contract checks" in captured.out
    assert "reward is not finite: nan" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_empty_dataset_fails_contract(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(rows=[]))

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "0/0 episodes passed contract checks" in captured.out
    assert "dataset is empty" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_passes_absolute_path_to_loader(monkeypatch, tmp_path, capsys):
    # a bare relative dir must reach the loader as an absolute path, otherwise it
    # matches the managed-slug pattern and would resolve remotely instead of locally
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv()
    seen = _patch_loader(monkeypatch, env)
    monkeypatch.chdir(tmp_path)

    assert cmd_env_test(_args("local-env")) == 0
    reference = seen["reference"]
    assert Path(reference).is_absolute()
    assert reference == str((tmp_path / "local-env" / "environment.py").resolve())


def test_env_test_malformed_prompt_fails_contract(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _BadPromptEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "policy=n/a" in captured.out
    assert "0/1 episodes passed contract checks" in captured.out
    assert "prompt is not well-formed" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_multi_turn_malformed_env_reply_fails_contract(monkeypatch, tmp_path, capsys):
    # a non-empty but malformed env reply must fail the episode: those messages become
    # chat-template input for the next turn in the real rollout and would break remotely
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _MalformedReplyMultiTurnEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "0/1 episodes passed contract checks" in captured.out
    assert "env_reply is not well-formed" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_multi_turn_stops_on_empty_env_reply(monkeypatch, tmp_path, capsys):
    # an empty env_reply is terminal in the worker loop; the driver must stop and score
    # rather than keep driving turns to the hard cap
    env_dir = _environment_dir(tmp_path)
    env = _EmptyReplyMultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    assert env.scored_state is not None
    assert env.scored_state["turn"] == 1
    out = capsys.readouterr().out
    assert "episode 1: policy=replay turns=1 reward=0.500000" in out
    assert "overall: PASS" in out


def test_env_test_replay_low_reward_warns_per_episode(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=0.0))

    # the per-episode warning still names the row, and with this the run's ONLY replayed answer,
    # the whole-run gate below it fails the command (see the all-zero test above).
    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=1 reward=0.000000" in captured.out
    assert "1/1 episodes passed contract checks" in captured.out
    assert "warning:" in captured.err
    assert "check the reward function" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_systemexit_from_reward_fails_contract(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SystemExitRewardEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "0/1 episodes passed contract checks" in captured.out
    assert "failed contract checks" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_split_flag_reaches_loader(monkeypatch, tmp_path):
    # the gate must validate the split a run actually trains on; without --split it always
    # loaded the environment's default split and could pass while the configured one was
    # never exercised.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())
    args = cli._build_parser().parse_args(["env", "test", str(env_dir), "--split", "validation"])

    assert args.func(args) == 0
    assert seen["kwargs"] == {"split": "validation"}


def test_env_test_param_flag_parses_toml_scalars(monkeypatch, tmp_path):
    # --param mirrors [environment.params], so values keep the types the config would give
    # them rather than arriving as strings.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())
    args = cli._build_parser().parse_args(
        [
            "env",
            "test",
            str(env_dir),
            "--param",
            "max_rows=5",
            "--param",
            "strict=true",
            "--param",
            "name=hard",
        ]
    )

    assert args.func(args) == 0
    assert seen["kwargs"] == {"max_rows": 5, "strict": True, "name": "hard"}


@pytest.mark.parametrize(
    "value",
    ["filters=[1,2", 'name="unterminated', "opts={a=1", "tags=['x'"],
)
def test_env_test_malformed_structured_param_fails_instead_of_becoming_a_string(
    monkeypatch, tmp_path, capsys, value
):
    # a value that opens a quote, array, or inline table is structured but malformed. keeping it
    # as a literal string would validate the environment against parameters the equivalent
    # [environment.params] entry could never load, so the offline check would pass on inputs
    # training rejects.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 1
    assert "kwargs" not in seen
    assert "is not a valid TOML value" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["filters=]", "filters=}", "filters=1,2]", "x=a=b"])
def test_env_test_malformed_param_without_an_opening_delimiter_is_rejected(
    monkeypatch, tmp_path, capsys, value
):
    # malformedness is not signalled by the first character: `filters=]` opens nothing yet is still
    # invalid TOML, so deciding by opening delimiter alone let it through as the literal string "]".
    # the test is whether the value reaches for TOML structure at all, not how it starts.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 1
    assert "kwargs" not in seen
    assert "is not a valid TOML value" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["cutoff=2026-01-01", "at=07:32:00", "when=2026-01-01T07:32:00"])
def test_env_test_json_incompatible_param_types_are_rejected(monkeypatch, tmp_path, capsys, value):
    # TOML has date/time types JSON does not, so these parse cleanly into datetime objects. the
    # equivalent [environment.params] keeps the object and the submit dies at json.dumps(body) in
    # ApiClient._request(), so passing here would approve a config that cannot be submitted at all.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 1
    assert "kwargs" not in seen
    assert "not JSON-serializable" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["threshold=nan", "threshold=inf", "threshold=-inf"])
def test_env_test_non_finite_param_floats_are_rejected(monkeypatch, tmp_path, capsys, value):
    # tomllib accepts nan/inf, and json.dumps does NOT raise on them -- it emits the non-standard
    # tokens NaN and Infinity. so this passed the gate and produced a request body that is not
    # JSON, which a strict parser rejects on submit and again when the value comes back in
    # RunStatus.spec. serializable is not the same test as valid.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 1
    assert "kwargs" not in seen
    assert "not JSON-serializable" in capsys.readouterr().err


@pytest.mark.parametrize(
    "value", ["threshold=NaN", "threshold=Inf", "threshold=-Inf", "threshold=NAN", "threshold=INF"]
)
def test_a_case_variant_of_a_non_finite_float_is_not_forwarded_as_text(
    monkeypatch, tmp_path, capsys, value
):
    # TOML spells the non-finite floats lowercase only, so a case variant fails the parse and used
    # to fall through the bare-word test as the literal STRING "NaN" -- never reaching the JSON
    # check that turns the lowercase spelling away. the gate then validated a str where the config
    # holds a float, or an env coercing it back got the very value that check exists to reject.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 1
    assert "kwargs" not in seen
    assert "lowercase inf and nan" in capsys.readouterr().err


@pytest.mark.parametrize(
    "value",
    [
        "threshold=Infinity",
        "threshold=infinity",
        "threshold=INFINITY",
        "threshold=-Infinity",
        "threshold=+infinity",
    ],
)
def test_a_full_length_infinity_is_not_forwarded_as_text(monkeypatch, tmp_path, capsys, value):
    # `infinity` is the same value written out, and it is not a TOML spelling in any case -- so it
    # reaches the bare-word test rather than the parse. matching only the `inf` abbreviation let it
    # through as the string "Infinity", which an env normalizing with float() turns straight back
    # into inf: the value the abbreviation is rejected for, reached by a spelling
    # [environment.params] cannot parse at all.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 1
    assert "kwargs" not in seen
    assert "lowercase inf and nan" in capsys.readouterr().err


@pytest.mark.parametrize(
    "spelling",
    [
        "dataset_path=/data/\udcff/set.jsonl",
        'dataset_path="/data/\udcff/set.jsonl"',
    ],
)
def test_a_surrogate_param_value_is_rejected_like_a_surrogate_name(
    monkeypatch, tmp_path, capsys, spelling
):
    # a lone surrogate reaches argv when the command line carries a byte that is not valid UTF-8. it
    # is no more expressible on the RIGHT of an assignment than on the left, but only the name was
    # guarded -- so the loader could open the path and the gate PASS, while no UTF-8 training TOML
    # could submit the run that was validated.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[spelling])) == 1
    assert "kwargs" not in seen
    assert "not valid UTF-8" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [("note=café", "café"), ("note=你好", "你好"), ("path=/data/ok.jsonl", "/data/ok.jsonl")],
)
def test_ordinary_non_ascii_values_are_unaffected_by_the_utf8_check(
    monkeypatch, tmp_path, spelling, expected
):
    # the check is only correct if it admits what a UTF-8 config CAN carry. a basic string holds any
    # encodable character, so rejecting non-ascii text would cost a spelling the config supports.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[spelling])) == 0
    assert seen["kwargs"][spelling.partition("=")[0]] == expected


@pytest.mark.parametrize("value", ["difficulty.level=3", "a.b.c=1"])
def test_env_test_param_keys_that_denote_toml_structure_are_rejected(
    monkeypatch, tmp_path, capsys, value
):
    # the left side of an [environment.params] entry is a TOML key, so `difficulty.level = 3` in a
    # config means {"difficulty": {"level": 3}}. forwarding the source spelling literally sent
    # {"difficulty.level": 3} instead, and an environment taking **kwargs swallows that without
    # ever exercising the nested parameter -- the gate passes on a call training never makes.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 1
    assert "kwargs" not in seen
    assert "TOML key syntax" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("value", "name"),
    [('"release.channel"=3', "release.channel"), ('"weird key"=1', "weird key")],
)
def test_a_quoted_param_key_names_itself_rather_than_nesting(monkeypatch, tmp_path, value, name):
    # a dot only nests when it is BARE: `"release.channel" = 3` is a quoted key and produces the
    # flat {"release.channel": 3}. classifying dots and quotes as structure outright left that valid
    # config with no --param spelling at all, so the command could not mirror it. quoting is also
    # exactly the text the config needs, so the flag and the config stay in step.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 0
    assert name in seen["kwargs"]


@pytest.mark.parametrize(
    ("value", "name", "expected"),
    [
        ('"a=b"=1', "a=b", 1),
        ("'a=b'=2", "a=b", 2),
        ('"pct=100"="done"', "pct=100", "done"),
    ],
)
def test_a_quoted_param_key_may_contain_an_equals_sign(
    monkeypatch, tmp_path, value, name, expected
):
    # `[environment.params] "a=b" = 1` loads as the flat key `a=b`, so the run really can receive
    # this name. splitting at the first `=` made the key `"a` and rejected the argument, leaving
    # that config with no --param spelling able to validate it.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 0
    assert seen["kwargs"][name] == expected


def test_a_null_spelling_is_not_forwarded_as_text(monkeypatch, tmp_path, capsys):
    # TOML has no null, so these bare words fail the parse, carry no structural character, and
    # forward as their own literal string -- truthy, where the spelling asked for absent. no
    # [environment.params] entry could produce that value either.
    env_dir = _environment_dir(tmp_path)

    for spelling in ("null", "NULL", "None", "none", "nil"):
        seen = _patch_loader(monkeypatch, _SingleTurnEnv())
        assert cmd_env_test(_args(env_dir, param=[f"value={spelling}"])) == 1
        assert "kwargs" not in seen
        assert "TOML has no null" in capsys.readouterr().err


def test_a_malformed_param_key_spelling_is_rejected(monkeypatch, tmp_path, capsys):
    # resolving the spelling through tomllib means an unterminated quote is a key TOML cannot read
    # at all, not a name. [environment.params] would reject it too, so forwarding it literally
    # would validate against a parameter no config could deliver.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=['unclosed"=1'])) == 1
    assert "kwargs" not in seen
    assert "not a valid TOML key" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["bad key=1", "a/b=1", "a@b=1", "k(x)=1", "café=1", "a😀b=1"])
def test_env_test_param_keys_a_quoted_config_key_can_carry_still_load(monkeypatch, tmp_path, value):
    # these are not BARE keys, but that is not the question -- a QUOTED key carries every one of
    # them and the schema loader reads it, so `"bad key" = 1` and `"café" = 1` are configs a run
    # really can receive. rejecting them blocked validating a working config while the error claimed
    # [environment.params] could not hold the name.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())
    key = value.split("=")[0]

    assert cmd_env_test(_args(env_dir, param=[value])) == 0
    assert key in seen["kwargs"]


def test_env_test_a_param_key_no_config_file_could_hold_is_rejected(monkeypatch, tmp_path, capsys):
    # what IS unexpressible is a name the UTF-8 config file cannot contain. a command line carrying
    # a byte that is not valid UTF-8 reaches argv as a lone surrogate, and no [environment.params]
    # entry could ever spell it -- so the run would never receive the parameter.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())
    key = b"a\xffb".decode("utf-8", "surrogateescape")

    assert cmd_env_test(_args(env_dir, param=[f"{key}=1"])) == 1
    assert "kwargs" not in seen
    assert "not valid UTF-8" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["max_rows=5", "MODE=fast", "n0=1", "keep-going=true"])
def test_env_test_valid_bare_key_params_still_load(monkeypatch, tmp_path, value):
    # the grammar check is only correct if it admits every name a config could hold: uppercase,
    # digits, underscores and dashes are all legal bare keys, and rejecting one would remove a
    # parameter spelling that works in [environment.params] today.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())
    key = value.split("=")[0]

    assert cmd_env_test(_args(env_dir, param=[value])) == 0
    assert key in seen["kwargs"]


def test_env_test_a_nested_param_passes_as_one_inline_table(monkeypatch, tmp_path):
    # the rejection above is only correct if the flag can still express the nested call, otherwise
    # it would be removing a capability rather than fixing a misforward. this is the spelling the
    # error message points at, so it has to produce the structure the config would.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=["difficulty={level = 3}"])) == 0
    assert seen["kwargs"]["difficulty"] == {"level": 3}


def test_env_test_param_with_trailing_assignment_is_rejected(monkeypatch, tmp_path, capsys):
    # a newline makes `v = <value>` a two-line document that tomllib accepts, so indexing only "v"
    # keeps max_rows and silently drops strict -- the gate would then validate a parameter set the
    # user never asked for. shell and CI quoting produce this without anyone typing a newline.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=["max_rows=5\nstrict=true"])) == 1
    assert "kwargs" not in seen
    err = capsys.readouterr().err
    assert "more than one assignment" in err
    assert "strict" in err


def test_env_test_bare_unquoted_param_still_falls_back_to_a_string(monkeypatch, tmp_path):
    # the common case stays convenient: unquoted text is not valid TOML but is what users type.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=["name=hard mode"])) == 0
    assert seen["kwargs"] == {"name": "hard mode"}


@pytest.mark.parametrize(
    "value",
    [
        "cutoff=2026-13-01",  # month 13
        "when=2026-01-32T00:00:00",  # day 32
        "at=12:99:00",  # minute 99
        "scale=1e",  # exponent with no digits
        "mask=0x",  # radix prefix with no digits
        "code=007",  # leading zeros are not a TOML integer
        "size=1_",  # trailing underscore separator
        "version=1.2.3",  # two dots is not a float
        "width=3px",  # number with a unit suffix
        "share=10%",
        "rate=.5",  # toml floats require a digit before the point
        "rate=.5e3",
    ],
)
def test_env_test_malformed_param_without_a_delimiter_is_rejected(
    monkeypatch, tmp_path, capsys, value
):
    # a token beginning with a digit, sign, or dot is scalar-shaped, not prose. if TOML cannot parse
    # it, reject it rather than forward a string that no equivalent config can express.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 1
    assert "kwargs" not in seen
    err = capsys.readouterr().err
    assert "is not a valid TOML value" in err, err
    # the remedy has to be named, because "3px" really is text to the user who typed it.
    assert "quote it" in err, err
    # and it has to survive a shell. suggesting `--param k="v"` is a remedy that fails when pasted:
    # the shell strips those quotes, argv sees `k=v` again, and the user hits the same error. the
    # quoting has to be spelled so the quotes reach argv.
    key = value.partition("=")[0]
    assert f"--param '{key}=" in err, err


@pytest.mark.parametrize("value", ["strict=False", "strict=True", "strict=TRUE", "flag=fAlse"])
def test_env_test_a_python_spelled_boolean_is_rejected_rather_than_sent_as_text(
    monkeypatch, tmp_path, capsys, value
):
    # TOML booleans are lowercase. rejecting Python-style case prevents ``False`` from becoming a
    # truthy string and reversing the run's behavior.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 1
    assert "kwargs" not in seen
    err = capsys.readouterr().err
    assert "in lowercase" in err, err
    # both remedies must be spelled out: the literal the user probably meant, and the shell-safe
    # quoting if they really did want the text.
    assert value.partition("=")[2].lower() in err, err
    assert f"--param '{value.partition('=')[0]}=" in err, err


@pytest.mark.parametrize("value", ["strict=false", "strict=true", "note=truthy", "note=False Ok"])
def test_env_test_real_booleans_and_prose_are_unaffected_by_the_case_check(
    monkeypatch, tmp_path, value
):
    # the check is only correct if it admits the lowercase literals TOML does accept, and leaves
    # ordinary prose alone -- rejecting either would cost a spelling [environment.params] supports.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())
    key, _, raw = value.partition("=")
    expected = {"false": False, "true": True}.get(raw, raw)

    assert cmd_env_test(_args(env_dir, param=[value])) == 0
    assert seen["kwargs"][key] == expected


@pytest.mark.parametrize(
    "value",
    [
        "name=v1.0",
        "path=/tmp/data",
        "tag=a-b",
        "when=x2026-01-01",
        "name=hard mode",
        "version=v1.2.3",  # dotted, but the leading char is prose
        "attr=a.b",
    ],
)
def test_env_test_text_that_merely_contains_digits_still_falls_back(monkeypatch, tmp_path, value):
    # the rejection is on how the token STARTS, not on whether digits appear in it. these are
    # ordinary prose values and must keep working unquoted, or the check would cost the convenience
    # the fallback exists for.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 0
    key, _, raw = value.partition("=")
    assert seen["kwargs"] == {key: raw}


@pytest.mark.parametrize("value", ['width="3px"', 'cutoff="2026-13-01"'])
def test_env_test_quoting_passes_a_scalar_looking_value_as_text(monkeypatch, tmp_path, value):
    # the escape hatch, and deliberately the same spelling [environment.params] needs: a genuinely
    # textual "3px" has to be quoted in the config too, so the flag and the config stay in step
    # rather than the flag accepting a spelling the config would reject.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 0
    key, _, raw = value.partition("=")
    assert seen["kwargs"] == {key: raw.strip('"')}


def test_env_test_well_formed_structured_params_still_load(monkeypatch, tmp_path):
    # the rejections above must not cost the valid structured forms [environment.params] supports,
    # or tightening the gate would just move the false failures to the other side.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())
    args = _args(
        env_dir,
        param=["filters=[1, 2]", "opts={a = 1}", 'name="hello world"', "ratio=0.5", "neg=-3"],
    )

    assert cmd_env_test(args) == 0
    assert seen["kwargs"] == {
        "filters": [1, 2],
        "opts": {"a": 1},
        "name": "hello world",
        "ratio": 0.5,
        "neg": -3,
    }


def test_env_test_split_flag_overrides_param_split(monkeypatch, tmp_path):
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, split="eval", param=["split=train"])) == 0
    assert seen["kwargs"] == {"split": "eval"}


@pytest.mark.parametrize("split", ["", "   "])
def test_env_test_explicitly_blank_split_is_rejected(monkeypatch, tmp_path, capsys, split):
    # `--split "$SPLIT"` with an unset variable is an explicit request for a split. treating it as
    # absent leaves a --param split=... in effect, so the gate validates a different split than the
    # command named -- passing on a dataset the run never trains on, which is this flag's whole job.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, split=split, param=["split=validation"])) == 1
    assert "kwargs" not in seen
    assert "--split requires a non-empty split name" in capsys.readouterr().err


def test_env_test_omitted_split_still_leaves_a_param_split_in_effect(monkeypatch, tmp_path):
    # not passing --split at all is different from passing it blank: there is no conflict to
    # resolve, so [environment.params]-style `--param split=` remains the user's stated intent.
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=["split=validation"])) == 0
    assert seen["kwargs"] == {"split": "validation"}


def test_env_test_malformed_param_fails_before_loading(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=["justakey"])) == 1
    assert "kwargs" not in seen
    captured = capsys.readouterr()
    assert "--param must be KEY=VALUE" in captured.err
    assert "overall: FAIL" in captured.err


def test_adapter_reward_with_error_exposes_what_reward_discards():
    """`reward` keeps only `score`, so the scorer's error needs its own accessor.

    A scorer whose runtime dependency is missing returns the documented 0.0 floor with `error` set.
    Reading `reward` alone reports that as an ordinary wrong answer; `reward_with_error` recovers the
    string that names the real cause.
    """
    from freesolo.datasets.types import TaskExample
    from freesolo.environments import EnvironmentSingleTurn, RewardResult

    from flash.envs.adapter import FreesoloEnvironment

    class _Env(EnvironmentSingleTurn):
        dataset: ClassVar[list] = [{"input": "find user", "output": "alice"}]

        def build_prompt_messages(self, example: TaskExample, prompt_text: str):
            return [{"role": "user", "content": example.input}]

        def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
            return RewardResult(
                score=0.0, threshold=1.0, error="ModuleNotFoundError: No module named 'pymongo'"
            )

    env = FreesoloEnvironment(_Env(), "env", source=None)
    example = env.dataset()[0]

    # the reward alone cannot tell a crashed scorer from a judged-wrong answer.
    assert env.reward("alice", example) == 0.0
    assert env.reward_with_error("alice", example) == (
        0.0,
        "ModuleNotFoundError: No module named 'pymongo'",
        "alice",
    )


def test_adapter_reward_with_error_is_empty_when_the_scorer_reported_none():
    from freesolo.datasets.types import TaskExample
    from freesolo.environments import EnvironmentSingleTurn, RewardResult

    from flash.envs.adapter import FreesoloEnvironment

    class _Env(EnvironmentSingleTurn):
        dataset: ClassVar[list] = [{"input": "what is 2 + 2?", "output": "4"}]

        def build_prompt_messages(self, example: TaskExample, prompt_text: str):
            return [{"role": "user", "content": example.input}]

        def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
            return RewardResult(score=1.0, threshold=1.0, success=True)

    env = FreesoloEnvironment(_Env(), "env", source=None)
    example = env.dataset()[0]

    assert env.reward("4", example) == 1.0
    assert env.reward_with_error("4", example) == (1.0, "", "4")


def test_env_test_scores_each_episode_exactly_once():
    """Reading the scorer's error must not cost a second scoring call.

    Scoring is not guaranteed to be pure: a rate-limited judge can answer differently the second
    time, so a re-score can report an error that did not produce the printed reward -- and bills a
    paid judge twice per episode. Both values come from one call.
    """
    from freesolo.datasets.types import TaskExample
    from freesolo.environments import EnvironmentSingleTurn, RewardResult

    from flash.envs.adapter import FreesoloEnvironment

    calls: list[str] = []

    class _Env(EnvironmentSingleTurn):
        dataset: ClassVar[list] = [{"input": "q", "output": "a"}]

        def build_prompt_messages(self, example: TaskExample, prompt_text: str):
            return [{"role": "user", "content": example.input}]

        def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
            calls.append(response_text)
            # a flaky judge: only the FIRST call reports the rate limit.
            if len(calls) == 1:
                return RewardResult(score=0.0, threshold=1.0, error="RateLimitError: 429")
            return RewardResult(score=0.0, threshold=1.0)

    env = FreesoloEnvironment(_Env(), "env", source=None)
    example = env.dataset()[0]

    reward, error, _scored = env.reward_with_error("a", example)
    assert len(calls) == 1, f"scored {len(calls)} times; a re-score would lose the real error"
    assert reward == 0.0
    # the error belongs to the call that produced this reward, not to a later, different one.
    assert error == "RateLimitError: 429"


def test_env_test_prints_the_scored_text_without_collapsing_it(monkeypatch, tmp_path, capsys):
    """The scored text is printed faithfully, since formatting IS the defect being diagnosed.

    `_preview` collapses whitespace and truncates at 200 characters, which hides exactly the faults
    that make a correct grader reject a gold answer: a stray newline, a trailing tab, or a wrapper
    that sits past the cutoff.
    """
    env_dir = _environment_dir(tmp_path)
    gold = "answer\n\t42 "
    env = _SingleTurnEnv(rows=[{"input": "q", "output": gold}], reward=0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    # repr, so the newline and tab that a collapsing preview would erase stay visible.
    assert "scored text: 'answer\\n\\t42 '" in captured.err


def test_env_test_shows_a_wrapper_past_the_preview_cutoff(monkeypatch, tmp_path, capsys):
    # a gold answer whose `\boxed{}` sits past _PREVIEW_CHARS is the case the truncating preview
    # hid -- and it is precisely the formatting evidence the reader needs.
    env_dir = _environment_dir(tmp_path)
    gold = "x" * 250 + "\\boxed{72}"
    env = _SingleTurnEnv(rows=[{"input": "q", "output": gold}], reward=0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "boxed{72}" in captured.err


def test_scaffolded_environment_documents_the_gold_completion_contract():
    """The guidance has to reach the generated file, not just the generator's source.

    A comment outside the template string is read by nobody running `flash env setup`.
    """
    from flash.cli.commands.env.setup import _STARTER_ENV_PY

    assert "gold" in _STARTER_ENV_PY
    assert "sft_completion" in _STARTER_ENV_PY
    # names the wrapper trap and where per-row scorer state belongs.
    assert "boxed" in _STARTER_ENV_PY
    assert "metadata" in _STARTER_ENV_PY
