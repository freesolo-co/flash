"""Tests for local offline environment contract validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import pytest

import flash.cli.parsing.main as cli
from flash.cli.commands.env.testing.test import cmd_env_test
from flash.envs.loading.adapter import FreesoloEnvironment
from flash.envs.loading.base import RolloutReward


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


class _ImageReplyMultiTurnEnv(_MultiTurnEnv):
    def __init__(self):
        super().__init__()
        self.reply_calls = 0
        self.state = None

    def new_rollout_state(self, example):
        self.state = super().new_rollout_state(example)
        return self.state

    def env_reply(self, messages, state):
        self.reply_calls += 1
        state["turn"] += 1
        reply = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/later.png"},
                }
            ],
        }
        messages.append(reply)
        return [reply]


class _NaturallyDoneImageReplyMultiTurnEnv(_ImageReplyMultiTurnEnv):
    max_turns = 4

    def dataset(self):
        return [
            {
                "input": "finish now",
                "output": [{"role": "assistant", "content": "finished"}],
            }
        ]

    def env_reply(self, messages, state):
        reply = super().env_reply(messages, state)
        state["done"] = True
        state["applied"] = True
        return reply


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

    monkeypatch.setattr("flash.envs.loading.loader.load_freesolo_environment", load)
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
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
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


def test_env_test_validates_episode_suite_without_executing_its_scorer(
    monkeypatch, tmp_path, capsys
):
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import EvalCase\n"
        "class Suite:\n"
        "    name = 'episode'\n"
        "    grades_episodes = True\n"
        "    def cases(self):\n"
        "        return [EvalCase(id='episode', input='finish the exchange', expected=[\n"
        "            {'role': 'assistant', 'content': 'first'},\n"
        "            {'role': 'assistant', 'content': 'second'},\n"
        "        ])]\n"
        "    def score(self, case, response, state):\n"
        "        if not state.get('done'):\n"
        "            raise AssertionError('episode scorer requires finished state')\n"
        "        return True\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class _CountingEnv(_MultiTurnEnv):
        def __init__(self):
            super().__init__()
            self.starts = 0

        def new_rollout_state(self, example):
            self.starts += 1
            return super().new_rollout_state(example)

    env = _CountingEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    assert env.starts == 2  # one dataset episode and one held-out initial state
    assert env.scored_state is not None
    assert env.scored_state["turn"] == 2
    captured = capsys.readouterr()
    assert "evaluation suite episode: 1/1 cases passed contract checks" in captured.out
    assert "mean_score=" not in captured.out
    assert "episode scorer requires finished state" not in captured.err
    assert "one generation per turn" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_keeps_uninspectable_episode_scorers_permissive(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import EvalCase\n"
        "class Scorer:\n"
        "    @property\n"
        "    def __signature__(self): raise ValueError('signature unavailable')\n"
        "    def __call__(self, *args): raise AssertionError('scorer must not run offline')\n"
        "class Suite:\n"
        "    name = 'episode'\n"
        "    grades_episodes = True\n"
        "    score = Scorer()\n"
        "    def cases(self): return [EvalCase(id='episode', input='held out')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    _patch_loader(monkeypatch, _MultiTurnEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "evaluation suite episode: 1/1 cases passed contract checks" in captured.out
    assert "signature unavailable" not in captured.err
    assert "scorer must not run offline" not in captured.err
    assert "overall: PASS" in captured.out


@pytest.mark.parametrize(
    "expected_clause",
    ["", ", expected='success'", ", expected={'status': 'success'}"],
    ids=["missing", "scalar", "structured"],
)
def test_env_test_episode_suite_does_not_require_sft_gold(
    monkeypatch, tmp_path, capsys, expected_clause
):
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import EvalCase\n"
        "class Suite:\n"
        "    name = 'episode'\n"
        "    grades_episodes = True\n"
        f"    def cases(self): return [EvalCase(id='episode', input='held out'{expected_clause})]\n"
        "    def score(self, case, response, state):\n"
        "        raise AssertionError('episode scorer must not run offline')\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class _NoEvaluationGoldEnv(_MultiTurnEnv):
        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "1"},
                        {"role": "assistant", "content": "2"},
                    ],
                }
            ]

        def sft_completion(self, example):
            if example["input"] == "held out":
                raise AssertionError("evaluation case requested an sft target")
            return super().sft_completion(example)

        def env_reply(self, messages, state):
            int(state["response_text"])
            return super().env_reply(messages, state)

        def reward(self, completion, example, state=None):
            if example["input"] == "held out":
                raise AssertionError("evaluation case requested environment reward")
            return super().reward(completion, example, state)

    _patch_loader(monkeypatch, _NoEvaluationGoldEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "evaluation suite episode: 1/1 cases passed contract checks" in captured.out
    assert "evaluation case requested an sft target" not in captured.err
    assert "evaluation case requested environment reward" not in captured.err
    assert "episode scorer must not run offline" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_checks_each_episode_cases_initial_rollout(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import EvalCase\n"
        "class Suite:\n"
        "    name = 'episode'\n"
        "    grades_episodes = True\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='ok', input='held out ok'),\n"
        "        EvalCase(id='broken', input='held out broken'),\n"
        "    ]\n"
        "    def score(self, case, response, state): return True\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class _HeldOutRolloutEnv(_MultiTurnEnv):
        def __init__(self):
            super().__init__()
            self.started_inputs = []

        def new_rollout_state(self, example):
            self.started_inputs.append(example["input"])
            if example["input"] == "held out broken":
                raise KeyError("no initial rollout for held out broken")
            return super().new_rollout_state(example)

    env = _HeldOutRolloutEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 1
    assert env.started_inputs == ["finish the exchange", "held out ok", "held out broken"]
    captured = capsys.readouterr()
    assert "no initial rollout for held out broken" in captured.err
    assert "overall: FAIL" in captured.err


@pytest.mark.parametrize(
    "scorer_source",
    [
        "def score(self, case): return True",
        "def score(self, case, response, *, state, required): return True",
        "def score(self, case, response, state, required, /): return True",
    ],
    ids=[
        "two-argument-call-does-not-bind",
        "state-keyword-misses-required",
        "state-position-misses-required",
    ],
)
def test_env_test_rejects_inspectable_episode_scorers_that_cannot_bind_online_call(
    monkeypatch, tmp_path, capsys, scorer_source
):
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import EvalCase\n"
        "class Suite:\n"
        "    name = 'episode'\n"
        "    grades_episodes = True\n"
        "    def cases(self): return [EvalCase(id='episode', input='held out')]\n"
        f"    {scorer_source}\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    _patch_loader(monkeypatch, _MultiTurnEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "evaluation suite episode failed contract checks" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_warns_when_episode_suite_cannot_receive_state(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import EvalCase\n"
        "class Suite:\n"
        "    name = 'last-response'\n"
        "    grades_episodes = True\n"
        "    def cases(self):\n"
        "        return [EvalCase(id='episode', input='finish the exchange', expected=[\n"
        "            {'role': 'assistant', 'content': 'first'},\n"
        "            {'role': 'assistant', 'content': 'second'},\n"
        "        ])]\n"
        "    def score(self, case, response): return response == 'second'\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    _patch_loader(monkeypatch, _MultiTurnEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "evaluation suite last-response: 1/1 cases passed contract checks" in captured.out
    assert captured.err.count("each episode will still be played out") == 1
    assert "only the episode's final response text, not the transcript" in captured.err


def test_env_test_rejects_episode_suite_for_single_turn_environment(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import EvalCase\n"
        "class Suite:\n"
        "    name = 'episode'\n"
        "    grades_episodes = True\n"
        "    def cases(self): return [EvalCase(id='a', input='what is 2 + 2?', expected='4')]\n"
        "    def score(self, case, response, state): return True\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "suite sets grades_episodes = True, but this environment is single-turn" in captured.err
    assert "each episode will still be played out" not in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_rejects_an_evaluation_case_whose_image_cannot_be_resolved(
    monkeypatch, tmp_path, capsys
):
    # `prompt_messages()` is only half of the prompt: env eval and every training worker then run
    # `normalize_prompt_images`. checking only the message envelope approved a case whose
    # package-relative image does not exist, and the online command then recorded a
    # prompt-construction failure for a suite this gate had reported `overall: PASS`.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
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
        "from flash.envs.meta.evaluations import BaseEvalSuite\n"
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
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase, EvalResult\n"
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
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
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
    # the echo fallback itself is not a fault and must not be reported as a broken grader: zero is
    # the CORRECT score for the deliberate junk echo replays. what is reported is the weaker, true
    # statement -- no gold answer was ever scored, so the run is evidence of nothing.
    assert "check the reward function, its runtime dependencies" not in captured.err
    assert "no row supplied gold text to replay" in captured.err
    # an absent answer must get the remedy that ASKS for one. paired with the tool-call test below,
    # which reaches this same branch and must get the opposite advice.
    assert "give the rows a gold answer whose assistant turns carry text" in captured.err
    assert "do NOT add assistant text" not in captured.err
    # exactly ONE warning, keeping the half of this test's original invariant that still holds: the
    # clean echo path emits nothing else. asserting only the expected wording would let a future
    # spurious warning (a role check or cap check misfiring on this shape) reach users unnoticed.
    assert captured.err.count("warning:") == 1


class _ToolCallGoldEnv(_SingleTurnEnv):
    """A gold completion whose payload is a native tool call, so `content` is null by construction.

    Not an authoring mistake: `freesolo.datasets.target_messages` produces exactly this from a
    record whose `output` carries tool calls, and SFT trains on it end to end. This command has no
    text to replay, so it echoes junk and scores zero -- the same observable path as a row with no
    gold answer at all, which is why the remedy has to be chosen on more than the policy.
    """

    def sft_completion(self, example):
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "paris"}'},
                    }
                ],
            }
        ]


def test_env_test_does_not_tell_a_tool_call_target_to_add_assistant_text(
    monkeypatch, tmp_path, capsys
):
    """The remedy must not ask an author to corrupt a correct target to satisfy a check.

    Adding assistant text to a native tool-call row changes what SFT trains on, so advice to do
    that is worse than silence. The run still deserves the warning -- nothing was measured -- but
    the action has to be "exercise the grader elsewhere", not "edit the row".
    """
    env_dir = _environment_dir(tmp_path)
    env = _ToolCallGoldEnv(rows=[{"input": "weather in paris?", "output": "x"}], reward=0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=echo turns=1 reward=0.000000" in captured.out
    assert "payload outside `content`" in captured.err
    assert "do NOT add assistant text" in captured.err
    assert "give the rows a gold answer whose assistant turns carry text" not in captured.err


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
    flash/envs/loading/adapter.py, so a flat ``env.reward`` can still hide separating turn rewards.
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

    from flash.envs.loading.adapter import FreesoloEnvironment

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
    #
    # the COUNT is pinned, not just the value, because for an env graded by a paid judge the number
    # IS the billing. exactly ONE call: the episode itself. this run is `sft`, which never trains
    # from `env.reward`, so neither probe is spent -- their answer could only soften a warning about
    # a number the algorithm does not read. that matches what `dev` bills for this shape, measured
    # on a counting grader; a set assertion would accept the 3 calls an earlier revision made here.
    assert graded == ["ENV_OVERRODE"]
    assert "scored text: 'ENV_OVERRODE'" in captured.err
    assert "scored text: 'RAW_TURN'" not in captured.err


def test_env_test_reports_an_empty_override_as_the_scored_text(monkeypatch, tmp_path, capsys):
    """An env that overrode the answer to nothing graded an empty string, and must say so.

    `step_episode` propagates its override on `is not None` (flash/envs/loading/adapter.py), so `""` really
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

    from flash.envs.loading.adapter import FreesoloEnvironment

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
    # the grader really did receive the empty override, so that is what must be reported. the count
    # is pinned for the same reason as the test above: this is an `sft` run, so the episode itself
    # is the whole spend and neither probe is billed. a set assertion would hide a second call.
    assert graded == [""]
    assert "scored text: ''" in captured.err
    # the replayed turn was never scored; naming it would send the reader to the wrong place.
    assert "scored text: 'RAW_TURN'" not in captured.err


def test_env_test_does_not_bill_the_probes_for_a_non_reward_driven_algorithm(
    monkeypatch, tmp_path, capsys
):
    """The advisory warning must not spend a paid judge on an algorithm that ignores the reward.

    Both probes exist to decide whether the grader separates, and separation only ever SUPPRESSES
    the warning. For sft and opd the reward is never read during training, so the answer cannot
    change the verdict -- but `_separates_on_turn_rewards` calls the env's own `score_episodes` and
    the junk probe drives a whole extra episode, so an unbounded rule billed the same judge three
    times for an all-zero sft run where `dev` billed it once.

    Paired with the grpo case below on the one input that differs, so a fix cannot pass by
    disabling the probes everywhere -- that would strip the blocking gate of its junk comparison.
    """
    env_dir = _environment_dir(tmp_path)
    graded: list[str] = []

    class _CountingEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            graded.append(str(completion))
            return 0.0

    _patch_loader(monkeypatch, _CountingEnv(rows=[{"input": "q", "output": "a"}], reward=0.0))

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    # the episode itself, and nothing else.
    assert len(graded) == 1
    # the warning is still emitted -- bounding the SPEND must not cost the diagnostic.
    assert "returned reward 0.000000" in captured.err


def test_env_test_still_probes_for_grpo(monkeypatch, tmp_path, capsys):
    """The paired positive: grpo trains from the reward, so the probes are worth their cost.

    Without this, bounding the probes by algorithm could be "fixed" by never probing at all, which
    would silently remove the junk comparison the LS-005 blocking gate decides on.
    """
    env_dir = _environment_dir(tmp_path)
    graded: list[str] = []

    class _CountingEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            graded.append(str(completion))
            return 0.0

    _patch_loader(monkeypatch, _CountingEnv(rows=[{"input": "q", "output": "a"}], reward=0.0))

    # every replayed gold answer scored zero and junk scores no worse, so the gate blocks.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    # the episode, the per-turn separation probe, and the junk probe.
    assert len(graded) > 1


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


def test_env_test_validates_images_added_by_an_in_loop_env_reply(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    env = _ImageReplyMultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "remote image URLs are not supported" in captured.err
    assert "overall: FAIL" in captured.err
    assert env.reply_calls == 1
    assert env.scored_state is None
    assert env.state["messages"][-1]["content"][0]["type"] == "image_url"


def test_env_test_skips_prompt_validation_after_natural_in_loop_termination(
    monkeypatch, tmp_path, capsys
):
    env_dir = _environment_dir(tmp_path)
    env = _NaturallyDoneImageReplyMultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=1 reward=0.500000" in captured.out
    assert "overall: PASS" in captured.out
    assert "remote image URLs are not supported" not in captured.err
    assert "env_reply is not well-formed" not in captured.err
    assert "never finished" not in captured.err
    assert env.reply_calls == 1
    assert env.scored_state is env.state
    assert env.state["done"] is True
    assert env.state["applied"] is True
    assert env.state["turn"] == 1 < env.max_turns
    assert env.state["messages"][-1]["content"][0]["type"] == "image_url"


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

    from flash.envs.loading.adapter import FreesoloEnvironment

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

    from flash.envs.loading.adapter import FreesoloEnvironment

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

    from flash.envs.loading.adapter import FreesoloEnvironment

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
    from flash.cli.commands.env.ops.setup import _STARTER_ENV_PY

    assert "gold" in _STARTER_ENV_PY
    assert "sft_completion" in _STARTER_ENV_PY
    # names the wrapper trap and where per-row scorer state belongs.
    assert "boxed" in _STARTER_ENV_PY
    assert "metadata" in _STARTER_ENV_PY


class _AssistantOnlySingleTurnEnv(_SingleTurnEnv):
    """A SINGLE-turn env whose gold completion is assistant turns alone: the ISSUE-015 defect.

    Single-turn is the shape where this is genuinely wrong. Nothing generates the intervening user
    turns, so the concatenation SFT trains on really is one question followed by three answers back
    to back. Each turn scores fine when replayed one at a time -- all this gate ever did -- so the
    defect lives only at the concatenation.

    A MULTI-turn env storing the same messages is correct and must not warn: there `env_reply`
    supplies the user turns at rollout time, which is why they are absent from the dataset.
    """

    def sft_completion(self, example):
        return [
            {"role": "assistant", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
        ]


def test_env_test_warns_when_sft_would_train_on_consecutive_assistant_turns(
    monkeypatch, tmp_path, capsys
):
    """ISSUE-015: an assistant-only gold completion renders as one reply, and passed silently.

    The defect exists only at the concatenation SFT trains on. This command replays the turns
    individually and scores each one, so every per-turn check passes and the run reported
    `overall: PASS` -- while the training text taught the model to dump every turn's answer into a
    single reply, the exact opposite of the multi-turn behaviour being taught.
    """
    env_dir = _environment_dir(tmp_path)
    env = _AssistantOnlySingleTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "consecutive same-role turns" in captured.err
    # the collision is inside the completion here, so the message must send the reader to
    # interleave its turns rather than to the prompt...
    assert "inside sft_completion" in captured.err
    # ...and it must use SFT_COMPLETION's own indices. a raw offset into the concatenation is not a
    # position the author can look up: with a prompt prepended, turn 1 would be reported as some
    # higher number that may not even exist in the file they open.
    assert "inside sft_completion (message 1 (assistant), message 2 (assistant))" in captured.err
    # the rendered role sequence is the evidence: it shows WHICH turns merged and on which side
    # of the prompt/completion boundary, which no preview of the message text conveys.
    assert "rendered roles: user | assistant assistant assistant" in captured.err


def test_env_test_does_not_warn_on_a_multi_turn_gold_answer(monkeypatch, tmp_path, capsys):
    """A multi-turn gold completion is consecutive assistant turns BY CONTRACT, not by mistake.

    `env_reply` produces the intervening user turns during the rollout, so they are deliberately
    absent from the dataset -- `_drive_multi_turn` depends on exactly that shape, replaying one
    reference turn per model turn. Warning here would fire on every correct multi-turn environment
    (including every multi-turn fixture in this file) and advise an edit that breaks the replay.
    """
    env_dir = _environment_dir(tmp_path)
    env = _MultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "consecutive same-role turns" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_does_not_warn_on_parallel_tool_result_turns(monkeypatch, tmp_path, capsys):
    """Two `tool` messages in a row are the required wire format for parallel tool calls.

    One assistant message carrying two tool calls is answered by one `tool` message per call, and
    chat templates render each as its own delimited block rather than merging them. Only the
    assistant role actually merges, so only it is checked -- flagging `tool` would teach users to
    edit a correct transcript into a broken one, or to ignore the warning entirely.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv()
    monkeypatch.setattr(
        env,
        "sft_completion",
        lambda example: [
            {"role": "assistant", "content": None},
            {"role": "tool", "content": "result a"},
            {"role": "tool", "content": "result b"},
            {"role": "assistant", "content": "4"},
        ],
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "consecutive same-role turns" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_stays_quiet_for_a_repeat_wholly_inside_the_prompt(monkeypatch, tmp_path, capsys):
    """A repeat wholly inside `prompt_messages` is not this warning's to report.

    `with_system_prompt` prepends the contract system message without merging an existing one, so a
    two-system prompt is a shape the adapter itself produces, and an ordinary RAG prompt puts the
    retrieved document in its own user message. Both render as separate delimited blocks and are
    correct. There is no edit to `sft_completion` -- here a single well-formed assistant turn that
    cannot contain a repeat at all -- that would change them, so naming it accused a correct file,
    and firing on every episode of a healthy env is what teaches people to ignore the warning where
    it is real.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv()
    monkeypatch.setattr(
        env,
        "prompt_messages",
        lambda example: [
            {"role": "assistant", "content": "preamble"},
            {"role": "assistant", "content": "still preamble"},
            {"role": "user", "content": example["input"]},
        ],
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "consecutive same-role turns" not in captured.err


def test_env_test_stays_quiet_for_an_assistant_prefill_the_completion_continues(
    monkeypatch, tmp_path, capsys
):
    """A prompt ending on an assistant prefill is the one seam that is MEANT to merge.

    The completion finishes an utterance the prompt began, so the two assistant turns rendering as
    one reply is the point. The remedy this warning would print -- interleave the user turn being
    answered -- destroys the prefill, which is why the assistant seam is exempt while a doubled
    `user` turn at the same boundary (a trajectory captured one turn early) still reports.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "who wrote Dune?", "output": "Frank Herbert"}])
    monkeypatch.setattr(
        env,
        "prompt_messages",
        lambda example: [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": "The author is"},
        ],
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "consecutive same-role turns" not in captured.err


def test_env_test_reports_an_internal_collapse_behind_a_legitimate_prefill(
    monkeypatch, tmp_path, capsys
):
    """A prefill seam is exempt, but it must not mask a real collapse further into the completion.

    This env has both: the prompt ends on an assistant prefill the completion continues (healthy,
    and the reason the seam is skipped), and the completion then runs two assistant turns together
    (the ISSUE-015 defect). Suppressing the whole episode because its first adjacency was excusable
    would lose the finding that matters, so the interior is still scanned and reported on its own
    index.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv()
    monkeypatch.setattr(
        env,
        "prompt_messages",
        lambda example: [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": "the answer is"},
        ],
    )
    monkeypatch.setattr(
        env,
        "sft_completion",
        lambda example: [
            {"role": "assistant", "content": "4"},
            {"role": "assistant", "content": "definitely 4"},
        ],
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "inside sft_completion (message 1 (assistant))" in captured.err
    # the prefill itself is not blamed: the remedy for it would break a working prompt.
    assert "prompt/completion boundary" not in captured.err


def test_env_test_does_not_warn_when_the_gold_completion_interleaves_user_turns(
    monkeypatch, tmp_path, capsys
):
    """The fix for ISSUE-015 must read as clean: several assistant turns are fine when separated.

    A completion legitimately holds many assistant turns as long as the env's own follow-up user
    prompts sit between them, which is what the rendered SFT string needs.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "count", "output": "one"}])
    monkeypatch.setattr(
        env,
        "sft_completion",
        lambda example: [
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "and then?"},
            {"role": "assistant", "content": "two"},
        ],
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "consecutive same-role turns" not in captured.err
    assert "overall: PASS" in captured.out


class _NeverFinishingMultiTurnEnv(_MultiTurnEnv):
    """An environment no trajectory can complete: gold burns the whole turn cap and never wins.

    This is the dead-environment shape (every move applied twice, so no board is ever solved). A
    partial-credit grader still pays a respectable score for the capped attempt, and that score
    still beats junk -- so every existing check passed it.
    """

    max_turns = 4

    def dataset(self):
        # a FULL-LENGTH gold answer, one reference turn per allowed turn. a shorter one would be
        # padded with junk, and junk cannot advance the env -- reaching the cap would then be the
        # padding's doing rather than the reference's, which is a different fault entirely.
        return [
            {
                "input": "solve the board",
                "output": [{"role": "assistant", "content": f"move {n}"} for n in range(1, 5)],
            }
        ]

    def env_reply(self, messages, state):
        # never declares the episode done: the driver can only stop at the hard cap.
        state["turn"] += 1
        reply = {"role": "user", "content": "keep going"}
        messages.append(reply)
        return [reply]

    def rollout_done(self, state, max_turns=None):
        # the dead environment: no sequence of moves ever satisfies the win condition.
        return False

    def reward(self, completion, example, state=None):
        self.scored_state = state
        # partial credit, comfortably above the junk floor -- which is why gold-vs-junk cleared.
        return 0.62


def test_env_test_warns_when_a_gold_replay_never_terminates(monkeypatch, tmp_path, capsys):
    """A dead environment passed: gold scored 0.62, beat junk, and burned the full turn cap.

    What actually surfaced the bug in the field was reading `turns=12` on a board with a five-move
    solution -- a comparison the verdict never made. The reward cannot show it: partial credit for
    a capped attempt is a perfectly ordinary-looking number.
    """
    env_dir = _environment_dir(tmp_path)
    env = _NeverFinishingMultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    # the run still passes -- this is advisory -- but it no longer passes SILENTLY.
    assert "overall: PASS" in captured.out
    assert "replay gold answer never finished: it used all 4 turn(s)" in captured.err
    assert "no rollout can either" in captured.err


def test_env_test_does_not_warn_when_the_environment_declares_the_episode_done(
    monkeypatch, tmp_path, capsys
):
    """An env that finishes well inside its budget must stay clean."""
    env_dir = _environment_dir(tmp_path)
    env = _MultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "never finished" not in captured.err
    assert "overall: PASS" in captured.out


class _SolvesOnItsLastAllowedTurnEnv(_MultiTurnEnv):
    """A healthy env on a tight budget: it solves the task with its final permitted turn.

    This is the case that exercises the turn-cap branch, which an env finishing early never
    reaches. The verdict cannot be read at the `break`: the last model turn is applied only by the
    deferred `env_reply` after the loop, so at the break the state still says unfinished.
    """

    max_turns = 3

    def dataset(self):
        return [
            {
                "input": "solve in three",
                "output": [{"role": "assistant", "content": f"move {n}"} for n in range(1, 4)],
            }
        ]

    def env_reply(self, messages, state):
        state["turn"] += 1
        state["done"] = state["turn"] >= 3
        reply = {"role": "user", "content": "next"}
        messages.append(reply)
        return [reply]

    def reward(self, completion, example, state=None):
        self.scored_state = state
        return 1.0


def test_env_test_does_not_warn_when_the_episode_finishes_on_its_last_allowed_turn(
    monkeypatch, tmp_path, capsys
):
    """A tightly-budgeted env that solves the task on its final turn is healthy, not capped.

    The loop breaks at the ceiling before the last turn has been applied, so asking there reports a
    working environment as one that never finishes -- sending the author to fix termination logic
    and move-application code that are both correct. The verdict is drawn after the deferred
    `env_reply`, once the state reflects everything the run would score.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SolvesOnItsLastAllowedTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=3 reward=1.000000" in captured.out
    assert "never finished" not in captured.err
    assert "overall: PASS" in captured.out


class _ShortGoldMultiTurnEnv(_MultiTurnEnv):
    """A WORKING env whose dataset row supplies a gold answer shorter than the episode.

    The environment really does terminate -- on its third turn, well inside the four-turn cap -- so
    the only thing unusual here is the short reference. That is what makes it the control: reaching
    the end of a healthy episode must not be reported as a termination fault.
    """

    max_turns = 4

    def dataset(self):
        return [
            {
                "input": "solve the board",
                "output": [{"role": "assistant", "content": "one move only"}],
            }
        ]

    def env_reply(self, messages, state):
        state["turn"] += 1
        # terminates on its own, unlike the dead env above: the short gold answer is the ONLY
        # difference, so a warning here would be blaming the reference for the padding.
        state["done"] = state["turn"] >= 3
        reply = {"role": "user", "content": "keep going"}
        messages.append(reply)
        return [reply]

    def rollout_done(self, state, max_turns=None):
        return bool(state.get("done"))

    def reward(self, completion, example, state=None):
        self.scored_state = state
        return 0.3


def test_env_test_does_not_blame_termination_for_a_gold_answer_that_ran_out(
    monkeypatch, tmp_path, capsys
):
    """A short gold answer reaches the cap because of the harness's own junk padding.

    Once the reference runs out the driver pads with `_junk_response`, and junk cannot advance the
    environment -- so hitting the ceiling is the padding's doing, not the reference's. Blaming
    episode termination here sends the author to audit working code, while the real fault (the row
    covers only part of the trajectory) goes unnamed.
    """
    env_dir = _environment_dir(tmp_path)
    env = _ShortGoldMultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    # the episode ran past the one gold turn on junk padding and still ended on the env's own
    # signal, inside the cap -- so there is nothing to report.
    assert "episode 1: policy=replay turns=3" in captured.out
    assert "never finished" not in captured.err


def test_env_test_warns_when_every_episode_scores_zero_under_sft(monkeypatch, tmp_path, capsys):
    """The blocking gate is grpo-only, so an all-zero sft run printed PASS and nothing else.

    This is the shape that reaches paid GPUs: one published environment serves all three
    algorithms, and an `output` shape that only the SFT rows satisfy leaves every GRPO/OPD rollout
    scoring 0.0. The warning's job here is to stop this run being read as clearance to train GRPO
    later - NOT to claim the sft run itself is broken. SFT never reads `env.reward`, so an all-zero
    scorer costs this run nothing, and a placeholder `reward()` on an sft-only environment is a
    deliberate and correct choice. Telling that author to debug their scorer is a false alarm on
    working code, so the wording is asserted, not just the fact that something was printed.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(
        rows=[{"input": "2 + 2?", "output": "4"}, {"input": "2 + 3?", "output": "5"}],
        reward=0.0,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "all 2 scored episode(s) returned reward 0.000000" in captured.err
    assert "sft does not train from `env.reward`" in captured.err
    assert "no evidence the reward function works" in captured.err
    # the grpo-only consequence must not be asserted about an sft run, and the uniformly-zero line
    # must not send the reader to fix a scorer this algorithm never calls. scoped to that line: the
    # per-episode low-reward warning legitimately says "check the reward function", because a gold
    # answer scoring zero is worth looking at whatever the algorithm.
    zero_line = next(
        line for line in captured.err.splitlines() if "scored episode(s) returned reward" in line
    )
    assert "zero advantage" not in zero_line
    assert "check the reward function" not in zero_line


def test_env_test_does_not_call_a_reasoning_markup_run_unmeasured(monkeypatch, tmp_path, capsys):
    """A `<think>` reference scoring zero here says nothing about the environment.

    This command has no run config, so `thinking` defaults off and it replays the tagged reference
    verbatim -- while a real run grades what `_scored_turn_text` leaves after stripping the span. A
    correct strict answer-only grader therefore scores 0.0 here on a WORKING environment, which is
    why `_carries_thinking_markup` keeps these episodes out of the blocking gate.

    The advisory warning has to abstain for the same reason: "this run measured nothing" is the very
    conclusion the gate withholds, and asserting it in a warning restates it with the blocking
    removed rather than the claim. The per-episode low-reward warning still prints the number, which
    is the part that is true.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(
        rows=[{"input": "2 + 2?", "output": "<think>hmm</think>4"}],
        reward=0.0,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    # the blocking gate stays out of it, exactly as designed...
    assert "cannot recognize its own reference answers" not in captured.err
    # ...and so does the advisory warning, which would otherwise assert the same thing.
    assert "measured nothing" not in captured.err
    # the observation that IS supported is still made.
    assert "replay gold answer scored low" in captured.err


def test_env_test_does_not_warn_when_one_episode_scores_nonzero(monkeypatch, tmp_path, capsys):
    """A single separating row is enough to prove the grader distinguishes answers.

    The warning must describe uniformity, not merely the presence of a zero: an environment that
    pays 0.0 for one row and 1.0 for another is working.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(
        rows=[{"input": "2 + 2?", "output": "4"}, {"input": "2 + 3?", "output": "5"}]
    )
    rewards = iter([0.0, 1.0])
    monkeypatch.setattr(env, "reward", lambda completion, example, state=None: next(rewards))
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "measured nothing" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_does_not_call_a_run_measured_nothing_when_junk_scores_worse(
    monkeypatch, tmp_path, capsys
):
    """A centered scale paying gold 0.0 and junk -1.0 separates by a full point.

    The blocking gate already exempts this shape (`centered rewards may legitimately score gold at
    zero`), and the advisory warning has to honour the same evidence: GRPO gets a real gradient
    here, so "this run measured nothing" would simply be false. False alarms on the healthy path are
    what teach people to ignore the warning on the broken path it was written for.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(
        rows=[{"input": "2 + 2?", "output": "4"}, {"input": "2 + 3?", "output": "5"}]
    )
    monkeypatch.setattr(
        env,
        "reward",
        lambda completion, example, state=None: 0.0 if completion == example["output"] else -1.0,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "cannot recognize its own reference answers" not in captured.err
    assert "measured nothing" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_does_not_call_a_run_measured_nothing_when_turn_rewards_separate(
    monkeypatch, tmp_path, capsys
):
    """`credit_assignment="per_turn"` trains from the turn vector; the scalar is a placeholder.

    The blocking gate consults `_separates_on_turn_rewards` for exactly this reason, and the
    advisory warning must too -- a turn vector of (1.0, 0.0) is the signal this training mode
    optimizes, so the run measured plenty.

    Multi-turn on purpose: the vector has to be one the PAID worker would accept, and two rewards
    only match an episode that emitted two assistant turns.
    """
    env_dir = _environment_dir(tmp_path)
    env = _MultiTurnEnv()
    monkeypatch.setattr(env, "reward", lambda completion, example, state=None: 0.0)
    monkeypatch.setattr(
        env,
        "rollout_rewards_many",
        lambda pairs: [RolloutReward(episode=0.0, turns=(1.0, 0.0)) for _ in pairs],
        raising=False,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "measured nothing" not in captured.err
    assert "overall: PASS" in captured.out


@pytest.mark.parametrize(
    ("turns", "why"),
    [
        ((0.0, 1.0, 0.5), "three rewards for a two-turn episode"),
        ((float("nan"), 1.0), "a non-finite value"),
        ((float("inf"), 1.0), "an infinite value"),
    ],
)
def test_env_test_ignores_a_turn_vector_the_paid_worker_would_reject(
    monkeypatch, tmp_path, capsys, turns, why
):
    """Only a vector the WORKER would use is evidence that per-turn credit separates.

    `_validated_reward` (flash/engine/worker/train/rl/rollout/scoring.py) discards a vector whose length
    disagrees with the assistant turns emitted, or that holds a non-finite value, and falls back to
    the flat episode reward. A run like that trains on all-zero rewards, so reading its vector as
    separation suppressed both the blocking gate and the warning on a run that measures nothing.
    """
    env_dir = _environment_dir(tmp_path)
    env = _MultiTurnEnv()
    monkeypatch.setattr(env, "reward", lambda completion, example, state=None: 0.0)
    monkeypatch.setattr(
        env,
        "rollout_rewards_many",
        lambda pairs: [RolloutReward(episode=0.0, turns=turns) for _ in pairs],
        raising=False,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1, why
    captured = capsys.readouterr()
    assert "cannot recognize its own reference answers" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_says_replayable_when_the_gold_answer_carries_no_text(
    monkeypatch, tmp_path, capsys
):
    """An image-only gold completion IS a gold answer; it just cannot be replayed as text.

    Echo is chosen whenever the gold answer yields no replay text, not only when the row supplied
    nothing. Telling this author to "give the rows a gold answer" sends them to add one they
    already have, and they re-run to the identical message.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "describe it", "output": "x"}], reward=0.0)
    monkeypatch.setattr(
        env,
        "sft_completion",
        lambda example: [
            {"role": "assistant", "content": [{"type": "image", "image": "board.png"}]}
        ],
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "no row supplied gold text to replay" in captured.err
    # the remedy must name the actual fault: the turns carry no text to replay.
    assert "whose assistant turns carry text" in captured.err


class _DeadEnvWithShortGoldEnv(_MultiTurnEnv):
    """The field case exactly: a five-move gold answer against a twelve-turn cap, unsolvable.

    Real datasets carry minimal solutions, not cap-length ones, so this -- not a gold answer padded
    out to the ceiling -- is the shape the warning has to catch.
    """

    max_turns = 12

    def dataset(self):
        return [
            {
                "input": "solve the board",
                "output": [{"role": "assistant", "content": f"move {n}"} for n in range(1, 6)],
            }
        ]

    def env_reply(self, messages, state):
        state["turn"] += 1
        reply = {"role": "user", "content": "keep going"}
        messages.append(reply)
        return [reply]

    def rollout_done(self, state, max_turns=None):
        # every move applied twice: no sequence of moves ever satisfies the win condition.
        return False

    def reward(self, completion, example, state=None):
        self.scored_state = state
        return 0.62


def test_env_test_warns_on_a_dead_environment_whose_gold_answer_is_short(
    monkeypatch, tmp_path, capsys
):
    """A five-move solution burning a twelve-turn cap is the signature that surfaced the bug.

    Excluding every short gold answer to avoid blaming the junk padding would silence exactly this
    case, since a minimal solution is shorter than the cap by definition. What distinguishes them is
    whether the environment had finished at the moment the gold answer ran out -- here it had not,
    and it never would have.
    """
    env_dir = _environment_dir(tmp_path)
    env = _DeadEnvWithShortGoldEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=12" in captured.out
    assert "replay gold answer never finished: it used all 12 turn(s)" in captured.err
    # the gold answer ran out at move 5 and the driver padded to 12, so the env is not the only
    # possible cause and must not be named as one. asserting the opening alone would pass on either
    # wording, which is how an over-strong claim survives a green suite.
    assert "either a reference that covers only part of a trajectory" in captured.err
    assert "check whether the dataset row is complete first" in captured.err
    assert "means no rollout can either" not in captured.err


class _CompleteGoldNonTerminatingEnv(_DeadEnvWithShortGoldEnv):
    """A non-terminating env whose gold answer covers EVERY turn up to the cap.

    The counterpart to the fixture above and the only shape that licenses blaming the environment:
    no junk padding ran, so the reference itself drove every turn and the episode still never
    ended. Same env, same cap, same grader -- only the dataset row's length differs, which is
    precisely the fact the two messages are allowed to distinguish.
    """

    def dataset(self):
        return [
            {
                "input": "solve the board",
                "output": [{"role": "assistant", "content": f"move {n}"} for n in range(1, 13)],
            }
        ]


def test_env_test_blames_the_environment_only_when_gold_covered_every_turn(
    monkeypatch, tmp_path, capsys
):
    """A complete reference that still never finishes leaves the environment as the explanation.

    Paired with the short-gold case above: identical environment and cap, opposite dataset
    coverage. Without this pair a fix could emit one message unconditionally and stay green.
    """
    env_dir = _environment_dir(tmp_path)
    env = _CompleteGoldNonTerminatingEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=12" in captured.out
    assert "replay gold answer never finished: it used all 12 turn(s)" in captured.err
    assert "means no rollout can either" in captured.err
    assert "covers only part of a trajectory" not in captured.err


class _OverlongGoldEnv(_MultiTurnEnv):
    """A HEALTHY env whose gold trajectory is LONGER than the turn cap.

    The driver replays a prefix and stops, so `replay_incomplete` is never set -- that flag means
    the driver ran PAST the reference and padded with junk, which is the opposite situation. The
    episode therefore reached the branch that blames the environment, even though the cap stopped
    the episode before the environment's termination logic could run at all.
    """

    max_turns = 3

    def dataset(self):
        return [
            {
                "input": "five steps",
                "output": [{"role": "assistant", "content": f"step {n}"} for n in range(1, 6)],
            }
        ]

    def env_reply(self, messages, state):
        state["turn"] += 1
        # would finish on the fifth turn, which the cap never lets it reach.
        state["done"] = state["turn"] >= 5
        reply = {"role": "user", "content": "next"}
        messages.append(reply)
        return [reply]

    def reward(self, completion, example, state=None):
        self.scored_state = state
        return 0.4


def test_env_test_names_a_cap_shorter_than_the_gold_trajectory(monkeypatch, tmp_path, capsys):
    """A cap/dataset mismatch has an exact cause, so it must not be reported as a guess.

    The other two unfinished shapes are genuinely ambiguous from here and say so. This one is not:
    the reference length and the cap are both known, and their comparison settles it. Reporting it
    as "either the row or your termination condition" would send an author to audit code that never
    ran, so the numbers and the remedy are both asserted.
    """
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _OverlongGoldEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "the gold trajectory is 5 turn(s) but the episode is capped at 3" in captured.err
    assert "cap/dataset mismatch, not an environment fault" in captured.err
    # neither ambiguous reading may be offered for a case whose cause is determined.
    assert "check the termination condition" not in captured.err
    assert "covers only part of a trajectory" not in captured.err


class _OverlongGoldEarlyDoneEnv(_OverlongGoldEnv):
    """Gold longer than the cap, but the env declares done BEFORE the cap is reached.

    Two turns are replayed, not three. The cap is what the episode was ALLOWED, which is not the
    same number as what was sent, so a report that quotes the cap as the replayed count overstates
    the replay and points at a turn that was never scored.
    """

    def env_reply(self, messages, state):
        state["turn"] += 1
        state["done"] = state["turn"] >= 2
        reply = {"role": "user", "content": "next"}
        messages.append(reply)
        return [reply]


def test_env_test_counts_the_turns_actually_replayed_not_the_cap(monkeypatch, tmp_path, capsys):
    """The truncation report must quote what was sent, not what was permitted.

    An environment that terminates before the cap replays fewer turns than the cap allows. Quoting
    the cap then claims a turn was replayed that never was, and an author reconciling the message
    against the episode line (`turns=2`) is chasing a discrepancy this command invented.
    """
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _OverlongGoldEarlyDoneEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "the gold trajectory is 5 turn(s) but the episode is capped at 3" in captured.err
    assert "only the first 2 were replayed" in captured.err
    assert "only the first 3 were replayed" not in captured.err
    assert "turns=2" in captured.out


class _OverlongGoldZeroRewardEnv(_OverlongGoldEnv):
    """Gold longer than the cap AND a grader that recognizes nothing."""

    def reward(self, completion, example, state=None):
        self.scored_state = state
        return 0.0


def test_env_test_still_blocks_a_zero_grader_behind_a_truncated_replay(
    monkeypatch, tmp_path, capsys
):
    """A cap/dataset mismatch explains the truncation, not the zero.

    The two findings are independent, so the advisory truncation report must not stand in for the
    blocking gate: a grader that scores its own reference zero is still broken whether or not the
    row outran the cap, and swallowing the gate here would turn a FAIL into a PASS.
    """
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _OverlongGoldZeroRewardEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "cap/dataset mismatch, not an environment fault" in captured.err
    assert "cannot recognize its own reference answers" in captured.err
    assert "overall: FAIL" in captured.err


class _FixedLengthEnv(_MultiTurnEnv):
    """A HEALTHY env that ends only by exhausting its budget and never sets `state["done"]`.

    A supported shape, not a defect: the real `rollout_done` (flash/envs/loading/adapter.py) returns True at
    `turn >= cap` regardless of `done`, so a fixed-length game trains fine. From this command it is
    indistinguishable from an environment no rollout can finish -- both replay every turn and both
    leave `done` unset -- so the warning may report it but must not tell its author their
    termination condition is broken.
    """

    max_turns = 3

    def dataset(self):
        return [
            {
                "input": "three rounds",
                "output": [{"role": "assistant", "content": f"r{n}"} for n in range(1, 4)],
            }
        ]

    def env_reply(self, messages, state):
        state["turn"] += 1
        reply = {"role": "user", "content": "next"}
        messages.append(reply)
        return [reply]

    def reward(self, completion, example, state=None):
        self.scored_state = state
        return 1.0


def test_env_test_does_not_blame_a_fixed_length_env_for_using_its_whole_budget(
    monkeypatch, tmp_path, capsys
):
    """A fixed-length episode must not be told its termination condition is broken.

    The warning still fires -- the shape genuinely cannot be distinguished from a dead environment
    here -- but the verdict has to stay open, and the reward that separates them in practice has to
    be on the line. Asserting the absence of the unconditional claim is the point: this env scores
    1.0 and is correct, so sending its author to audit `step_episode` is a false alarm on working
    code, which is what teaches people to ignore the warning on the broken case.
    """
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _FixedLengthEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=3" in captured.out
    assert "replay gold answer never finished: it used all 3 turn(s)" in captured.err
    # the alternative is named, and the reward that distinguishes the two is quoted.
    assert "fixed-length episode that ends by using its whole budget" in captured.err
    assert "reward=1.000000" in captured.err


class _RealRolloutDoneDeadEnv(_DeadEnvWithShortGoldEnv):
    """The dead environment, judged by the REAL `rollout_done` instead of a fake override.

    Every other multi-turn fixture here overrides `rollout_done` to a bare `False`, which no
    published environment does: the adapter's own implementation returns True from `turn >= cap`
    ALONE (flash/envs/loading/adapter.py), independently of whether the episode was ever won. Consulting it
    to decide the turn-cap warning therefore made the warning unsatisfiable against real
    environments while these fakes kept the tests green -- the warning was dead code that passed.

    Binding the adapter's own function rather than restating its logic is deliberate: a copy would
    drift the moment `rollout_done` changed, which is the same way the gap opened.
    """

    rollout_done = FreesoloEnvironment.rollout_done


def test_env_test_warns_on_a_dead_environment_under_real_rollout_done_semantics(
    monkeypatch, tmp_path, capsys
):
    """The turn-cap warning must survive an env whose `rollout_done` is True at the ceiling.

    This is the production shape: at `turn >= cap` the real adapter reports done for a healthy
    environment and for one no rollout can ever finish, so the two are indistinguishable by that
    signal. `state["done"]` is what separates them, and only a fixture using the real method proves
    the warning reads it.
    """
    env_dir = _environment_dir(tmp_path)
    env = _RealRolloutDoneDeadEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=12" in captured.out
    assert "replay gold answer never finished: it used all 12 turn(s)" in captured.err


class _RealRolloutDoneHealthyEnv(_MultiTurnEnv):
    """A healthy env carrying the real `rollout_done`, to prove the warning is not unconditional.

    Same method as the dead fixture above and a two-turn gold answer that finishes the episode on
    its own, so any warning here would be firing on a correct environment.
    """

    max_turns = 4
    rollout_done = FreesoloEnvironment.rollout_done


def test_env_test_stays_quiet_for_a_healthy_env_under_real_rollout_done_semantics(
    monkeypatch, tmp_path, capsys
):
    """A working environment must stay silent even though its `rollout_done` is the real one."""
    env_dir = _environment_dir(tmp_path)
    env = _RealRolloutDoneHealthyEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "never finished" not in captured.err


class _PerExampleCapDeadEnv(_MultiTurnEnv):
    """A dead environment whose PER-EXAMPLE budget is well below the dataset-wide cap.

    `rollout_done` gives `state["max_episode_turns"]` precedence over `env.max_turns`
    (flash/envs/loading/adapter.py), so this episode is stopped by its own budget at turn 3 while the
    dataset-wide ceiling is 12. Deriving the ceiling from `env.max_turns` alone left the exhaustion
    flag False for exactly these episodes, so a board no move can solve reached `overall: PASS`
    with nothing said -- the silent pass this warning exists to catch.
    """

    max_turns = 12
    rollout_done = FreesoloEnvironment.rollout_done

    def dataset(self):
        return [
            {
                "input": "solve the board",
                "output": [{"role": "assistant", "content": f"move {n}"} for n in range(1, 3)],
            }
        ]

    def new_rollout_state(self, example):
        prompt = [{"role": "user", "content": example["input"]}]
        return {
            "prompt": prompt,
            "messages": list(prompt),
            "done": False,
            "turn": 0,
            "max_episode_turns": 3,
        }

    def env_reply(self, messages, state):
        # never declares the episode done: no sequence of moves satisfies the win condition.
        state["turn"] += 1
        reply = {"role": "user", "content": "keep going"}
        messages.append(reply)
        return [reply]

    def reward(self, completion, example, state=None):
        self.scored_state = state
        return 0.62


def test_env_test_warns_when_a_per_example_cap_stops_an_unfinished_replay(
    monkeypatch, tmp_path, capsys
):
    """The exhaustion verdict must use the EFFECTIVE ceiling, not the dataset-wide one.

    A row may set `max_episode_turns` below `env.max_turns`, and the adapter gives that budget
    precedence. Comparing the turn count against `env.max_turns` alone silenced the warning for
    every such episode, which is the one shape where the budget is tightest and a non-terminating
    reference is most likely to go unnoticed.
    """
    env_dir = _environment_dir(tmp_path)
    env = _PerExampleCapDeadEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=3" in captured.out
    assert "replay gold answer never finished: it used all 3 turn(s)" in captured.err


class _PerExampleCapHealthyEnv(_PerExampleCapDeadEnv):
    """The same per-example budget, on an environment that actually finishes inside it.

    Differs from the dead fixture above in exactly two ways: the episode is declared over on the
    last turn the budget allows, and the gold answer is long enough to reach it. Lowering the
    ceiling to the per-example cap must not turn "used its whole budget" into evidence of a defect:
    a run that finishes on its final allowed turn is healthy, and warning at it would fire on a
    correct environment.

    The gold answer must cover all three turns. Inheriting the dead fixture's two-move dataset
    makes this a different scenario entirely -- a reference that runs out before the episode does,
    which the driver then pads with junk -- and that case is reported by design.
    """

    def dataset(self):
        return [
            {
                "input": "solve the board",
                "output": [{"role": "assistant", "content": f"move {n}"} for n in range(1, 4)],
            }
        ]

    def env_reply(self, messages, state):
        state["turn"] += 1
        state["done"] = state["turn"] >= 3
        reply = {"role": "user", "content": "keep going"}
        messages.append(reply)
        return [reply]

    def reward(self, completion, example, state=None):
        self.scored_state = state
        return 1.0


def test_env_test_does_not_warn_when_an_episode_finishes_on_its_per_example_cap(
    monkeypatch, tmp_path, capsys
):
    """Using the whole per-example budget is not the same as never finishing.

    The paired negative for the test above: same cap, same turn count, opposite completion signal.
    Without both, a fix that simply lowered the ceiling would pass the positive case while warning
    on every environment that uses its budget fully.
    """
    env_dir = _environment_dir(tmp_path)
    env = _PerExampleCapHealthyEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "never finished" not in captured.err


class _MalformedFinalReplyEnv(_PerExampleCapDeadEnv):
    """An env whose terminal reply is not suitable for a future model prompt."""

    def env_reply(self, messages, state):
        state["turn"] += 1
        reply = (
            {"role": "user", "content": 123}
            if state["turn"] >= 3
            else {"role": "user", "content": "keep going"}
        )
        messages.append(reply)
        return [reply]


class _ImageFinalReplyEnv(_PerExampleCapDeadEnv):
    def __init__(self):
        super().__init__()
        self.reply_calls = 0
        self.state = None

    def new_rollout_state(self, example):
        self.state = super().new_rollout_state(example)
        return self.state

    def env_reply(self, messages, state):
        self.reply_calls += 1
        state["turn"] += 1
        content = (
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/final.png"},
                }
            ]
            if state["turn"] >= 3
            else "keep going"
        )
        reply = {"role": "user", "content": content}
        messages.append(reply)
        return [reply]


def test_env_test_does_not_validate_the_deferred_final_reply_as_a_prompt(
    monkeypatch, tmp_path, capsys
):
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _MalformedFinalReplyEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "env_reply is not well-formed" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_allows_images_added_by_the_deferred_final_reply(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    env = _ImageFinalReplyEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "remote image URLs are not supported" not in captured.err
    assert "overall: PASS" in captured.out
    assert env.reply_calls == 3
    assert env.scored_state is env.state
    assert env.state["messages"][-1]["content"][0]["type"] == "image_url"


class _NoFinalReplyEnv(_PerExampleCapDeadEnv):
    """A healthy env with nothing left to observe on its final turn."""

    def env_reply(self, messages, state):
        state["turn"] += 1
        if state["turn"] >= 3:
            # legitimate: no further observation to make.
            return []
        reply = {"role": "user", "content": "keep going"}
        messages.append(reply)
        return [reply]


def test_env_test_allows_an_empty_deferred_final_env_reply(monkeypatch, tmp_path, capsys):
    """Returning no final observation is legal and still applies the final action."""
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _NoFinalReplyEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "env_reply is not well-formed" not in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_warns_when_the_completion_repeats_the_prompts_last_user_turn(
    monkeypatch, tmp_path, capsys
):
    """An off-by-one trajectory capture duplicates the question in the trained string.

    Restricting the check to the assistant role would miss it: the doubled turn here is `user`, and
    it merges in the rendered text exactly as a doubled assistant turn does. Only `tool` is exempt,
    because parallel tool calls legitimately render as back-to-back tool results.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "2 + 2?", "output": "4"}])
    monkeypatch.setattr(
        env,
        "sft_completion",
        lambda example: [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ],
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    # reported even though it sits AT the seam: only an assistant seam is a legitimate prefill, and
    # a doubled user turn there has no such form -- it is the completion restating the question.
    assert "message 0 (user), which repeats the prompt's last turn" in captured.err
    # the remedy names the real fault rather than telling them to interleave user turns.
    assert "captured one turn early" in captured.err


def test_env_test_scores_the_junk_probe_once_per_run(monkeypatch, tmp_path, capsys):
    """The probe drives a whole extra episode through user code and may bill a paid judge.

    Both the blocking gate and the advisory warning need its answer, so they share one result: two
    passes would double the cost and, since scoring is not guaranteed to be pure, could return two
    different answers to the same question.
    """
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "2 + 2?", "output": "4"}], reward=0.0)
    scored: list[str] = []
    monkeypatch.setattr(
        env,
        "reward",
        lambda completion, example, state=None: scored.append(completion) or 0.0,
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) != 0
    # one real episode plus exactly one junk probe.
    assert scored == ["4", "test"]


def test_replay_state_rejects_a_state_with_no_prompt_rather_than_using_its_transcript():
    # `prompt` and `messages` are not two spellings of one field: `new_rollout_state` seeds
    # `messages` with a COPY of `prompt` and appends every turn onto it. falling back to `messages`
    # when `prompt` is absent therefore records the transcript-so-far as the frozen prefix -- on any
    # state past turn zero that silently includes model turns the prompt never had. every producer
    # sets `prompt`, so its absence is a corrupt state and must be named as one.
    from flash.cli.commands.env.testing.test import _new_multi_turn_replay_state

    class _Env:
        def new_rollout_state(self, _example):
            return {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }

    with pytest.raises(ValueError, match="prompt is not well-formed"):
        _new_multi_turn_replay_state(_Env(), {}, {})
