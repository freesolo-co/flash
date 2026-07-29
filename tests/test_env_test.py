"""Tests for local offline environment contract validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import flash.cli as cli
from flash.cli.env_test import _CONTROL_CANDIDATES, cmd_env_test


class _SingleTurnEnv:
    multi_turn = False
    max_turns = 8

    def __init__(self, *, rows=None, reward=1.0, wrong_reward=None):
        self.rows = [{"input": "what is 2 + 2?", "output": "4"}] if rows is None else rows
        self.reward_value = reward
        # a working grader separates a correct completion from a wrong one. default one full
        # point below the gold score so the flat-reward gate sees real ranking; pass
        # wrong_reward=reward to model a grader that cannot tell them apart.
        self.wrong_reward = reward - 1.0 if wrong_reward is None else wrong_reward
        self.completions: list[str] = []

    def dataset(self):
        return self.rows

    def prompt_messages(self, example):
        return [{"role": "user", "content": example["input"]}]

    def sft_completion(self, example):
        return [{"role": "assistant", "content": example.get("output", "")}]

    def reward(self, completion, example, state=None):
        self.completions.append(completion)
        gold = example.get("output", "")
        if gold and completion != gold:
            return self.wrong_reward
        return self.reward_value

    @property
    def replayed(self) -> list[str]:
        """Completions the driver actually replayed, without the gate's negative controls."""
        return [c for c in self.completions if c not in _CONTROL_CANDIDATES]


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
    assert env.replayed == ["4"]
    out = capsys.readouterr().out
    assert "episode 1: policy=replay turns=1 reward=1.000000" in out
    assert "1/1 episodes passed contract checks" in out
    assert "overall: PASS" in out


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
    assert env.replayed == ["4"]
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
    assert len(env.replayed) == 3
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


def test_env_test_grader_that_cannot_rank_completions_fails(monkeypatch, tmp_path, capsys):
    # a grader that hands a deliberately wrong answer the same score as its own gold answer
    # cannot rank completions at all. the contract checks still pass, so this must fail on the
    # reward signal itself, otherwise identically-zero advantages reach a gpu and the run can
    # never learn.
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=0.0, wrong_reward=0.0))

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=1 reward=0.000000" in captured.out
    assert "1/1 episodes passed contract checks" in captured.out
    assert "overall: PASS" not in captured.out
    assert "all 1 replayed episode(s) scored every deliberately wrong answer" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_substring_grader_gold_inside_a_control_still_passes(
    monkeypatch, tmp_path, capsys
):
    # the graders this repo ships by default accept a completion when the gold text occurs
    # anywhere inside it (BaseEnvironment.grade, and the exact_match_reward written by
    # `flash env setup`). a gold answer that is a word of a control string would then score that
    # control CORRECT, and reading the equal scores as "cannot rank" would fail a working
    # environment. only controls disjoint from the gold text may be scored.
    env_dir = _environment_dir(tmp_path)

    class _SubstringEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            gold = str(example.get("output") or "").strip()
            return 1.0 if gold and gold in completion else 0.0

    # each of these is a word of the English control candidate
    for gold in ("test", "answer", "wrong"):
        env = _SubstringEnv(rows=[{"input": "say the word", "output": gold}])
        _patch_loader(monkeypatch, env)

        assert cmd_env_test(_args(env_dir)) == 0, gold
        captured = capsys.readouterr()
        assert "overall: PASS" in captured.out, gold
        assert "overall: FAIL" not in captured.err, gold
        # whatever controls were scored, none may contain the gold answer
        for control in env.completions:
            if control != gold:
                assert gold not in control, (gold, control)


def test_env_test_permissive_grader_is_not_reported_as_unrankable(monkeypatch, tmp_path, capsys):
    # an open-ended task ("respond with a sentence") can legitimately accept a wrong english
    # sentence while still ranking other completions below it. scoring several controls rather
    # than one keeps such a grader passing, because the degenerate fillers still fail it.
    env_dir = _environment_dir(tmp_path)

    class _AnyProseEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            return 1.0 if " " in completion.strip() else 0.0

    # this gold shares no word with the english control, so that control really is scored here
    env = _AnyProseEnv(rows=[{"input": "say something", "output": "quick brown fox jumps"}])
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "overall: FAIL" not in captured.err
    # the permissive english control scored as high as the gold answer; only the degenerate
    # fillers separate, so the gate must have scored more than the first usable control.
    assert len(env.completions) > 2


@pytest.mark.parametrize("gold", [0.0, -0.1, 5.0])
def test_env_test_gold_reward_value_alone_never_fails_the_gate(monkeypatch, tmp_path, capsys, gold):
    # the reward contract accepts any finite scalar, so an env may legitimately score its gold
    # answer 0.0 or -0.1 with worse completions below it. only the gold-vs-wrong comparison may
    # fail the gate; the absolute value carries no information about the grader.
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=gold))

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert f"episode 1: policy=replay turns=1 reward={gold:.6f}" in captured.out
    assert "overall: PASS" in captured.out
    assert "check the reward function" not in captured.err


def test_env_test_partial_flat_rewards_warn_but_pass(monkeypatch, tmp_path, capsys):
    # one row the grader cannot separate is a hard row, not a broken grader: warn, but keep
    # passing so a strict reward function is not blocked from training.
    env_dir = _environment_dir(tmp_path)
    rows = [{"input": "q1", "output": "4"}, {"input": "q2", "output": "8"}]

    class _FirstRowOnlyEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            if example["output"] != "4":
                return 0.0
            return 1.0 if completion == "4" else 0.0

    _patch_loader(monkeypatch, _FirstRowOnlyEnv(rows=rows))

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "2/2 episodes passed contract checks" in captured.out
    assert "overall: PASS" in captured.out
    assert "check the reward function" in captured.err
    assert "all 2 replayed episode(s)" not in captured.err


def test_env_test_unrepresentable_gold_turn_is_excluded_from_the_gate(monkeypatch, tmp_path, capsys):
    # a native tool-call turn (content=None + tool_calls) cannot be replayed as text: the driver
    # sends an empty turn stripped of the call. a grader that correctly rejects that mutilated
    # transcript scores zero, which says nothing about the reward function, so it must not trip
    # the all-zero gate.
    env_dir = _environment_dir(tmp_path)

    class _ToolCallGoldEnv(_SingleTurnEnv):
        def sft_completion(self, example):
            return [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f"}}],
                },
                {"role": "assistant", "content": example.get("output", "")},
            ]

    _patch_loader(monkeypatch, _ToolCallGoldEnv(reward=0.0, wrong_reward=0.0))

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert "overall: PASS" in captured.out
    assert "all 1 replayed episode(s)" not in captured.err


def test_env_test_null_content_turn_without_tool_calls_still_feeds_the_gate(
    monkeypatch, tmp_path, capsys
):
    # a bare content=None turn with no tool_calls carries no payload, so replaying it as the
    # empty string loses nothing. marking it partial would excuse the episode from the gate and
    # let a grader that cannot rank completions pass.
    env_dir = _environment_dir(tmp_path)

    class _NullTurnGoldEnv(_SingleTurnEnv):
        def sft_completion(self, example):
            return [
                {"role": "assistant", "content": None},
                {"role": "assistant", "content": example.get("output", "")},
            ]

    _patch_loader(monkeypatch, _NullTurnGoldEnv(reward=0.0, wrong_reward=0.0))

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "all 1 replayed episode(s) scored every deliberately wrong answer" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_multi_message_single_turn_gold_is_excluded_from_the_gate(
    monkeypatch, tmp_path, capsys
):
    # the real single-turn scorer grades only the LAST assistant message
    # (flash.multimodal.assistant_completion_text), but the driver joins every assistant turn.
    # a grader that correctly accepts only the final answer therefore scores the joined text
    # like any wrong answer, which says nothing about the reward function.
    env_dir = _environment_dir(tmp_path)

    class _TrajectoryGoldEnv(_SingleTurnEnv):
        def sft_completion(self, example):
            return [
                {"role": "assistant", "content": "let me compute that"},
                {"role": "assistant", "content": example.get("output", "")},
            ]

        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            # only the exact final answer scores; the joined trajectory does not
            return 1.0 if completion == example.get("output", "") else 0.0

    _patch_loader(monkeypatch, _TrajectoryGoldEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert "overall: PASS" in captured.out
    assert "all 1 replayed episode(s)" not in captured.err


def test_env_test_multi_turn_episodes_never_reach_the_reward_gate(monkeypatch, tmp_path, capsys):
    # a multi-turn reward reads the accumulated rollout state, so swapping in one wrong
    # completion string cannot produce a comparable wrong episode and there is no control to
    # compare against. such episodes carry no evidence either way and must not fail the gate,
    # including when the env drives more turns than the gold transcript provides and the driver
    # pads the tail with echo filler.
    env_dir = _environment_dir(tmp_path)

    class _ShortGoldMultiTurnEnv(_MultiTurnEnv):
        def sft_completion(self, example):
            # one gold turn, but env_reply keeps the rollout going for two
            return [{"role": "assistant", "content": "first"}]

        def reward(self, completion, example, state=None):
            self.scored_state = state
            return 0.0

    _patch_loader(monkeypatch, _ShortGoldMultiTurnEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "check the reward function" not in captured.err


def test_env_test_sft_only_environment_is_not_failed_by_the_reward_gate(
    monkeypatch, tmp_path, capsys
):
    # `env test` takes no algorithm argument and the environment declares none, so it cannot
    # know a placeholder scorer will ever reach GRPO. the SFT worker trains on
    # prompt_messages/sft_completion and never calls env.reward (flash/engine/worker/sft.py),
    # so a grader that rejects the off-distribution control must not block an SFT push.
    env_dir = _environment_dir(tmp_path)

    class _SftOnlyEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            if completion in _CONTROL_CANDIDATES:
                raise NotImplementedError("this environment has no reward function")
            return 0.0

    _patch_loader(monkeypatch, _SftOnlyEnv())

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert "overall: PASS" in captured.out
    assert "overall: FAIL" not in captured.err


def test_env_test_echo_policy_zero_reward_still_passes(monkeypatch, tmp_path, capsys):
    # an echo episode has no gold answer to score, so a zero reward says nothing about the
    # grader. the zero-reward gate must only judge replayed gold answers.
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "say anything", "output": ""}], reward=0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=echo turns=1 reward=0.000000" in captured.out
    assert "overall: PASS" in captured.out
    assert "scored 0.0" not in captured.err


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


def test_env_test_split_flag_overrides_param_split(monkeypatch, tmp_path):
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, split="eval", param=["split=train"])) == 0
    assert seen["kwargs"] == {"split": "eval"}


def test_env_test_malformed_param_fails_before_loading(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=["justakey"])) == 1
    assert "kwargs" not in seen
    captured = capsys.readouterr()
    assert "--param must be KEY=VALUE" in captured.err
    assert "overall: FAIL" in captured.err
