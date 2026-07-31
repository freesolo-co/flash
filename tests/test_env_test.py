"""Tests for local offline environment contract validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import flash.cli as cli
from flash.cli.env_test import (
    _CONTROL_CANDIDATES,
    _SYNTHETIC_CONTROL_ALPHABET,
    _control_is_disjoint,
    _fmt_credited_turns,
    _group_separates,
    _Score,
    _synthetic_controls,
    cmd_env_test,
)


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

    def prompt_messages(self, example):
        # every real env has one: BaseEnvironment defines it (flash/envs/base.py:45) and the grpo
        # path calls it unconditionally (rl.py:237), so a multi_turn env that also scores natively
        # is reached through it.
        return [{"role": "user", "content": example["input"]}]

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
        state.setdefault("messages", []).append(reply)
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
        state.setdefault("messages", []).append(reply)
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
        state.setdefault("messages", []).append(reply)
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
        state.setdefault("messages", []).append(reply)
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
    namespace = argparse.Namespace(
        path=str(path), algorithm=None, credit_assignment=None, split=None, param=[]
    )
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
    # only the gold replay is asserted: the negative controls each drive their own rollout through
    # the same loop, so they append further turns after it.
    assert env.recorded[:3] == ["first", "", "third"]
    out = capsys.readouterr().out
    assert "overall: PASS" in out


def test_env_test_steps_the_env_on_the_final_turn_before_scoring(monkeypatch, tmp_path, capsys):
    # the worker now applies the last assistant turn before scoring (_final_env_step in
    # flash/engine/multiturn_rollout.py). without the same close-out here the command scores a
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


def test_env_test_rejects_a_mistyped_algorithm_before_resolving_the_path(tmp_path, capsys):
    # a mistyped --algorithm is a fact about the INVOCATION, so it has to be reported as one.
    # validating it after the path resolved meant a typo'd flag surfaced as whatever the path check
    # happened to say -- here "does not exist", which sends the user to fix the wrong argument
    # (codex[bot]). normalize_algorithm raises ValueError, which main() renders as an `error:` line.
    missing = tmp_path / "no-such-env"

    with pytest.raises(ValueError, match="unsupported algorithm: gpro"):
        cmd_env_test(_args(missing, algorithm="gpro"))

    captured = capsys.readouterr()
    assert "does not exist" not in captured.err
    assert "overall: FAIL" not in captured.err


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

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=1 reward=0.000000" in captured.out
    assert "1/1 episodes passed contract checks" in captured.out
    assert "overall: PASS" not in captured.out
    assert "all 1 replayed episode(s) scored every deliberately wrong answer" in captured.err
    assert "overall: FAIL" in captured.err


def test_a_tie_above_zero_is_reported_without_failing(monkeypatch, tmp_path, capsys):
    # a healthy grader can legitimately tie gold with every control: a safety scorer awarding 1 to
    # any response without a prohibited phrase does exactly that, while sampled unsafe completions
    # score 0 and give real advantages. the controls are only LEXICALLY disjoint from gold, so
    # nothing here establishes them as known-negative and the tie must not be failed on
    # (codex[bot]).
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=1.0, wrong_reward=1.0))

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "overall: FAIL" not in captured.err
    assert "cannot rank completions" in captured.err
    assert "not failed on" in captured.err


def test_a_tie_at_zero_is_still_conclusive(monkeypatch, tmp_path, capsys):
    # the LS-005 signature this gate exists for. a grader returning nothing for its own reference
    # answer is broken or missing a dependency however the reward is shaped, so unlike a non-zero
    # tie it stays a verdict.
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=0.0, wrong_reward=0.0))

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    assert "overall: FAIL" in capsys.readouterr().err


def test_controls_separating_only_from_each_other_are_not_flat(monkeypatch, tmp_path, capsys):
    # at a turn gold emitted nothing, the per-turn trainer drops gold and centres the CONTROLS
    # against one another, so controls scoring differently there earn real advantages. comparing
    # each control only against gold missed that: `_overlap` removes the coordinate from every gold
    # pairing, leaving the group falsely flat (codex[bot]).
    gold = _Score(episode=0.5, turns=(1.0, 1.0), emitted=(False, True))
    spread = [
        _Score(episode=0.5, turns=(1.0, 1.0), emitted=(True, True)),
        _Score(episode=0.5, turns=(0.0, 1.0), emitted=(True, True)),
    ]

    # gold is excluded at turn 0 and ties at turn 1, so no control separates from GOLD...
    assert not any(control.separates_from(gold, per_turn=True) for control in spread)
    # ...yet the controls differ where the trainer centres them, so the group is not flat.
    assert _group_separates(gold, spread, per_turn=True)

    tied = [
        _Score(episode=0.5, turns=(1.0, 1.0), emitted=(True, True)),
        _Score(episode=0.5, turns=(1.0, 1.0), emitted=(True, True)),
    ]
    assert not _group_separates(gold, tied, per_turn=True)


def test_the_flat_warning_prints_only_the_turns_it_compared(monkeypatch, tmp_path, capsys):
    # the raw vectors differ at a turn no member emitted at, which the comparison already dropped.
    # printing them claimed rewards were identical where they visibly differ (cursor).
    gold = _Score(episode=0.5, turns=(9.0, 1.0), emitted=(False, True))
    controls = [_Score(episode=0.5, turns=(3.0, 1.0), emitted=(False, True))]

    assert _fmt_credited_turns(gold, controls) == "(1.000000)"


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


def test_env_test_infinite_control_reward_fails_the_episode(monkeypatch, tmp_path, capsys):
    # infinity is NOT trl's unscorable marker -- isnan(inf) is false -- so it reaches the group as a
    # real number and contaminates every advantage in it. that is the same reward contract the gold
    # answer is already failed for, and excluding the episode as merely inconclusive would let
    # `overall: PASS` hide it.
    env_dir = _environment_dir(tmp_path)

    class _InfiniteControlEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            if completion != example.get("output", ""):
                return float("inf")
            return 1.0

    _patch_loader(monkeypatch, _InfiniteControlEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "0/1 episodes passed contract checks" in captured.out
    assert "reward is not finite for a non-reference completion" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_nan_single_turn_control_is_dropped_rather_than_failed(
    monkeypatch, tmp_path, capsys
):
    # NaN is the trainer's supported unscorable marker: trl excludes such a row from the group
    # baseline and zeroes its advantage. a grammar-constrained scorer marking a synthetic control
    # unscorable is behaving as designed, so failing the episode rejected a working scorer for the
    # single reason that a fixed control is not valid input for it (codex[bot]). the multi-turn path
    # already dropped these; this is the same rule on the single-turn path.
    env_dir = _environment_dir(tmp_path)

    class _GrammarEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            if completion == example.get("output", ""):
                return 1.0
            # a control inside the grammar still ranks; one outside it is unscorable.
            if set(completion) <= {"z"} or set(completion) <= {"0"}:
                return float("nan")
            return 0.0

    _patch_loader(monkeypatch, _GrammarEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "reward is not finite" not in captured.err, captured.err
    # the surviving control still separates, so the episode is not read as flat either.
    assert "cannot rank completions" not in captured.err, captured.err


def test_env_test_sft_placeholder_reward_warns_instead_of_failing(monkeypatch, tmp_path, capsys):
    # the sft worker builds rows from dataset()/prompt_messages()/sft_completion() and never calls
    # reward() (flash/engine/worker/sft.py), so an sft-only environment may ship a placeholder
    # scorer returning one constant. failing that would reject a working environment.
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=0.0, wrong_reward=0.0))

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "overall: FAIL" not in captured.err
    assert "cannot rank completions" in captured.err


def test_env_test_unknown_algorithm_warns_instead_of_failing(monkeypatch, tmp_path, capsys):
    # without --algorithm the command cannot know whether reward() will ever be consumed, so the
    # finding is reported without failing rather than blocking a possibly-sft environment.
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=0.0, wrong_reward=0.0))

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "overall: FAIL" not in captured.err
    assert "pass --algorithm to fail on this instead of warning" in captured.err


def test_env_test_inverted_reward_direction_is_reported_but_does_not_fail(
    monkeypatch, tmp_path, capsys
):
    # a grader whose sign is inverted scores wrong answers ABOVE gold, which grpo would maximize
    # straight away from the references. worth reporting -- but not worth failing on, because the
    # controls are only LEXICALLY disjoint from the gold text and a healthy grader rewarding an
    # open-ended property they share produces the identical picture (codex[bot]). so: warn, pass.
    env_dir = _environment_dir(tmp_path)

    class _InvertedEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            return 0.0 if completion == example.get("output", "") else 1.0

    _patch_loader(monkeypatch, _InvertedEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert "scored a deliberately wrong answer higher than the gold answer" in captured.err
    assert "overall: PASS" in captured.out


def test_env_test_a_grader_rewarding_length_is_not_failed_as_inverted(
    monkeypatch, tmp_path, capsys
):
    # the concrete case the verdict could not tell from an inverted sign: every fixed control runs
    # 64-67 characters, so a correct open-ended grader paying by response length outranks a short
    # gold reference with all of them (codex[bot]). maximizing that reward is right, and failing the
    # run on it would block a working environment.
    env_dir = _environment_dir(tmp_path)

    class _LengthRewardEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            return float(len(completion))

    _patch_loader(monkeypatch, _LengthRewardEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    # and the reader is told why the ordering is not conclusive, rather than just that it happened.
    assert "open-ended property" in captured.err


def test_env_test_control_rejected_on_one_row_still_counts_as_separation(
    monkeypatch, tmp_path, capsys
):
    # grpo's reward_fn catches a raising env scorer and scores that completion 0.0
    # (flash/engine/worker/rl.py), so a rejected control is real evidence of gold-vs-wrong
    # separation, not an inconclusive result. dropping the row would leave a later tied row to
    # fail the whole sample alone.
    env_dir = _environment_dir(tmp_path)
    rows = [{"input": "q1", "output": "4"}, {"input": "q2", "output": "8"}]

    class _RejectsControlEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            if example["output"] == "4":
                if completion != "4":
                    raise ValueError("cannot parse this completion")
                return 1.0
            return 0.0

    _patch_loader(monkeypatch, _RejectsControlEnv(rows=rows))

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "2/2 episodes passed contract checks" in captured.out
    assert "overall: PASS" in captured.out
    assert "overall: FAIL" not in captured.err


def test_env_test_gold_ending_in_an_empty_turn_is_excluded_from_the_gate(
    monkeypatch, tmp_path, capsys
):
    # assistant_completion_text returns a trailing empty content verbatim, so training would grade
    # "" here while the driver joins the earlier text. that mismatch means the reward below is not
    # the reward the run would compute, so the episode must not feed the flat-reward gate.
    env_dir = _environment_dir(tmp_path)

    class _EmptyLastTurnEnv(_SingleTurnEnv):
        def sft_completion(self, example):
            return [
                {"role": "assistant", "content": example.get("output", "")},
                {"role": "assistant", "content": ""},
            ]

    _patch_loader(monkeypatch, _EmptyLastTurnEnv(reward=0.0, wrong_reward=0.0))

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert "overall: PASS" in captured.out
    assert "overall: FAIL" not in captured.err


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


def test_env_test_unrepresentable_gold_turn_is_excluded_from_the_gate(
    monkeypatch, tmp_path, capsys
):
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

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
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


def test_env_test_echo_padded_multi_turn_episode_is_excluded_from_the_gate(
    monkeypatch, tmp_path, capsys
):
    # the env drives more turns than the gold transcript provides, so the driver pads the tail
    # with echo filler and the reward grades a transcript no correct policy would produce. that
    # is not the gold reward, so the episode carries no evidence either way and must not fail
    # the gate even though the grader here is genuinely flat.
    env_dir = _environment_dir(tmp_path)

    class _ShortGoldMultiTurnEnv(_MultiTurnEnv):
        def sft_completion(self, example):
            # one gold turn, but env_reply keeps the rollout going for two
            return [{"role": "assistant", "content": "first"}]

        def reward(self, completion, example, state=None):
            self.scored_state = state
            return 0.0

    _patch_loader(monkeypatch, _ShortGoldMultiTurnEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "check the reward function" not in captured.err


def test_env_test_multi_turn_grader_that_cannot_rank_completions_fails(
    monkeypatch, tmp_path, capsys
):
    # a multi-turn reward reads the accumulated rollout state, so the control is driven through
    # the same rollout loop answering the wrong text at every turn rather than assembled. a
    # grader that scores that wrong episode exactly as high as the replayed gold one cannot rank
    # completions and would reach a paid run with no learning signal.
    env_dir = _environment_dir(tmp_path)

    class _FlatMultiTurnEnv(_MultiTurnEnv):
        def reward(self, completion, example, state=None):
            self.scored_state = state
            return 0.0

    _patch_loader(monkeypatch, _FlatMultiTurnEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "check the reward function" in captured.err
    assert "cannot rank completions" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_multi_turn_control_is_a_driven_rollout(monkeypatch, tmp_path, capsys):
    # a multi-turn reward reads the accumulated rollout state, so the control has to be driven
    # through the same loop rather than assembled from one wrong string. assert that directly on
    # what the grader was handed: every scored control transcript must be a full rollout, the
    # same length as the gold one and wrong at every turn. an assembled control would score a
    # single-turn state the real trainer never produces, and its reward would not be comparable
    # to the gold reward the gate ranks it against.
    env_dir = _environment_dir(tmp_path)

    class _TranscriptScoringEnv(_MultiTurnEnv):
        def __init__(self):
            super().__init__()
            self.scored_transcripts = []

        def reward(self, completion, example, state=None):
            self.scored_state = state
            said = [
                message["content"]
                for message in state["messages"]
                if message.get("role") == "assistant"
            ]
            self.scored_transcripts.append(said)
            gold = [turn["content"] for turn in example["output"]]
            return float(sum(1 for text in said if text in gold))

    env = _TranscriptScoringEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "check the reward function" not in captured.err
    assert "reward direction" not in captured.err

    gold_transcript, *control_transcripts = env.scored_transcripts
    assert gold_transcript == ["first", "second"]
    assert control_transcripts, "no negative control was scored"
    for transcript in control_transcripts:
        assert len(transcript) == len(gold_transcript)
        assert set(transcript).isdisjoint(gold_transcript)


def test_env_test_multi_turn_rollouts_are_scored_in_one_listwise_call(
    monkeypatch, tmp_path, capsys
):
    # the worker hands its whole rollout request list to score_rollouts at once
    # (flash/engine/multiturn_rollout.py), so an env that ranks its candidates against each other --
    # a pairwise judge, a batch normalizer -- only produces training's numbers when it sees them
    # together. scoring each rollout in its own singleton call hands such an env a list of one,
    # where every candidate is trivially top-ranked, and the identical scores that come back read
    # as a grader that cannot rank (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _ListwiseRankingEnv(_MultiTurnEnv):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            self.batch_sizes.append(len(items))
            # rank within the batch: the score depends on the OTHER candidates, so a singleton call
            # cannot reproduce it. alone, every rollout is the best in its list and scores 1.0.
            correct = []
            for example, state in items:
                gold = {turn["content"] for turn in example["output"]}
                said = [m["content"] for m in state["messages"] if m.get("role") == "assistant"]
                correct.append(sum(1 for text in said if text in gold))
            best = max(correct)
            return [RolloutReward(episode=1.0 if hits == best else 0.0) for hits in correct]

    env = _ListwiseRankingEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "cannot rank completions" not in captured.err
    # the gold rollout is scored WITH its controls and only once: grading it alone as well would
    # show a stateful or listwise grader a request list the real run never submits.
    assert env.batch_sizes == [1 + len(_CONTROL_CANDIDATES)]


def test_gold_rollouts_no_control_batch_reached_are_graded_in_one_call(
    monkeypatch, tmp_path, capsys
):
    # the other end of the same listwise contract. episodes that never reach a control batch -- an
    # echo policy here -- still need a gold reward, and grading them one call each hands a listwise
    # rollout_rewards_many a request list the real run never submits: the worker submits its whole
    # rollout list to score_rollouts at once (flash/engine/multiturn_rollout.py), so a grader that
    # cannot score a singleton works under GRPO and fails its contract check here (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _BatchOnlyEnv(_MultiTurnEnv):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def dataset(self):
            # three episodes, so a per-episode call and a single batched call differ observably.
            return [{"input": f"prompt {index}", "output": []} for index in range(3)]

        def sft_completion(self, example):
            # no assistant turn to replay, so the policy is echo and no control batch runs.
            return [{"role": "user", "content": "no gold answer here"}]

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            self.batch_sizes.append(len(items))
            if len(items) < 2:
                raise ValueError("this grader normalizes across the batch and cannot score one")
            return [RolloutReward(episode=0.5) for _ in items]

    env = _BatchOnlyEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    # one call carrying all three, not three calls carrying one.
    assert env.batch_sizes == [3]


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


def test_env_test_grades_through_scores_breakdown_when_present(monkeypatch, tmp_path, capsys):
    # the grpo worker prefers scores_breakdown(...)["total"] and only falls back to reward()
    # (flash/engine/worker/rl.py). an env whose real composite grader lives there while reward() is
    # an inherited placeholder must be judged on the scorer training actually calls: grading it on
    # the placeholder would report a working environment as unable to rank.
    env_dir = _environment_dir(tmp_path)

    class _BreakdownEnv(_SingleTurnEnv):
        def __init__(self):
            super().__init__()
            self.breakdown_completions = []

        def reward(self, completion, example, state=None):
            # the placeholder: flat, and would trip the gate if it were the one consulted.
            self.completions.append(completion)
            return 0.0

        def scores_breakdown(self, completion, example, state=None):
            self.breakdown_completions.append(completion)
            gold = example.get("output", "")
            return {"total": 1.0 if gold and completion == gold else 0.0, "format": 0.0}

    env = _BreakdownEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=1 reward=1.000000" in captured.out
    assert "overall: PASS" in captured.out
    assert "cannot rank completions" not in captured.err
    # the composite grader saw the gold answer and every control; the placeholder saw nothing.
    assert "4" in env.breakdown_completions
    assert env.completions == []


def test_env_test_opd_placeholder_reward_warns_instead_of_failing(monkeypatch, tmp_path, capsys):
    # the opd worker consumes dataset()/prompt_messages()/sft_completion() and never reads a reward
    # (no .reward( or scores_breakdown call exists in flash/engine/worker/opd.py), so a placeholder
    # scorer is not a defect there. gating on samples_on_policy instead of on who consumes the
    # reward would fail this working environment.
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=0.0, wrong_reward=0.0))

    assert cmd_env_test(_args(env_dir, algorithm="opd")) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "overall: FAIL" not in captured.err
    assert "cannot rank completions" in captured.err


def test_env_test_multi_turn_scorer_error_fails_instead_of_scoring_zero(
    monkeypatch, tmp_path, capsys
):
    # the multi-turn path calls score_rollouts directly (flash/engine/multiturn_rollout.py) with no
    # except branch, so a scorer that raises there aborts the run. swallowing it here as 0.0 would
    # report the wrong episode as cleanly separated and pass an env that cannot survive its first
    # rollout.
    env_dir = _environment_dir(tmp_path)

    class _ControlRaisesEnv(_MultiTurnEnv):
        def reward(self, completion, example, state=None):
            self.scored_state = state
            said = [
                message["content"]
                for message in state["messages"]
                if message.get("role") == "assistant"
            ]
            if any(text in _CONTROL_CANDIDATES for text in said):
                raise RuntimeError("scorer cannot handle this rollout")
            return 0.5

    _patch_loader(monkeypatch, _ControlRaisesEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "0/1 episodes passed contract checks" in captured.out
    assert "scorer cannot handle this rollout" in captured.err
    assert "overall: FAIL" in captured.err


def test_env_test_text_free_gold_turn_still_reaches_the_reward_gate(monkeypatch, tmp_path, capsys):
    # a text-free gold turn (a native tool call) carries no text a control could collide with, so it
    # cannot make a control unusable. keeping it in control selection would trip the empty-gold
    # guard, return None for the episode, and silently exclude it from the gate -- letting this flat
    # grader pass unexamined.
    env_dir = _environment_dir(tmp_path)
    env = _TextFreeMultiTurnEnv()
    _patch_loader(monkeypatch, env)

    # reported, not failed on: the tie sits at a non-zero value, which a grader rewarding a
    # property these lexically-disjoint controls happen to share would also produce.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert "cannot rank completions" in captured.err
    assert "overall: FAIL" not in captured.err


def test_env_test_per_turn_credit_separates_a_flat_episode_score(monkeypatch, tmp_path, capsys):
    # under credit_assignment="per_turn" the trainer credits each assistant turn by its own
    # group-relative reward (GRPOPerTurnTrainer, selected in flash/engine/worker/rl.py), so an env
    # may hold the episode scalar constant while the per-turn vector still ranks gold above wrong.
    # reading the scalar alone would call that env unable to rank while training learns from it.
    env_dir = _environment_dir(tmp_path)

    class _PerTurnCreditEnv(_MultiTurnEnv):
        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            rewards = []
            for example, state in items:
                gold = {turn["content"] for turn in example["output"]}
                said = [
                    message["content"]
                    for message in state["messages"]
                    if message.get("role") == "assistant"
                ]
                # constant episode score; all the ranking signal lives in the per-turn vector.
                rewards.append(
                    RolloutReward(
                        episode=0.5,
                        turns=tuple(1.0 if text in gold else 0.0 for text in said),
                    )
                )
            return rewards

    _patch_loader(monkeypatch, _PerTurnCreditEnv())

    args = _args(env_dir, algorithm="grpo", credit_assignment="per_turn")
    assert cmd_env_test(args) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=2 reward=0.500000" in captured.out
    assert "overall: PASS" in captured.out
    assert "cannot rank completions" not in captured.err


def test_env_test_per_turn_vectors_do_not_rescue_a_flat_run_by_default(
    monkeypatch, tmp_path, capsys
):
    # the same environment under the DEFAULT credit assignment. train.credit_assignment defaults to
    # per_episode (flash/spec.py), where select_grpo_trainer returns the ordinary scalar trainer and
    # these vectors are never read -- the run sees one constant 0.5 for gold and wrong alike and
    # computes zero advantages. reading the vectors regardless passed an env that cannot train
    # (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _PerTurnCreditEnv(_MultiTurnEnv):
        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            rewards = []
            for example, state in items:
                gold = {turn["content"] for turn in example["output"]}
                said = [
                    message["content"]
                    for message in state["messages"]
                    if message.get("role") == "assistant"
                ]
                rewards.append(
                    RolloutReward(
                        episode=0.5,
                        turns=tuple(1.0 if text in gold else 0.0 for text in said),
                    )
                )
            return rewards

    _patch_loader(monkeypatch, _PerTurnCreditEnv())

    # the vectors are still not read, so the finding stands -- but the constant is 0.5, not zero,
    # so it is reported rather than failed on.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "cannot rank completions" in captured.err
    assert "overall: FAIL" not in captured.err


def test_env_test_thinking_reference_is_excluded_from_the_reward_gate(
    monkeypatch, tmp_path, capsys
):
    # a gold answer carrying reasoning markup is graded by the run as graded_text leaves it, which
    # strips the <think> span under a thinking config (flash/engine/worker/rl.py). this command has
    # no run config to know whether thinking is on, so it cannot reproduce that text: an exact-answer
    # grader scores the raw reference and every control zero, and treating that as conclusive would
    # fail a working environment. the episode carries no evidence and must be excluded.
    env_dir = _environment_dir(tmp_path)

    class _ThinkingGoldEnv(_SingleTurnEnv):
        def sft_completion(self, example):
            return [{"role": "assistant", "content": "<think>work</think>4"}]

        def reward(self, completion, example, state=None):
            # an exact-answer grader: the raw reference still carries the reasoning span, so it
            # scores zero exactly as every deliberately wrong control does.
            self.completions.append(completion)
            return 1.0 if completion == example.get("output", "") else 0.0

    _patch_loader(monkeypatch, _ThinkingGoldEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert "overall: PASS" in captured.out
    assert "cannot rank completions" not in captured.err


def test_a_control_credited_above_gold_at_one_shared_turn_is_inverted():
    # a crossing pair. the control is admitted only when it is disjoint from EVERY gold turn, so
    # there is no turn at which it legitimately scores higher -- and build_per_turn_advantages
    # centres each index separately, so turn 1 hands the deliberately wrong text a positive
    # advantage and training reinforces it there. requiring dominance called this "neither is worse"
    # and let it pass (codex[bot]).
    gold = _Score(episode=1.0, turns=(1.0, 0.0))
    control = _Score(episode=1.0, turns=(0.0, 1.0))

    assert control.outranks(gold, per_turn=True)
    # separation is a different question and still true: both turns carry nonzero advantage.
    assert gold.separates_from(control, per_turn=True)
    assert control.separates_from(gold, per_turn=True)


def test_a_control_no_better_at_any_shared_turn_is_not_inverted():
    # the complement: worse at turn 0, tied at turn 1. no turn credits it above gold, so the sign is
    # right and only the flat gate is left to ask anything.
    gold = _Score(episode=1.0, turns=(1.0, 1.0))
    control = _Score(episode=1.0, turns=(0.0, 1.0))

    assert not control.outranks(gold, per_turn=True)
    assert control.separates_from(gold, per_turn=True)


def test_the_sum_does_not_decide_either_direction():
    # a sum rule gets this one wrong in the opposite direction from the crossing case: the control's
    # sum is far larger, but what makes it inverted is turn 1 specifically, not the total. the
    # verdict must come from the per-index comparison the trainer performs.
    gold = _Score(episode=1.0, turns=(1.0, 0.0))
    control = _Score(episode=1.0, turns=(0.0, 5.0))

    assert sum(control.turns or ()) > sum(gold.turns or ())
    assert control.outranks(gold, per_turn=True)
    # ...and a larger sum built ONLY from turns gold also wins is not inverted at all.
    tied_high = _Score(episode=1.0, turns=(0.5, 0.0))
    assert sum(tied_high.turns or ()) < sum(gold.turns or ())
    assert not tied_high.outranks(gold, per_turn=True)


def test_identical_per_turn_vectors_are_still_flat():
    # the gate must keep firing on a genuinely unrankable grader.
    gold = _Score(episode=1.0, turns=(1.0, 0.0))
    control = _Score(episode=1.0, turns=(1.0, 0.0))

    assert not gold.separates_from(control, per_turn=True)
    assert not control.outranks(gold, per_turn=True)
    assert not gold.outranks(control, per_turn=True)


def test_identical_vectors_are_flat_however_far_apart_the_episode_scores_are():
    # build_per_turn_advantages REPLACES the episode advantages with centred turn rewards, so an
    # env whose episode scalars differ but whose vectors match trains on exactly zero advantage.
    # consulting the scalar first reported that as separation (codex[bot]).
    gold = _Score(episode=1.0, turns=(1.0, 0.0))
    control = _Score(episode=0.0, turns=(1.0, 0.0))

    assert not gold.separates_from(control, per_turn=True)
    assert not control.outranks(gold, per_turn=True)
    # ...and the same pair IS separable in the default mode, where the scalar is what trains.
    assert gold.separates_from(control, per_turn=False)


@pytest.mark.parametrize(
    ("gold", "control", "separates"),
    [
        # a shorter control still overlaps at turn 0, where it is worse, so the turns the two
        # rollouts BOTH reached are evidence exactly as build_per_turn_advantages reads them.
        (_Score(episode=1.0, turns=(1.0, 0.0)), _Score(episode=1.0, turns=(0.0,)), True),
        # ...and an overlap that matches is still flat, so length alone decides nothing.
        (_Score(episode=1.0, turns=(1.0, 0.0)), _Score(episode=1.0, turns=(1.0,)), False),
        # a missing vector is the one real absence: there is no turn to compare against.
        (_Score(episode=1.0, turns=(1.0, 0.0)), _Score(episode=1.0, turns=None), False),
    ],
)
def test_unequal_turn_vectors_are_compared_over_their_overlap(gold, control, separates):
    # build_per_turn_advantages walks to the group's LONGEST vector and centres only the members
    # present at each index, so a control that terminated early is still credited on the turns it
    # took. discarding the whole comparison let a flat grader through on a short control (codex).
    assert control.separates_from(gold, per_turn=True) is separates
    assert gold.separates_from(control, per_turn=True) is separates


def test_a_shorter_control_that_is_worse_over_the_overlap_is_not_inverted():
    # the overlap is what the trainer centres, so an early-terminating wrong answer that is worse
    # everywhere it reached credits itself nowhere -- the sign is right, not incomparable.
    gold = _Score(episode=1.0, turns=(1.0, 1.0))
    control = _Score(episode=1.0, turns=(0.0,))

    assert not control.outranks(gold, per_turn=True)
    assert control.separates_from(gold, per_turn=True)


def test_a_missing_vector_falls_back_to_the_episode_scalar():
    # `_overlap` returns None when either side has no vector, which is also the case where
    # build_per_turn_advantages falls the whole group back to episode advantages -- so the scalar
    # is exactly the right thing to read there.
    gold = _Score(episode=1.0, turns=(1.0, 0.0))
    control = _Score(episode=2.0, turns=None)

    assert control.outranks(gold, per_turn=True)
    assert control.outranks(gold, per_turn=False)
    assert control.separates_from(gold, per_turn=True)


@pytest.mark.parametrize(
    "control",
    [
        _Score(episode=1.0, turns=(0.0, 1.0)),
        _Score(episode=1.0, turns=(0.0, 0.0)),
    ],
)
def test_per_turn_vectors_are_not_evidence_under_default_credit_assignment(control):
    # train.credit_assignment defaults to per_episode, where select_grpo_trainer returns the
    # ordinary scalar trainer and these vectors are never read. treating them as separation there
    # passes an env whose real run sees one constant episode score and computes zero advantages.
    gold = _Score(episode=1.0, turns=(1.0, 0.0))

    assert not control.separates_from(gold, per_turn=False)
    assert not control.outranks(gold, per_turn=False)
    assert not gold.outranks(control, per_turn=False)


def test_env_test_control_with_a_shorter_turn_vector_is_compared_over_the_overlap(
    monkeypatch, tmp_path, capsys
):
    # end-to-end companion to the overlap unit tests. a control that stops the rollout early scores
    # a SHORTER per-turn vector than the two-turn gold replay at the same episode score. the trainer
    # walks to the group's longest vector but centres each index against only the members present
    # there (build_per_turn_advantages), so the shared turn 0 is exactly where the credit lives --
    # and here the control is worse at it. discarding the comparison for the length mismatch made
    # this env's only evidence invisible (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _ShortControlRolloutEnv(_MultiTurnEnv):
        def env_reply(self, messages, state):
            # a wrong first turn ends the exchange, so the control rollout emits one turn where the
            # gold replay emits two.
            said = [m["content"] for m in state["messages"] if m.get("role") == "assistant"]
            if said and said[-1] != "first":
                state["done"] = True
                return []
            return super().env_reply(messages, state)

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            rewards = []
            for example, state in items:
                gold = {turn["content"] for turn in example["output"]}
                said = [m["content"] for m in state["messages"] if m.get("role") == "assistant"]
                # identical episode score either way; the vectors differ in LENGTH and at turn 0.
                rewards.append(
                    RolloutReward(
                        episode=0.5,
                        turns=tuple(1.0 if text in gold else 0.0 for text in said),
                    )
                )
            return rewards

    _patch_loader(monkeypatch, _ShortControlRolloutEnv())

    args = _args(env_dir, algorithm="grpo", credit_assignment="per_turn")
    assert cmd_env_test(args) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "cannot rank completions" not in captured.err


def test_env_test_gold_answer_colliding_with_every_fixed_control_is_still_controlled(
    monkeypatch, tmp_path, capsys
):
    # a gold answer drawing on several alphabets disqualifies the whole fixed control set at once:
    # "answer" rejects the english candidate, "z" the repeated-z one, "0" the repeated-zero one.
    # returning None there excluded the episode, so a constant grader passed unexamined.
    env_dir = _environment_dir(tmp_path)

    class _ConstantGraderEnv(_SingleTurnEnv):
        def dataset(self):
            return [{"input": "q", "output": "answer z 0"}]

        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            return 1.0

    env = _ConstantGraderEnv()
    _patch_loader(monkeypatch, env)

    # every fixed candidate is unusable for this gold answer...
    assert not any(_control_is_disjoint(c, "answer z 0") for c in _CONTROL_CANDIDATES)
    # ...so without a fallback the episode carried no control at all and the constant grader was
    # never examined. it is examined now; the constant is 1.0, so the finding is reported without
    # failing, and the report is the evidence the episode reached the gate.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "cannot rank completions" in captured.err


def test_synthetic_controls_are_disjoint_from_the_gold_answer():
    controls = _synthetic_controls(["answer z 0"])

    assert controls
    for control in controls:
        assert _control_is_disjoint(control, "answer z 0")


def test_synthetic_controls_supply_enough_alphabets_for_the_inversion_verdict():
    # the inversion verdict reads unanimity across controls on mutually exclusive alphabets, so a
    # lone fallback could never reach it and an inverted grader passed unexamined for any gold text
    # that disqualified the whole fixed set (codex[bot]).
    controls = _synthetic_controls(["answer z 0"])

    assert len(controls) > 1
    assert len({control[0] for control in controls}) == len(controls)


def test_an_inverted_grader_is_reported_when_the_gold_text_disqualifies_every_fixed_control(
    monkeypatch, tmp_path, capsys
):
    # the end of the same finding: the report needs unanimity across controls, so a single fallback
    # left it unreachable and an inverted grader went unmentioned for any gold text that
    # disqualified the whole fixed set -- "answer z 0" being enough to do it (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _InvertedGraderEnv(_SingleTurnEnv):
        def dataset(self):
            return [{"input": "q", "output": "answer z 0"}]

        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            # the sign is backwards: the gold answer scores lowest, every wrong answer above it.
            return 0.0 if completion == "answer z 0" else 1.0

    _patch_loader(monkeypatch, _InvertedGraderEnv())

    assert not any(_control_is_disjoint(c, "answer z 0") for c in _CONTROL_CANDIDATES)
    # reported, not failed on: an ordering alone cannot separate an inverted sign from a correct
    # reward for a property the controls share (see the length test above).
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "scored a deliberately wrong answer higher than the gold answer" in captured.err


def test_synthetic_controls_are_empty_when_the_gold_text_uses_every_character():
    # no character is left to build a provably-wrong control from, so the episode must still be
    # excluded rather than controlled against text that might be correct.
    exhaustive = _SYNTHETIC_CONTROL_ALPHABET

    assert _synthetic_controls([exhaustive]) == []


class _NonFiniteControlScorerEnv(_SingleTurnEnv):
    """A grader that scores its own reference finitely and anything else infinite.

    `_score_control` maps a raised exception to 0.0 for single-turn (mirroring reward_fn), so a
    non-finite score is the shape that actually reaches the caller. Infinity rather than NaN,
    because NaN is the trainer's supported unscorable marker and is dropped as inconclusive, while
    infinity reaches the group as a real number. Legitimate for sft/opd/opsd, whose workers never
    call the scorer at all; a real defect for grpo, whose reward_fn grades arbitrary policy
    completions.
    """

    def reward(self, completion, example, state=None):
        self.completions.append(completion)
        if completion != example.get("output", ""):
            return float("inf")
        return 1.0


def test_env_test_control_that_cannot_be_graded_does_not_fail_a_non_reward_algorithm(
    monkeypatch, tmp_path, capsys
):
    # scoring the control raised, and that propagated out as a failed EPISODE -- so the run reported
    # overall: FAIL and the algorithm gate below never got to soften it. the failure is a fact about
    # the control, not about the episode the driver had already replayed successfully.
    env_dir = _environment_dir(tmp_path)
    env = _NonFiniteControlScorerEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert "overall: PASS" in captured.out
    # the episode is not silently dropped: the missing evidence is reported.
    assert "could not score a deliberately wrong answer" in captured.err
    # ...and it is not counted as flat, which would be a finding the evidence does not support.
    assert "cannot rank completions" not in captured.err


def test_env_test_control_that_cannot_be_graded_still_fails_grpo(monkeypatch, tmp_path, capsys):
    # grpo's reward_fn does grade arbitrary completions, so a scorer that cannot score them is a
    # real defect and must keep failing. this is what stops the fix above from swallowing a live bug.
    env_dir = _environment_dir(tmp_path)
    env = _NonFiniteControlScorerEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "overall: FAIL" in captured.err


def test_env_test_one_control_without_a_vector_falls_the_group_back_to_the_scalar(
    monkeypatch, tmp_path, capsys
):
    # `build_per_turn_advantages` checks `any(row.turns is None for row in group)` and falls the
    # WHOLE group back to episode advantages, so a single control with no vector means no member
    # trains on turn credit. the gate must follow the trainer: read the episode scalars for every
    # comparison in the group, not just for the pair that is missing a vector.
    #
    # here the vectors alone would report an inversion -- gold (1, 0) against a crossing control
    # (0, 1) -- while the episode scalars agree gold is ahead of both. what actually trains is the
    # scalar, which ranks correctly, so failing this env would be a false positive built on numbers
    # the run never reads.
    env_dir = _environment_dir(tmp_path)

    class _MixedVectorEnv(_MultiTurnEnv):
        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            rewards = []
            for index, (example, state) in enumerate(items):
                gold_turns = {turn["content"] for turn in example["output"]}
                said = [
                    message["content"]
                    for message in state["messages"]
                    if message.get("role") == "assistant"
                ]
                if any(text in gold_turns for text in said):
                    rewards.append(RolloutReward(episode=1.0, turns=(1.0, 0.0)))
                elif index % 2:
                    # one control emits no vector at all, which is what collapses the group.
                    rewards.append(RolloutReward(episode=0.0, turns=None))
                else:
                    rewards.append(RolloutReward(episode=0.0, turns=(0.0, 1.0)))
            return rewards

    _patch_loader(monkeypatch, _MixedVectorEnv())

    args = _args(env_dir, algorithm="grpo", credit_assignment="per_turn")
    assert cmd_env_test(args) == 0, capsys.readouterr().err
    captured = capsys.readouterr()
    assert "reward direction looks inverted" not in captured.err, captured.err
    # and the scalars do separate, so this is not the flat finding either.
    assert "cannot rank completions" not in captured.err, captured.err


def test_env_test_control_that_cannot_be_graded_still_fails_when_the_algorithm_is_unset(
    monkeypatch, tmp_path, capsys
):
    # unset is unknown intent, not a promise that nothing grades, so the exemption must not extend
    # to it -- otherwise omitting --algorithm would hide a scorer grpo is about to break on.
    env_dir = _environment_dir(tmp_path)
    env = _NonFiniteControlScorerEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "overall: FAIL" in captured.err


def test_env_test_flat_reward_still_warns_for_a_non_reward_algorithm(monkeypatch, tmp_path, capsys):
    # a scorer that GRADES the controls fine but ranks them equal is a different case: the evidence
    # exists, so the flat finding must still be reported (as a warning, since sft never reads it).
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(reward=1.0, wrong_reward=1.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    captured = capsys.readouterr()
    assert "cannot rank completions" in captured.err
    assert [c for c in env.completions if c in _CONTROL_CANDIDATES]


def test_env_test_reports_a_grader_that_credits_a_wrong_answer_at_one_turn(
    monkeypatch, tmp_path, capsys
):
    # end-to-end companion to the crossing-vector unit test. the control matches no gold turn, so
    # its (0, 1) against gold's (1, 0) is not a tie the sums obscure -- turn 1 gives the wrong text
    # a positive advantage and training reinforces it there. the equal episode scalar and equal sum
    # are exactly why the two scalar readings both missed it (codex[bot]). the per-turn ordering has
    # to REACH the report; whether an ordering is conclusive is settled above.
    env_dir = _environment_dir(tmp_path)

    class _CrossingPerTurnEnv(_MultiTurnEnv):
        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            rewards = []
            for example, state in items:
                gold = {turn["content"] for turn in example["output"]}
                said = [
                    message["content"]
                    for message in state["messages"]
                    if message.get("role") == "assistant"
                ]
                # gold replays both reference turns and earns (1, 0); a control matches neither and
                # earns (0, 1). identical episode score, identical sum, different vectors.
                correct = [1.0 if text in gold else 0.0 for text in said]
                turns = (1.0, 0.0) if any(correct) else (0.0, 1.0)
                rewards.append(RolloutReward(episode=0.5, turns=turns))
            return rewards

    _patch_loader(monkeypatch, _CrossingPerTurnEnv())

    args = _args(env_dir, algorithm="grpo", credit_assignment="per_turn")
    assert cmd_env_test(args) == 0
    captured = capsys.readouterr()
    assert "reward direction may be inverted" in captured.err
    # the warning must quote the vectors it compared: the episode scalars are equal here, so
    # printing them would read "0.500000 scored higher than 0.500000".
    assert "gold turns (1.000000, 0.000000)" in captured.err, captured.err
    assert "wrong answer (0.000000, 1.000000)" in captured.err, captured.err
    # and not the flat finding: the vectors do differ.
    assert "cannot rank completions" not in captured.err


def test_a_grader_rewarding_an_open_ended_property_is_not_called_inverted(
    monkeypatch, tmp_path, capsys
):
    # a grader may reward a property of the completion rather than a match against the reference,
    # and then a lexically disjoint control can be genuinely better: one point per "z" scores
    # "z" * 64 above a gold "pizza", and training toward it is what the env asks for. reading that
    # single win as an inverted sign rejected a healthy environment (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _PropertyEnv(_SingleTurnEnv):
        def __init__(self):
            super().__init__(rows=[{"input": "say something with a z", "output": "pizza"}])

        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            return float(completion.casefold().count("z"))

    _patch_loader(monkeypatch, _PropertyEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    captured = capsys.readouterr()
    assert "reward direction" not in captured.err, captured.err
    # and it is not read as flat either: the controls do score differently from gold.
    assert "cannot rank completions" not in captured.err, captured.err


def test_env_turns_that_do_not_reproduce_exclude_the_episode_from_the_gate(
    monkeypatch, tmp_path, capsys
):
    # the assistant turns are only half a multi-turn transcript. an env whose own replies differ
    # from the reference trajectory hands the grader a different episode under the same assistant
    # strings, so a correct grader scores that supposed gold rollout like the controls and the gate
    # reports a flat grader for an env that ranks fine (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _InterleavedEnv(_MultiTurnEnv):
        """A multi-turn env whose reference trajectory records the observation it replied with."""

        observation = "observation-A"

        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "user", "content": "observation-A"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def env_reply(self, messages, state):
            state["turn"] += 1
            state["done"] = state["turn"] >= 2
            reply = {"role": "user", "content": self.observation}
            state.setdefault("messages", []).append(reply)
            return [reply]

        def reward(self, completion, example, state=None):
            # scores the WHOLE transcript, so a differing observation lands it with the controls.
            said = [
                message["content"]
                for message in state["messages"]
                if message.get("role") != "system"
            ]
            gold = [turn["content"] for turn in example["output"]]
            # the prompt opens the transcript, so the completion starts after it; env_reply appends
            # one more observation past the final assistant turn, so compare gold as a prefix.
            return 1.0 if said[1:][: len(gold)] == gold else 0.0

    class _DivergentEnv(_InterleavedEnv):
        # a stochastic env: this run's observation is not the one the reference recorded.
        observation = "observation-B"

    _patch_loader(monkeypatch, _DivergentEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    captured = capsys.readouterr()
    assert "cannot rank completions" not in captured.err, captured.err

    # the control: an env that does reproduce its observation stays in the gate, so the exclusion
    # above is about the divergence rather than about interleaved transcripts in general.
    _patch_loader(monkeypatch, _InterleavedEnv())
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    assert "cannot rank completions" not in capsys.readouterr().err


def test_env_turn_reproduction_is_judged_by_position_not_by_the_transcript_tail(
    monkeypatch, tmp_path, capsys
):
    # the turn loop appends one more env_reply after the final assistant turn, so the reference's
    # observations do not sit at the end of the driven transcript. comparing against the tail
    # shifted every one of them by that trailing reply, which broke the exclusion in both
    # directions: a faithful replay was dropped from the gate, and a divergence earlier in the
    # trajectory was hidden whenever the trailing reply happened to match (cursor).
    env_dir = _environment_dir(tmp_path)

    class _RecordedObservationEnv(_MultiTurnEnv):
        """Reference trajectory records the observation replied at that point in the exchange."""

        replies = ("observation-1", "observation-2")

        def __init__(self):
            super().__init__()
            self.replied = 0

        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "user", "content": "observation-1"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def new_rollout_state(self, example):
            # the gold and control rollouts share this instance, and each drives the reply
            # sequence from its start.
            self.replied = 0
            return super().new_rollout_state(example)

        def env_reply(self, messages, state):
            state["turn"] += 1
            state["done"] = state["turn"] >= 2
            reply = {"role": "user", "content": self.replies[self.replied]}
            self.replied += 1
            state.setdefault("messages", []).append(reply)
            return [reply]

        def reward(self, completion, example, state=None):
            # flat on purpose: this env really cannot rank, and the gate has to be able to say so.
            return 0.5

    # a faithful replay: "observation-1" comes back where the reference recorded it, and the
    # trailing "observation-2" is past everything the reference claims. the episode belongs in the
    # gate, so the flat grader above must still be reported.
    _patch_loader(monkeypatch, _RecordedObservationEnv())
    # the finding is what proves the episode reached the gate. the flat 0.5 is non-zero, so it is
    # reported rather than failed on; an excluded episode would be silent instead.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert "cannot rank completions" in capsys.readouterr().err

    class _LateMatchEnv(_RecordedObservationEnv):
        # diverges at the turn the reference recorded, then replies with the recorded text one turn
        # too late. the tail comparison read that coincidence as a faithful replay.
        replies = ("observation-X", "observation-1")

        def reward(self, completion, example, state=None):
            # scores the whole transcript, so the differing observation lands gold with the
            # controls -- a flat-grader report this env has not earned.
            said = [
                message["content"]
                for message in state["messages"]
                if message.get("role") != "system"
            ]
            gold = [turn["content"] for turn in example["output"]]
            return 1.0 if said[1:][: len(gold)] == gold else 0.0

    _patch_loader(monkeypatch, _LateMatchEnv())
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    assert "cannot rank completions" not in capsys.readouterr().err


def test_a_system_turn_in_the_reference_is_compared_on_both_sides(monkeypatch, tmp_path, capsys):
    # the reference and the driven transcript must be filtered by the same rule. dropping system
    # turns from the driven side alone left the reference's system turn with nothing to line up
    # against, shifting every later observation by one and marking an exact replay unreproduced --
    # excluding a healthy env from the gate for having a system turn at all (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _SystemObservationEnv(_MultiTurnEnv):
        """Replies with a system turn ahead of the observation, and records both in the reference."""

        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "system", "content": "tool budget exceeded"},
                        {"role": "user", "content": "observation-1"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def env_reply(self, messages, state):
            state["turn"] += 1
            state["done"] = state["turn"] >= 2
            replies = [
                {"role": "system", "content": "tool budget exceeded"},
                {"role": "user", "content": "observation-1"},
            ]
            state.setdefault("messages", []).extend(replies)
            return replies

        def reward(self, completion, example, state=None):
            # flat on purpose: the env genuinely cannot rank, so the gate has to still say so. that
            # report is the evidence the episode was not silently dropped -- an excluded episode and
            # a healthy one are otherwise indistinguishable from the outside, both being silent.
            return 0.5

    _patch_loader(monkeypatch, _SystemObservationEnv())
    # as above: the report is the evidence the episode was not silently dropped, and the non-zero
    # constant keeps it a warning.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert "cannot rank completions" in capsys.readouterr().err


def test_an_extra_environment_turn_is_not_a_faithful_replay(monkeypatch, tmp_path, capsys):
    # comparing the driven side as a flat prefix accepted an env that interleaved a message the
    # reference never recorded: reference a1/x/a2 against a driven a1/x/<extra>/a2 matched, though
    # the grader received a materially different episode (codex[bot]). the episode must be excluded
    # from the flat gate rather than counted as gold.
    env_dir = _environment_dir(tmp_path)

    class _InterleavingEnv(_MultiTurnEnv):
        """Emits an unrecorded tool message ahead of the observation the reference records."""

        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "user", "content": "observation-1"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def env_reply(self, messages, state):
            state["turn"] += 1
            state["done"] = state["turn"] >= 2
            replies = [
                {"role": "user", "content": "observation-1"},
                # never recorded in the reference above
                {"role": "tool", "content": "retrying upstream call"},
            ]
            state.setdefault("messages", []).extend(replies)
            return replies

        def reward(self, completion, example, state=None):
            # flat, so the gate would report it were this episode counted. the exclusion is what
            # keeps the run silent.
            return 0.5

    _patch_loader(monkeypatch, _InterleavingEnv())
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert "cannot rank completions" not in capsys.readouterr().err


def test_a_prompt_ending_in_an_assistant_turn_still_replays_faithfully(
    monkeypatch, tmp_path, capsys
):
    # a prefill prompt -- one ending in an assistant turn -- is copied through unfiltered by
    # flash/engine/multiturn_rollout.py, and the rollout state's transcript starts as a copy of the
    # prompt. skipping the prompt's NON-ASSISTANT messages by count therefore left that trailing
    # turn to open a block, shifting every reference block by one. an exact replay was marked
    # partial_replay, excluded from the control gate, and an all-zero grader could pass unreported
    # (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _PrefillEnv(_MultiTurnEnv):
        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    # the gold transcript must RECORD an observation. a completion of pure
                    # assistant turns makes every reference block empty, `any(reference)` false,
                    # and _env_turns_reproduce returns True before it aligns anything -- so the
                    # defect is unreachable and a test built on it cannot fail either way.
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "user", "content": "continue"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def new_rollout_state(self, example):
            prompt = [
                {"role": "user", "content": example["input"]},
                # the prefill: part of the PROMPT, not of the completion.
                {"role": "assistant", "content": "let me start"},
            ]
            return {"prompt": prompt, "messages": list(prompt), "done": False, "turn": 0}

        def reward(self, completion, example, state=None):
            # flat at zero, which is the conclusive signature. it can only be reported if the
            # episode reached the gate, so a shifted alignment silently passes instead.
            return 0.0

    _patch_loader(monkeypatch, _PrefillEnv())
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    assert "cannot rank completions" in capsys.readouterr().err


def test_the_same_text_split_into_blocks_is_still_a_faithful_replay(monkeypatch, tmp_path, capsys):
    # _message_text joins text blocks, so the text is already carried. counting them in the shape
    # too marked a faithful replay partial_replay purely for expressing the same string differently,
    # dropping it from the control gate and letting a flat-zero grader pass unreported (cursor).
    env_dir = _environment_dir(tmp_path)

    class _SplitTextEnv(_MultiTurnEnv):
        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        # the gold transcript records the observation as ONE plain string.
                        {"role": "user", "content": "continue"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def env_reply(self, messages, state):
            state["turn"] += 1
            state["done"] = state["turn"] >= 2
            # the live env emits the same string split across two text blocks. same payload, same
            # extracted text, and nothing an image-aware check should object to.
            reply = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "cont"},
                    {"type": "text", "text": "inue"},
                ],
            }
            state.setdefault("messages", []).append(reply)
            return [reply]

        def reward(self, completion, example, state=None):
            return 0.0

    _patch_loader(monkeypatch, _SplitTextEnv())

    # a faithful replay, so the flat-zero grader must still be caught.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    assert "cannot rank completions" in capsys.readouterr().err


def test_a_function_call_turn_is_not_a_faithful_replay(monkeypatch, tmp_path, capsys):
    # function_call is the older shape of the same payload tool_calls carries, and both are live in
    # the openai schema. checking only the newer name marked such a turn representable, replayed it
    # as its empty content string, and admitted the mutilated rollout to the gate (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _FunctionCallEnv(_MultiTurnEnv):
        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {
                            "role": "assistant",
                            "content": None,
                            "function_call": {"name": "calc", "arguments": '{"a": 1}'},
                        },
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def reward(self, completion, example, state=None):
            return 0.0

    _patch_loader(monkeypatch, _FunctionCallEnv())

    # the structured call cannot be replayed as text, so the episode is excluded rather than
    # counted as evidence -- leaving nothing controlled and no flat grader to report.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert "cannot rank completions" not in capsys.readouterr().err


def test_an_extra_message_beside_the_recorded_final_observation_is_not_faithful(
    monkeypatch, tmp_path, capsys
):
    # the gold transcript's last block states what the env replied after its final assistant turn,
    # and one env_reply call produced it. accepting the driven block as a mere PREFIX let an env
    # slip an unexpected message in alongside that reply, though the grader scores the complete
    # state and could score the supposed gold rollout like the controls (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _ExtraFinalMessageEnv(_MultiTurnEnv):
        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        # recorded final block: exactly one observation.
                        {"role": "user", "content": "continue"},
                    ],
                }
            ]

        def env_reply(self, messages, state):
            state["turn"] += 1
            state["done"] = state["turn"] >= 1
            # the recorded reply, plus one the transcript never claimed.
            replies = [
                {"role": "user", "content": "continue"},
                {"role": "system", "content": "unexpected"},
            ]
            state.setdefault("messages", []).extend(replies)
            return replies

        def reward(self, completion, example, state=None):
            return 0.0

    _patch_loader(monkeypatch, _ExtraFinalMessageEnv())

    # not a faithful replay, so the episode is excluded and the flat-zero grader is not reported.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert "cannot rank completions" not in capsys.readouterr().err


def test_an_observation_that_dropped_an_image_is_not_a_faithful_replay(
    monkeypatch, tmp_path, capsys
):
    # _message_text keeps only text blocks, so an observation carrying an image read identically to
    # one that never had it. the grader reads the whole rollout state, so it saw a transcript the
    # replay did not reproduce, and the episode was admitted to the control gate as faithful
    # (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _DroppedImageEnv(_MultiTurnEnv):
        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {
                            "role": "user",
                            # the gold transcript records an image alongside the text.
                            "content": [
                                {"type": "text", "content": None, "text": "continue"},
                                {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                            ],
                        },
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def env_reply(self, messages, state):
            state["turn"] += 1
            state["done"] = state["turn"] >= 2
            # the live env replies with the SAME text but no image. identical under (role, text).
            reply = {"role": "user", "content": [{"type": "text", "text": "continue"}]}
            state.setdefault("messages", []).append(reply)
            return [reply]

        def reward(self, completion, example, state=None):
            return 0.0

    _patch_loader(monkeypatch, _DroppedImageEnv())

    # the replay is not faithful, so the episode must be excluded rather than counted as evidence.
    # with no controlled episode left, the flat-zero grader is not reported and the run passes.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert "cannot rank completions" not in capsys.readouterr().err


def test_a_reordered_image_is_not_a_faithful_replay(monkeypatch, tmp_path, capsys):
    # one step past the dropped-image case: filtering every text block out of the shape also erased
    # WHERE the non-text blocks sat, so [image, text] and [text, image] both read as
    # ("image_url",). an env that reorders an image relative to its caption then admitted a
    # materially different transcript to the control gate, where an order-aware grader scores the
    # supposed gold replay like the controls and reports a working env as flat (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _ReorderedImageEnv(_MultiTurnEnv):
        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {
                            "role": "user",
                            # the gold records the image BEFORE its caption.
                            "content": [
                                {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                                {"type": "text", "text": "continue"},
                            ],
                        },
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def env_reply(self, messages, state):
            state["turn"] += 1
            state["done"] = state["turn"] >= 2
            # the live env emits the same blocks in the opposite order: identical text, identical
            # set of block kinds, different transcript.
            reply = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "continue"},
                    {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                ],
            }
            state.setdefault("messages", []).append(reply)
            return [reply]

        def reward(self, completion, example, state=None):
            return 0.0

    _patch_loader(monkeypatch, _ReorderedImageEnv())

    # excluded rather than counted as evidence, so the flat-zero grader is not reported against it.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert "cannot rank completions" not in capsys.readouterr().err


def test_a_completion_only_message_list_is_aligned_without_skipping_real_turns(
    monkeypatch, tmp_path, capsys
):
    # the production driver takes the opening from `prompt` when it is present and does NOT require
    # `messages` to duplicate it (flash/engine/multiturn_rollout.py:171-175). skipping len(prompt)
    # unconditionally therefore removed real completion messages from such a state, shifting every
    # block and marking an exact replay partial_replay -- dropping it from the control gate, where a
    # flat-zero grader then passes unreported (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _CompletionOnlyStateEnv(_MultiTurnEnv):
        def new_rollout_state(self, example):
            state = super().new_rollout_state(example)
            state["messages"] = []  # `prompt` carries the opening; `messages` records turns only
            return state

        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "user", "content": "continue"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def reward(self, completion, example, state=None):
            # flat: the run must REPORT this, which it can only do if the replay was admitted.
            return 0.0

    _patch_loader(monkeypatch, _CompletionOnlyStateEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    assert "cannot rank completions" in capsys.readouterr().err


def test_the_prompt_prefix_is_recognized_through_a_regenerated_per_call_key(
    monkeypatch, tmp_path, capsys
):
    # the prefix check must read the prompt the same way the rest of the comparison does: role, text,
    # and block shape. comparing whole dicts would fail on any per-call key the env regenerates when
    # it seeds its transcript -- the prefix would then go unrecognized, no messages would be skipped,
    # and the prompt itself would be compared against the reference's observations, marking an exact
    # replay partial_replay. that is the same false exclusion `tool_call_id` already caused one level
    # up, and it drops the episode from the control gate.
    env_dir = _environment_dir(tmp_path)

    class _ReseededPromptEnv(_MultiTurnEnv):
        def new_rollout_state(self, example):
            state = super().new_rollout_state(example)
            # the transcript is seeded from the prompt but stamped afresh, exactly as an env that
            # mints an id per rollout would. identical under (role, text, shape); not `==`.
            state["messages"] = [
                {**dict(message), "request_id": f"req-{index}"}
                for index, message in enumerate(state["prompt"])
            ]
            return state

        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "user", "content": "continue"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def reward(self, completion, example, state=None):
            # flat, so it is only REPORTED if the replay was admitted to the gate.
            return 0.0

    _patch_loader(monkeypatch, _ReseededPromptEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    assert "cannot rank completions" in capsys.readouterr().err


def test_a_per_call_tool_id_does_not_make_a_faithful_replay_partial(monkeypatch, tmp_path, capsys):
    # the opposite direction, pinning what the comparison must NOT read. a fresh tool_call_id per
    # call is ordinary and two faithful runs of the same env differ in it, so comparing whole
    # payloads would mark an exact replay partial_replay and drop it from the gate -- the same false
    # exclusion that let a flat-zero grader pass unreported.
    env_dir = _environment_dir(tmp_path)

    class _VolatileToolIdEnv(_MultiTurnEnv):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "tool", "content": "42", "tool_call_id": "call_recorded"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def env_reply(self, messages, state):
            state["turn"] += 1
            state["done"] = state["turn"] >= 2
            self.calls += 1
            # same role, same text, same block shape -- a fresh id, as a real tool call issues.
            reply = {"role": "tool", "content": "42", "tool_call_id": f"call_live_{self.calls}"}
            state.setdefault("messages", []).append(reply)
            return [reply]

        def reward(self, completion, example, state=None):
            return 0.0

    _patch_loader(monkeypatch, _VolatileToolIdEnv())

    # a faithful replay, so the flat-zero grader must still be caught.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    assert "cannot rank completions" in capsys.readouterr().err


def test_an_unscorable_control_still_decides_the_groups_reward_path(monkeypatch, tmp_path, capsys):
    # an unscorable control is dropped from the comparisons because it earns no advantage -- but it
    # is still a MEMBER of the group the trainer builds, and build_per_turn_advantages demotes the
    # whole group to episode scalars as soon as one member has no turn vector
    # (grpo_perturn_trainer.py:59-63). judging the path on the survivors alone read per_turn where
    # production falls back to the tied scalars and produces no learning signal (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _UnscorableControlEnv(_MultiTurnEnv):
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

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            rewards = []
            for example, state in items:
                gold = {turn["content"] for turn in example["output"]}
                said = [
                    message["content"]
                    for message in state["messages"]
                    if message.get("role") == "assistant"
                ]
                if all(text in gold for text in said) and said:
                    # gold: tied at zero on the episode scalar, separated only per turn.
                    rewards.append(RolloutReward(episode=0.0, turns=(1.0, 1.0)))
                elif said and said[0].startswith("z"):
                    # one control the grader cannot score. NaN is the trainer's unscorable marker,
                    # and it arrives here with NO turn vector -- which is what demotes the group.
                    rewards.append(RolloutReward(episode=float("nan"), turns=None))
                else:
                    rewards.append(RolloutReward(episode=0.0, turns=(0.0, 0.0)))
            return rewards

    _patch_loader(monkeypatch, _UnscorableControlEnv())

    # the group really trains on the tied ZERO episode scalars, so this must be reported, not passed
    # on the per-turn vectors the trainer never reaches.
    assert cmd_env_test(_args(env_dir, algorithm="grpo", credit_assignment="per_turn")) == 1
    assert "cannot rank completions" in capsys.readouterr().err


def test_the_reference_completion_is_read_once_per_episode(monkeypatch, tmp_path, capsys):
    # sft_completion is not required to be pure: one that samples a stored trajectory or consumes an
    # iterator answers differently each call. it was called twice per episode -- once to pick the
    # assistant strings to replay, once again to build the observations to compare against -- so an
    # exact replay was checked against a trajectory it never replayed, marked partial_replay, and
    # dropped from the control gate, letting a flat-zero grader pass unreported (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _StatefulReferenceEnv(_MultiTurnEnv):
        """Answers sft_completion from a different stored trajectory on each call."""

        def __init__(self):
            super().__init__()
            self.calls = 0

        def sft_completion(self, example):
            self.calls += 1
            # both trajectories replay the SAME assistant turns; they differ only in what the env
            # is recorded as having said back, which is exactly what the observation check reads.
            observation = "continue" if self.calls == 1 else "something else entirely"
            return [
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": observation},
                {"role": "assistant", "content": "second"},
            ]

        def reward(self, completion, example, state=None):
            # flat at zero, which is conclusive -- but only reportable if the episode reaches the
            # gate. a spurious partial_replay excludes it and the run passes silently instead.
            return 0.0

    env = _StatefulReferenceEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    assert "cannot rank completions" in capsys.readouterr().err


def test_a_mixed_tie_sample_is_not_described_as_entirely_non_zero(monkeypatch, tmp_path, capsys):
    # `not conclusive` is `scored_zero != controlled`, which a MIXED sample satisfies too. claiming
    # every score was non-zero was then false, and pointed at the wrong episodes (cursor).
    env_dir = _environment_dir(tmp_path)

    class _MixedTieEnv(_MultiTurnEnv):
        """Ties every control with gold, at zero on the first episode and above it on the rest."""

        def __init__(self):
            super().__init__()
            self.episodes = 0

        def dataset(self):
            return [
                dict(row, input=f"{row['input']} {i}")
                for i, row in enumerate([_MultiTurnEnv.dataset(self)[0]] * 2)
            ]

        def reward(self, completion, example, state=None):
            # constant within an episode (so it ties) and different across them.
            return 0.0 if str(example["input"]).endswith("0") else 0.5

    _patch_loader(monkeypatch, _MixedTieEnv())
    cmd_env_test(_args(env_dir, algorithm="grpo"))
    err = capsys.readouterr().err
    assert "every score was the same non-zero value" not in err, err
    assert "tied above zero" in err, err


def test_an_unscorable_multi_turn_control_is_dropped_rather_than_failed(
    monkeypatch, tmp_path, capsys
):
    # score_rollouts canonicalizes a non-finite multi-turn episode to NaN deliberately: it is the
    # trainer's supported marker for a row the group baseline excludes and whose advantage is then
    # zeroed. converting it into a contract failure rejected an env that merely marks a malformed
    # episode unscorable, even when its other controls rank fine (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _GrammarEnv(_MultiTurnEnv):
        def reward(self, completion, example, state=None):
            said = [
                message["content"]
                for message in state["messages"]
                if message.get("role") == "assistant"
            ]
            gold = {turn["content"] for turn in example["output"]}
            if any(text in gold for text in said):
                return 1.0
            # outside the env's grammar: unscorable, not wrong.
            if any(set(text) <= {"z"} or set(text) <= {"0"} for text in said):
                return float("nan")
            return 0.0

    _patch_loader(monkeypatch, _GrammarEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "reward is not finite" not in captured.err, captured.err
    # the surviving control still separates, so the episode is not read as flat either.
    assert "cannot rank completions" not in captured.err, captured.err


def test_replay_fidelity_compares_the_role_of_an_observation_not_only_its_text(
    monkeypatch, tmp_path, capsys
):
    # the reference records an observation as a `tool` message and the live env emits the identical
    # string as `user`. that is a materially different transcript -- it renders differently under
    # the chat template, and a role-aware grader scores it differently -- so reducing both sides to
    # content alone reported a faithful replay for an episode the grader never saw (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _RoleEnv(_MultiTurnEnv):
        """Reference records its observation as a tool message; `reply_role` is what it emits."""

        reply_role = "tool"

        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "tool", "content": "observation-1"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def env_reply(self, messages, state):
            state["turn"] += 1
            state["done"] = state["turn"] >= 2
            reply = {"role": self.reply_role, "content": "observation-1"}
            state.setdefault("messages", []).append(reply)
            return [reply]

        def reward(self, completion, example, state=None):
            # role-aware, so a `user` observation is a different episode and scores like a control.
            observed = [
                (message.get("role"), message.get("content"))
                for message in state["messages"]
                if message.get("role") != "system"
            ]
            gold = [(turn["role"], turn["content"]) for turn in example["output"]]
            return 1.0 if observed[1:][: len(gold)] == gold else 0.0

    class _RoleShiftedEnv(_RoleEnv):
        # same strings, different role: the grader sees an episode the reference never recorded.
        reply_role = "user"

    _patch_loader(monkeypatch, _RoleShiftedEnv())
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    # excluded from the gate, so the flat finding is not raised against a grader that ranks fine.
    assert "cannot rank completions" not in capsys.readouterr().err

    # the control: the same env replying with the recorded role does reproduce, so the exclusion
    # above is about the role rather than about interleaved observations in general. it is flat by
    # construction there -- gold and controls all miss the reference -- and saying so proves the
    # episode reached the gate instead of being dropped.
    _patch_loader(monkeypatch, _RoleEnv())
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    assert "cannot rank completions" not in capsys.readouterr().err


def test_a_rollout_state_exposing_only_messages_is_accepted(monkeypatch, tmp_path, capsys):
    # the production driver accepts `state.get("prompt") or state.get("messages")`
    # (flash/engine/multiturn_rollout.py:171-175), so an env keeping its transcript under `messages`
    # alone is a supported shape. requiring `prompt` afterwards failed the episode before the reward
    # gate ever ran (codex[bot]). and the opening has to be SNAPSHOTTED: by the end of the loop that
    # same list holds every driven turn, so a re-read would report the whole transcript as prompt.
    env_dir = _environment_dir(tmp_path)

    class _MessagesOnlyEnv(_MultiTurnEnv):
        def new_rollout_state(self, example):
            state = super().new_rollout_state(example)
            del state["prompt"]
            return state

        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "user", "content": "continue"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def reward(self, completion, example, state=None):
            said = [
                message["content"]
                for message in state["messages"]
                if message.get("role") == "assistant"
            ]
            return 1.0 if said == ["first", "second"] else 0.0

    _patch_loader(monkeypatch, _MessagesOnlyEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "prompt is not well-formed" not in captured.err, captured.err
    # the reported prompt is the opening alone. the driven turns were appended to that same list, so
    # a re-read would show "first" here.
    prompt_line = [line for line in captured.out.splitlines() if line.strip().startswith("prompt:")]
    assert prompt_line, captured.out
    assert "first" not in prompt_line[0], prompt_line[0]


def test_every_rollout_of_the_run_is_scored_in_one_call(monkeypatch, tmp_path, capsys):
    # the worker submits its whole rollout request list to score_rollouts at once
    # (flash/engine/multiturn_rollout.py:687-695), and that list spans the generation batch rather
    # than one example. scoring per episode handed a listwise rollout_rewards_many a shorter list
    # than training ever gives it, which is enough to change the numbers (codex[bot]).
    env_dir = _environment_dir(tmp_path)
    rows = [
        {
            "input": f"exchange {index}",
            "output": [
                {"role": "assistant", "content": f"first-{index}"},
                {"role": "assistant", "content": f"second-{index}"},
            ],
        }
        for index in (1, 2)
    ]

    class _BatchCountingEnv(_MultiTurnEnv):
        def __init__(self):
            super().__init__()
            self.batches: list[int] = []

        def dataset(self):
            return rows

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            self.batches.append(len(items))
            rewards = []
            for example, state in items:
                gold = {turn["content"] for turn in example["output"]}
                said = [
                    message["content"]
                    for message in state["messages"]
                    if message.get("role") == "assistant"
                ]
                rewards.append(RolloutReward(episode=1.0 if set(said) <= gold else 0.0))
            return rewards

    env = _BatchCountingEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    # one call, holding both episodes' gold rollouts and every control.
    assert len(env.batches) == 1, env.batches
    assert env.batches[0] == len(rows) * (1 + len(_CONTROL_CANDIDATES)), env.batches


def test_a_reward_coordinate_for_a_text_free_turn_is_not_read_as_separation(
    monkeypatch, tmp_path, capsys
):
    # build_per_turn_advantages skips any turn whose span is zero-width
    # (flash/engine/worker/grpo_perturn_trainer.py:67-75), so a reward coordinate belonging to a
    # turn that emitted nothing is one no advantage is computed from. counting it read separation
    # into a group the trainer resolves to zero advantage everywhere (codex[bot]): gold replays
    # ("", "answer") for (1, 0) against controls' (0, 0), and turn 0 -- gold's empty one -- is the
    # only place they differ.
    env_dir = _environment_dir(tmp_path)

    class _EmptyFirstTurnEnv(_MultiTurnEnv):
        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": ""},
                        {"role": "assistant", "content": "answer"},
                    ],
                }
            ]

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            rewards = []
            for _example, state in items:
                said = [
                    message["content"]
                    for message in state["messages"]
                    if message.get("role") == "assistant"
                ]
                # credit only at the turn gold left empty, and tie the episode scalar so nothing
                # else can carry the separation.
                gold = said[:1] == [""]
                rewards.append(RolloutReward(episode=0.5, turns=(1.0, 0.0) if gold else (0.0, 0.0)))
            return rewards

    _patch_loader(monkeypatch, _EmptyFirstTurnEnv())

    args = _args(env_dir, algorithm="grpo", credit_assignment="per_turn")
    assert cmd_env_test(args) == 1
    assert "cannot rank completions" in capsys.readouterr().err


def test_a_reward_coordinate_for_a_turn_that_emitted_text_still_separates(
    monkeypatch, tmp_path, capsys
):
    # the control for the exclusion above: the same shape with gold's first turn carrying text is
    # a turn the trainer does credit, so it must still count as separation.
    env_dir = _environment_dir(tmp_path)

    class _TextFirstTurnEnv(_MultiTurnEnv):
        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            rewards = []
            for example, state in items:
                gold = {turn["content"] for turn in example["output"]}
                said = [
                    message["content"]
                    for message in state["messages"]
                    if message.get("role") == "assistant"
                ]
                correct = said[:1] and said[0] in gold
                rewards.append(
                    RolloutReward(episode=0.5, turns=(1.0, 0.0) if correct else (0.0, 0.0))
                )
            return rewards

    _patch_loader(monkeypatch, _TextFirstTurnEnv())

    args = _args(env_dir, algorithm="grpo", credit_assignment="per_turn")
    assert cmd_env_test(args) == 0, capsys.readouterr().err
    assert "cannot rank completions" not in capsys.readouterr().err


def test_a_reward_returning_a_non_number_fails_rather_than_scoring_zero(
    monkeypatch, tmp_path, capsys
):
    # the worker's guard covers the env CALL raising -- it scores that 0.0 and carries on. it does
    # NOT coerce what reward() returns (flash/engine/worker/rl.py:460-471), so a non-numeric value
    # is appended to the reward list as-is and aborts the run in trl. catching the conversion error
    # here as a valid 0.0 control passed an exact-match grader that returns 1.0 for gold and an
    # accidental string for everything else, which breaks on its first sampled completion
    # (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _BadReturnEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            return 1.0 if completion == example.get("output") else "oops"

    _patch_loader(monkeypatch, _BadReturnEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "is not a number" in captured.err
    assert "overall: FAIL" in captured.err
    assert "overall: PASS" not in captured.out


def test_a_reward_raising_is_still_scored_zero_rather_than_failed(monkeypatch, tmp_path, capsys):
    # the control for the test above, and the behaviour that must survive it. an env that RAISES on
    # an arbitrary completion is scored 0.0 by the worker and keeps training, so the control is real
    # evidence of separation and the episode must still pass.
    env_dir = _environment_dir(tmp_path)

    class _RaisingControlEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            if completion != example.get("output"):
                raise RuntimeError("no parse")
            return 1.0

    _patch_loader(monkeypatch, _RaisingControlEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "is not a number" not in captured.err


def test_a_non_numeric_breakdown_total_is_still_scored_zero(monkeypatch, tmp_path, capsys):
    # the asymmetry is deliberate and mirrors the worker: it coerces the breakdown total INSIDE its
    # own guard, so a bad total there really is scored 0.0 and the run continues. only the plain
    # reward() return reaches trl uncoerced.
    env_dir = _environment_dir(tmp_path)

    class _BadBreakdownEnv(_SingleTurnEnv):
        def scores_breakdown(self, completion, example, state=None):
            self.completions.append(completion)
            return {"total": 1.0 if completion == example.get("output") else "oops"}

    _patch_loader(monkeypatch, _BadBreakdownEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "is not a number" not in captured.err


def test_a_thinking_run_grades_the_state_the_worker_would_build(monkeypatch, tmp_path, capsys):
    # with thinking on, the worker strips the reasoning span before grading and passes a
    # raw/completion/thinking dict alongside it (flash/engine/worker/rl.py:444-461). passing None
    # and the unprocessed text graded a contract the run does not use, so a scorer reading its state
    # looked flat here while ranking correctly in production -- the wrong verdict either way
    # (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _StateReadingEnv(_SingleTurnEnv):
        def __init__(self):
            super().__init__()
            self.states: list[dict | None] = []

        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            self.states.append(state)
            # only ranks when handed the production state shape.
            if not isinstance(state, dict):
                return 0.0
            return 1.0 if completion == example.get("output") else 0.0

    env = _StateReadingEnv()
    _patch_loader(monkeypatch, env)
    args = cli._build_parser().parse_args(
        ["env", "test", str(env_dir), "--algorithm", "grpo", "--thinking"]
    )

    assert args.func(args) == 0, capsys.readouterr().err
    assert "overall: PASS" in capsys.readouterr().out
    assert env.states
    assert all(isinstance(state, dict) for state in env.states)
    assert set(env.states[0]) == {"raw", "completion", "thinking"}


def test_the_graded_text_of_a_thinking_run_has_its_reasoning_removed(monkeypatch, tmp_path, capsys):
    # the state is only half of it: the worker grades the answer with the <think> span stripped, so
    # a gold reference carrying reasoning must reach the scorer as its answer alone.
    env_dir = _environment_dir(tmp_path)

    class _ThinkingGoldEnv(_SingleTurnEnv):
        def __init__(self):
            super().__init__()
            self.states: dict[str, dict | None] = {}

        def sft_completion(self, example):
            return [{"role": "assistant", "content": "<think>weighing it up</think>4"}]

        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            self.states[completion] = state
            return 1.0 if completion == "4" else 0.0

    env = _ThinkingGoldEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo", thinking=True)) == 0
    assert env.replayed[0] == "4"
    # the gold answer's own state, keyed rather than taken from the last call: under --thinking the
    # gold reaches the ranking gate, so its negative controls are graded after it.
    gold_state = env.states["4"]
    assert gold_state["raw"] == "<think>weighing it up</think>4"
    assert gold_state["thinking"] == "weighing it up"


def test_a_run_without_thinking_still_grades_the_raw_text_with_no_state(
    monkeypatch, tmp_path, capsys
):
    # the control for the two above. thinking is off by default and the worker then passes None and
    # grades the completion verbatim, so the flag must change nothing unless it is set.
    env_dir = _environment_dir(tmp_path)

    class _StateRecordingEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            self.last_state = state
            return super().reward(completion, example, state)

    env = _StateRecordingEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert env.last_state is None


def test_a_thinking_run_gates_a_gold_answer_carrying_reasoning(monkeypatch, tmp_path, capsys):
    # under --thinking the run grades strip_think(gold), and so does this command, so a gold answer
    # carrying a <think> span replays faithfully and belongs in the ranking gate. excluding it there
    # set partial_replay and skipped control scoring, which is how a flat-zero grader on a thinking
    # run still reported overall: PASS -- the miss this gate exists to catch (cursor).
    env_dir = _environment_dir(tmp_path)

    class _FlatThinkingGoldEnv(_SingleTurnEnv):
        def sft_completion(self, example):
            return [{"role": "assistant", "content": "<think>work</think>4"}]

        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            return 0.0  # flat at zero: only REPORTED if the replay reached the gate

    _patch_loader(monkeypatch, _FlatThinkingGoldEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo", thinking=True)) == 1
    captured = capsys.readouterr()
    assert "cannot rank completions" in captured.err
    assert "overall: FAIL" in captured.err


def test_a_run_without_thinking_still_excludes_a_gold_answer_carrying_reasoning(
    monkeypatch, tmp_path, capsys
):
    # the control for the above, and the case the exclusion was written for. without thinking the run
    # grades the raw text, so the reasoning markup really does reach an exact-answer grader, which
    # scores it and every control zero. that is a property of the replay, not of the reward function.
    env_dir = _environment_dir(tmp_path)

    class _FlatThinkingGoldEnv(_SingleTurnEnv):
        def sft_completion(self, example):
            return [{"role": "assistant", "content": "<think>work</think>4"}]

        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            return 0.0

    _patch_loader(monkeypatch, _FlatThinkingGoldEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert "cannot rank completions" not in capsys.readouterr().err


def test_a_multi_turn_gold_carrying_reasoning_is_excluded_even_under_thinking(
    monkeypatch, tmp_path, capsys
):
    # the second control, pinning where the thinking allowance stops. a multi-turn transcript is
    # scored as a whole episode through reward_from_messages (flash/engine/worker/rl.py:436-438),
    # which never strips reasoning -- so gold markup reaches that grader verbatim whatever the run's
    # thinking mode says, and the turn stays unreproducible.
    env_dir = _environment_dir(tmp_path)

    class _ThinkingMultiTurnEnv(_MultiTurnEnv):
        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "<think>work</think>first"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def reward(self, completion, example, state=None):
            return 0.0  # flat at zero, so an admitted replay would be reported

    _patch_loader(monkeypatch, _ThinkingMultiTurnEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo", thinking=True)) == 0
    assert "cannot rank completions" not in capsys.readouterr().err


def test_a_tagless_thinking_gold_answer_reports_which_reading_was_assumed(
    monkeypatch, tmp_path, capsys
):
    # strip_think reads a completion with no closing </think> differently depending on whether the
    # RENDERED prompt already opened the span (rl.py:367-371 derives that from the chat template).
    # this command grades it as its own text; under a prompt-opening template the run grades "".
    # named rather than excluded: a plain gold answer is exactly this shape, so excluding it would
    # empty the gate for every thinking env (codex[bot]).
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo", thinking=True)) == 0
    captured = capsys.readouterr()
    assert "no closing </think>" in captured.err
    # reported, never failed on, and the gate still ran on the episode.
    assert "overall: PASS" in captured.out


def test_a_closed_thinking_span_reads_the_same_either_way_and_is_not_reported(
    monkeypatch, tmp_path, capsys
):
    # the control: a gold answer carrying </think> resolves to the same graded text under both
    # readings, so there is no ambiguity to report and the warning must stay silent.
    env_dir = _environment_dir(tmp_path)

    class _ClosedThinkingGoldEnv(_SingleTurnEnv):
        def sft_completion(self, example):
            return [{"role": "assistant", "content": "<think>work</think>4"}]

    _patch_loader(monkeypatch, _ClosedThinkingGoldEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo", thinking=True)) == 0
    assert "no closing </think>" not in capsys.readouterr().err


def test_a_run_without_thinking_never_reports_the_template_reading(monkeypatch, tmp_path, capsys):
    # the second control: without thinking the worker grades the raw text and strip_think is not
    # called at all, so no template reading enters the verdict and there is nothing to warn about.
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert "no closing </think>" not in capsys.readouterr().err


def test_a_raising_tools_hook_fails_a_grpo_run_on_per_episode_credit(monkeypatch, tmp_path, capsys):
    # the grpo worker calls env.tools() unguarded while building the tool loop (rl.py:816), outside
    # the try/except that scores a raising reward as 0.0, so the run aborts during initialization.
    # checking it only on the per-turn path surfaced it nowhere for a default run, and env test
    # reported PASS for a configuration that dies before its first step (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _RaisingToolsEnv(_MultiTurnEnv):
        is_tool_env = True

        def tools(self):
            raise RuntimeError("tool schema failed to load")

    _patch_loader(monkeypatch, _RaisingToolsEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo", credit_assignment="per_episode")) == 1
    captured = capsys.readouterr()
    assert "env.tools() raised (RuntimeError: tool schema failed to load)" in captured.err
    assert "overall: FAIL" in captured.err
    # refused before any episode runs, like the credit-assignment refusal beside it.
    assert "episode 1:" not in captured.out


def test_a_raising_tools_hook_is_not_a_finding_for_a_run_that_never_builds_the_tool_loop(
    monkeypatch, tmp_path, capsys
):
    # the control: only the grpo worker reaches rl.py's tool setup. an sft run never calls tools(),
    # so a raising hook is not that run's problem and must not fail it.
    env_dir = _environment_dir(tmp_path)

    class _RaisingToolsEnv(_MultiTurnEnv):
        is_tool_env = True

        def tools(self):
            raise RuntimeError("tool schema failed to load")

    _patch_loader(monkeypatch, _RaisingToolsEnv())

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0
    assert "env.tools() raised" not in capsys.readouterr().err


def test_a_completion_only_transcript_repeating_the_prompt_keeps_its_first_turn(
    monkeypatch, tmp_path, capsys
):
    # provenance is observed, not inferred. a prompt ending in an assistant prefill whose text the
    # first replayed turn repeats makes a completion-only transcript OPEN with the same observations
    # as the prompt, so a content test answers len(prompt) and drops a real completion message --
    # shifting every block, marking an exact replay partial_replay, and letting a flat-zero grader
    # pass unreported (codex[bot]). _run_rollout reads `messages` on the FRESH state instead.
    env_dir = _environment_dir(tmp_path)

    class _PrefillEchoEnv(_MultiTurnEnv):
        def new_rollout_state(self, example):
            # the opening is an assistant prefill the model continues from, and the gold's first
            # turn happens to repeat it -- so the completion transcript OPENS with a message the
            # prompt also holds. `prompt` carries the opening; `messages` records driven turns only.
            prompt = [{"role": "assistant", "content": "first"}]
            return {"prompt": prompt, "messages": [], "done": False, "turn": 0}

        def dataset(self):
            return [
                {
                    "input": "finish the exchange",
                    "output": [
                        {"role": "assistant", "content": "first"},
                        {"role": "user", "content": "continue"},
                        {"role": "assistant", "content": "second"},
                    ],
                }
            ]

        def reward(self, completion, example, state=None):
            return 0.0  # flat at zero, so an admitted replay is REPORTED

    _patch_loader(monkeypatch, _PrefillEchoEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    assert "cannot rank completions" in capsys.readouterr().err


def test_per_turn_credit_is_refused_for_a_native_tool_environment(monkeypatch, tmp_path, capsys):
    # a tool env exposing tools is driven through trl's tool loop rather than a rollout_func
    # (flash/engine/worker/rl.py:814-827), and select_grpo_trainer refuses per-turn credit on that
    # path. reporting the per-turn reward vectors as evidence passed an environment for a run that
    # raises before its first step (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _ToolEnv(_MultiTurnEnv):
        is_tool_env = True

        def tools(self):
            return [lambda x: x]

    _patch_loader(monkeypatch, _ToolEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo", credit_assignment="per_turn")) == 1
    captured = capsys.readouterr()
    assert "is not supported for tool-calling" in captured.err
    assert "overall: FAIL" in captured.err
    # refused before any episode runs: there is no reward evidence worth gathering for a
    # configuration that cannot reach a first training step.
    assert "episode 1:" not in captured.out


def test_a_tool_environment_on_per_episode_credit_is_not_refused(monkeypatch, tmp_path, capsys):
    # the control: the refusal is about the per-turn COMBINATION, not about tool envs. the default
    # mode trains fine on that path and must still be tested.
    #
    # it carries a reward_from_messages because that -- not reward() -- is the hook this run's
    # scorer calls (flash/engine/worker/rl.py:433-438), and only a ranking one leaves the refusal
    # as the single thing this control can fail on.
    env_dir = _environment_dir(tmp_path)

    class _ToolEnv(_MultiTurnEnv):
        is_tool_env = True

        def tools(self):
            return [lambda x: x]

        def reward_from_messages(self, completion, example):
            # the completion interleaves this env's own replies with the generated turns, so the
            # gold comparison is over the assistant messages only.
            said = [m.get("content") for m in completion if m.get("role") == "assistant"]
            return 1.0 if said == [m.get("content") for m in example["output"]] else 0.0

    _patch_loader(monkeypatch, _ToolEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo", credit_assignment="per_episode")) == 0
    captured = capsys.readouterr()
    assert "is not supported for tool-calling" not in captured.err
    assert "episode 1:" in captured.out


def test_a_raising_tools_hook_is_a_finding_for_a_single_turn_tool_environment(
    monkeypatch, tmp_path, capsys
):
    # rl.py:816 calls tools() for every is_tool_env, with no multi_turn in the condition, so a
    # single-turn tool env whose tools() raises aborts during initialization exactly as a
    # multi-turn one does. gating this command's call on multi_turn reported PASS for it
    # (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _SingleTurnToolEnv(_SingleTurnEnv):
        is_tool_env = True

        def tools(self):
            raise RuntimeError("tool registry unavailable")

    _patch_loader(monkeypatch, _SingleTurnToolEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "env.tools() raised (RuntimeError: tool registry unavailable)" in captured.err
    assert "overall: FAIL" in captured.err
    assert "episode 1:" not in captured.out


def test_a_native_tool_environment_is_graded_by_reward_from_messages(monkeypatch, tmp_path, capsys):
    # a tool env exposing tools is driven through trl's tool loop, whose message completion is
    # scored by reward_from_messages (rl.py:433-438) -- score_rollouts is never reached. grading it
    # through the rollout path let a placeholder rollout scorer pass an env whose real grader is
    # flat (codex[bot]). here the unused rollout scorer ranks and the real one does not, so only
    # reading the right hook can fail this run.
    env_dir = _environment_dir(tmp_path)

    class _FlatNativeGraderEnv(_MultiTurnEnv):
        is_tool_env = True

        def tools(self):
            return [lambda x: x]

        def reward_from_messages(self, completion, example):
            return 0.0

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            return [RolloutReward(episode=1.0) for _ in items]

    env = _FlatNativeGraderEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    assert "cannot rank completions" in capsys.readouterr().err


def test_a_native_tool_grader_is_handed_the_generated_messages_without_the_prompt(
    monkeypatch, tmp_path, capsys
):
    # trl's completion is the generated messages only -- the prompt is not part of it -- so the
    # opening is dropped as a positional prefix before the grader sees it.
    env_dir = _environment_dir(tmp_path)
    seen: list[list[dict]] = []

    class _RecordingNativeEnv(_MultiTurnEnv):
        is_tool_env = True

        def tools(self):
            return [lambda x: x]

        def reward_from_messages(self, completion, example):
            seen.append(completion)
            said = [m.get("content") for m in completion if m.get("role") == "assistant"]
            return 1.0 if said == [m.get("content") for m in example["output"]] else 0.0

    _patch_loader(monkeypatch, _RecordingNativeEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert seen, "the native grader was never called"
    for completion in seen:
        assert {"role": "user", "content": "finish the exchange"} not in completion


def test_a_raising_native_tool_grader_is_scored_zero_rather_than_failing_the_episode(
    monkeypatch, tmp_path, capsys
):
    # reward_fn wraps this call in `except Exception` and scores 0.0 (rl.py:462-471), so the run
    # survives a raising grader. failing here would report a run-surviving grader as run-aborting.
    # it is still reported: every episode ties at zero, which is the flat-reward finding.
    env_dir = _environment_dir(tmp_path)

    class _RaisingNativeEnv(_MultiTurnEnv):
        is_tool_env = True

        def tools(self):
            return [lambda x: x]

        def reward_from_messages(self, completion, example):
            raise RuntimeError("grader dependency missing")

    _patch_loader(monkeypatch, _RaisingNativeEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "cannot rank completions" in captured.err
    # scored, not aborted: the episode still reports a reward.
    assert "episode 1: policy=replay turns=2 reward=0.000000" in captured.out


def test_a_non_grpo_run_never_calls_tools_on_a_native_tool_environment(monkeypatch, tmp_path):
    # sft/opd/opsd read dataset/prompt/sft_completion and never build a tool loop, so tools() is
    # not their run's problem -- _grpo_rejection leaves it unvalidated for them, and the scoring
    # dispatch must not call it either.
    env_dir = _environment_dir(tmp_path)

    class _ExplodingToolsEnv(_MultiTurnEnv):
        is_tool_env = True

        def tools(self):
            raise AssertionError("tools() must not be called for a non-grpo run")

    _patch_loader(monkeypatch, _ExplodingToolsEnv())

    assert cmd_env_test(_args(env_dir, algorithm="sft")) == 0


def test_a_tool_environment_exposing_no_tools_still_allows_per_turn_credit(
    monkeypatch, tmp_path, capsys
):
    # the other control, and the reason this checks tools() rather than the is_tool_env flag alone:
    # a tool env with no tools degrades to the rollout_func path, which supports per-turn credit.
    env_dir = _environment_dir(tmp_path)

    class _ToollessToolEnv(_MultiTurnEnv):
        is_tool_env = True

        def tools(self):
            return []

    _patch_loader(monkeypatch, _ToollessToolEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo", credit_assignment="per_turn")) == 0
    assert "is not supported for tool-calling" not in capsys.readouterr().err


class _SingleTurnNativeToolEnv(_SingleTurnEnv):
    """A single-turn tool env exposing tools: native-scored in training, like a multi-turn one.

    `reward()` is a RANKING placeholder and `reward_from_messages` is the real, flat grader, so
    only reading the hook the run actually uses can fail this env.
    """

    is_tool_env = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graded: list[list[dict]] = []

    def tools(self):
        return [lambda x: x]

    def reward(self, completion, example, state=None):  # ranking placeholder, never called
        return 1.0 if completion == example.get("output", "") else 0.0

    def reward_from_messages(self, completion, example):
        self.graded.append([dict(m) for m in completion])
        return 0.0


def test_a_single_turn_native_tool_environment_is_graded_by_reward_from_messages(
    monkeypatch, tmp_path, capsys
):
    # rl.py:433 joins the two with `or` -- `is_message_completion and (is_multi_turn or
    # is_tool_env)` -- so a single-turn tool env exposing tools is scored as an episode by
    # reward_from_messages exactly as a multi-turn one is (tests/test_grpo_params.py:820).
    # grading it through reward() let this ranking placeholder report PASS for an env whose real
    # grader is flat (codex[bot]).
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnNativeToolEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert env.graded, "the native grader was never called"
    assert "cannot rank completions" in captured.err
    assert "reward=0.000000" in captured.out


def test_a_single_turn_native_tool_grader_is_handed_the_gold_trajectory(monkeypatch, tmp_path):
    # what trl's tool loop hands the grader is the generated messages, so the gold trajectory is
    # forwarded whole rather than flattened to its last assistant text.
    env_dir = _environment_dir(tmp_path)
    rows = [{"input": "look it up", "output": "4"}]
    env = _SingleTurnNativeToolEnv(rows=rows)
    trajectory = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"name": "lookup", "arguments": {"q": "x"}}],
        },
        {"role": "tool", "name": "lookup", "content": "4"},
        {"role": "assistant", "content": "4"},
    ]
    env.sft_completion = lambda example: [dict(m) for m in trajectory]
    _patch_loader(monkeypatch, env)

    cmd_env_test(_args(env_dir, algorithm="grpo"))

    assert env.graded, "the native grader was never called"
    assert env.graded[0] == trajectory, "the trajectory was flattened before it reached the grader"
    # the controls keep that trajectory and are wrong only in the final answer, so what they test
    # is whether the grader can RANK rather than whether it tolerates a bare one-message envelope.
    assert all(len(seen) == len(trajectory) for seen in env.graded[1:])
    assert [seen[-1]["content"] for seen in env.graded[1:]] != ["4"] * (len(env.graded) - 1)


def test_a_single_turn_non_tool_environment_is_still_graded_by_reward(monkeypatch, tmp_path):
    # the negative control on the dispatch: an ordinary single-turn env has no tool loop, so it
    # must keep reaching reward()/scores_breakdown. an over-eager native route breaks every
    # ordinary env, which has no reward_from_messages at all.
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert env.replayed == ["4"], "the ordinary single-turn grader was not called"


def test_the_scorer_dispatch_reuses_the_validated_tools_snapshot(monkeypatch, tmp_path, capsys):
    # production calls the hook once (rl.py:816). a one-shot tools() -- tools on the first call,
    # empty after -- was native-scored in training but rollout-scored here, so an unused ranking
    # rollout scorer produced PASS for an env whose real reward_from_messages is flat (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _OneShotToolsEnv(_MultiTurnEnv):
        is_tool_env = True

        def __init__(self):
            super().__init__()
            self.tools_calls = 0
            self.native_graded = 0

        def tools(self):
            self.tools_calls += 1
            return [lambda x: x] if self.tools_calls == 1 else []

        def reward_from_messages(self, completion, example):
            self.native_graded += 1
            return 0.0

    env = _OneShotToolsEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    assert env.tools_calls == 1, (
        f"tools() was called {env.tools_calls} times, production calls it once"
    )
    assert env.native_graded, "the second snapshot rerouted scoring away from the native grader"
    assert "cannot rank completions" in capsys.readouterr().err


def test_a_native_tool_environment_never_reaches_the_rollout_hooks(monkeypatch, tmp_path, capsys):
    # `use_rollout_func = is_multi_turn and not (is_tool_env and tools)` (rl.py:820): a multi-turn
    # tool env exposing tools gets use_rollout_func=False and trl drives the tool loop, so
    # new_rollout_state/record_model_turn/rollout_done/env_reply are never called for it. requiring
    # them here failed a valid env whose unused rollout hooks simply raise (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _NoRolloutHooksEnv(_MultiTurnEnv):
        is_tool_env = True

        def __init__(self):
            super().__init__()
            self.graded: list[list[dict]] = []

        def tools(self):
            return [lambda x: x]

        def new_rollout_state(self, example):
            raise AssertionError("the rollout hooks are unused for a native tool env")

        def reward_from_messages(self, completion, example):
            self.graded.append([dict(m) for m in completion])
            # a working native grader: the gold trajectory outranks any wrong answer.
            said = [m.get("content") for m in completion if m.get("role") == "assistant"]
            return 1.0 if said == ["first", "second"] else 0.0

    env = _NoRolloutHooksEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    assert env.graded, "the native grader was never called"
    assert "overall: PASS" in capsys.readouterr().out


def test_the_env_reply_transcript_is_built_rather_than_read_from_state(monkeypatch, tmp_path):
    # production builds its own message list from `state.get("prompt") or state.get("messages")`
    # and appends each turn and reply to it (flash/engine/multiturn_rollout.py:175-210). reading
    # `state["messages"]` instead raised KeyError for an env keeping only `prompt`, and handed a
    # promptless transcript to one using `messages` for completion-only state (codex[bot]).
    env_dir = _environment_dir(tmp_path)

    class _PromptOnlyStateEnv(_MultiTurnEnv):
        def __init__(self):
            super().__init__()
            self.seen: list[list[dict]] = []

        def new_rollout_state(self, example):
            # no `messages` key at all: the opening lives under `prompt` and the env tracks its
            # own turns. reading state["messages"] raises KeyError here.
            return {
                "prompt": [{"role": "user", "content": example["input"]}],
                "done": False,
                "turn": 0,
                "said": [],
            }

        def record_model_turn(self, state, content):
            state["said"].append(content)
            state["response_text"] = content
            return {"role": "assistant", "content": content}

        def env_reply(self, messages, state):
            self.seen.append([dict(m) for m in messages])
            state["turn"] += 1
            state["done"] = state["turn"] >= 2
            return [{"role": "user", "content": "continue"}]

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            return [
                RolloutReward(
                    episode=1.0 if state["said"] == ["first", "second"] else 0.0, turns=None
                )
                for _example, state in items
            ]

    env = _PromptOnlyStateEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0
    assert env.seen, "env_reply was never called"
    # the transcript opens with the prompt and grows by the assistant turn, exactly as production
    # assembles it -- not the completion-only list the state happens to keep.
    assert env.seen[0] == [
        {"role": "user", "content": "finish the exchange"},
        {"role": "assistant", "content": "first"},
    ], env.seen[0]


def test_a_listwise_grader_is_not_failed_on_a_group_smaller_than_the_run_uses(
    monkeypatch, tmp_path, capsys
):
    # production groups `group_size` completions per prompt (num_generations = group_size, default
    # 8: flash/engine/recipe.py:133, flash/engine/worker/rl.py:199,567). the group here is the gold
    # replay plus the two or three controls provably wrong for it, so a scorer reading its group
    # can answer differently for that reason alone (codex[bot]). the tie is ABOVE zero: that is the
    # band this note speaks for. a tie AT zero is the grader scoring its own reference answer
    # nothing, which no group shape explains, and the test below holds it failing.
    env_dir = _environment_dir(tmp_path)

    class _GroupSizeReadingEnv(_MultiTurnEnv):
        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            # ranks fine at the run's group size and ties at anything smaller.
            episode = 1.0 if len(items) >= 8 else 0.5
            return [RolloutReward(episode=episode, turns=None) for _ in items]

    _patch_loader(monkeypatch, _GroupSizeReadingEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 0, capsys.readouterr().err
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "scores rollouts listwise" in captured.err, captured.err
    assert "group_size (default 8)" in captured.err, captured.err


def test_a_listwise_grader_tied_at_zero_is_still_failed(monkeypatch, tmp_path, capsys):
    # the negative control on that note: scoring listwise explains a tie ABOVE zero, never a tie AT
    # it. a grader that scores its own gold replay nothing has failed at a ranking no group shape
    # asks it to make, so the note is attached and the failure stands.
    env_dir = _environment_dir(tmp_path)

    class _FlatListwiseEnv(_MultiTurnEnv):
        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            return [RolloutReward(episode=0.0, turns=None) for _ in items]

    _patch_loader(monkeypatch, _FlatListwiseEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "scores rollouts listwise" in captured.err, captured.err
    assert "overall: FAIL" in captured.err, captured.err


def test_the_thinking_ambiguity_warning_is_not_raised_for_a_native_tool_run(
    monkeypatch, tmp_path, capsys
):
    # reward_from_messages is handed the messages whole and never strips reasoning, so no template
    # reading enters a native score and the warning described a reading that path does not perform
    # (cursor).
    env_dir = _environment_dir(tmp_path)
    rows = [{"input": "think it through", "output": "reasoning about it\n4"}]
    env = _SingleTurnNativeToolEnv(rows=rows)
    env.reward_from_messages = lambda completion, example: (
        1.0 if completion[-1].get("content") == "reasoning about it\n4" else 0.0
    )
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, algorithm="grpo", thinking=True)) == 0
    assert "no closing </think>" not in capsys.readouterr().err
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
    # to fall through the bare-word test as the literal STRING "NaN" -- never reaching the
    # JSON check that turns the lowercase spelling away. the gate then validated a str where the
    # config holds a float, or an env coercing it back got the very value that check exists to
    # reject (codex[bot]).
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
    # [environment.params] cannot parse at all (codex[bot]).
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
    # could submit the run that was validated (codex[bot]).
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
    # flat {"release.channel": 3}. classifying dots and quotes as structure outright left that
    # valid config with no --param spelling at all, so the command could not mirror it (codex[bot]).
    # quoting is also exactly the text the config needs, so the flag and the config stay in step.
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
    # that config with no --param spelling able to validate it (codex[bot]).
    env_dir = _environment_dir(tmp_path)
    seen = _patch_loader(monkeypatch, _SingleTurnEnv())

    assert cmd_env_test(_args(env_dir, param=[value])) == 0
    assert seen["kwargs"][name] == expected


def test_a_null_spelling_is_not_forwarded_as_text(monkeypatch, tmp_path, capsys):
    # TOML has no null, so these bare words fail the parse, carry no structural character, and
    # forward as their own literal string -- truthy, where the spelling asked for absent. no
    # [environment.params] entry could produce that value either (codex[bot]).
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
def test_env_test_param_keys_a_quoted_config_key_can_carry_still_load(
    monkeypatch, tmp_path, value
):
    # these are not BARE keys, but that is not the question -- a QUOTED key carries every one of
    # them and the schema loader reads it, so `"bad key" = 1` and `"café" = 1` are configs a run
    # really can receive. rejecting them blocked validating a working config while the error
    # claimed [environment.params] could not hold the name (cursor).
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
    # carrying no structural character does not make a token prose. these hold none, so they
    # forwarded as literal strings ("2026-13-01", "1e", ...) while the equivalent
    # [environment.params] entry fails to load -- the gate approving a config that cannot be
    # written. the tell is a leading digit or sign: every TOML scalar except the bare
    # true/false/inf/nan words starts with one, so a token that starts that way and does not parse
    # is a malformed number or date rather than text (codex[bot]).
    #
    # a leading `.` is the same tell: TOML requires a digit before the point, so `.5` is exactly as
    # malformed as the `+.5` the signs already caught -- accepting one and rejecting the other left
    # the same number admitted or refused on whether it carried a sign (cursor).
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
    # quoting has to be spelled so the quotes reach argv (cursor).
    key = value.partition("=")[0]
    assert f"--param '{key}=" in err, err


@pytest.mark.parametrize("value", ["strict=False", "strict=True", "strict=TRUE", "flag=fAlse"])
def test_env_test_a_python_spelled_boolean_is_rejected_rather_than_sent_as_text(
    monkeypatch, tmp_path, capsys, value
):
    # the booleans are the TOML scalars that do not start with a digit or sign, so the
    # leading-character test cannot see them. TOML spells them lowercase only, which makes a
    # python-style `strict=False` parse-fail and fall through as the STRING "False" -- and a
    # non-empty string is truthy, so an env branching on `if strict` reads it as ENABLED while the
    # config spelling `false` disables it. the gate would pass on the opposite of what the run
    # trains with (codex[bot]).
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
