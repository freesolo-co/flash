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

    def load(reference):
        seen["reference"] = reference
        return env

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", load)
    return seen


def _args(path, **overrides):
    namespace = argparse.Namespace(path=str(path), algorithm=None, credit_assignment=None)
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
            messages.append(reply)
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
            messages.append(reply)
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
            messages.extend(replies)
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
            messages.extend(replies)
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
            messages.append(reply)
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
