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
    _Score,
    _synthetic_control,
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


def test_env_test_non_finite_control_reward_fails_the_episode(monkeypatch, tmp_path, capsys):
    # a scorer that returns NaN for an unexpected completion breaks the same reward contract the
    # gold answer is already failed for: the policy reaches this scorer with completions no more
    # expected than the controls, and non-finite rewards there yield unusable samples. excluding
    # the episode as merely inconclusive would let `overall: PASS` hide that.
    env_dir = _environment_dir(tmp_path)

    class _NanControlEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            if completion != example.get("output", ""):
                return float("nan")
            return 1.0

    _patch_loader(monkeypatch, _NanControlEnv())

    assert cmd_env_test(_args(env_dir)) == 1
    captured = capsys.readouterr()
    assert "0/1 episodes passed contract checks" in captured.out
    assert "reward is not finite for a non-reference completion" in captured.err
    assert "overall: FAIL" in captured.err


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


def test_env_test_inverted_reward_direction_fails(monkeypatch, tmp_path, capsys):
    # a grader whose sign is inverted scores wrong answers ABOVE gold. grpo maximizes the reward,
    # so the run would train directly away from the references. differing scores are not enough:
    # the gold answer must not rank below a deliberately wrong one.
    env_dir = _environment_dir(tmp_path)

    class _InvertedEnv(_SingleTurnEnv):
        def reward(self, completion, example, state=None):
            self.completions.append(completion)
            return 0.0 if completion == example.get("output", "") else 1.0

    _patch_loader(monkeypatch, _InvertedEnv())

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert "the reward direction is inverted" in captured.err
    assert "overall: FAIL" in captured.err


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

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "1/1 episodes passed contract checks" in captured.out
    assert "cannot rank completions" in captured.err
    assert "overall: FAIL" in captured.err


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

    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "cannot rank completions" in captured.err
    assert "overall: FAIL" in captured.err


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


def test_per_turn_vectors_with_equal_sums_are_not_reported_as_flat():
    # build_per_turn_advantages centres each turn index against its own group mean, so (1, 0)
    # against (0, 1) produces nonzero opposing advantages at BOTH turns. reducing the vectors to
    # one scalar made the sums tie and reported usable per-turn credit as a grader that cannot rank.
    gold = _Score(episode=1.0, turns=(1.0, 0.0))
    control = _Score(episode=1.0, turns=(0.0, 1.0))

    assert gold.separates_from(control, per_turn=True)
    assert control.separates_from(gold, per_turn=True)
    # neither DOMINATES: worse at one turn and better at another is not "worse" in any direction
    # the trainer reads, so the inverted finding must not fire either.
    assert not gold.ranks_below(control, per_turn=True)
    assert not control.ranks_below(gold, per_turn=True)


def test_per_turn_vector_that_dominates_still_ranks():
    # the ranking direction still has to work: no better anywhere and worse somewhere IS worse.
    gold = _Score(episode=1.0, turns=(1.0, 1.0))
    control = _Score(episode=1.0, turns=(0.0, 1.0))

    assert control.ranks_below(gold, per_turn=True)
    assert gold.outranks(control, per_turn=True)
    assert control.separates_from(gold, per_turn=True)


def test_a_larger_sum_is_not_a_ranking():
    # the companion to the equal-sum case, and the one a sum rule gets WRONG rather than ties on:
    # the control's sum is the larger, but it is worse at turn 0 and better at turn 1, so the group
    # centring gives it advantages in both directions and it outranks the gold answer at neither.
    # reading the sums here reports a working grader as inverted.
    gold = _Score(episode=1.0, turns=(1.0, 0.0))
    control = _Score(episode=1.0, turns=(0.0, 5.0))

    assert sum(control.turns or ()) > sum(gold.turns or ())
    assert not gold.ranks_below(control, per_turn=True)
    assert not control.ranks_below(gold, per_turn=True)
    assert gold.separates_from(control, per_turn=True)


def test_identical_per_turn_vectors_are_still_flat():
    # the gate must keep firing on a genuinely unrankable grader.
    gold = _Score(episode=1.0, turns=(1.0, 0.0))
    control = _Score(episode=1.0, turns=(1.0, 0.0))

    assert not gold.separates_from(control, per_turn=True)
    assert not gold.ranks_below(control, per_turn=True)
    assert not control.ranks_below(gold, per_turn=True)


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


def test_a_shorter_control_that_is_worse_over_the_overlap_still_ranks_below():
    # the ranking direction reads the same overlap, so an early-terminating wrong answer that is
    # worse everywhere it reached is worse -- not incomparable.
    gold = _Score(episode=1.0, turns=(1.0, 1.0))
    control = _Score(episode=1.0, turns=(0.0,))

    assert control.ranks_below(gold, per_turn=True)
    assert gold.outranks(control, per_turn=True)


def test_differing_episode_scores_rank_whatever_the_turn_shapes():
    # the episode scalar alone settles it, so turn shape is irrelevant -- and it settles it in
    # either credit mode, since the scalar is what per_episode training reads.
    gold = _Score(episode=1.0, turns=(1.0, 0.0))
    control = _Score(episode=0.0, turns=None)

    assert control.ranks_below(gold, per_turn=True)
    assert control.ranks_below(gold, per_turn=False)
    assert control.separates_from(gold, per_turn=False)


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
    assert not control.ranks_below(gold, per_turn=False)
    assert not gold.ranks_below(control, per_turn=False)


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
    # ...so without a fallback the episode carried no control at all and the constant grader passed.
    assert cmd_env_test(_args(env_dir, algorithm="grpo")) == 1
    captured = capsys.readouterr()
    assert "cannot rank completions" in captured.err


def test_synthetic_control_is_disjoint_from_the_gold_answer():
    control = _synthetic_control(["answer z 0"])

    assert control is not None
    assert _control_is_disjoint(control, "answer z 0")


def test_synthetic_control_is_none_when_the_gold_text_uses_every_character():
    # no character is left to build a provably-wrong control from, so the episode must still be
    # excluded rather than controlled against text that might be correct.
    exhaustive = _SYNTHETIC_CONTROL_ALPHABET

    assert _synthetic_control([exhaustive]) is None


class _NonFiniteControlScorerEnv(_SingleTurnEnv):
    """A grader that scores its own reference finitely and anything else NaN.

    `_score_control` maps a raised exception to 0.0 for single-turn (mirroring reward_fn), so a
    non-finite score is the shape that actually reaches the caller. Legitimate for sft/opd/opsd,
    whose workers never call the scorer at all; a real defect for grpo, whose reward_fn grades
    arbitrary policy completions.
    """

    def reward(self, completion, example, state=None):
        self.completions.append(completion)
        if completion != example.get("output", ""):
            return float("nan")
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


def test_env_test_equal_sum_per_turn_vectors_are_not_failed_as_flat(monkeypatch, tmp_path, capsys):
    # end-to-end companion to the _Score unit tests: a grader whose gold and control vectors have
    # the same sum but differ positionally ((1, 0) against (0, 1)) does rank under the per-turn
    # trainer, since each turn index is centred against its own group mean. reducing the vectors to
    # one scalar made the sums tie and failed a working environment.
    env_dir = _environment_dir(tmp_path)

    class _EqualSumPerTurnEnv(_MultiTurnEnv):
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

    _patch_loader(monkeypatch, _EqualSumPerTurnEnv())

    args = _args(env_dir, algorithm="grpo", credit_assignment="per_turn")
    assert cmd_env_test(args) == 0
    captured = capsys.readouterr()
    assert "overall: PASS" in captured.out
    assert "cannot rank completions" not in captured.err
    # and not misread as inverted either: neither vector dominates the other.
    assert "reward direction" not in captured.err
