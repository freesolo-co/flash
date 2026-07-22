"""Tests for local offline environment contract validation."""

from __future__ import annotations

import argparse

import flash.cli as cli
from flash.cli.env_test import cmd_env_test


class _SingleTurnEnv:
    multi_turn = False
    max_turns = 8

    def __init__(self, *, rows=None, reward=1.0):
        self.rows = (
            [{"input": "what is 2 + 2?", "output": "4"}] if rows is None else rows
        )
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


def _args(path, *, episodes=3):
    return argparse.Namespace(path=str(path), episodes=episodes)


def test_env_test_single_turn_replays_reference_and_passes(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv()
    seen = _patch_loader(monkeypatch, env)
    args = cli._build_parser().parse_args(
        ["env", "test", str(env_dir), "--episodes", "1"]
    )

    assert args.func is cmd_env_test
    assert args.func(args) == 0
    assert seen["reference"] == str(env_dir / "environment.py")
    assert env.completions == ["4"]
    out = capsys.readouterr().out
    assert "episode 1: policy=replay turns=1 reward=1.000000" in out
    assert "1/1 episodes passed contract checks" in out
    assert "overall: PASS" in out


def test_env_test_auto_falls_back_to_echo_for_empty_reference(
    monkeypatch, tmp_path, capsys
):
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv(rows=[{"input": "say anything", "output": ""}], reward=0.0)
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, episodes=1)) == 0
    assert env.completions == ["test"]
    captured = capsys.readouterr()
    assert "episode 1: policy=echo turns=1 reward=0.000000" in captured.out
    assert "1/1 episodes passed contract checks" in captured.out
    assert "WARNING" not in captured.err


def test_env_test_multi_turn_terminates_and_scores(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    env = _MultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, episodes=1)) == 0
    assert env.scored_state is not None
    assert env.scored_state["turn"] == 2
    out = capsys.readouterr().out
    assert "episode 1: policy=replay turns=2 reward=0.500000" in out
    assert "1/1 episodes passed contract checks" in out


def test_env_test_multi_turn_honors_per_example_cap_past_hard_cap(
    monkeypatch, tmp_path, capsys
):
    env_dir = _environment_dir(tmp_path)
    env = _PerExampleCapMultiTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, episodes=1)) == 0
    assert env.scored_state is not None
    assert env.scored_state["turn"] == 12
    out = capsys.readouterr().out
    assert "episode 1: policy=replay turns=12 reward=0.500000" in out
    assert "overall: PASS" in out


def test_env_test_zero_episodes_runs_one(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    env = _SingleTurnEnv()
    _patch_loader(monkeypatch, env)

    assert cmd_env_test(_args(env_dir, episodes=0)) == 0
    assert env.completions == ["4"]
    out = capsys.readouterr().out
    assert "1/1 episodes passed contract checks" in out
    assert "overall: PASS" in out


def test_env_test_nan_reward_fails_contract(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=float("nan")))

    assert cmd_env_test(_args(env_dir, episodes=1)) == 1
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
