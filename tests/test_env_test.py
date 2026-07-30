"""Tests for local offline environment contract validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import flash.cli as cli
from flash.cli.env_test import cmd_env_test


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


def test_env_test_replay_low_reward_warns_but_passes(monkeypatch, tmp_path, capsys):
    env_dir = _environment_dir(tmp_path)
    _patch_loader(monkeypatch, _SingleTurnEnv(reward=0.0))

    assert cmd_env_test(_args(env_dir)) == 0
    captured = capsys.readouterr()
    assert "episode 1: policy=replay turns=1 reward=0.000000" in captured.out
    assert "1/1 episodes passed contract checks" in captured.out
    assert "overall: PASS" in captured.out
    assert "warning:" in captured.err
    assert "check the reward function" in captured.err


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


@pytest.mark.parametrize("value", ["difficulty.level=3", '"weird key"=1', "a.b.c=1"])
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
