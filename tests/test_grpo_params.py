"""Datums-parity GRPO knobs + init-from-adapter wiring (CPU-only, no GPU/network).

The SDK ships the GRPO recipe knobs (group_size/temperature/advantage_clip/
kl_penalty_coef/thinking_length_penalty_coef) plus the optimizer/batching knobs
(learning_rate/batch_size/max_length/save_every) in the job spec's ``[train]`` table
(TrainSpec) — NOT ``[environment.params]``, which is forwarded verbatim to the verifiers
env's ``load_environment`` — and an optional ``train.init_from_adapter``; these tests
cover the pure plumbing the worker uses to honor them (the GPU trainer wiring itself is
exercised by the live smokes).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoslm.config_schema import spec_from_dict
from autoslm.worker_spec import JobSpec


class _Tok:
    """Whitespace tokenizer stub: one token per word."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_think_token_count_counts_the_think_span() -> None:
    import autoslm.engine.worker as w

    tok = _Tok()
    assert w.think_token_count("<think>a b c</think>the answer", tok) == 3
    assert w.think_token_count("no reasoning here", tok) == 0
    # an unclosed block (budget exhausted) counts everything after <think>
    assert w.think_token_count("pre <think>a b c d", tok) == 4
    assert w.think_token_count(None, tok) == 0
    assert w.think_token_count("<think></think>x", tok) == 0


def test_grpo_overrides_reads_train_knobs(monkeypatch) -> None:
    import autoslm.engine.worker as w

    knobs = {
        "group_size": 4,
        "temperature": 0.7,
        "max_tokens": 256,
        "advantage_clip": 1.5,
        "kl_penalty_coef": 0.02,
        "thinking_length_penalty_coef": 0.001,
    }
    # GRPO knobs live in [train]/TrainSpec, NOT [environment.params].
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3-0.6B",
            "algorithm": "grpo",
            "environment": {"id": "owner/env"},
            "train": {"seeds": [0], **knobs},
        }
    )
    monkeypatch.setattr(w, "JOB_SPEC", spec)
    assert w.grpo_overrides() == knobs
    # A leftover grpo_config in environment.params must NOT be read by the worker.
    poisoned = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3-0.6B",
            "algorithm": "grpo",
            "environment": {"id": "owner/env", "params": {"grpo_config": knobs}},
            "train": {"seeds": [0]},
        }
    )
    monkeypatch.setattr(w, "JOB_SPEC", poisoned)
    assert w.grpo_overrides() == {}
    # only the knobs actually set are returned (a partial set omits the rest)
    monkeypatch.setattr(
        w,
        "JOB_SPEC",
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3-0.6B",
                "algorithm": "grpo",
                "environment": {"id": "owner/env"},
                "train": {"seeds": [0], "group_size": 2},
            }
        ),
    )
    assert w.grpo_overrides() == {"group_size": 2}
    # no [train] knobs -> empty (recipe defaults apply downstream)
    monkeypatch.setattr(
        w,
        "JOB_SPEC",
        JobSpec.from_dict({"model": "Qwen/Qwen3-0.6B", "algorithm": "grpo"}),
    )
    assert w.grpo_overrides() == {}
    monkeypatch.setattr(w, "JOB_SPEC", None)
    assert w.grpo_overrides() == {}


def test_train_grpo_knobs_parse_and_roundtrip() -> None:
    from autoslm.config_schema import spec_from_dict

    raw = {
        "model": "Qwen/Qwen3-0.6B",
        "algorithm": "grpo",
        "model_policy": "allow",
        "environment": {"id": "owner/env"},
        "gpu": {"type": "cheapest", "allow_unvalidated": True},
        "train": {
            "seeds": [0],
            "steps": 10,
            "group_size": 4,
            "temperature": 0.7,
            "max_tokens": 256,
            "kl_penalty_coef": 0.02,
            "advantage_clip": 1.5,
            "thinking_length_penalty_coef": 0.001,
        },
    }
    spec = spec_from_dict(raw, run_id="grpo-x")
    assert spec.train.group_size == 4
    assert spec.train.temperature == 0.7
    assert spec.train.max_tokens == 256
    assert spec.train.kl_penalty_coef == 0.02
    assert spec.train.advantage_clip == 1.5
    assert spec.train.thinking_length_penalty_coef == 0.001
    # survives the JSON round-trip the worker reconstructs from
    rt = JobSpec.from_dict(spec.to_dict()).train
    assert rt.group_size == 4
    assert rt.thinking_length_penalty_coef == 0.001
    # GRPO knobs are NOT in environment.params (that goes verbatim to load_environment)
    assert spec.environment.params == {}


def test_opt_int_float_reject_bools() -> None:
    """A JSON boolean must NOT silently coerce to a numeric train knob: bool is an int
    subclass in Python, so ``int(True)`` would become 1. JobSpec.from_dict (via
    _opt_int/_opt_float) rejects it, matching config_schema._opt_num."""
    import pytest

    from autoslm.worker_spec import _opt_float, _opt_int

    for bad in (True, False):
        with pytest.raises(TypeError):
            _opt_int(bad)
        with pytest.raises(TypeError):
            _opt_float(bad)

    # Genuine numbers (and None) still parse.
    assert _opt_int(None) is None
    assert _opt_int(4) == 4
    assert _opt_float(None) is None
    assert _opt_float(0.7) == 0.7

    # A bool train knob propagates through JobSpec.from_dict as an error, not a 0/1 coercion.
    with pytest.raises(TypeError):
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3-0.6B",
                "algorithm": "grpo",
                "environment": {"id": "owner/env"},
                "train": {"steps": 10, "group_size": True},
            }
        )


def test_verifiers_adapter_forwards_only_env_kwargs(monkeypatch) -> None:
    # environment.params is forwarded to vf.load_environment WITHOUT autoslm-reserved
    # keys (a stray grpo_config/mode/records/eval_* must be dropped, not passed through).
    import sys
    import types

    captured = {}

    def fake_load_environment(env_id, **kwargs):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return object()

    fake_vf = types.SimpleNamespace(
        load_environment=fake_load_environment,
        ToolEnv=type("ToolEnv", (), {}),
        MultiTurnEnv=type("MultiTurnEnv", (), {}),
        SingleTurnEnv=type("SingleTurnEnv", (), {}),
        JudgeRubric=type("JudgeRubric", (), {}),
    )
    monkeypatch.setitem(sys.modules, "verifiers", fake_vf)

    from autoslm.envs import registry

    registry.load_environment(
        "owner/env",
        params={
            "difficulty": "hard",  # a real verifiers-env kwarg -> forwarded
            "grpo_config": {"group_size": 4},  # reserved -> dropped
            "mode": "train",  # reserved -> dropped
            "records": [1, 2],  # reserved -> dropped
            "eval_seed": 99,  # adapter-consumed -> not forwarded
        },
    )
    assert captured["env_id"] == "env"
    assert captured["kwargs"] == {"difficulty": "hard"}
    for forbidden in ("grpo_config", "mode", "records", "eval_seed"):
        assert forbidden not in captured["kwargs"]


def test_init_from_adapter_parses_and_roundtrips() -> None:
    raw = {
        "model": "Qwen/Qwen3-0.6B",
        "algorithm": "grpo",
        "model_policy": "allow",
        "environment": {"id": "owner/env"},
        "gpu": {"type": "cheapest", "allow_unvalidated": True},
        "train": {"seeds": [0], "steps": 10, "init_from_adapter": "sft/run-x/seed0"},
    }
    spec = spec_from_dict(raw, run_id="grpo-x")
    assert spec.train.init_from_adapter == "sft/run-x/seed0"
    # survives the JSON round-trip the worker reconstructs from
    assert JobSpec.from_dict(spec.to_dict()).train.init_from_adapter == "sft/run-x/seed0"
    # absent -> empty string (train fresh from base)
    raw["train"].pop("init_from_adapter")
    assert spec_from_dict(raw, run_id="grpo-y").train.init_from_adapter == ""


def test_optimizer_and_batching_knobs_roundtrip() -> None:
    # The SDK's SftConfig/GrpoConfig optimizer/batching knobs must survive config_schema
    # (server validation) AND the worker's JobSpec.from_dict, or the worker would silently
    # train with recipe defaults while W&B reports the user's values.
    raw = {
        "model": "Qwen/Qwen3-0.6B",
        "algorithm": "grpo",
        "model_policy": "allow",
        "environment": {"id": "owner/env"},
        "gpu": {"type": "cheapest", "allow_unvalidated": True},
        "train": {
            "seeds": [0],
            "learning_rate": 3e-5,
            "batch_size": 16,
            "max_length": 2048,
            "save_every": 5,
            "max_tokens": 512,
            "stop_sequences": ["</answer>", "\n\n"],
        },
    }
    spec = spec_from_dict(raw, run_id="grpo-z")
    for s in (spec, JobSpec.from_dict(spec.to_dict())):  # server parse + worker re-parse
        assert s.train.learning_rate == 3e-5
        assert s.train.batch_size == 16
        assert s.train.max_length == 2048
        assert s.train.save_every == 5
        assert s.train.max_tokens == 512
        assert s.train.stop_sequences == ("</answer>", "\n\n")
    # omitted optimizer knobs stay None so the worker applies its recipe defaults
    bare = spec_from_dict({**raw, "train": {"seeds": [0]}}, run_id="grpo-w")
    assert bare.train.learning_rate is None
    assert bare.train.batch_size is None
    assert bare.train.stop_sequences == ()
    # a bare-string stop_sequences is ONE stop, never split into characters
    one = spec_from_dict(
        {**raw, "train": {"seeds": [0], "stop_sequences": "</answer>"}}, run_id="grpo-s"
    )
    assert one.train.stop_sequences == ("</answer>",)
    assert JobSpec.from_dict(one.to_dict()).train.stop_sequences == ("</answer>",)
    # an empty string means "no stop configured" -> (), not ("",); empty list entries drop
    empty = spec_from_dict({**raw, "train": {"seeds": [0], "stop_sequences": ""}}, run_id="grpo-e")
    assert empty.train.stop_sequences == ()
    dropped = spec_from_dict(
        {**raw, "train": {"seeds": [0], "stop_sequences": ["x", ""]}}, run_id="grpo-d"
    )
    assert dropped.train.stop_sequences == ("x",)


def test_optimizer_knob_validation_rejects_bad_values() -> None:
    # config_schema is the server's 400 layer: nonsensical/malformed knobs must raise
    # ConfigError at parse time, not TypeError (500) or a silently-misbehaving worker.
    from autoslm.config_schema import ConfigError

    base = {
        "model": "Qwen/Qwen3-0.6B",
        "algorithm": "grpo",
        "model_policy": "allow",
        "environment": {"id": "owner/env"},
        "gpu": {"type": "cheapest", "allow_unvalidated": True},
    }
    bad_cases = [
        {"batch_size": 0},  # must be >= 1
        {"batch_size": -4},
        {"max_length": 0},
        {"save_every": 0},
        {"group_size": 0},
        {"learning_rate": 0},  # must be > 0
        {"learning_rate": -1e-5},
        {"temperature": -0.1},  # must be >= 0
        {"kl_penalty_coef": -1},
        {"batch_size": 1.5},  # non-integer
        {"batch_size": "16"},  # wrong type (string)
        {"learning_rate": [1]},  # wrong type (list) -> 400, not a 500 TypeError
        {"stop_sequences": {"a": 1}},  # dict not allowed
        {"stop_sequences": [1, 2]},  # non-string entries
        {"learning_rate": float("nan")},  # non-finite -> 400, not a silent NaN to the optimizer
        {"learning_rate": float("inf")},
        {"temperature": float("inf")},
        {"batch_size": float("inf")},  # int knob: must 400, not OverflowError(500) from int(inf)
        {"max_tokens": float("nan")},
    ]
    for bad in bad_cases:
        with pytest.raises(ConfigError):
            spec_from_dict({**base, "train": {"seeds": [0], **bad}}, run_id="bad")
