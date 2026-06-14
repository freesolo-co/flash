"""Datums-parity GRPO knobs + init-from-adapter wiring (CPU-only, no GPU/network).

The SDK ships the GRPO recipe knobs (group_size/temperature/advantage_clip/
kl_penalty_coef/thinking_length_penalty_coef) plus the optimizer/batching knobs
(learning_rate/batch_size/max_length/save_every) in the job spec's ``[train]``
(TrainSpec) and an optional ``train.init_from_adapter``; these tests cover the pure
plumbing the worker uses to honor them (the GPU trainer wiring itself is exercised by
the live smokes).
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
        "advantage_clip": 1.5,
        "kl_penalty_coef": 0.02,
        "thinking_length_penalty_coef": 0.001,
    }
    # The SDK ships the GRPO recipe knobs in [train] (TrainSpec fields), not env params.
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
    # only the knobs actually set are returned (max_tokens omitted here)
    assert "max_tokens" not in w.grpo_overrides()
    # no knobs -> empty (recipe defaults apply downstream)
    monkeypatch.setattr(
        w,
        "JOB_SPEC",
        JobSpec.from_dict({"model": "Qwen/Qwen3-0.6B", "algorithm": "grpo"}),
    )
    assert w.grpo_overrides() == {}
    monkeypatch.setattr(w, "JOB_SPEC", None)
    assert w.grpo_overrides() == {}


def test_init_from_adapter_parses_and_roundtrips() -> None:
    raw = {
        "model": "Qwen/Qwen3-0.6B",
        "algorithm": "grpo",
        "model_policy": "allow",
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
    ]
    for bad in bad_cases:
        with pytest.raises(ConfigError):
            spec_from_dict({**base, "train": {"seeds": [0], **bad}}, run_id="bad")
