"""Datums-parity GRPO knobs + init-from-adapter wiring (CPU-only, no GPU/network).

The SDK ships the GRPO recipe knobs (group_size/temperature/advantage_clip/
kl_penalty_coef/thinking_length_penalty_coef) in the job spec's environment params and
an optional ``train.init_from_adapter``; these tests cover the pure plumbing the worker
uses to honor them (the GPU trainer wiring itself is exercised by the live smokes).
"""

from __future__ import annotations

import os
import sys

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


def test_grpo_overrides_reads_env_params(monkeypatch) -> None:
    import autoslm.engine.worker as w

    knobs = {
        "group_size": 4,
        "temperature": 0.7,
        "advantage_clip": 1.5,
        "kl_penalty_coef": 0.02,
        "thinking_length_penalty_coef": 0.001,
    }
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3-0.6B",
            "algorithm": "grpo",
            "environment": {"id": "freesolo", "params": {"grpo_config": knobs}},
            "train": {"seeds": [0]},
        }
    )
    monkeypatch.setattr(w, "JOB_SPEC", spec)
    assert w.grpo_overrides() == knobs
    # no grpo_config -> empty (recipe defaults apply downstream)
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
