"""Config linter advice (CPU-only, fully offline).

``lint_spec`` is the soft advisory layer over ``schema``'s hard ConfigError validation: it
flags configs that PARSE but encode a common mistake (tiny rank on a big model, a
two-completion GRPO group, a near-zero rollout temperature, a thinking run with a short
budget, a too-hot learning rate, an eval/checkpoint cadence past the run length). These tests
feed it intentionally-bad setups and a couple of clean ones, asserting the right advice fires
and — critically — that a healthy/default config produces NO advice.
"""

from __future__ import annotations

import json

from flash.catalog import ModelInfo, resolve_model
from flash.lint import Advice, lint_spec
from flash.schema import spec_from_dict

BASE_RAW = {
    "model": "Qwen/Qwen3.5-0.8B",
    "algorithm": "grpo",
    "environment": {"id": "primeintellect/gsm8k"},
    "train": {"steps": 10, "lora_rank": 8, "seeds": [0], "hf_repo": "owner/runs"},
    "gpu": {"type": "RTX 4090"},
}


def _raw(**overrides) -> dict:
    raw = json.loads(json.dumps(BASE_RAW))
    for key, value in overrides.items():
        section, _, leaf = key.partition(".")
        if leaf:
            raw.setdefault(section, {})[leaf] = value
        else:
            raw[section] = value
    return raw


def _advise(**overrides) -> list[Advice]:
    """Parse a config (catalog model -> offline resolve) and return its advice."""
    spec = spec_from_dict(_raw(**overrides))
    info = resolve_model(spec.model, spec.algorithm, policy=spec.model_policy, gpu=spec.gpu.type)
    return lint_spec(spec, info)


def _fields(advice: list[Advice]) -> set[str]:
    return {a.field for a in advice}


# ---------------------------------------------------------------------------
# the no-advice guarantee: a healthy config must stay quiet
# ---------------------------------------------------------------------------


def test_clean_default_config_yields_no_advice() -> None:
    # 0.8B model + lora_rank=8 (small model, so rank is fine), every GRPO knob unset (recipe
    # defaults apply downstream). The linter must say nothing.
    assert _advise() == []


def test_explicit_recipe_defaults_yield_no_advice() -> None:
    # Setting every knob EXPLICITLY to its recipe default must also stay quiet (off-by-one
    # guard: the thresholds must not fire on the defaults themselves). Use the 4B catalog model
    # so rank 32 is unambiguously fine.
    advice = _advise(
        model="Qwen/Qwen3.5-4B",
        **{
            "gpu.type": "RTX 5090",
            "train.lora_rank": 32,
            "train.lora_alpha": 64,
            "train.group_size": 8,
            "train.temperature": 1.0,
            "train.learning_rate": 1e-5,
            "train.max_tokens": 320,
            "train.save_every": 5,
            "train.eval_every_steps": 5,
        },
    )
    assert advice == []


# ---------------------------------------------------------------------------
# individual checks fire on the matching mistake
# ---------------------------------------------------------------------------


def test_thinking_short_completion_budget_grpo() -> None:
    # 4B is thinking-"hybrid"; thinking=true + a 320-token budget truncates reasoning.
    advice = _advise(model="Qwen/Qwen3.5-4B", thinking=True, **{"gpu.type": "RTX 5090", "train.max_tokens": 320})
    assert "train.max_tokens" in _fields(advice)
    # a generous thinking budget does NOT warn
    ok = _advise(model="Qwen/Qwen3.5-4B", thinking=True, **{"gpu.type": "RTX 5090", "train.max_tokens": 2048})
    assert "train.max_tokens" not in _fields(ok)


def test_thinking_short_completion_only_when_thinking() -> None:
    # the same short budget with thinking OFF is fine (non-thinking default is 320)
    advice = _advise(model="Qwen/Qwen3.5-4B", **{"gpu.type": "RTX 5090", "train.max_tokens": 320})
    assert "train.max_tokens" not in _fields(advice)


def test_low_lora_rank_for_large_model() -> None:
    # 9.7B model + rank 8 -> underfit risk.
    advice = _advise(model="Qwen/Qwen3.5-9B", **{"gpu.type": "RTX 5090", "train.lora_rank": 8})
    assert "train.lora_rank" in _fields(advice)
    # a healthy rank on the same big model is quiet
    ok = _advise(model="Qwen/Qwen3.5-9B", **{"gpu.type": "RTX 5090", "train.lora_rank": 32})
    assert "train.lora_rank" not in _fields(ok)
    # the SAME low rank on a small model is fine (capacity is plenty)
    small = _advise(**{"train.lora_rank": 8})  # 0.8B base
    assert "train.lora_rank" not in _fields(small)


def test_small_group_size_for_grpo() -> None:
    advice = _advise(**{"train.group_size": 2})
    assert "train.group_size" in _fields(advice)
    assert "train.group_size" not in _fields(_advise(**{"train.group_size": 8}))


def test_group_size_not_warned_under_sft() -> None:
    # group_size is a GRPO-only knob; the SFT worker ignores it, so SFT must not warn.
    advice = _advise(algorithm="sft", **{"train.group_size": 2, "train.epochs": 2})
    assert "train.group_size" not in _fields(advice)


def test_low_grpo_temperature() -> None:
    advice = _advise(**{"train.temperature": 0})
    assert "train.temperature" in _fields(advice)
    assert "train.temperature" not in _fields(_advise(**{"train.temperature": 1.0}))


def test_learning_rate_high_for_grpo_vs_very_high() -> None:
    grpo_hot = _advise(**{"train.learning_rate": 1e-4})
    lr_advice = [a for a in grpo_hot if a.field == "train.learning_rate"]
    assert len(lr_advice) == 1  # exactly one message, no double-fire
    assert "high for GRPO" in lr_advice[0].message

    very_hot = _advise(**{"train.learning_rate": 2e-3})
    lr_advice = [a for a in very_hot if a.field == "train.learning_rate"]
    assert len(lr_advice) == 1
    assert "very high" in lr_advice[0].message


def test_sft_default_learning_rate_is_quiet() -> None:
    # SFT's recipe LR is 1e-4; an SFT run at 1e-4 is the default and must NOT warn (the GRPO
    # "high" rule is algorithm-gated, and 1e-4 is below the universal 1e-3 threshold).
    advice = _advise(algorithm="sft", **{"train.epochs": 2, "train.learning_rate": 1e-4})
    assert "train.learning_rate" not in _fields(advice)


def test_eval_and_save_cadence_past_run_length() -> None:
    advice = _advise(**{"train.steps": 10, "train.eval_every_steps": 999, "train.save_every": 999})
    assert {"train.eval_every_steps", "train.save_every"} <= _fields(advice)
    # in-range cadences are quiet
    ok = _advise(**{"train.steps": 100, "train.eval_every_steps": 20, "train.save_every": 25})
    assert not ({"train.eval_every_steps", "train.save_every"} & _fields(ok))


def test_kitchen_sink_fires_multiple() -> None:
    # The motivating example: every common mistake at once should produce several warnings.
    advice = _advise(
        model="Qwen/Qwen3.5-9B",
        thinking=True,
        **{
            "gpu.type": "RTX 5090",
            "train.steps": 50,
            "train.lora_rank": 8,
            "train.group_size": 2,
            "train.temperature": 0,
            "train.max_tokens": 320,
            "train.learning_rate": 1e-4,
            "train.eval_every_steps": 200,
        },
    )
    assert {
        "train.lora_rank",
        "train.group_size",
        "train.temperature",
        "train.max_tokens",
        "train.learning_rate",
        "train.eval_every_steps",
    } <= _fields(advice)


# ---------------------------------------------------------------------------
# open-model / unknown size: size-based checks degrade gracefully
# ---------------------------------------------------------------------------


def test_unknown_model_size_skips_rank_check() -> None:
    # The open-model policy can yield params="unknown size"; the rank-vs-size check must simply
    # not fire (no crash, no false positive). Build the spec + a synthetic unknown-size info
    # directly so the test stays fully offline.
    spec = spec_from_dict(_raw(**{"train.lora_rank": 8}))
    info = ModelInfo(
        id=spec.model,
        display_name=spec.model,
        params="unknown size",
        algos=("sft", "grpo"),
        min_vram_gb=24,
        thinking="unknown",
    )
    assert "train.lora_rank" not in _fields(lint_spec(spec, info))


def test_advice_to_dict_shape() -> None:
    advice = _advise(**{"train.group_size": 2})
    d = advice[0].to_dict()
    assert set(d) == {"field", "message", "severity"}
    assert d["severity"] == "warning"
