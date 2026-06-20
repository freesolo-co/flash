"""`slm train --estimate`: map a training config to a pre-flight cost estimate. No network."""

from __future__ import annotations

import copy
import types

from flash.cli.main.commands import _runconfig_from_spec, _spec_steps, cmd_train
from flash.engine.recipe import RECIPE
from flash.schema import spec_from_dict

GRPO_RAW = {
    "model": "Qwen/Qwen3.5-9B",
    "algorithm": "grpo",
    "environment": {"id": "primeintellect/gsm8k"},
    "train": {
        "steps": 50,
        "group_size": 8,
        "batch_size": 16,
        "max_tokens": 512,
        "max_length": 2048,
        "seeds": [0],
        "hf_repo": "owner/runs",
    },
    "gpu": {"type": "RTX 5090"},
}


def _spec(**overrides):
    raw = copy.deepcopy(GRPO_RAW)
    for key, value in overrides.items():
        section, _, leaf = key.partition(".")
        if leaf:
            raw.setdefault(section, {})[leaf] = value
        else:
            raw[section] = value
    return spec_from_dict(raw)


def test_runconfig_from_grpo_spec_maps_fields():
    cfg = _runconfig_from_spec(_spec())
    assert cfg.model_id == "Qwen/Qwen3.5-9B"
    assert cfg.method == "grpo"
    assert cfg.steps == 50
    assert cfg.batch_size == 16
    assert cfg.group_size == 8
    assert cfg.completion_len == 512  # GRPO max_tokens
    assert cfg.seq_len == 2048
    assert cfg.gpu == "RTX 5090"
    assert cfg.environment == "primeintellect/gsm8k"


def test_multi_seed_scales_steps_and_setup():
    cfg = _runconfig_from_spec(_spec(**{"train.seeds": [0, 1, 2]}))
    assert cfg.setup_repeats == 3
    assert cfg.steps == 50 * 3  # per-seed steps x seeds (each seed re-pays cold start)


def test_grpo_uses_recipe_steps_when_omitted():
    spec = _spec()
    object.__setattr__(spec.train, "steps", None)
    assert _spec_steps(spec) == RECIPE.rl.num_steps


def test_sft_steps_derived_from_examples():
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "environment": {"id": "acme/sft-data"},
            "train": {
                "max_examples": 320,
                "batch_size": 16,
                "epochs": 2,
                "seeds": [0],
                "hf_repo": "owner/runs",
            },
            "gpu": {"type": "RTX 4090"},
        }
    )
    assert _spec_steps(spec) == 40  # ceil(320 / 16) * 2 epochs
    cfg = _runconfig_from_spec(spec)
    assert cfg.method == "sft"
    assert cfg.group_size is None  # SFT carries no completions-per-prompt
    assert cfg.completion_len is None


def test_cmd_train_estimate_prints_breakdown_without_submitting(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLASH_SKIP_NET", "1")
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "primeintellect/gsm8k"\n'
        "[train]\n"
        "steps = 50\n"
        'hf_repo = "owner/runs"\n'
        "[gpu]\n"
        'type = "RTX 5090"\n'
    )
    args = types.SimpleNamespace(
        config=str(cfg), overrides=[], extra_configs=[], estimate=True, dry_run=False, background=False
    )
    # --estimate is fully local: it must NOT touch the control-plane client.
    rc = cmd_train(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "TOTAL" in out
    assert "$" in out
    assert "RTX 5090" in out
