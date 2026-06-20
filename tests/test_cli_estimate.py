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
    # batch_size 16 is a multiple of the per-device micro-batch (4) so realized == requested:
    # epochs(2) x ceil(320 / 16) = 40.
    assert _spec_steps(spec) == 40
    cfg = _runconfig_from_spec(spec)
    assert cfg.method == "sft"
    assert cfg.group_size is None  # SFT carries no completions-per-prompt
    assert cfg.completion_len is None


def _sft_spec(**train):
    raw = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "sft",
        "environment": {"id": "acme/sft-data"},
        "train": {"seeds": [0], "hf_repo": "owner/runs", **train},
        "gpu": {"type": "RTX 4090"},
    }
    return spec_from_dict(raw)


def _worker_sft_steps(*, examples, requested_batch, epochs, max_steps=0):
    """Independent re-derivation of the worker's per-seed SFT optimizer-step count (engine.worker:
    fixed per-device micro-batch 4 + ceil grad-accum -> realized global batch), to pin _spec_steps
    against what actually runs."""
    from flash.engine.vram import _sft_per_device_bs

    per_device = max(1, min(_sft_per_device_bs(), requested_batch))
    grad_accum = max(1, -(-requested_batch // per_device))
    realized = per_device * grad_accum
    n = max(1, -(-examples // realized) * epochs)
    return min(n, max_steps) if max_steps > 0 else n


def test_sft_steps_default_epochs_mirror_the_worker():
    # No [train].epochs -> the worker uses RECIPE.sft.num_epochs (2), NOT 1. The estimate must too.
    spec = _sft_spec(max_examples=320, batch_size=16)  # epochs omitted
    assert spec.train.epochs is None
    assert RECIPE.sft.num_epochs == 2
    assert _spec_steps(spec) == _worker_sft_steps(examples=320, requested_batch=16, epochs=2) == 40


def test_sft_steps_use_worker_realized_grad_accum_batch():
    # batch_size 6 is NOT a multiple of the micro-batch (4): the worker realizes per_device(4) x
    # grad_accum(ceil(6/4)=2) = 8, so steps = epochs(2) x ceil(320/8) = 80 -- NOT the raw-batch
    # ceil(320/6)*2 = 108 the old derivation produced.
    spec = _sft_spec(max_examples=320, batch_size=6, epochs=2)
    assert _spec_steps(spec) == _worker_sft_steps(examples=320, requested_batch=6, epochs=2) == 80


def test_sft_ignores_train_steps_for_step_count():
    # train.steps is a GRPO concept; SFT must derive from epochs/examples/realized-batch and NOT
    # honor a stray train.steps.
    spec = _sft_spec(max_examples=320, batch_size=16, epochs=2, steps=9999)
    assert _spec_steps(spec) == 40


def test_sft_steps_unpinned_examples_is_documented_floor_not_100():
    # No max_examples -> the worker trains the full dataset (size unknown locally). The estimate
    # uses the documented assumed-examples floor, NOT the old hardcoded 100 optimizer steps.
    from flash.cli.main.commands import _SFT_ASSUMED_EXAMPLES_WHEN_UNPINNED

    spec = _sft_spec(batch_size=16, epochs=2)  # max_examples omitted
    assert spec.train.max_examples is None
    expected = _worker_sft_steps(
        examples=_SFT_ASSUMED_EXAMPLES_WHEN_UNPINNED, requested_batch=16, epochs=2
    )
    assert _spec_steps(spec) == expected
    assert _spec_steps(spec) != 100


def test_sft_max_steps_caps_the_derived_count():
    spec = _sft_spec(max_examples=10_000, batch_size=16, epochs=2, max_steps=5)
    assert _spec_steps(spec) == 5


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
