"""`flash train --cost`: map a training config to a pre-flight cost."""

from __future__ import annotations

import copy
import types

import pytest

from flash.cli.commands import _cmd_train_cost, cmd_train
from flash.cost.spec import runconfig_from_spec as _runconfig_from_spec
from flash.cost.spec import spec_steps as _spec_steps
from flash.cost.types import RunConfig
from flash.engine.recipe import RECIPE
from flash.schema import ConfigError, spec_from_dict

GRPO_RAW = {
    "model": "Qwen/Qwen3.5-9B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {
        "epochs": 1,
        "max_examples": 800,
        "group_size": 8,
        "batch_size": 16,
        "max_completion_tokens": 512,
        "max_context_tokens": 2048,
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
    spec = _spec()
    cfg = _runconfig_from_spec(spec)
    assert cfg.model_id == "Qwen/Qwen3.5-9B"
    assert cfg.method == "grpo"
    assert cfg.steps == 50
    assert cfg.batch_size == 16
    assert cfg.group_size == 8
    assert cfg.completion_len == 512  # GRPO max_completion_tokens
    assert cfg.seq_len == 2048
    assert cfg.environment == "github:freesolo-co/envs@main:gsm8k/environment.py"


def test_grpo_default_epochs_mirror_recipe():
    spec = _spec()
    object.__setattr__(spec.train, "epochs", None)
    assert RECIPE.rl.num_epochs == 1
    assert _spec_steps(spec) == 50


def test_grpo_epochs_derive_steps_from_max_examples():
    spec = _spec(**{"train.epochs": 2, "train.max_examples": 33})
    assert _spec_steps(spec) == 5  # ceil(33 rows * 2 epochs / batch_size 16)


def test_grpo_epochs_need_max_examples_for_cost():
    spec = _spec(**{"train.max_examples": None, "train.epochs": 2})
    assert _spec_steps(spec) == 2


def test_grpo_positive_max_steps_is_authoritative():
    assert _spec_steps(_spec(**{"train.max_steps": 73})) == 73
    assert _spec_steps(_spec(**{"train.max_steps": 0})) == 50


def test_required_save_density_adds_wall_time_and_cost_without_changing_steps():
    from flash.cost.analytical import estimate_cost

    for method in ("sft", "grpo"):
        common = {
            "model_id": "Qwen/Qwen3.5-4B",
            "method": method,
            "steps": 10,
            "seq_len": 1024,
            "batch_size": 4,
        }
        if method == "grpo":
            common.update(completion_len=128, group_size=2)
        baseline = RunConfig(**common)
        sparse = RunConfig(**common, save_at_steps=(5,))
        dense = RunConfig(**common, save_at_steps=(2, 4, 6, 8))

        baseline_estimate = estimate_cost(baseline)
        sparse_estimate = estimate_cost(sparse)
        dense_estimate = estimate_cost(dense)

        assert baseline.steps == sparse.steps == dense.steps == 10
        assert baseline_estimate.train_seconds < sparse_estimate.train_seconds
        assert sparse_estimate.train_seconds < dense_estimate.train_seconds
        assert baseline_estimate.total_usd < sparse_estimate.total_usd < dense_estimate.total_usd


def test_required_save_overhead_uses_contractual_commit_counts():
    from flash.cost.analytical import (
        REQUIRED_SAVE_COMMIT_FLOOR_S,
        REQUIRED_SAVE_S_PER_MODEL_B_AT_RANK32,
        required_save_overhead_seconds,
    )
    from flash.cost.facts import total_params_b

    model_id = "Qwen/Qwen3.5-4B"
    save_at_steps = (2, 4, 6)
    common = {
        "model_id": model_id,
        "steps": 10,
        "seq_len": 1024,
        "batch_size": 4,
        "lora_rank": 32,
        "save_at_steps": save_at_steps,
    }
    serialize_per_save = REQUIRED_SAVE_S_PER_MODEL_B_AT_RANK32 * total_params_b(model_id)

    for method, commits_per_save in (("sft", 2), ("grpo", 2), ("opd", 1)):
        config = RunConfig(method=method, **common)
        expected = len(save_at_steps) * (
            commits_per_save * REQUIRED_SAVE_COMMIT_FLOOR_S + serialize_per_save
        )
        assert required_save_overhead_seconds(config) == pytest.approx(expected)


def test_opd_required_saves_add_overhead_without_changing_steps():
    from flash.cost.analytical import estimate_cost

    common = {
        "model_id": "Qwen/Qwen3.5-4B",
        "method": "opd",
        "steps": 10,
        "seq_len": 1024,
        "batch_size": 4,
        "completion_len": 128,
        "group_size": 1,
    }
    baseline = RunConfig(**common)
    withsave = RunConfig(**common, save_at_steps=(2, 4, 6))

    # opd publishes a deployable adapter at each exact save, so exact saves cost wall/dollars too.
    assert baseline.steps == withsave.steps == 10
    assert estimate_cost(withsave).train_seconds > estimate_cost(baseline).train_seconds
    assert estimate_cost(withsave).total_usd > estimate_cost(baseline).total_usd


def test_partial_reprice_counts_reached_saves_and_drops_future_saves():
    from flash.runner import charge_usd_for_spec

    def partial_charge(save_at_steps):
        raw = copy.deepcopy(GRPO_RAW)
        raw["train"].update({"max_steps": 100, "save_at_steps": save_at_steps})
        return charge_usd_for_spec(spec_from_dict(raw), steps=10, fallback=-1.0)

    # cancel at step 10: step 5 landed and remains priced, while steps 50/100 are dropped before the
    # reduced run config is built so neither estimate falls back.
    reached_save_charge = partial_charge([5, 50, 100])
    future_only_charge = partial_charge([50, 100])

    assert reached_save_charge != -1.0
    assert future_only_charge != -1.0
    assert reached_save_charge > future_only_charge > 0.0


def test_opd_epochs_derive_steps_from_max_examples():
    raw = copy.deepcopy(GRPO_RAW)
    raw["algorithm"] = "opd"
    raw["train"].update({"epochs": 2, "max_examples": 17, "batch_size": 8, "group_size": 1})
    spec = spec_from_dict(raw)
    assert _spec_steps(spec) == 5  # ceil(17 rows * 2 epochs / batch_size 8)


def test_opd_positive_max_steps_is_authoritative():
    raw = copy.deepcopy(GRPO_RAW)
    raw["algorithm"] = "opd"
    raw["train"].update(
        {"epochs": 2, "max_examples": 17, "batch_size": 8, "group_size": 1, "max_steps": 31}
    )
    assert _spec_steps(spec_from_dict(raw)) == 31


def test_opd_runconfig_carries_selected_teacher_and_prices_it():
    """runconfig_from_spec resolves [train].teacher_model to the Fireworks model id so the estimate
    prices the CHOSEN teacher; a cheaper teacher lowers teacher_api_usd vs the default GLM 5.2, and
    sft/grpo carry no teacher."""
    from flash.cost.analytical import estimate_cost

    def _opd(teacher=None):
        raw = copy.deepcopy(GRPO_RAW)
        raw["model"] = "Qwen/Qwen3.5-4B"
        raw["algorithm"] = "opd"
        raw["train"].update({"epochs": 1, "max_examples": 40, "batch_size": 8, "group_size": 1})
        if teacher is not None:
            raw["train"]["teacher_model"] = teacher
        return spec_from_dict(raw)

    # Omitted teacher_model -> default GLM 5.2 provider id.
    assert _runconfig_from_spec(_opd()).teacher_model == "accounts/fireworks/models/glm-5p2"
    # A selected alias resolves to its Fireworks model id.
    kimi_cfg = _runconfig_from_spec(_opd("kimi-k2.6"))
    assert kimi_cfg.teacher_model == "accounts/fireworks/models/kimi-k2p6"

    # kimi-k2.6 input price ($0.95/M) < glm-5.2 ($1.40/M), so its teacher-API estimate is smaller.
    default_teacher_usd = estimate_cost(_runconfig_from_spec(_opd())).teacher_api_usd
    kimi_teacher_usd = estimate_cost(kimi_cfg).teacher_api_usd
    assert 0 < kimi_teacher_usd < default_teacher_usd

    # sft/grpo carry no teacher.
    assert _runconfig_from_spec(_spec()).teacher_model == ""


def test_sft_steps_derived_from_examples():
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "environment": {"id": "github:acme/envs@main:sft-data/environment.py"},
            "train": {
                "max_examples": 320,
                "batch_size": 16,
                "epochs": 2,
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
        "environment": {"id": "github:acme/envs@main:sft-data/environment.py"},
        "train": {"hf_repo": "owner/runs", **train},
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
    return max_steps if max_steps > 0 else n


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


def test_sft_steps_unpinned_requires_max_examples():
    # --cost no longer imports/counts the environment. SFT must pin max_examples explicitly.
    with pytest.raises(ConfigError, match=r"max_examples.*positive"):
        _sft_spec(batch_size=16, epochs=2)


def test_sft_steps_max_examples_zero_requires_positive_cap_for_cost():
    # max_examples = 0 still means "no cap" to the worker, but --cost needs a positive row count.
    with pytest.raises(ConfigError, match=r"max_examples.*positive"):
        _sft_spec(max_examples=0, batch_size=16, epochs=2)


def test_sft_steps_pinned_examples_are_used_as_the_cost_row_count(capsys):
    # An explicit [train].max_examples prices exactly that, with no environment fallback.
    spec = _sft_spec(max_examples=320, batch_size=16, epochs=2)
    assert _spec_steps(spec) == _worker_sft_steps(examples=320, requested_batch=16, epochs=2) == 40
    assert "could not count" not in capsys.readouterr().err


def test_sft_runconfig_does_not_count_env_train_tokens():
    spec = _sft_spec(max_examples=320, batch_size=16, epochs=2)
    cfg = _runconfig_from_spec(spec)
    assert cfg.steps == 40
    assert cfg.train_tokens is None


def test_runconfig_preserves_positional_seq_len_compatibility():
    cfg = RunConfig("Qwen/Qwen3.5-4B", "sft", 10, 2048)
    assert cfg.seq_len == 2048
    assert cfg.train_tokens is None


def test_sft_positive_max_steps_is_authoritative():
    below_derived = _sft_spec(max_examples=10_000, batch_size=16, epochs=2, max_steps=5)
    above_derived = _sft_spec(max_examples=16, batch_size=16, epochs=1, max_steps=9)
    assert _spec_steps(below_derived) == 5
    assert _spec_steps(above_derived) == 9


def test_sft_steps_honor_big_vocab_per_device_cap():
    # For a sub-3B short-ctx SFT the worker vocab-sizes the per-device micro-batch (the big-vocab
    # logits cap), which with CEIL'd grad-accum changes the REALIZED global batch -- so the priced
    # step count must mirror the capped batch, not the fixed pd=4 one. Qwen3.5-0.8B (0.9B, ~248k
    # vocab) at a 1024 ctx leaves CE un-fused -> per_device caps 4->1, so batch 6 realizes 1x6=6
    # (not 4x2=8): steps = epochs(2) x ceil(320/6) = 108, NOT the uncapped ceil(320/8)*2 = 80.
    import math

    from flash.catalog import vocab_size_for
    from flash.engine.vram import sft_logits_fused, sft_per_device, sft_realized_batch

    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "environment": {"id": "github:acme/envs@main:sft-data/environment.py"},
        "train": {
            "hf_repo": "owner/runs",
            "max_examples": 320,
            "batch_size": 6,
            "epochs": 2,
            "max_context_tokens": 1024,
        },
        "gpu": {"type": "RTX 4090"},
    }
    spec = spec_from_dict(raw)
    v = vocab_size_for("Qwen/Qwen3.5-0.8B")
    fused = sft_logits_fused(0.9, 1024)
    assert fused is False
    assert sft_per_device(6, seq_len=1024, vocab=v, fused=fused) == 1
    assert sft_realized_batch(6, seq_len=1024, vocab=v, fused=fused) == 6
    assert _spec_steps(spec) == math.ceil(320 / 6) * 2 == 108
    assert _spec_steps(spec) != 80  # the pre-fix uncapped (pd=4 -> realized 8) step count


def test_cmd_train_cost_prints_breakdown_without_submitting(tmp_path, capsys):
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 800\n"
        "batch_size = 16\n"
        'hf_repo = "owner/runs"\n'
        "[gpu]\n"
        'type = "RTX 5090"\n'
    )
    args = types.SimpleNamespace(
        config=str(cfg),
        overrides=[],
        extra_configs=[],
        cost=True,
        dry_run=False,
        background=False,
    )
    # --cost is local: it must NOT touch the control-plane client. GRPO needs no env load, and
    # estimate_cost sizes VRAM offline, so no network is required for a listed model.
    rc = cmd_train(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "TOTAL" in out
    assert "$" in out
    assert "GPU" in out  # the breakdown names the chosen (provisional cheapest-fit) class


@pytest.mark.parametrize(("gpu_type", "expects_warning"), [("B200", True), ("H200", False)])
def test_cmd_train_cost_warns_only_for_b200(tmp_path, capsys, gpu_type, expects_warning):
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 800\n"
        "batch_size = 16\n"
        'hf_repo = "owner/runs"\n'
        "[gpu]\n"
        f'exact_type = "{gpu_type}"\n'
    )
    args = types.SimpleNamespace(config=str(cfg), overrides=[], extra_configs=[])

    rc = _cmd_train_cost(args)
    captured = capsys.readouterr()

    assert rc == 0
    assert "TOTAL" in captured.out
    assert gpu_type in captured.out
    warning = "warning: this estimate assumes peak-flops throughput; B200 (sm100) kernels"
    h200_comparison = (
        'if your configuration also fits H200, pin [gpu] exact_type = "H200" to compare.'
    )
    assert (warning in captured.err) is expects_warning
    assert (h200_comparison in captured.err) is expects_warning
    if not expects_warning:
        assert captured.err == ""


def test_cmd_train_cost_rejects_context_above_serving_cap(tmp_path):
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-4B"\n'
        'algorithm = "sft"\n'
        "[environment]\n"
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 8\n"
        "max_context_tokens = 33000\n"
        'hf_repo = "owner/runs"\n'
    )
    args = types.SimpleNamespace(
        config=str(cfg), overrides=[], extra_configs=[], cost=True, dry_run=False, background=False
    )

    with pytest.raises(
        ValueError,
        match=r"train\.max_context_tokens=33000 exceeds Qwen/Qwen3\.5-4B's serving max_model_len=32768",
    ):
        cmd_train(args)


def test_cmd_train_cost_rejects_unlisted_model(tmp_path):
    """Cost is catalog-only: ``--cost`` on a non-catalog model errors cleanly (no open-model
    sizing)."""
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "some-org/unlisted-7b"\n'
        'model_policy = "allow"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 10\n"
        'hf_repo = "owner/runs"\n'
    )
    args = types.SimpleNamespace(
        config=str(cfg), overrides=[], extra_configs=[], cost=True, dry_run=False, background=False
    )
    with pytest.raises((KeyError, ValueError)):
        cmd_train(args)
