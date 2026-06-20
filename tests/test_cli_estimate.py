"""`slm train --estimate`: map a training config to a pre-flight cost estimate. No network."""

from __future__ import annotations

import copy
import os
import types

from flash.cli.main.commands import cmd_train
from flash.cost.spec import runconfig_from_spec as _runconfig_from_spec
from flash.cost.spec import spec_steps as _spec_steps
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
        config=str(cfg),
        overrides=[],
        extra_configs=[],
        estimate=True,
        dry_run=False,
        background=False,
    )
    # --estimate is fully local: it must NOT touch the control-plane client.
    rc = cmd_train(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "TOTAL" in out
    assert "$" in out
    assert "RTX 5090" in out


# --- --estimate is network-free even for an unlisted (open-model-policy) model ---------------
# An UNLISTED model under model_policy="allow" parses via resolve_model -> check_fit and sizes via
# resolve_gpu_policy -> model_required_vram_gb; both reach engine.vram.fetch_hf_params_b, which probes
# the Hugging Face API (constructs HfApi) UNLESS FLASH_SKIP_NET is set. cmd_train's --estimate branch
# forces FLASH_SKIP_NET on for the whole flow (parse + size), so estimation never does network I/O
# even when the user hasn't set it. We detect the network probe by recording HfApi construction:
# fetch_hf_params_b builds HfApi only AFTER its skip-net short-circuit AND swallows exceptions
# internally, so a side-effect FLAG (not a raise) is the reliable detector.
_UNLISTED_ESTIMATE_TOML = (
    'model = "acme/unlisted-7b"\n'
    'model_policy = "allow"\n'
    'algorithm = "grpo"\n'
    "[environment]\n"
    'id = "primeintellect/gsm8k"\n'
    "[train]\n"
    "steps = 50\n"
    'hf_repo = "owner/runs"\n'
    "[gpu]\n"
    'type = "auto"\n'  # policy word -> also exercises resolve_gpu_policy's sizing probe
)


def _record_hfapi(monkeypatch) -> list[int]:
    """Replace huggingface_hub.HfApi with a recorder; returns a list that gains an entry per
    construction (== one network probe). model_info raises so any caller that DID reach the
    network falls back as if offline (matching the real best-effort path)."""
    calls: list[int] = []

    class _Recorder:
        def __init__(self, *a, **k):
            calls.append(1)

        def model_info(self, *a, **k):
            raise RuntimeError("offline (test: no network)")

    monkeypatch.setattr("huggingface_hub.HfApi", _Recorder, raising=True)
    return calls


def _estimate_args(cfg) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        config=str(cfg),
        overrides=[],
        extra_configs=[],
        estimate=True,
        dry_run=False,
        background=False,
    )


def test_estimate_unlisted_model_does_no_network_without_flash_skip_net(
    tmp_path, monkeypatch, capsys
):
    # Pre-fix this fails: cmd_train parsed the spec BEFORE checking --estimate, so the open-model
    # spec-parse + GPU sizing each constructed HfApi (network) when FLASH_SKIP_NET was unset.
    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)
    calls = _record_hfapi(monkeypatch)
    cfg = tmp_path / "run.toml"
    cfg.write_text(_UNLISTED_ESTIMATE_TOML)

    rc = cmd_train(_estimate_args(cfg))

    assert rc == 0
    assert calls == [], "--estimate must not construct HfApi (no network) for an unlisted model"
    out = capsys.readouterr().out
    assert "TOTAL" in out
    assert "$" in out
    # The estimate still produced a fully-validated spec sized via the offline heuristic.
    assert "acme/unlisted-7b" in out


def test_estimate_restores_preexisting_flash_skip_net(tmp_path, monkeypatch):
    # The scoped env guard must restore a PRE-EXISTING FLASH_SKIP_NET value afterward (not leave the
    # forced "1" behind, and not delete a value the caller had set).
    monkeypatch.setenv("FLASH_SKIP_NET", "preexisting")
    _record_hfapi(monkeypatch)
    cfg = tmp_path / "run.toml"
    cfg.write_text(_UNLISTED_ESTIMATE_TOML)

    assert cmd_train(_estimate_args(cfg)) == 0
    assert os.environ.get("FLASH_SKIP_NET") == "preexisting"


def test_estimate_does_not_set_flash_skip_net_when_unset(tmp_path, monkeypatch):
    # When FLASH_SKIP_NET was UNSET, the guard must leave it unset afterward (no leak of the forced
    # value into the rest of the process / a later real run).
    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)
    _record_hfapi(monkeypatch)
    cfg = tmp_path / "run.toml"
    cfg.write_text(_UNLISTED_ESTIMATE_TOML)

    assert cmd_train(_estimate_args(cfg)) == 0
    assert "FLASH_SKIP_NET" not in os.environ
