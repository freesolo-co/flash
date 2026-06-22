"""Open-model policy + VRAM estimator unit tests (CPU-only, no network)."""

from __future__ import annotations

import math

import pytest

from flash.catalog import resolve_model
from flash.engine.vram import VramEstimate, check_fit, estimate_vram_gb
from flash.schema import ConfigError, spec_from_dict
from tests._helpers.specs import raw_spec as _raw


def test_catalog_policy_rejects_unlisted_with_hint():
    with pytest.raises(ValueError, match="model_policy"):
        resolve_model("some-org/some-model", "sft", policy="catalog")


def test_catalog_model_resolves_normally():
    info = resolve_model("Qwen/Qwen3.5-4B", "grpo", policy="catalog")
    assert info.id == "Qwen/Qwen3.5-4B"


def test_minicpm5_in_catalog():
    info = resolve_model("openbmb/MiniCPM5-1B", "grpo", policy="catalog")
    assert info.recommended_gpu == "RTX 4090"
    assert "grpo" in info.algos
    assert "sft" in info.algos


def test_allow_policy_accepts_fitting_model(monkeypatch):
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: 1.2, raising=True)
    info = resolve_model("acme/tiny-1b", "grpo", policy="allow", gpu="RTX 4090")
    assert info.thinking == "unknown"
    assert "open-model policy" in info.notes


def test_allow_policy_blocks_provably_too_big(monkeypatch):
    monkeypatch.setattr(
        "flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: 70.0, raising=True
    )
    with pytest.raises(ValueError, match="does not fit"):
        resolve_model("acme/huge-70b", "sft", policy="allow", gpu="RTX 4090")


def test_allow_policy_disaggregated_clears_colocate_too_big(monkeypatch, capsys):
    # A 20B model is too_big for colocated GRPO on an 80 GB A100, but with [train].inference_gpus=2
    # it fits as a disaggregated split (server bf16 weights sharded TP across 2 cards + a separate
    # trainer). The open-model fit check must use that disaggregated sizing — matching the
    # disaggregated-aware resolve_gpu_policy / submit-time allocator — not reject it on the colocate
    # total (which would block a model the GPU policy already sized for).
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id: 20.0, raising=True)
    monkeypatch.setattr(
        "flash.providers.allocator._params_b_for_vram", lambda model_id: 20.0, raising=True
    )
    # Sanity: colocated, the same model IS rejected (no disaggregated context).
    with pytest.raises(ValueError, match="does not fit"):
        resolve_model("acme/big-20b", "grpo", policy="allow", gpu="A100 PCIe")
    # With a disaggregated split it resolves.
    info = resolve_model(
        "acme/big-20b", "grpo", policy="allow", gpu="A100 PCIe", train={"inference_gpus": 2}
    )
    assert info.id == "acme/big-20b"
    assert "disaggregated split" in capsys.readouterr().out


def test_allow_policy_disaggregated_raises_disk_floor(monkeypatch):
    # A disaggregated split materializes TWO bf16 checkpoint copies (trainer + standalone
    # `trl vllm-serve`) on the same node plus the HF download/Xet temp/checkpoint-save peak, so the
    # synthesized open-model disk floor must exceed the colocate single-checkpoint heuristic
    # (~2 GB/param + 64) — otherwise a paid multi-GPU node provisions and then dies with "No space
    # left on device". Mirrors the curated 35B entry's validated 300 GB floor.
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id: 20.0, raising=True)
    monkeypatch.setattr(
        "flash.providers.allocator._params_b_for_vram", lambda model_id: 20.0, raising=True
    )
    # the 20B model as a disaggregated split -> the elevated two-copies + temp/save floor.
    split = resolve_model(
        "acme/big-20b", "grpo", policy="allow", gpu="A100 PCIe", train={"inference_gpus": 2}
    )
    assert split.min_disk_gb == 20 * 6 + 96  # 216 GB: trainer + server copies + headroom
    # and it must exceed the colocate single-checkpoint heuristic (~2 GB/param + 64) for this size.
    assert split.min_disk_gb > 20 * 2 + 64

    # a small model that DOES fit colocated keeps the lower single-checkpoint floor (no regression).
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id: 1.2, raising=True)
    small = resolve_model("acme/tiny-1b", "grpo", policy="allow", gpu="RTX 4090")
    # ceil (not truncate): a fractional param estimate rounds the disk floor UP so it never
    # under-provisions (1.2 * 2 = 2.4 -> 3, not 2).
    assert small.min_disk_gb == math.ceil(1.2 * 2) + 64


def test_allow_policy_disaggregated_fits_verdict_still_elevates_disk(monkeypatch):
    # A disaggregated split (inference_gpus>0) materializes multiple model copies on the same node
    # REGARDLESS of the per-card VRAM verdict: a model that comfortably "fits" each role's card still
    # needs the elevated trainer+server+download disk floor. The floor must key off inference_gpus,
    # not the (per-card) VRAM verdict — otherwise a `fits` split under-provisions disk and dies with
    # "No space left on device" on a paid multi-GPU node.
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id: 9.0, raising=True)
    monkeypatch.setattr(
        "flash.providers.allocator._params_b_for_vram", lambda model_id: 9.0, raising=True
    )
    # 9B on an 80 GB A100 fits colocated (verdict == "fits"), but the split must still get 6*p + 96.
    split = resolve_model(
        "acme/mid-9b", "grpo", policy="allow", gpu="A100 PCIe", train={"inference_gpus": 2}
    )
    assert split.min_disk_gb == 9 * 6 + 96  # 150 GB, not the colocate 9*2 + 64
    assert split.min_disk_gb > 9 * 2 + 64


def test_allow_policy_unknown_size_warns_but_allows(monkeypatch, capsys):
    monkeypatch.setattr(
        "flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: None, raising=True
    )
    info = resolve_model("acme/mystery", "sft", policy="allow", gpu="RTX 5090")
    assert info.id == "acme/mystery"
    assert "warning" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Estimator sanity: calibrated against catalog anchors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("params_b", "algo", "quant", "gpu", "expected"),
    [
        (4.0, "grpo", "bf16", "RTX 5090", "fits"),  # Qwen3-4B colocate on 32 GB (measured)
        (4.0, "sft", "bf16", "RTX 4090", "fits"),
        (9.65, "sft", "bf16", "RTX 5090", "fits"),  # Qwen3.5-9B SFT
        (36.0, "sft", "bf16", "RTX 5090", "too_big"),  # 72 GB of weights
    ],
)
def test_estimator_anchors(monkeypatch, params_b, algo, quant, gpu, expected):
    est = check_fit("anchor/model", algo, gpu, quant=quant, params_b=params_b)
    assert isinstance(est, VramEstimate)
    assert est.verdict == expected, est.describe()


def test_grpo_needs_more_than_sft():
    assert estimate_vram_gb(4.0, "grpo") > estimate_vram_gb(4.0, "sft")


# ---------------------------------------------------------------------------
# Config schema plumbing
# ---------------------------------------------------------------------------
def test_spec_model_policy_default_catalog():
    spec = spec_from_dict(_raw())
    assert spec.model_policy == "catalog"


def test_spec_user_model_policy_is_ignored():
    # model_policy is not a user knob: managed runs always use the curated catalog, so a
    # user-supplied "allow" in the config is ignored (the policy stays "catalog").
    spec = spec_from_dict(_raw(model="Qwen/Qwen3.5-4B", model_policy="allow"))
    assert spec.model_policy == "catalog"


def test_spec_unlisted_model_under_catalog_policy_fails():
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(model="acme/unlisted"))
    assert "model_policy" in str(ei.value)
