"""Tests for the GPU table (mapping, validation gate, cheapest policy) — no network."""

from __future__ import annotations

import pytest


def test_canonical_gpu_aliases():
    from flash.providers.base import canonical_gpu

    for alias in ("RTX 5090", "rtx5090", "5090", "RTX_5090", "NVIDIA GeForce RTX 5090"):
        assert canonical_gpu(alias) == "RTX 5090"
    for alias in ("RTX 4090", "rtx4090", "4090", "RTX_4090"):
        assert canonical_gpu(alias) == "RTX 4090"
    assert canonical_gpu("NVIDIA A100-SXM4-40GB") == "A100 SXM 40GB"


def test_ambiguous_provider_aliases_are_not_user_canonicalizable():
    from flash.providers.base import UnsupportedGpuError, canonical_gpu, vast_gpu_for_offer

    for alias in ("A100 SXM4", "NVIDIA A100 SXM4"):
        with pytest.raises(UnsupportedGpuError):
            canonical_gpu(alias)

    assert canonical_gpu("A100 SXM 40GB") == "A100 SXM 40GB"
    assert canonical_gpu("A100 SXM") == "A100 SXM"
    assert vast_gpu_for_offer("A100 SXM4", 40 * 1024) == "A100 SXM 40GB"
    assert vast_gpu_for_offer("A100 SXM4", 80 * 1024) == "A100 SXM"


def test_unknown_gpu_rejected():
    from flash.providers.base import UnsupportedGpuError, canonical_gpu

    for bad in ("", "TPU v5", "RTX 9090", "Tesla T4"):
        with pytest.raises(UnsupportedGpuError):
            canonical_gpu(bad)


def test_providers_for():
    from flash.providers.base import providers_for

    # Consumer GeForce cards are RunPod + Vast (Vast's verified-datacenter market carries GeForce;
    # Lambda's datacenter fleet does not). Order follows the registry: runpod, lambda, vast.
    assert providers_for("RTX 4090") == ("runpod", "vast")
    assert providers_for("RTX 5090") == ("runpod", "vast")
    # Datacenter cards span the instance-based complements where the hardware exists.
    assert providers_for("H100") == ("runpod", "lambda", "vast")
    assert providers_for("A100 PCIe") == ("runpod", "vast")
    assert providers_for("RTX Pro 6000") == ("runpod",)  # no vast_name (RunPod-only)
    # Provider-exclusive / mixed instance classes.
    assert providers_for("A10") == ("lambda",)  # Lambda-only (no RunPod A10, no Vast A10)
    assert providers_for("A100 SXM 40GB") == ("lambda", "vast")  # instance-only 40 GB SXM4


def test_vast_gpu_for_offer_accepts_h100_pcie_alias():
    # Codex Mslmq: Vast lists PCIe H100s as "H100 PCIE" (distinct from the SXM "H100 SXM"). Both must
    # map to the canonical H100 class, else a PCIe-only market is dropped as unknown capacity.
    from flash.providers.base import vast_gpu_for_offer

    h100_ram = 80 * 1024  # MB
    assert vast_gpu_for_offer("H100 SXM", h100_ram) == "H100"
    assert vast_gpu_for_offer("H100 PCIE", h100_ram) == "H100"
    # an unmanaged name still maps to nothing (the Ampere+ floor / unknown-board guard holds)
    assert vast_gpu_for_offer("Tesla T4", 16 * 1024) is None


def test_removed_gpu_class_is_unmanaged():
    # RTX A6000 was a retired class kept only to resolve legacy records; it is now fully removed, so
    # it is not a managed class and resolves nowhere (no special retired-but-known status remains).
    from flash.providers.base import KNOWN, UnsupportedGpuError, canonical_gpu

    assert "RTX A6000" not in KNOWN
    with pytest.raises(UnsupportedGpuError):
        canonical_gpu("RTX A6000")


def test_expanded_gpu_table():
    from flash.providers.base import GPU_INFO, canonical_gpu, get_gpu_info, gpu_short

    # Cheap-capacity classes the cheapest policy exists for are all mapped.
    assert canonical_gpu("A100") == "A100 PCIe"
    assert canonical_gpu("h100") == "H100"
    assert get_gpu_info("A100 PCIe").vram_gb == 80
    # endpoint-name tokens stay single-word safe
    assert gpu_short("A100 PCIe") == "a100pcie"
    assert gpu_short("A100 SXM 40GB") == "a100sxm40"
    # architecture floor: nothing below Ampere (sm80)
    assert all(int(g.sm.removeprefix("sm")) >= 80 for g in GPU_INFO.values())


def test_sm_capability_and_fp8_kv_support():
    """fp8 KV cache is a cc >= 8.9 feature the OPD/GRPO workers enable off get_device_capability();
    sizing infers it from a class's ``sm`` string. Ada (4090, sm89), Hopper (H100/H200, sm90) and
    Blackwell (B200 sm100, RTX Pro 6000/5090 sm120) qualify; Ampere (A100 sm80, A10 sm86) does not."""
    from flash.providers.base import _sm_capability, max_non_fp8_kv_vram_gb, supports_fp8_kv

    assert _sm_capability("sm80") == (8, 0)
    assert _sm_capability("sm89") == (8, 9)
    assert _sm_capability("sm90") == (9, 0)
    assert _sm_capability("sm100") == (10, 0)
    assert _sm_capability("sm120") == (12, 0)
    assert _sm_capability("bogus") == (0, 0)  # unparseable -> pre-fp8
    for name in ("RTX 4090", "H100", "H200", "B200", "RTX Pro 6000", "RTX 5090"):
        assert supports_fp8_kv(name), name
    for name in ("A100 PCIe", "A100 SXM", "A10"):
        assert not supports_fp8_kv(name), name
    # the largest validated card WITHOUT fp8 KV is the 80 GB A100 -> a run needing more can only land
    # on a modern (fp8-capable) card.
    assert max_non_fp8_kv_vram_gb() == 80


def test_blackwell_min_cuda_pin():
    from flash.providers.base import min_cuda_modern

    assert min_cuda_modern("RTX 5090") == "13.0"
    assert min_cuda_modern("RTX Pro 6000") == "13.0"
    assert min_cuda_modern("RTX 4090") == "12.8"
    assert min_cuda_modern("A100 SXM") == "12.8"


def test_cheapest_gpu_policy(monkeypatch):
    from flash.providers import base as gpus
    from flash.providers.runpod import pricing

    # Validated-only by default: the cheapest validated enum class that fits each VRAM tier
    # wins on static rates. 24 GB is the floor now (sub-24 GB classes dropped), and the cheapest
    # validated card is the 24 GB RTX 4090 ($0.69).
    assert (
        gpus.cheapest_gpu(16) == "RTX 4090"
    )  # no sub-24 GB tier -> cheapest validated card ($0.69)
    # cheapest VALIDATED >=24G.
    assert gpus.cheapest_gpu(24) == "RTX 4090"
    assert gpus.cheapest_gpu(32) == "RTX 5090"  # 32G 5090 ($0.99) is the cheapest validated >=32G
    assert (
        gpus.cheapest_gpu(48) == "A100 PCIe"
    )  # cheapest validated >=48G is the 80G A100 PCIe ($1.39)
    # The error names the REAL constraint: this helper filters to validated RunPod classes,
    # so a fitting unvalidated class doesn't make the message a lie.
    with pytest.raises(gpus.UnsupportedGpuError, match="no validated RunPod-provisionable GPU"):
        gpus.cheapest_gpu(4096)
    # static rates cover every RunPod-provisionable class
    rates = pricing.static_rates()
    assert set(rates) == {name for name, info in gpus.GPU_INFO.items() if info.enum_member}


def test_provisional_gpu_cheapest_for_model(monkeypatch):
    from flash.providers.base import provisional_gpu

    # provisional_gpu remains the offline auto-sizing path and returns the cheapest fitting
    # validated class. 0.8B GRPO -> cheapest validated >=24G (RTX 4090). 9B is now bf16 (QLoRA dropped: the
    # 4-bit vLLM-rollout merge broke GRPO learning), so colocated 9B GRPO needs an 80G-class card
    # -> the cheapest validated 80G class (A100 PCIe).
    assert provisional_gpu("Qwen/Qwen3.5-0.8B", algorithm="grpo") == "RTX 4090"
    assert provisional_gpu("Qwen/Qwen3.5-9B", algorithm="grpo") == "A100 PCIe"
    assert provisional_gpu("Qwen/Qwen3.6-27B", algorithm="sft") == "A100 PCIe"
    assert provisional_gpu("Qwen/Qwen3.6-27B", algorithm="grpo") == "B200"


def test_config_gpu_type_is_empty_for_auto_and_preserves_pins():
    from flash.schema import ConfigError, spec_from_dict

    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
        "train": {"epochs": 1, "max_examples": 8},
        "gpu": {},
    }
    assert spec_from_dict(raw, run_id="x").gpu.type == ""

    for klass in ("RTX 4090", "A100 SXM", "H100"):
        raw["gpu"] = {"type": klass}
        assert spec_from_dict(raw, run_id="x").gpu.type == klass

    raw["gpu"] = {"type": "cheapest"}
    with pytest.raises(ConfigError, match="unsupported gpu"):
        spec_from_dict(raw, run_id="x")


def test_flash_gpu_enum_members():
    pytest.importorskip("runpod_flash")

    from flash.providers.runpod.gpus import flash_gpu

    assert flash_gpu("RTX 5090").name == "NVIDIA_GEFORCE_RTX_5090"
    assert flash_gpu("4090").name == "NVIDIA_GEFORCE_RTX_4090"
    # B200 is newly added; resolving it fails fast (AttributeError) if the installed runpod_flash SDK
    # doesn't expose the NVIDIA_B200 enum member, so CI catches an SDK/version gap before a live run.
    assert flash_gpu("B200").name == "NVIDIA_B200"


def test_gpu_short():
    from flash.providers.base import gpu_short

    assert gpu_short("RTX 5090") == "5090"
    assert gpu_short("rtx_4090") == "4090"


def test_config_defaults_gpu_to_auto():
    from flash.schema import spec_from_dict

    raw = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "sft",
        "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
        "train": {"epochs": 1, "max_examples": 8},
    }
    spec = spec_from_dict(raw, run_id="x")
    assert spec.gpu.type == ""


def test_build_worker_env():
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    spec = JobSpec(
        run_id="r1",
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=8),
        seed=0,
    )
    env = build_worker_env(spec, 0)
    assert env["RUN_ID"] == "r1"
    assert env["BENCH_HF_MODEL"] == "Qwen/Qwen3.5-4B"
    assert "RL_STEPS" not in env
    assert "SFT_EPOCHS" not in env


def test_grpo_kv_floor_escalates_large_group_long_context():
    """vLLM KV-cache init preflight: a rollout whose concurrent-group KV cannot fit under the
    colocate utilization cap on a small card must size onto a bigger one — previously such runs
    passed preflight and died at vLLM init with 'No available memory for the cache blocks'."""
    from flash.catalog import MODELS
    from flash.engine.vram import grpo_kv_floor_gb, model_required_vram_gb

    info = MODELS["Qwen/Qwen3.5-4B"]
    floor = grpo_kv_floor_gb(
        info.params_b,
        4096,
        16,
        active_params_b=info.active_params_b,
        model_info=info,
    )
    # the architecture-aware cache and profiled overhead still exceed the rtx 5090 class.
    assert floor > 32
    need = model_required_vram_gb(
        info.id, "grpo", train={"group_size": 16, "max_context_tokens": 4096}
    )
    assert need >= floor

    # The validated lean default (group 8, short context) stays on the 32 GB tier.
    assert model_required_vram_gb("Qwen/Qwen3.5-4B", "grpo", train={"group_size": 8}) <= 36


def test_opd_kv_floor_uses_fp8_above_non_fp8_ceiling():
    from flash.catalog import MODELS
    from flash.engine.vram import grpo_kv_floor_gb, model_required_vram_gb
    from flash.providers.base import max_non_fp8_kv_vram_gb

    info = MODELS["Qwen/Qwen3.5-2B"]
    concurrency = 8 * 16
    ceiling = max_non_fp8_kv_vram_gb()
    bf16_floor = grpo_kv_floor_gb(
        info.params_b,
        4096,
        concurrency,
        active_params_b=info.active_params_b,
        model_info=info,
        preserve_legacy_floor=True,
    )
    fp8_floor = grpo_kv_floor_gb(
        info.params_b,
        4096,
        concurrency,
        active_params_b=info.active_params_b,
        fp8_kv=True,
        model_info=info,
        preserve_legacy_floor=True,
    )
    need = model_required_vram_gb(
        info.id,
        "opd",
        train={"batch_size": 8, "group_size": 16, "max_context_tokens": 4096},
    )

    assert bf16_floor > ceiling
    assert fp8_floor > ceiling
    assert fp8_floor <= need < bf16_floor


def test_pinned_revision_retains_calibrated_vram_floors(monkeypatch):
    from flash.catalog import MODELS
    from flash.engine import vram

    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    monkeypatch.setattr(
        vram,
        "_validated_revision_geometry",
        lambda model_id, revision, model_info: (model_info.params_b, model_info.vocab_size),
    )

    grpo_need = vram.model_required_vram_gb(
        info.id,
        "grpo",
        model_revision="a" * 40,
    )
    sft_need = vram.model_required_vram_gb(
        info.id,
        "sft",
        model_revision="a" * 40,
    )

    assert grpo_need >= info.grpo_min_vram_gb
    assert sft_need >= info.sft_min_vram_gb
