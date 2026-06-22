"""Tests for the GPU table (mapping, validation gate, cheapest policy) — no network."""

from __future__ import annotations

import pytest


def test_canonical_gpu_aliases():
    from flash.providers.base import canonical_gpu

    for alias in ("RTX 5090", "rtx5090", "5090", "RTX_5090", "NVIDIA GeForce RTX 5090"):
        assert canonical_gpu(alias) == "RTX 5090"
    for alias in ("RTX 4090", "rtx4090", "4090", "RTX_4090"):
        assert canonical_gpu(alias) == "RTX 4090"


def test_unknown_gpu_rejected():
    from flash.providers.base import UnsupportedGpuError, canonical_gpu

    for bad in ("", "TPU v5", "RTX 9090", "Tesla T4"):  # junk + sub-Ampere reject
        with pytest.raises(UnsupportedGpuError):
            canonical_gpu(bad)


def test_providers_for():
    from flash.providers.base import providers_for

    # RTX 4090 is provisionable on BOTH substrates; RTX Pro 6000 is RunPod-only
    # (no vast_name); L40S / RTX Pro 4000 are vast-only (no Flash enum member).
    assert providers_for("RTX 4090") == ("runpod", "vast")
    assert providers_for("RTX Pro 6000") == ("runpod",)
    assert providers_for("L40S") == ("vast",)
    assert providers_for("RTX Pro 4000") == ("vast",)


def test_expanded_gpu_table():
    from flash.providers.base import GPU_INFO, canonical_gpu, get_gpu_info, gpu_short

    # Cheap-capacity classes the cheapest policy exists for are all mapped.
    assert canonical_gpu("A100") == "A100 PCIe"
    assert canonical_gpu("rtx a5000") == "RTX A5000"
    assert canonical_gpu("NVIDIA GeForce RTX 3090") == "RTX 3090"
    assert canonical_gpu("h100") == "H100"
    assert get_gpu_info("A40").vram_gb == 48
    # endpoint-name tokens stay single-word safe
    assert gpu_short("A100 PCIe") == "a100pcie"
    assert gpu_short("RTX 6000 Ada") == "6000ada"
    # architecture floor: nothing below Ampere (sm80)
    assert all(int(g.sm.removeprefix("sm")) >= 80 for g in GPU_INFO.values())


def test_blackwell_min_cuda_pin():
    from flash.providers.base import min_cuda_modern

    assert min_cuda_modern("RTX 5090") == "13.0"
    assert min_cuda_modern("RTX Pro 6000") == "13.0"
    assert min_cuda_modern("RTX 4090") == "12.8"
    assert min_cuda_modern("A100 SXM") == "12.8"


def test_cheapest_gpu_policy(monkeypatch):
    from flash.providers import base as gpus
    from flash.providers.runpod import pricing

    # Validated-only by default: the cheapest LIVE-VALIDATED enum class that fits each VRAM tier
    # wins on static rates (unvalidated cheaper cards — RTX 2000 Ada @ 16G, A40 @ 48G — excluded,
    # because the deployed control plane would reject a submit for them).
    assert gpus.cheapest_gpu(16) == "RTX A5000"  # cheapest VALIDATED >=16G (RTX 2000 Ada excluded)
    assert gpus.cheapest_gpu(24) == "RTX A5000"  # cheapest VALIDATED >=24G enum class
    assert gpus.cheapest_gpu(32) == "RTX 5090"  # validated >=32G (A40 is unvalidated)
    assert gpus.cheapest_gpu(48) == "A100 PCIe"  # cheapest validated >=48G (80G A100 PCIe)
    # The error names the REAL constraint: this helper filters to RunPod-provisionable VALIDATED
    # classes, so a fitting Vast-only / unvalidated class doesn't make the message a lie.
    with pytest.raises(gpus.UnsupportedGpuError, match="no validated RunPod-provisionable GPU"):
        gpus.cheapest_gpu(4096)
    # static fallback rates cover every known class
    rates = pricing.live_rates()
    assert set(rates) >= set(gpus.GPU_INFO)


def test_provisional_gpu_cheapest_for_model(monkeypatch):
    from flash.providers.base import provisional_gpu

    # GPU pinning is gone: provisional_gpu returns the cheapest fitting VALIDATED class for the
    # model. 0.8B GRPO -> cheapest validated >=24G (A5000). 9B GRPO needs the 32G tier, whose
    # cheapest VALIDATED class is RTX 5090 (the cheaper 48G A40 is unvalidated and excluded).
    assert provisional_gpu("Qwen/Qwen3.5-0.8B", algorithm="grpo") == "RTX A5000"
    assert provisional_gpu("Qwen/Qwen3.5-9B", algorithm="grpo") == "RTX 5090"


def test_config_cheapest_policy_validated_pool(monkeypatch):
    from flash.schema import spec_from_dict

    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "environment": {"id": "primeintellect/gsm8k"},
        "train": {"epochs": 1, "seeds": [0], "hf_repo": "owner/runs"},
        "gpu": {"type": "cheapest"},
    }
    spec = spec_from_dict(raw, run_id="x")
    # Cheapest fitting VALIDATED class for a small model (the unvalidated RTX 2000 Ada is excluded
    # — the deployed control plane would reject it).
    assert spec.gpu.type == "RTX A5000"
    # GPU pinning is gone: a config's gpu.type is IGNORED — the schema always resolves the
    # cheapest fitting VALIDATED class, no matter what class the config names.
    for klass in ("L4", "L40S", "RTX 3090", "A40"):
        raw["gpu"] = {"type": klass}
        assert spec_from_dict(raw, run_id="x").gpu.type == "RTX A5000"


def test_flash_gpu_enum_members():
    from flash.providers.runpod.gpus import flash_gpu

    assert flash_gpu("RTX 5090").name == "NVIDIA_GEFORCE_RTX_5090"
    assert flash_gpu("4090").name == "NVIDIA_GEFORCE_RTX_4090"


def test_gpu_short():
    from flash.providers.base import gpu_short

    assert gpu_short("RTX 5090") == "5090"
    assert gpu_short("rtx_4090") == "4090"


def test_config_defaults_gpu_from_model():
    from flash.schema import spec_from_dict

    raw = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "sft",  # 9B is SFT-only (colocated GRPO does not fit 32 GB bf16)
        "environment": {"id": "primeintellect/gsm8k"},
        "train": {"epochs": 1, "seeds": [0], "hf_repo": "owner/runs"},
    }
    spec = spec_from_dict(raw, run_id="x")
    # 9B 4-bit QLoRA SFT fits a 24 GB card; validated pool -> cheapest validated class (A5000).
    assert spec.gpu.type == "RTX A5000"


def test_build_worker_env():
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    spec = JobSpec(
        run_id="r1",
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(steps=20, seeds=(0,)),
    )
    env = build_worker_env(spec, 0)
    assert env["RUN_ID"] == "r1"
    assert env["BENCH_HF_MODEL"] == "Qwen/Qwen3.5-4B"
    assert env["RL_STEPS"] == "20"
