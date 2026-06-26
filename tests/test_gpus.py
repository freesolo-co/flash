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

    # Consumer cards are RunPod-only (datacenter clouds don't carry GeForce).
    assert providers_for("RTX 4090") == ("runpod",)
    assert providers_for("RTX 5090") == ("runpod",)
    assert providers_for("L4") == ("runpod",)
    # Datacenter cards span the instance-based complements where the hardware exists.
    assert providers_for("RTX A6000") == ("runpod", "lambda")
    assert providers_for("H100") == ("runpod", "lambda")
    assert providers_for("A100 PCIe") == ("runpod",)
    assert providers_for("RTX Pro 6000") == ("runpod",)
    # Provider-exclusive classes.
    assert providers_for("A10") == ("lambda",)  # Lambda-only


def test_expanded_gpu_table():
    from flash.providers.base import GPU_INFO, canonical_gpu, get_gpu_info, gpu_short

    # Cheap-capacity classes the cheapest policy exists for are all mapped.
    assert canonical_gpu("A100") == "A100 PCIe"
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

    # Validated-only by default: the cheapest validated enum class that fits each VRAM tier
    # wins on static rates. 24 GB is the floor now (sub-24 GB classes dropped), and the cheapest
    # validated card is the 48 GB RTX A6000 ($0.49), so anything that fits <=48 GB resolves to it.
    # A40 (48G) stays unvalidated (no RunPod capacity to smoke it in-window).
    assert gpus.cheapest_gpu(16) == "RTX A6000"  # no sub-24 GB tier -> cheapest validated card ($0.49)
    # cheapest VALIDATED >=24G.
    assert gpus.cheapest_gpu(24) == "RTX A6000"
    assert gpus.cheapest_gpu(32) == "RTX A6000"  # 48G A6000 validated, $0.49 < 5090 $0.99
    assert gpus.cheapest_gpu(48) == "RTX A6000"  # cheapest validated >=48G ($0.49, was A100 PCIe)
    # The error names the REAL constraint: this helper filters to validated RunPod classes,
    # so a fitting unvalidated class doesn't make the message a lie.
    with pytest.raises(gpus.UnsupportedGpuError, match="no validated RunPod-provisionable GPU"):
        gpus.cheapest_gpu(4096)
    # static rates cover every RunPod-provisionable class
    rates = pricing.static_rates()
    assert set(rates) == {name for name, info in gpus.GPU_INFO.items() if info.enum_member}


def test_provisional_gpu_cheapest_for_model(monkeypatch):
    from flash.providers.base import provisional_gpu

    # GPU pinning is gone: provisional_gpu returns the cheapest fitting VALIDATED class for the
    # model. 0.8B GRPO -> cheapest validated >=24G (RTX A6000). 9B is now bf16 (QLoRA dropped: the
    # 4-bit vLLM-rollout merge broke GRPO learning), so colocated 9B GRPO needs an 80G-class card
    # -> the cheapest validated 80G class (A100 PCIe).
    assert provisional_gpu("Qwen/Qwen3.5-0.8B", algorithm="grpo") == "RTX A6000"
    assert provisional_gpu("Qwen/Qwen3.5-9B", algorithm="grpo") == "A100 PCIe"


def test_config_cheapest_policy_validated_pool(monkeypatch):
    from flash.schema import spec_from_dict

    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
        "train": {"epochs": 1, "seeds": [0], "hf_repo": "owner/runs"},
        "gpu": {"type": "cheapest"},
    }
    spec = spec_from_dict(raw, run_id="x")
    # Cheapest fitting VALIDATED class for a small model: 0.8B SFT needs ~12 GB, but 24 GB is the
    # floor now (sub-24 GB classes dropped), so it resolves to the cheapest validated card,
    # RTX A6000 ($0.49).
    assert spec.gpu.type == "RTX A6000"
    # GPU pinning is gone: a config's gpu.type is IGNORED — the schema always resolves the
    # cheapest fitting VALIDATED class, no matter what class the config names.
    for klass in ("L4", "A100 SXM", "RTX 4090", "A40"):
        raw["gpu"] = {"type": klass}
        assert spec_from_dict(raw, run_id="x").gpu.type == "RTX A6000"


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
        "algorithm": "sft",
        "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
        "train": {"epochs": 1, "seeds": [0], "hf_repo": "owner/runs"},
    }
    spec = spec_from_dict(raw, run_id="x")
    # 9B is bf16 (QLoRA dropped): bf16 SFT needs ~29 GB but the catalog min_vram floor is 48 GB,
    # so the cheapest validated class is the 48 GB RTX A6000.
    assert spec.gpu.type == "RTX A6000"


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
