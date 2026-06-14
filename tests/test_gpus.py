"""Tests for the GPU table (mapping, validation gate, cheapest policy) — no network."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest


def test_canonical_gpu_aliases():
    from autoslm.flash.gpus import canonical_gpu

    for alias in ("RTX 5090", "rtx5090", "5090", "RTX_5090", "NVIDIA GeForce RTX 5090"):
        assert canonical_gpu(alias) == "RTX 5090"
    for alias in ("RTX 4090", "rtx4090", "4090", "RTX_4090"):
        assert canonical_gpu(alias) == "RTX 4090"


def test_unknown_gpu_rejected():
    from autoslm.flash.gpus import UnsupportedGpuError, canonical_gpu

    for bad in ("", "TPU v5", "RTX 9090", "Tesla T4"):  # junk + sub-Ampere reject
        with pytest.raises(UnsupportedGpuError):
            canonical_gpu(bad)


def test_vast_only_classes():
    from autoslm.flash.gpus import UnsupportedGpuError, canonical_gpu, flash_gpu, providers_for

    # L40S is provisionable on Vast but absent from the Flash SDK's GpuType enum:
    # canonical as a class, rejected only at the RunPod mapping with a providers hint.
    assert canonical_gpu("L40S") == "L40S"
    assert providers_for("L40S") == ("vast",)
    with pytest.raises(UnsupportedGpuError, match="not available on RunPod"):
        flash_gpu("L40S")
    assert providers_for("RTX 4090") == ("runpod", "vast")
    assert providers_for("RTX Pro 6000") == ("runpod",)


def test_vast_gpu_for_offer():
    from autoslm.flash.gpus import vast_gpu_for_offer

    # plain name matches, with under-reported board RAM tolerated (L4: 23034 MB)
    assert vast_gpu_for_offer("RTX 3090", 24576) == "RTX 3090"
    assert vast_gpu_for_offer("L4", 23034) == "L4"
    assert vast_gpu_for_offer("RTX 6000Ada", 49140) == "RTX 6000 Ada"
    # A100 SXM4 covers two VRAM variants under one Vast name
    assert vast_gpu_for_offer("A100 SXM4", 40960) == "A100 SXM 40GB"
    assert vast_gpu_for_offer("A100 SXM4", 81920) == "A100 SXM"
    # Ampere+ floor: anything not in the managed table maps to nothing
    assert vast_gpu_for_offer("Tesla T4", 16384) is None
    assert vast_gpu_for_offer("RTX 2080 Ti", 11264) is None


def test_is_validated_per_provider():
    from autoslm.flash.gpus import is_validated

    assert is_validated("RTX 4090")  # any-provider
    assert is_validated("RTX 4090", "runpod")
    assert not is_validated("RTX 4090", "vast")  # no live Vast smoke yet
    assert not is_validated("L4")
    # live-smoked on Vast 2026-06-12; never smoked on RunPod
    assert is_validated("RTX 3090", "vast")
    assert not is_validated("RTX 3090", "runpod")
    assert is_validated("RTX Pro 4000", "vast")


def test_expanded_gpu_table():
    from autoslm.flash.gpus import GPU_INFO, canonical_gpu, get_gpu_info, gpu_short

    # Cheap-capacity classes the cheapest policy exists for are all mapped.
    assert canonical_gpu("A100") == "A100 PCIe"
    assert canonical_gpu("rtx a5000") == "RTX A5000"
    assert canonical_gpu("NVIDIA GeForce RTX 3090") == "RTX 3090"
    assert canonical_gpu("h200") == "H200"
    assert get_gpu_info("A40").vram_gb == 48
    # endpoint-name tokens stay single-word safe
    assert gpu_short("A100 PCIe") == "a100pcie"
    assert gpu_short("RTX 6000 Ada") == "6000ada"
    # architecture floor: nothing below Ampere (sm80)
    assert all(int(g.sm.removeprefix("sm")) >= 80 for g in GPU_INFO.values())


def test_blackwell_min_cuda_pin():
    from autoslm.flash.gpus import min_cuda_modern

    assert min_cuda_modern("RTX 5090") == "13.0"
    assert min_cuda_modern("B200") == "13.0"
    assert min_cuda_modern("RTX Pro 6000") == "13.0"
    assert min_cuda_modern("RTX 4090") == "12.8"
    assert min_cuda_modern("A100 SXM") == "12.8"


def test_cheapest_gpu_policy(monkeypatch):
    from autoslm.flash import gpus, pricing

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")  # static rates only
    # RTX A5000 (validated 2026-06-11) is the cheapest 24GB-capable class.
    assert gpus.cheapest_gpu(24) == "RTX A5000"
    assert gpus.cheapest_gpu(32) == "RTX 5090"
    # With unvalidated classes allowed, cheaper Ampere cards win on static rates.
    assert gpus.cheapest_gpu(16, include_unvalidated=True) == "RTX 2000 Ada"
    assert gpus.cheapest_gpu(48, include_unvalidated=True) == "A40"
    with pytest.raises(gpus.UnsupportedGpuError):
        gpus.cheapest_gpu(4096)
    # static fallback rates cover every known class
    rates = pricing.live_rates()
    assert set(rates) >= set(gpus.GPU_INFO)


def test_resolve_gpu_policy(monkeypatch):
    from autoslm.flash.gpus import resolve_gpu_policy

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    monkeypatch.delenv("AUTOSLM_GPU_ALLOW_UNVALIDATED", raising=False)
    # catalog model needing 24GB -> cheapest validated 24GB class
    assert resolve_gpu_policy("cheapest", "Qwen/Qwen3-4B-Instruct-2507") == "RTX A5000"
    assert resolve_gpu_policy("cheapest", "Qwen/Qwen3-8B") == "RTX 5090"
    # concrete names pass through canonicalization
    assert resolve_gpu_policy("rtx5090", "Qwen/Qwen3-8B") == "RTX 5090"


def test_config_cheapest_policy_and_unvalidated_gate(monkeypatch):
    from autoslm.config_schema import ConfigError, spec_from_dict

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    monkeypatch.delenv("AUTOSLM_GPU_ALLOW_UNVALIDATED", raising=False)
    raw = {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "algorithm": "sft",
        "environment": {"id": "gsm8k"},
        "train": {"epochs": 1, "seeds": [0]},
        "gpu": {"type": "cheapest"},
    }
    spec = spec_from_dict(raw, run_id="x")
    assert spec.gpu.type == "RTX A5000"  # cheapest validated 24GB class
    # class validated on NO provider blocked without opt-in...
    raw["gpu"] = {"type": "L4"}
    with pytest.raises(ConfigError):
        spec_from_dict(raw, run_id="x")
    # ...allowed with the config opt-in
    raw["gpu"] = {"type": "L4", "allow_unvalidated": True}
    spec = spec_from_dict(raw, run_id="x")
    assert spec.gpu.type == "L4"
    # vast-validated class passes the gate when the provider is consistent
    raw["gpu"] = {"type": "RTX 3090"}
    assert spec_from_dict(raw, run_id="x").gpu.type == "RTX 3090"
    # ...but pinning it to a provider it is NOT validated on still blocks
    raw["gpu"] = {"type": "RTX 3090", "provider": "runpod"}
    with pytest.raises(ConfigError):
        spec_from_dict(raw, run_id="x")


def test_flash_gpu_enum_members():
    from autoslm.flash.gpus import flash_gpu

    assert flash_gpu("RTX 5090").name == "NVIDIA_GEFORCE_RTX_5090"
    assert flash_gpu("4090").name == "NVIDIA_GEFORCE_RTX_4090"


def test_gpu_for_model_thresholds():
    from autoslm.flash.gpus import gpu_for_model

    # >=32GB catalog models -> 5090; smaller -> 4090
    assert gpu_for_model("Qwen/Qwen3.5-9B") == "RTX 5090"
    assert gpu_for_model("Qwen/Qwen3.5-2B") == "RTX 4090"


def test_gpu_short():
    from autoslm.flash.gpus import gpu_short

    assert gpu_short("RTX 5090") == "5090"
    assert gpu_short("rtx_4090") == "4090"


def test_config_rejects_unsupported_gpu():
    from autoslm.config_schema import ConfigError, spec_from_dict

    raw = {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "algorithm": "grpo",
        "environment": {"id": "gsm8k"},
        "train": {"steps": 1, "seeds": [0]},
        "gpu": {"type": "L40S"},  # not in the Flash SDK's GpuType enum
    }
    with pytest.raises(ConfigError):
        spec_from_dict(raw, run_id="x")


def test_config_defaults_gpu_from_model():
    from autoslm.config_schema import spec_from_dict

    raw = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "sft",  # 9B is SFT-only (colocated GRPO does not fit 32 GB bf16)
        "environment": {"id": "gsm8k"},
        "train": {"epochs": 1, "seeds": [0]},
    }
    spec = spec_from_dict(raw, run_id="x")
    assert spec.gpu.type == "RTX 5090"  # 9B needs >=32GB


def test_build_worker_env():
    from autoslm.flash.train import build_worker_env
    from autoslm.worker_spec import JobSpec, TrainSpec

    spec = JobSpec(
        run_id="r1",
        model="Qwen/Qwen3-4B-Instruct-2507",
        algorithm="grpo",
        train=TrainSpec(steps=20, seeds=(0,), eval_examples=50),
    )
    env = build_worker_env(spec, 0)
    assert env["RUN_ID"] == "r1"
    assert env["BENCH_HF_MODEL"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert env["EVAL_NUM"] == "50"
    assert env["RL_STEPS"] == "20"
