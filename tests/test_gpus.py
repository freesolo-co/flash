"""Tests for the GPU table (mapping, validation gate, cheapest policy) — no network."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest


def test_canonical_gpu_aliases():
    from autoslm.providers.base import canonical_gpu

    for alias in ("RTX 5090", "rtx5090", "5090", "RTX_5090", "NVIDIA GeForce RTX 5090"):
        assert canonical_gpu(alias) == "RTX 5090"
    for alias in ("RTX 4090", "rtx4090", "4090", "RTX_4090"):
        assert canonical_gpu(alias) == "RTX 4090"


def test_unknown_gpu_rejected():
    from autoslm.providers.base import UnsupportedGpuError, canonical_gpu

    for bad in ("", "TPU v5", "RTX 9090", "Tesla T4"):  # junk + sub-Ampere reject
        with pytest.raises(UnsupportedGpuError):
            canonical_gpu(bad)


def test_providers_for():
    from autoslm.providers.base import providers_for

    # RTX 4090 is provisionable on BOTH substrates; RTX Pro 6000 is RunPod-only
    # (no vast_name); L40S / RTX Pro 4000 are vast-only (no Flash enum member).
    assert providers_for("RTX 4090") == ("runpod", "vast")
    assert providers_for("RTX Pro 6000") == ("runpod",)
    assert providers_for("L40S") == ("vast",)
    assert providers_for("RTX Pro 4000") == ("vast",)


def test_is_validated_per_provider():
    from autoslm.providers.base import is_validated

    assert is_validated("RTX 4090")  # any-provider
    assert is_validated("RTX 4090", "runpod")
    assert not is_validated("L4")
    # RTX 3090 has a RunPod enum member but no live smoke yet
    assert not is_validated("RTX 3090", "runpod")


def test_expanded_gpu_table():
    from autoslm.providers.base import GPU_INFO, canonical_gpu, get_gpu_info, gpu_short

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
    from autoslm.providers.base import min_cuda_modern

    assert min_cuda_modern("RTX 5090") == "13.0"
    assert min_cuda_modern("B200") == "13.0"
    assert min_cuda_modern("RTX Pro 6000") == "13.0"
    assert min_cuda_modern("RTX 4090") == "12.8"
    assert min_cuda_modern("A100 SXM") == "12.8"


def test_cheapest_gpu_policy(monkeypatch):
    from autoslm.providers import base as gpus
    from autoslm.providers.runpod import pricing

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
    from autoslm.providers.base import resolve_gpu_policy

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    monkeypatch.delenv("AUTOSLM_GPU_ALLOW_UNVALIDATED", raising=False)
    # small (12GB) model -> cheapest validated class (A5000); 32GB model -> 5090
    assert resolve_gpu_policy("cheapest", "Qwen/Qwen3.5-0.8B") == "RTX A5000"
    assert resolve_gpu_policy("cheapest", "Qwen/Qwen3.5-9B") == "RTX 5090"
    # concrete names pass through canonicalization
    assert resolve_gpu_policy("rtx5090", "Qwen/Qwen3.5-9B") == "RTX 5090"


def test_config_cheapest_policy_and_unvalidated_gate(monkeypatch):
    from autoslm.schema import ConfigError, spec_from_dict

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    monkeypatch.delenv("AUTOSLM_GPU_ALLOW_UNVALIDATED", raising=False)
    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "environment": {"id": "primeintellect/gsm8k"},
        "train": {"epochs": 1, "seeds": [0], "hf_repo": "owner/runs"},
        "gpu": {"type": "cheapest"},
    }
    spec = spec_from_dict(raw, run_id="x")
    assert spec.gpu.type == "RTX A5000"  # cheapest validated class for a small model
    # class validated on NO provider blocked without opt-in...
    raw["gpu"] = {"type": "L4"}
    with pytest.raises(ConfigError):
        spec_from_dict(raw, run_id="x")
    # ...allowed with the config opt-in
    raw["gpu"] = {"type": "L4", "allow_unvalidated": True}
    spec = spec_from_dict(raw, run_id="x")
    assert spec.gpu.type == "L4"
    # RTX 3090 is validated on vast (validated on ANY provider) so it parses with no
    # opt-in when the provider is left "auto".
    raw["gpu"] = {"type": "RTX 3090"}
    assert spec_from_dict(raw, run_id="x").gpu.type == "RTX 3090"
    # ...but pinning it to runpod (where it is NOT validated) is blocked without opt-in
    raw["gpu"] = {"type": "RTX 3090", "provider": "runpod"}
    with pytest.raises(ConfigError):
        spec_from_dict(raw, run_id="x")
    # ...and allowed with the opt-in
    raw["gpu"] = {"type": "RTX 3090", "provider": "runpod", "allow_unvalidated": True}
    assert spec_from_dict(raw, run_id="x").gpu.type == "RTX 3090"


def test_flash_gpu_enum_members():
    from autoslm.providers.runpod.gpus import flash_gpu

    assert flash_gpu("RTX 5090").name == "NVIDIA_GEFORCE_RTX_5090"
    assert flash_gpu("4090").name == "NVIDIA_GEFORCE_RTX_4090"


def test_gpu_short():
    from autoslm.providers.base import gpu_short

    assert gpu_short("RTX 5090") == "5090"
    assert gpu_short("rtx_4090") == "4090"


def test_config_rejects_unsupported_gpu():
    from autoslm.schema import ConfigError, spec_from_dict

    raw = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "environment": {"id": "primeintellect/gsm8k"},
        "train": {"steps": 1, "seeds": [0], "hf_repo": "owner/runs"},
        "gpu": {"type": "L40S"},  # not a managed GPU class
    }
    with pytest.raises(ConfigError):
        spec_from_dict(raw, run_id="x")


def test_config_defaults_gpu_from_model():
    from autoslm.schema import spec_from_dict

    raw = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "sft",  # 9B is SFT-only (colocated GRPO does not fit 32 GB bf16)
        "environment": {"id": "primeintellect/gsm8k"},
        "train": {"epochs": 1, "seeds": [0], "hf_repo": "owner/runs"},
    }
    spec = spec_from_dict(raw, run_id="x")
    assert spec.gpu.type == "RTX 5090"  # 9B needs >=32GB


def test_build_worker_env():
    from autoslm.providers.runpod.train import build_worker_env
    from autoslm.spec import JobSpec, TrainSpec

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
