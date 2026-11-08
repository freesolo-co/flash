"""Approved GPU alternatives never silently become smaller or multi-GPU shapes."""

import pytest

from flash.serving.lambda_cloud.config import select_region, validate_gpu
from flash.serving.lambda_cloud.profiles import engine_profile
from flash.serving.src.engine.model_config import engine_overrides_for

SMALL = "Qwen/Qwen3.5-0.8B"
MOE = "Qwen/Qwen3.6-35B-A3B"


def offer(gpu, price=300, count=1, regions=("us-west-1",), architecture="x86_64"):
    return {
        "instance_type": {
            "gpu_description": gpu,
            "price_cents_per_hour": price,
            "specs": {"gpus": count},
            "architecture": architecture,
        },
        "regions_with_capacity_available": [{"name": r} for r in regions],
    }


def test_select_cheapest_available_approved_gpu_without_guessing_instance_ids():
    catalog = {
        "provider-a100": offer("A100 (80 GB SXM4)", 250),
        "provider-h100": offer("H100 (80 GB SXM5)", 400),
    }
    assert select_region(catalog, SMALL) == ("provider-a100", "us-west-1")
    catalog["provider-a100"]["regions_with_capacity_available"] = []
    assert select_region(catalog, SMALL) == ("provider-h100", "us-west-1")
    assert select_region({"b": offer("B200 (180 GB SXM6)")}, MOE) == ("b", "us-west-1")
    assert select_region({"h": offer("H200 (141 GB SXM5)")}, MOE) == ("h", "us-west-1")


@pytest.mark.parametrize(
    "candidate",
    [
        offer("A100 (40 GB SXM4)"),
        offer("A100 (80 GB SXM4)", count=8),
        offer("H100 (80 GB SXM5)", architecture="arm64"),
        offer("H100 (80 GB SXM5)", regions=("asia-south-1",)),
        offer("H100 (80 GB SXM5)", price=0),
        offer("H100"),
        offer("GH200 (96 GB)"),
    ],
)
def test_unapproved_or_unverifiable_shape_is_blocked(candidate):
    with pytest.raises(RuntimeError, match="no US capacity"):
        select_region({"not-approved": candidate}, SMALL)


@pytest.mark.parametrize(
    ("name", "memory", "count"),
    [
        ("NVIDIA A100-SXM4-40GB", 40, 1),
        ("NVIDIA A100-SXM4-80GB", 80, 8),
        ("NVIDIA B200", 180, 1),
        ("NVIDIA H100 MIG", 40, 1),
    ],
)
def test_worker_rejects_wrong_actual_hardware(name, memory, count):
    with pytest.raises(RuntimeError, match="unapproved GPU"):
        validate_gpu(SMALL, name, memory * 2**30, count)


def test_worker_reports_actual_approved_shape():
    assert validate_gpu(SMALL, "NVIDIA A100-SXM4-80GB", 79 * 2**30, 1) == "A100-80GB"
    assert validate_gpu(SMALL, "NVIDIA H100 80GB HBM3", 79 * 2**30, 1) == "H100"
    assert validate_gpu(MOE, "NVIDIA H200", 140 * 2**30, 1) == "H200"


@pytest.mark.parametrize(
    "model", [SMALL, "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base"]
)
def test_ampere_profile_uses_passing_fp8_path_without_changing_admission(model):
    before = engine_overrides_for(model)
    result = engine_profile(model, "A100-80GB")
    assert result["quantization"] == "fp8_per_tensor"
    assert result["serve_model_id"] == model
    assert result["model_revision"] == before["tokenizer_revision"]
    assert "re:.*visual.*" in result["quantization_config"]["ignore"]
    for key in ("max_loras", "max_lora_rank", "max_model_len", "max_num_seqs", "enforce_eager"):
        assert result[key] == before[key]
    assert engine_overrides_for(model) == before


def test_moe_fp8_pin_and_unapproved_profile():
    result = engine_profile(MOE, "H200")
    assert result["serve_model_id"] == MOE + "-FP8"
    assert result["model_revision"] == "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2"
    assert result["quantization"] == "fp8"
    with pytest.raises(ValueError, match="Unapproved"):
        engine_profile(MOE, "A100-80GB")


@pytest.mark.parametrize(
    "model", ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.8-27B", MOE]
)
def test_a10_is_never_selected_for_larger_models(model):
    with pytest.raises(RuntimeError, match="no US capacity"):
        select_region({"a10": offer("A10 (24 GB PCIe)", 129)}, model)


def test_a10_is_cheapest_small_model_option():
    catalog = {"a10": offer("A10 (24 GB PCIe)", 129), "h100": offer("H100 (80 GB SXM5)", 429)}
    for model in (SMALL, "Qwen/Qwen3.5-4B"):
        assert select_region(catalog, model) == ("a10", "us-west-1")
        assert validate_gpu(model, "NVIDIA A10", 22.06 * 2**30, 1) == "A10"
