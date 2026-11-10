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


def test_select_cheapest_available_approved_gpu_without_guessing_instance_ids(monkeypatch):
    from flash.serving.app.inference_provider import MODEL_POLICIES, ModelPolicy

    monkeypatch.setitem(MODEL_POLICIES, SMALL, ModelPolicy())
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
    assert validate_gpu(SMALL, "NVIDIA A10", 22.06 * 2**30, 1) == "A10"
    with pytest.raises(RuntimeError, match="unapproved GPU"):
        validate_gpu(SMALL, "NVIDIA H100 80GB HBM3", 79 * 2**30, 1)
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


@pytest.mark.parametrize("gpu", ["H200", "B200"])
def test_moe_preserves_qualified_bf16_lora_profile(gpu):
    result = engine_profile(MOE, gpu)
    assert result == engine_overrides_for(MOE)
    assert result["serve_model_id"] == MOE
    assert result["quantization"] is None
    assert result["enforce_eager"] is False
    assert result["max_loras"] == 6
    assert result["max_lora_rank"] == 64
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


@pytest.mark.parametrize("a100", ["A100 (40 GB SXM4)", "A100 (80 GB SXM4)"])
def test_nine_base_uses_available_a100_or_h100(a100):
    model = "Qwen/Qwen3.5-9B-Base"
    catalog = {"a100": offer(a100, 200), "h100": offer("H100 (80 GB SXM5)", 400)}
    assert select_region(catalog, model) == ("a100", "us-west-1")
    catalog["a100"]["regions_with_capacity_available"] = []
    assert select_region(catalog, model) == ("h100", "us-west-1")
    catalog["h100"]["regions_with_capacity_available"] = []
    with pytest.raises(RuntimeError, match="no US capacity"):
        select_region(catalog, model)


def test_small_model_does_not_fall_back_to_more_expensive_hardware():
    with pytest.raises(RuntimeError, match="no US capacity"):
        select_region({"h100": offer("H100 (80 GB SXM5)", 400)}, SMALL)


def test_rejected_offer_does_not_prevent_trying_other_available_shapes():
    model = "Qwen/Qwen3.5-9B-Base"
    catalog = {"a100": offer("A100 (40 GB SXM4)", 200), "h100": offer("H100 (80 GB SXM5)", 400)}
    assert select_region(catalog, model, excluded={("a100", "us-west-1")}) == ("h100", "us-west-1")
