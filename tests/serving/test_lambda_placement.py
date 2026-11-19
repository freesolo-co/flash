"""Approved GPU alternatives never silently become smaller or multi-GPU shapes."""

import pytest

from flash.serving.lambda_cloud.config import placements, select_region, validate_gpu
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
        "provider-h100-pcie": offer("H100 (80 GB PCIe)", 250),
        "provider-h100": offer("H100 (80 GB SXM5)", 400),
    }
    assert select_region(catalog, SMALL) == ("provider-h100-pcie", "us-west-1")
    catalog["provider-h100-pcie"]["regions_with_capacity_available"] = []
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
    assert validate_gpu(SMALL, "NVIDIA H100 80GB HBM3", 79 * 2**30, 1) == "H100"
    with pytest.raises(RuntimeError, match="unapproved GPU"):
        validate_gpu(SMALL, "NVIDIA A10", 22.06 * 2**30, 1)
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


LAMBDA_MODELS = [
    SMALL,
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.8-27B",
    MOE,
]
AMPERE = {
    "gpu_1x_a10": offer("A10 (24 GB PCIe)", 129),
    "gpu_1x_a100_sxm4": offer("A100 (40 GB SXM4)", 199),
    "gpu_1x_a100": offer("A100 (40 GB PCIe)", 179),
    "gpu_1x_a100_80gb_sxm4": offer("A100 (80 GB SXM4)", 249),
}


@pytest.mark.parametrize("model", LAMBDA_MODELS)
def test_lambda_never_rents_a10_or_a100(model):
    """Even when only cheap Ampere offers have capacity, no Lambda VM is created."""
    assert not {"A10", "A100-40GB", "A100-80GB"} & set(placements(model))
    with pytest.raises(RuntimeError, match="no US capacity"):
        select_region(AMPERE, model)
    for name, gib in [
        ("NVIDIA A10", 22.06),
        ("NVIDIA A100-SXM4-40GB", 40),
        ("NVIDIA A100-SXM4-80GB", 80),
    ]:
        with pytest.raises(RuntimeError, match="unapproved GPU"):
            validate_gpu(model, name, gib * 2**30, 1)


@pytest.mark.parametrize("model", [m for m in LAMBDA_MODELS if m != MOE])
def test_dense_qwen_picks_h100_even_when_ampere_is_cheaper(model):
    catalog = dict(AMPERE, gpu_1x_h100_pcie=offer("H100 (80 GB PCIe)", 249))
    assert select_region(catalog, model) == ("gpu_1x_h100_pcie", "us-west-1")
    assert validate_gpu(model, "NVIDIA H100 PCIe", 79 * 2**30, 1) == "H100"


@pytest.mark.parametrize("gpu", ["A10", "A100-40GB", "A100-80GB"])
def test_policy_cannot_request_ampere_on_lambda(gpu):
    from flash.serving.app.inference_provider import ModelPolicy

    with pytest.raises(ValueError, match="lambda_gpus"):
        ModelPolicy(lambda_gpus=(gpu,))


def test_ampere_stays_qualified_for_modal_engine_profiles():
    # Modal A10G/A100-80GB fallback containers still build their engine profile.
    assert engine_profile(SMALL, "A10")["quantization"] == "fp8_per_tensor"
    assert engine_profile("Qwen/Qwen3.5-9B", "A100-80GB")["quantization"] == "fp8_per_tensor"


def test_rejected_offer_does_not_prevent_trying_other_available_shapes():
    model = "Qwen/Qwen3.5-9B-Base"
    catalog = {
        "h100-pcie": offer("H100 (80 GB PCIe)", 250),
        "h100": offer("H100 (80 GB SXM5)", 400),
    }
    excluded = {("h100-pcie", "us-west-1")}
    assert select_region(catalog, model, excluded=excluded) == ("h100", "us-west-1")
