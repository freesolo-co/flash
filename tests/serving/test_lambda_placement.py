"""Lambda rents the cheapest qualified single GPU; unqualified or unverifiable shapes are blocked."""

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
        offer("RTX 6000 (24 GB)"),
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
        ("Quadro RTX 6000", 24, 1),
        ("NVIDIA A100-SXM4-80GB", 80, 8),
        ("NVIDIA B200", 180, 2),
        ("NVIDIA H100 MIG", 40, 1),
    ],
)
def test_worker_rejects_wrong_actual_hardware(name, memory, count):
    with pytest.raises(RuntimeError, match="unapproved GPU"):
        validate_gpu(SMALL, name, memory * 2**30, count)


def test_worker_reports_actual_approved_shape():
    assert validate_gpu(SMALL, "NVIDIA H100 80GB HBM3", 79 * 2**30, 1) == "H100"
    assert validate_gpu(SMALL, "NVIDIA A10", 22.06 * 2**30, 1) == "A10"
    assert validate_gpu(SMALL, "NVIDIA B200", 180 * 2**30, 1) == "B200"
    assert validate_gpu(MOE, "NVIDIA H200", 140 * 2**30, 1) == "H200"
    # A10 is only qualified for the small models; 9B needs at least 40 GB.
    with pytest.raises(RuntimeError, match="unapproved GPU"):
        validate_gpu("Qwen/Qwen3.5-9B-Base", "NVIDIA A10", 22.06 * 2**30, 1)


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
    "gpu_1x_a6000": offer("A6000 (48 GB)", 109),
    "gpu_1x_a100_sxm4": offer("A100 (40 GB SXM4)", 199),
    "gpu_1x_a100": offer("A100 (40 GB PCIe)", 179),
    "gpu_1x_a100_80gb_sxm4": offer("A100 (80 GB SXM4)", 249),
}
# Shapes no hosted model may ever rent: Turing has no FP8 kernels, GH200 is arm64.
NEVER = {
    "gpu_1x_rtx6000": offer("RTX 6000 (24 GB)", 69),
    "gpu_1x_gh200": offer("GH200 (96 GB)", 259, architecture="arm64"),
}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (SMALL, "gpu_1x_a6000"),
        ("Qwen/Qwen3.5-4B", "gpu_1x_a6000"),
        ("Qwen/Qwen3.5-9B", "gpu_1x_a6000"),
        ("Qwen/Qwen3.5-9B-Base", "gpu_1x_a6000"),
        ("Qwen/Qwen3.8-27B", "gpu_1x_a100_80gb_sxm4"),
    ],
)
def test_dense_qwen_rents_the_cheapest_qualified_shape_including_ampere(model, expected):
    assert select_region(dict(AMPERE, **NEVER), model) == (expected, "us-west-1")


def test_9b_base_uses_any_available_shape_when_others_are_sold_out():
    import copy

    model = "Qwen/Qwen3.5-9B-Base"
    catalog = copy.deepcopy(dict(AMPERE, **NEVER))
    for name in ("gpu_1x_a6000", "gpu_1x_a100", "gpu_1x_a100_sxm4", "gpu_1x_a100_80gb_sxm4"):
        catalog[name]["regions_with_capacity_available"] = []
    catalog["gpu_1x_b200_sxm6"] = offer("B200 (180 GB SXM6)", 799)
    # A10 (24 GB) is too small for 9B, so the only fit is the B200.
    assert select_region(catalog, model) == ("gpu_1x_b200_sxm6", "us-west-1")
    assert validate_gpu(model, "NVIDIA A100-SXM4-40GB", 40 * 2**30, 1) == "A100-40GB"
    assert validate_gpu(model, "NVIDIA RTX A6000", 48 * 2**30, 1) == "A6000"


@pytest.mark.parametrize("model", LAMBDA_MODELS)
def test_lambda_never_rents_turing_or_arm_shapes(model):
    with pytest.raises(RuntimeError, match="no US capacity"):
        select_region(NEVER, model)
    with pytest.raises(RuntimeError, match="unapproved GPU"):
        validate_gpu(model, "Quadro RTX 6000", 24 * 2**30, 1)


def test_moe_keeps_its_hopper_blackwell_only_placements():
    assert placements(MOE) == ("H200", "B200")
    with pytest.raises(RuntimeError, match="no US capacity"):
        select_region(AMPERE, MOE)


@pytest.mark.parametrize("gpu", ["T4", "RTX6000", "GH200", "L40S"])
def test_policy_cannot_request_an_unknown_lambda_shape(gpu):
    from flash.serving.app.inference_provider import ModelPolicy

    with pytest.raises(ValueError, match="lambda_gpus"):
        ModelPolicy(lambda_gpus=(gpu,))


def test_policy_can_narrow_to_ampere_on_lambda(monkeypatch):
    from flash.serving.app import inference_provider

    model = "Qwen/Qwen3.5-9B-Base"
    monkeypatch.setitem(
        inference_provider.MODEL_POLICIES,
        model,
        inference_provider.ModelPolicy(max_capacity=4, lambda_gpus=("A100-40GB",)),
    )
    assert placements(model) == ("A100-40GB",)
    assert select_region(dict(AMPERE, **NEVER), model)[0] in {"gpu_1x_a100", "gpu_1x_a100_sxm4"}


def test_ampere_engine_profiles_use_weight_only_fp8():
    assert engine_profile(SMALL, "A10")["quantization"] == "fp8_per_tensor"
    assert engine_profile("Qwen/Qwen3.5-9B", "A6000")["quantization"] == "fp8_per_tensor"
    assert engine_profile("Qwen/Qwen3.5-9B-Base", "A100-40GB")["quantization"] == "fp8_per_tensor"
    with pytest.raises(ValueError, match="Unapproved"):
        engine_profile("Qwen/Qwen3.5-9B", "A10")


def test_rejected_offer_does_not_prevent_trying_other_available_shapes():
    model = "Qwen/Qwen3.5-9B-Base"
    catalog = {
        "h100-pcie": offer("H100 (80 GB PCIe)", 250),
        "h100": offer("H100 (80 GB SXM5)", 400),
    }
    excluded = {("h100-pcie", "us-west-1")}
    assert select_region(catalog, model, excluded=excluded) == ("h100", "us-west-1")
