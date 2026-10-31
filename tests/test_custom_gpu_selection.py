"""Custom runs share training sizing and allocation, including rank-dependent MoE state."""

from types import SimpleNamespace

import pytest

from flash.core.catalog import MODELS
from flash.providers.core import allocator
from flash.providers.core.base import GPU_INFO, Candidate, UnsupportedGpuError, provisional_gpu
from flash.schema import ConfigError, spec_from_dict

MOE = "Qwen/Qwen3.6-35B-A3B"


def custom_spec(model=MOE, *, rank=64, context=4096, gpu=None):
    return spec_from_dict(
        {
            "model": model,
            "algorithm": "custom",
            "custom": {"file": "algorithm.py"},
            "train": {"lora_rank": rank, "max_context_tokens": context},
            "gpu": gpu or {},
        }
    )


@pytest.fixture
def modal_capacity(monkeypatch):
    candidates = [
        Candidate("modal", gpu.name, gpu.hourly_usd, gpu.vram_gb)
        for gpu in GPU_INFO.values()
        if gpu.validated
    ]
    monkeypatch.setattr(allocator, "available_providers", lambda: ("modal",))
    monkeypatch.setattr(
        allocator,
        "get_provider",
        lambda _: SimpleNamespace(live_candidates=lambda *args: candidates),
    )
    return candidates


def test_moe_rank_64_routes_to_b200_and_accounts_for_every_expert(modal_capacity):
    low = custom_spec(rank=16)
    high = custom_spec(rank=64)
    low_need = allocator.required_vram_gb(MOE, "custom", train=low.train)
    high_need = allocator.required_vram_gb(MOE, "custom", train=high.train)
    # Two fused routed tensors per layer: gate/up (2048 -> 1024), down (512 -> 2048).
    # Every expert has adapters, even though only eight experts are active per token.
    expert_params_per_rank = 40 * 256 * ((2048 + 1024) + (512 + 2048))
    state_growth_gb = (64 - 16) * expert_params_per_rank * 16 / 1e9
    assert high_need - low_need >= state_growth_gb
    assert high_need > GPU_INFO["H200"].vram_gb
    assert high_need <= GPU_INFO["B200"].vram_gb
    for spec, expected in ((low, "H200"), (high, "B200")):
        result = allocator.allocate(MOE, "custom", train=spec.train)
        assert result.gpu == expected
        assert result.gpu_count == 1
        assert provisional_gpu(MOE, "custom", train=spec.train) == expected


@pytest.mark.parametrize("gpu", ["RTX Pro 6000", "H200", ["B200", "H200"]])
def test_rank_64_rejects_undersized_pins_and_fallbacks_before_allocation(gpu):
    with pytest.raises(ConfigError, match="requires at least"):
        custom_spec(gpu={"type": gpu})


def test_direct_allocation_cannot_rent_an_undersized_pin(modal_capacity):
    spec = custom_spec()
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate(MOE, "custom", train=spec.train, gpu_type="RTX Pro 6000")
    modal_capacity[:] = [c for c in modal_capacity if c.gpu != "B200"]
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate(MOE, "custom", train=spec.train)


def test_pinned_worker_revision_retains_rank_dependent_expert_state(monkeypatch, modal_capacity):
    from flash.engine.plan import vram

    info = MODELS[MOE]
    monkeypatch.setattr(
        vram,
        "fetch_hf_model_geometry",
        lambda *args, **kwargs: (
            info.params_b,
            info.vocab_size,
            info.hidden_size,
            info.num_layers,
            info.num_attention_heads,
        ),
    )
    spec = custom_spec()
    result = allocator.allocate(MOE, "custom", train=spec.train, model_revision="custom-test")
    assert result.gpu == "B200"


def test_context_sizes_full_logits_instead_of_sft_chunked_loss(modal_capacity):
    model = "Qwen/Qwen3.5-0.8B"
    short = custom_spec(model, rank=16, context=512)
    long = custom_spec(model, rank=16, context=4096)
    short_need = allocator.required_vram_gb(model, "custom", train=short.train)
    long_need = allocator.required_vram_gb(model, "custom", train=long.train)
    full_logits_growth = (4096 - 512) * MODELS[model].vocab_size * 16 / 1e9
    assert long_need - short_need >= full_logits_growth
    assert allocator.allocate(model, "custom", train=short.train).gpu == "L4"
    assert allocator.allocate(model, "custom", train=long.train).gpu == "L40S"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("rank", range(1, 65))
@pytest.mark.parametrize("context", [2, 512, 1024, 2048, 4096])
def test_catalog_custom_runs_choose_cheapest_fitting_single_card(model, rank, context, monkeypatch):
    from flash.engine.plan import vram
    from flash.providers.modal.pricing import hourly_rate
    from flash.providers.modal.provider import ModalProvider

    # Exercise the production candidate filtering and prices, without renting hardware.
    monkeypatch.setattr(allocator, "available_providers", lambda: ("modal",))
    monkeypatch.setattr(allocator, "get_provider", lambda _: ModalProvider())
    info = MODELS[model]
    monkeypatch.setattr(
        vram,
        "fetch_hf_model_geometry",
        lambda *args, **kwargs: (
            info.params_b,
            info.vocab_size,
            info.hidden_size,
            info.num_layers,
            info.num_attention_heads,
        ),
    )
    spec = custom_spec(model, rank=rank, context=context)
    need = allocator.required_vram_gb(model, "custom", train=spec.train)
    # Independently bound resident BF16 weights plus FP32 adapters, gradients and AdamW state.
    adapter_params = rank * sum((i + o) * count for i, o, count in info.lora_target_shapes)
    assert need >= info.params_b * 2 + adapter_params * 16 / 1e9
    expected = min(
        (gpu for gpu in GPU_INFO.values() if gpu.validated and gpu.vram_gb >= need),
        key=lambda gpu: hourly_rate(gpu.name),
    )
    for revision in ("", "custom-matrix-revision"):
        result = allocator.allocate(model, "custom", train=spec.train, model_revision=revision)
        assert result.gpu == expected.name
        assert result.gpu_count == 1
        assert (
            provisional_gpu(model, "custom", train=spec.train, model_revision=revision)
            == expected.name
        )
    for gpu in GPU_INFO.values():
        if gpu.vram_gb < need:
            with pytest.raises(ConfigError, match="requires at least"):
                custom_spec(model, rank=rank, context=context, gpu={"type": gpu.name})
            with pytest.raises(UnsupportedGpuError):
                allocator.allocate(model, "custom", train=spec.train, gpu_type=gpu.name)
        else:
            pinned = custom_spec(model, rank=rank, context=context, gpu={"type": gpu.name})
            assert (
                allocator.allocate(
                    model, "custom", train=pinned.train, gpu_type=pinned.gpu.type
                ).gpu
                == gpu.name
            )


def test_custom_sizing_defaults_match_parsed_worker_defaults():
    spec = spec_from_dict({"model": MOE, "algorithm": "custom", "custom": {"file": "algorithm.py"}})
    assert allocator.required_vram_gb(MOE, "custom") == allocator.required_vram_gb(
        MOE, "custom", train=spec.train
    )
