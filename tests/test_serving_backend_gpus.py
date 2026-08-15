"""Serving GPU table and fit estimator for the self-hosted Modal backend."""

from __future__ import annotations

import pytest

from flash.core.catalog import MODELS
from flash.serve.backend.gpus import (
    MODAL_GPUS,
    MODAL_GPUS_BY_NAME,
    cheapest_fitting,
    default_gpu,
    estimate_fit,
    recommend,
    serving_dtype,
)


def test_every_catalog_serving_gpu_is_a_known_modal_card():
    """The catalog's production choice must be requestable from Modal by that exact string.

    This is what lets the generated app default to Freesolo's validated configuration instead of
    something computed. A catalog entry naming a card Modal does not rent would silently fall back.
    """
    for info in MODELS.values():
        if info.serving is None:
            continue
        assert info.serving.gpu in MODAL_GPUS_BY_NAME, (
            f"{info.id} serves on {info.serving.gpu!r}, which is not a Modal GPU class"
        )


def test_catalog_gpu_fits_every_model():
    """The estimator must agree with production on all six models.

    Freesolo runs these exact model/card pairs, so a card the estimator rejects would mean the
    estimate is wrong, not that production is. This is the strongest available check short of a
    real GPU: independent arithmetic reproducing six validated choices.
    """
    for info in MODELS.values():
        gpu = default_gpu(info)
        if gpu is None:
            continue
        fit = estimate_fit(info, gpu)
        assert fit.fits, (
            f"{info.id} does not fit its own production card {gpu.name}: "
            f"needs {fit.total_gb:.1f}G of {gpu.vram_gb}G"
        )
        assert fit.is_catalog_default


def test_pre_ampere_cards_are_excluded():
    """T4 must never be offered.

    vLLM DOWNGRADES bf16 to fp16 on pre-Ampere silicon rather than refusing, so a T4 would serve
    at quietly degraded quality with no error anywhere for the user to see. Excluded by compute
    capability rather than by name, so a future pre-Ampere card cannot reappear.
    """
    offered = {gpu.name for gpu in MODAL_GPUS}
    assert "T4" not in offered
    for fit in recommend(MODELS["Qwen/Qwen3.5-4B"]):
        assert fit.gpu.sm >= 80


def test_reproduces_the_production_h200_lora_canary():
    """The 35B's real-GPU canary is the one hard serving measurement available to check against.

    Production recorded that six hot rank-64 adapters plus 32k context fit the 141 GB H200 and that
    EIGHT overflow it. An estimator that cannot separate those two is not measuring the thing that
    decides the card. Both corrections are load-bearing here: quoting this MoE at fp8 halves its
    weights on paper, and sizing against nameplate VRAM instead of vLLM's utilization budget hides
    the overflow -- either mistake alone makes eight adapters look fine.
    """
    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    h200 = MODAL_GPUS_BY_NAME["H200"]
    assert estimate_fit(info, h200, max_loras=6, max_lora_rank=64).fits
    assert not estimate_fit(info, h200, max_loras=8, max_lora_rank=64).fits


def test_serving_dtype_uses_explicit_catalog_quantization():
    dense = MODELS["Qwen/Qwen3.5-4B"]
    moe = MODELS["Qwen/Qwen3.6-35B-A3B"]
    assert dense.serving.quantization == "fp8"
    assert moe.serving.quantization is None
    assert serving_dtype(dense) == "fp8"
    assert serving_dtype(moe) == "bf16"
    assert estimate_fit(MODELS["Qwen/Qwen3.6-35B-A3B"], MODAL_GPUS_BY_NAME["H200"]).dtype == "bf16"


def test_fit_is_measured_against_the_engine_budget_not_nameplate_vram():
    """vLLM claims gpu_memory_utilization of the card and never gets the rest.

    Sizing on nameplate VRAM overstates every card by ~10%, which is exactly the margin the tight
    configurations live in.
    """
    fit = estimate_fit(MODELS["Qwen/Qwen3.5-4B"], MODAL_GPUS_BY_NAME["L4"])
    assert fit.budget_gb < fit.gpu.vram_gb
    assert fit.free_gb == pytest.approx(fit.budget_gb - fit.total_gb, rel=1e-6)


def test_bf16_needs_about_twice_the_weight_memory_of_fp8():
    info = MODELS["Qwen/Qwen3.5-4B"]
    fp8 = estimate_fit(info, MODAL_GPUS_BY_NAME["L4"], dtype="fp8")
    bf16 = estimate_fit(info, MODAL_GPUS_BY_NAME["L4"], dtype="bf16")
    assert bf16.weights_gb == pytest.approx(fp8.weights_gb * 2, rel=0.01)


def test_fp8_is_what_makes_4b_fit_the_l4():
    """The dtype decision is load-bearing, not a preference.

    Production serves 4B on an L4 by online-quantizing the public bf16 checkpoint to fp8 at load.
    In bf16 the same model does not fit that card, so an estimator defaulting to bf16 would send
    self-hosters to a needlessly expensive GPU.
    """
    info = MODELS["Qwen/Qwen3.5-4B"]
    l4 = MODAL_GPUS_BY_NAME["L4"]
    assert estimate_fit(info, l4, dtype="fp8", kv_dtype="fp8").fits
    assert not estimate_fit(info, l4, dtype="bf16", kv_dtype="bf16").fits


def test_largest_model_only_fits_the_largest_cards():
    fits = [fit.gpu.name for fit in recommend(MODELS["Qwen/Qwen3.6-35B-A3B"]) if fit.fits]
    assert fits == ["H200", "B200"]


def test_headroom_degrades_as_the_card_shrinks():
    info = MODELS["Qwen/Qwen3.5-9B"]
    order = ["no", "tight", "good", "ample"]
    small = estimate_fit(info, MODAL_GPUS_BY_NAME["L4"])
    large = estimate_fit(info, MODAL_GPUS_BY_NAME["H200"])
    assert order.index(small.headroom) < order.index(large.headroom)


def test_negative_headroom_reports_no_and_does_not_fit():
    fit = estimate_fit(MODELS["Qwen/Qwen3.6-27B"], MODAL_GPUS_BY_NAME["L4"])
    assert fit.free_gb < 0
    assert fit.headroom == "no"
    assert not fit.fits


def test_longer_context_costs_kv_memory():
    info = MODELS["Qwen/Qwen3.5-4B"]
    short = estimate_fit(info, MODAL_GPUS_BY_NAME["H100"], context_len=4096)
    long = estimate_fit(info, MODAL_GPUS_BY_NAME["H100"], context_len=32768)
    assert long.kv_gb > short.kv_gb
    assert long.free_gb < short.free_gb


def test_lora_pool_scales_with_hot_adapter_count_and_rank():
    """vLLM PRE-ALLOCATES max_loras x max_lora_rank at engine init, so both are linear VRAM levers.

    Sizing on the adapters actually loaded would under-count and recommend a card that OOMs at
    startup, before any adapter is registered.
    """
    info = MODELS["Qwen/Qwen3.5-4B"]
    gpu = MODAL_GPUS_BY_NAME["H100"]
    base = estimate_fit(info, gpu, max_loras=4, max_lora_rank=32)
    more = estimate_fit(info, gpu, max_loras=16, max_lora_rank=32)
    higher = estimate_fit(info, gpu, max_loras=4, max_lora_rank=128)
    assert more.lora_gb == pytest.approx(base.lora_gb * 4, rel=0.01)
    assert higher.lora_gb == pytest.approx(base.lora_gb * 4, rel=0.01)


def test_speed_bands_separate_the_affordable_cards():
    """The speed column has to inform the choice a self-hoster is actually making.

    B200 has ~27x L4's bandwidth, so a linear scale collapses every affordable card into one band
    and the column becomes decoration.
    """
    speeds = {fit.gpu.name: fit.speed for fit in recommend(MODELS["Qwen/Qwen3.5-4B"])}
    assert speeds["L40S"] != speeds["L4"]
    assert speeds["H100"] != speeds["L40S"]


def test_fp8_native_support_is_reported_not_assumed():
    """A100 is sm80: it serves fp8 through a weight-only fallback, not native tensor cores."""
    info = MODELS["Qwen/Qwen3.5-4B"]
    assert estimate_fit(info, MODAL_GPUS_BY_NAME["L4"], dtype="fp8").fp8_native
    assert not estimate_fit(info, MODAL_GPUS_BY_NAME["A100-80GB"], dtype="fp8").fp8_native
    # in bf16 the question does not arise, so it must not be reported as a limitation.
    assert estimate_fit(info, MODAL_GPUS_BY_NAME["A100-80GB"], dtype="bf16").fp8_native


def test_recommendations_are_ordered_cheapest_first():
    fits = recommend(MODELS["Qwen/Qwen3.5-2B"])
    prices = [fit.gpu.usd_hr for fit in fits]
    assert prices == sorted(prices)


def test_cheapest_fitting_picks_the_first_card_that_fits():
    fits = recommend(MODELS["Qwen/Qwen3.5-9B"])
    pick = cheapest_fitting(fits)
    assert pick is not None
    assert pick.gpu.name == "L40S"
    assert all(not fit.fits for fit in fits[: fits.index(pick)])


def test_model_without_a_serving_entry_has_no_default_gpu():
    """No serving entry means no validated choice, so the caller must fall back and say so."""

    class _Bare:
        params_b = 1.0
        serving = None

    assert default_gpu(_Bare()) is None
