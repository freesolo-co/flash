"""Cost estimator: model-size facts. No network on the managed path.

A curated model is sized from the catalog's ``params_b`` stat, read directly (no parsing of the
``params`` display string) and with no network call. For the MoE the per-token FLOPs/step-time term
reads the smaller ``active_params_b`` while memory/size terms (VRAM, disk, download) keep total
``params_b``.

An UNCATALOGED model reaches the estimator only under ``model_policy="allow"`` (self-hosted planes),
and is sized from HF safetensors metadata via the same ``resolve_params_b`` the VRAM path uses, so
cost and sizing cannot disagree. Unsizeable by both = rejected; the quote never guesses.
"""

from __future__ import annotations

import pytest

from flash.catalog import MODELS
from flash.cost.facts import active_params_b, download_weight_gb, model_quant, total_params_b


@pytest.mark.parametrize("model_id", list(MODELS))
def test_total_params_is_the_catalog_stat(model_id):
    assert total_params_b(model_id) == pytest.approx(MODELS[model_id].params_b)


@pytest.mark.parametrize("model_id", list(MODELS))
def test_active_params_defaults_to_total_for_dense(model_id):
    # A dense model (active_params_b unset) bills FLOPs against its full param count.
    info = MODELS[model_id]
    if info.active_params_b:
        pytest.skip(f"{model_id} is an MoE (active_params_b set)")
    assert active_params_b(model_id) == pytest.approx(info.params_b)


def test_active_params_uses_the_active_count_for_the_moe():
    # The MoE's per-token FLOPs size is the ~3B active count, far below its 35B total — so cost/step
    # time isn't ~10x overstated. Memory/size terms still use the 35B total (asserted just below).
    moe = "Qwen/Qwen3.6-35B-A3B"
    assert active_params_b(moe) == pytest.approx(3.0)
    assert active_params_b(moe) < total_params_b(moe)


def test_moe_memory_terms_still_use_total_params():
    # Download (and VRAM/disk, which read total_params_b too) size the FULL checkpoint — all experts
    # are materialized on the GPU — so they must NOT shrink to the active count.
    moe = "Qwen/Qwen3.6-35B-A3B"
    assert download_weight_gb(moe) == pytest.approx(total_params_b(moe) * 2.0)
    assert download_weight_gb(moe) == pytest.approx(70.0)


def test_quant_lookup():
    assert model_quant("Qwen/Qwen3.5-9B") == "bf16"
    assert model_quant("Qwen/Qwen3.5-4B") == "bf16"


def test_download_weight_gb_is_total_params_bf16():
    # Download is always the full bf16 checkpoint (2 bytes/param).
    nine = "Qwen/Qwen3.5-9B"
    assert download_weight_gb(nine) == pytest.approx(total_params_b(nine) * 2.0)


def test_an_unsizeable_model_is_rejected():
    # Fail closed: uncataloged AND unsizeable by HF (the autouse _offline fixture returns None)
    # raises rather than pricing an unknown model as free.
    with pytest.raises(ValueError, match="could not size model"):
        total_params_b("nobody/never-heard-of-it")
    with pytest.raises(ValueError, match="could not size model"):
        active_params_b("nobody/never-heard-of-it")


def test_an_open_policy_model_is_priced_from_huggingface(monkeypatch):
    """An authorized open-model run must be PRICEABLE, not just parseable.

    Regression: `model_policy="allow"` parsed and authorized, then died in prepare_job ->
    estimate_for_spec -> total_params_b, which was catalog-only. The open-model path was
    reachable and unusable -- a quoting error, not a policy one.
    """
    import flash.engine.vram as vram

    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda _m, **_k: 8.03, raising=False)
    assert total_params_b("meta-llama/Llama-3.1-8B") == pytest.approx(8.03)
    # No curated MoE routing data for an unlisted model, so it prices as dense (active == total).
    assert active_params_b("meta-llama/Llama-3.1-8B") == pytest.approx(8.03)
    assert download_weight_gb("meta-llama/Llama-3.1-8B") == pytest.approx(16.06)


def test_a_catalog_model_is_never_sized_over_the_network(monkeypatch):
    """The curated entry answers first: adding the HF fallback must not put a network call on the
    managed path, where every model is cataloged."""
    import flash.engine.vram as vram

    def _boom(*_args, **_kwargs):
        raise AssertionError("a catalog model must be sized from the catalog, with no HF call")

    monkeypatch.setattr(vram, "fetch_hf_params_b", _boom, raising=False)
    assert total_params_b("Qwen/Qwen3.5-9B") == pytest.approx(MODELS["Qwen/Qwen3.5-9B"].params_b)
