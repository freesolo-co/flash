"""FP8 wiring is structural, so pin what the engine is actually built with.

``engine_boot.engine_args_for`` returns the AsyncEngineArgs kwargs as a plain dict, so these assert
the resolved values for a real catalog model rather than matching source text: quantization and KV
cache stay overridable per model, calculate_kv_scales stays off, and a build-specific arg is
forwarded only when this vLLM exposes it.

The image-level ``VLLM_*`` env still lives in ``modal_app.py`` and is pinned by source below,
because it is literal image configuration with no function to call.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODAL_APP = Path(__file__).resolve().parents[2] / "flash" / "serving" / "app" / "modal_app.py"

# Every catalog model currently ships a pre-quantized serve_model_id, so the online-quantization
# default is exercised by clearing it rather than by picking a different model.
MODEL = "Qwen/Qwen3.5-9B"


@pytest.fixture
def engine_args():
    """``engine_args_for`` for a catalog model, with ad-hoc override tweaks applied on top.

    The conftest installs the vLLM stub at import, so this resolves against the same
    ``AsyncEngineArgs`` field set the engine would see on a build missing newer args.
    """
    from flash.serving.src.engine.boot import engine_args_for
    from flash.serving.src.engine.model_config import engine_overrides_for
    from flash.serving.src.store import settings as cfg

    def _for(model: str, **overrides):
        resolved = {**engine_overrides_for(model), **overrides}
        return engine_args_for(model, resolved, cfg)

    return _for


def test_engine_args_wires_fp8_weights_and_kv(engine_args) -> None:
    from flash.serving.src.store import settings as cfg

    # FP8 weights: with no pre-quantized checkpoint the base is online-quantized to cfg.QUANTIZATION
    # and serves itself.
    kwargs = engine_args(MODEL, serve_model_id=None)
    assert kwargs["quantization"] == cfg.QUANTIZATION
    assert kwargs["model"] == MODEL
    # FP8 KV cache, default cfg.KV_CACHE_DTYPE.
    assert kwargs["kv_cache_dtype"] == cfg.KV_CACHE_DTYPE


def test_prequantized_base_passes_no_quantization(engine_args) -> None:
    # A pre-quantized serve_model_id checkpoint is served directly and vLLM auto-detects its FP8, so
    # passing quantization on top would re-quantize an already-quantized checkpoint.
    kwargs = engine_args(MODEL)
    assert kwargs["model"] == "Freesolo-Co/Qwen3.5-9B-FP8"
    assert kwargs["quantization"] is None


def test_a_model_can_opt_out_to_bf16_explicitly(engine_args) -> None:
    # `.get` (not a truthiness test) is load-bearing: a model overriding quantization to None must
    # reach vLLM as None (the documented 35B H200 bf16 fallback), not fall back to the global FP8
    # default. serve_model_id is cleared first, or the default would already be None and this could
    # not tell the two apart.
    from flash.serving.src.store import settings as cfg

    assert engine_args(MODEL, serve_model_id=None)["quantization"] == cfg.QUANTIZATION
    opted_out = engine_args(MODEL, serve_model_id=None, quantization=None)
    assert opted_out["quantization"] is None


def test_engine_never_enables_calculate_kv_scales(engine_args) -> None:
    # Warmup-estimated KV scales corrupt the Qwen3 GDN-hybrid's recurrent state — uncalibrated
    # dynamic e4m3 is the safe path, so the flag must never be passed to the engine.
    assert "calculate_kv_scales" not in engine_args(MODEL), (
        "do not pass calculate_kv_scales — it corrupts the GDN-hybrid KV scales"
    )


def test_moe_backend_override_is_guarded(engine_args, monkeypatch, capsys) -> None:
    # The catalog currently leaves the 35B MoE backend unset/auto, but the engine keeps a guarded
    # forwarding path for future canaries or emergency overrides. It must never crash an older build
    # on an unknown kwarg.
    assert engine_args(MODEL, moe_backend="triton")["moe_backend"] == "triton"

    monkeypatch.setattr("flash.serving.src.engine.boot._async_engine_arg_names", lambda _t: set())
    assert "moe_backend" not in engine_args(MODEL, moe_backend="triton")
    assert "no moe_backend arg" in capsys.readouterr().out


def test_max_num_batched_tokens_override_is_guarded(engine_args, monkeypatch, capsys) -> None:
    # The 35B needs max_num_batched_tokens=4096 because its attention block size (2096 tokens)
    # exceeds vLLM's 2048 default. Forward it only when this AsyncEngineArgs build exposes the field.
    assert engine_args(MODEL, max_num_batched_tokens=4096)["max_num_batched_tokens"] == 4096

    monkeypatch.setattr("flash.serving.src.engine.boot._async_engine_arg_names", lambda _t: set())
    assert "max_num_batched_tokens" not in engine_args(MODEL, max_num_batched_tokens=4096)
    assert "no max_num_batched_tokens arg" in capsys.readouterr().out


def test_reasoning_parser_forwarding_fails_closed_on_incompatible_vllm(
    engine_args, monkeypatch
) -> None:
    assert engine_args(MODEL, reasoning_parser="qwen3")["reasoning_parser"] == "qwen3"
    # An unset parser must not be forwarded at all, so an older build never sees the kwarg.
    assert "reasoning_parser" not in engine_args(MODEL, reasoning_parser=None)

    def _incompatible(*_args, **_kwargs):
        raise RuntimeError("vllm build cannot forward reasoning_parser")

    monkeypatch.setattr(
        "flash.serving.src.engine.boot._require_reasoning_api_compatibility", _incompatible
    )
    with pytest.raises(RuntimeError):
        engine_args(MODEL, reasoning_parser="qwen3")


def test_image_keeps_deepgemm_out_of_moe_backend_race() -> None:
    # The validated 35B path leaves moe_backend unset/auto; VLLM_MOE_USE_DEEP_GEMM=0 keeps the image
    # off the DeepGEMM path that previously crash-looped and lets vLLM select the working backend.
    assert '"VLLM_MOE_USE_DEEP_GEMM": "0"' in MODAL_APP.read_text()


def test_image_persists_vllm_compile_cache_on_the_volume() -> None:
    # VLLM_CACHE_ROOT defaults to ~/.cache/vllm (vllm/envs.py) -- ephemeral container storage -- so
    # leaving it unset made every scale-from-zero recompile the model: vLLM writes torch.compile
    # artifacts under VLLM_CACHE_ROOT/torch_compile_cache/ (vllm/compilation/backends.py). It must
    # resolve to the SAME persistent volume mount as the weight caches, not merely be set: pointing
    # it anywhere off HOSTING_CACHE_MOUNT silently restores the recompile-every-cold-start behavior.
    src = MODAL_APP.read_text()
    tree = ast.parse(src)
    roots = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=False)
        if isinstance(key, ast.Constant) and key.value == "VLLM_CACHE_ROOT"
        for node in [value]
    ]
    assert roots, "VLLM_CACHE_ROOT must be set in the image env"
    (root,) = roots
    # An f-string interpolating HOSTING_CACHE_MOUNT is the only accepted form, matching HF_HUB_CACHE.
    assert isinstance(root, ast.JoinedStr), "VLLM_CACHE_ROOT must interpolate HOSTING_CACHE_MOUNT"
    names = {
        node.id
        for part in root.values
        if isinstance(part, ast.FormattedValue)
        for node in ast.walk(part.value)
        if isinstance(node, ast.Name)
    }
    assert "HOSTING_CACHE_MOUNT" in names, (
        "VLLM_CACHE_ROOT must live under the persistent volume mount, or the compile cache is "
        "discarded on every cold start"
    )


def test_image_disables_hf_xet_for_deterministic_download() -> None:
    # The HF Xet/CAS download path intermittently fails engine init with "CAS Client Error: Format
    # error: I/O error: error decoding response body" (huggingface_hub xet_get inside vLLM
    # download_weights_from_hf) and crash-looped 27B FP8 serving boots; HF_HUB_DISABLE_XET=1 forces
    # the deterministic HTTP downloader (verified: a matched re-serve with it set booted and served).
    src = MODAL_APP.read_text()
    assert '"HF_HUB_DISABLE_XET": "1"' in src
    # Superseded by disabling Xet: the high-performance Xet env entry is moot once the path is off.
    assert '"HF_XET_HIGH_PERFORMANCE"' not in src
