"""FP8 wiring is structural, so lock exact engine-arg expressions by parsing the source.

Functional tests execute ``_load()`` with a vLLM stub; this file keeps AST-level guards around the
AsyncEngineArgs call so quantization/KV-cache settings stay overridable per model and
calculate_kv_scales stays off.

The engine (``_LoraEngineImpl._load`` and its AsyncEngineArgs wiring) lives in the
``src.lora_engine`` module; the image-level ``VLLM_*`` env still lives in ``modal_app.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

MODAL_APP = Path(__file__).resolve().parents[2] / "flash" / "serving" / "modal_app.py"
LORA_ENGINE = Path(__file__).resolve().parents[2] / "flash" / "serving" / "src" / "lora_engine.py"


def _engine_args_call() -> ast.Call:
    tree = ast.parse(LORA_ENGINE.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AsyncEngineArgs"
        ):
            return node
    raise AssertionError("AsyncEngineArgs(...) call not found in lora_engine.py")


def _kwargs(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def _is_ov_get_defaulting_to(value: ast.expr, *, cfg_attr: str) -> bool:
    """True if ``value`` is ``overrides.get("<key>", cfg.<cfg_attr>)`` — a per-model override that
    falls back to the baked-in global. Using ``.get`` (not ``in``) is what lets a model override
    the value to ``None`` for pre-quantized checkpoint auto-detection or an explicit bf16 fallback."""
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)):
        return False
    if value.func.attr != "get" or not isinstance(value.func.value, ast.Name):
        return False
    if value.func.value.id != "overrides" or len(value.args) != 2:
        return False
    default = value.args[1]
    return (
        isinstance(default, ast.Attribute)
        and default.attr == cfg_attr
        and isinstance(default.value, ast.Name)
        and default.value.id == "cfg"
    )


def _is_ov_get_of(value: ast.expr, key: str) -> bool:
    """True if ``value`` is ``overrides.get("<key>", ...)`` — a per-model override lookup (any default)."""
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "get"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "overrides"
        and len(value.args) == 2
        and isinstance(value.args[0], ast.Constant)
        and value.args[0].value == key
    )


def test_engine_args_wires_fp8_weights_and_kv() -> None:
    kwargs = _kwargs(_engine_args_call())
    src = LORA_ENGINE.read_text()
    # FP8 weights: quantization is a per-model override lookup. Its default (_quant_default) is None
    # for a pre-quant serve_model_id base (vLLM auto-detects the checkpoint's FP8) else cfg.QUANTIZATION.
    assert "quantization" in kwargs, "engine must pass quantization (FP8)"
    assert _is_ov_get_of(kwargs["quantization"], "quantization")
    assert '_quant_default = None if overrides.get("serve_model_id") else cfg.QUANTIZATION' in src
    # The served checkpoint is serve_model_id when set, else the base model.
    assert '_served_model = overrides.get("serve_model_id") or self.base_model' in src
    assert _engine_arg_is_name(kwargs.get("model"), "_served_model")
    # FP8 KV cache, default cfg.KV_CACHE_DTYPE (unchanged shape).
    assert "kv_cache_dtype" in kwargs, "engine must pass kv_cache_dtype (FP8)"
    assert _is_ov_get_defaulting_to(kwargs["kv_cache_dtype"], cfg_attr="KV_CACHE_DTYPE")


def _engine_arg_is_name(value: ast.expr | None, name: str) -> bool:
    return isinstance(value, ast.Name) and value.id == name


def test_engine_never_enables_calculate_kv_scales() -> None:
    # Warmup-estimated KV scales corrupt the Qwen3 GDN-hybrid's recurrent state — uncalibrated
    # dynamic e4m3 is the safe path, so the flag must never be passed to the engine.
    assert "calculate_kv_scales" not in _kwargs(_engine_args_call()), (
        "do not pass calculate_kv_scales — it corrupts the GDN-hybrid KV scales"
    )


def test_moe_backend_override_is_guarded() -> None:
    # The catalog currently leaves the 35B MoE backend unset/auto, but the engine keeps a guarded
    # forwarding path for future canaries or emergency overrides. It must never crash an older build
    # on an unknown kwarg — same guard as language_model_only.
    src = LORA_ENGINE.read_text()
    assert '_extra["moe_backend"]' in src
    assert '_arg_supported("moe_backend")' in src


def test_max_num_batched_tokens_override_is_guarded() -> None:
    # The 35B needs max_num_batched_tokens=4096 because its attention block size (2096 tokens)
    # exceeds vLLM's 2048 default. Forward it only when this AsyncEngineArgs build exposes the field.
    src = LORA_ENGINE.read_text()
    assert '_extra["max_num_batched_tokens"]' in src
    assert '_arg_supported("max_num_batched_tokens")' in src


def test_reasoning_parser_forwarding_fails_closed_on_incompatible_vllm() -> None:
    src = LORA_ENGINE.read_text()
    assert '_extra["reasoning_parser"] = self.reasoning_parser' in src
    assert "_require_reasoning_api_compatibility(" in src


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
