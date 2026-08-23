"""Owned prequantized FP8 checkpoint mapping for serving.

Serving loads a PRE-QUANTIZED FP8 checkpoint for EVERY catalog base model — no online quantization
and no dependence on community repos:

- **FP8** (``FP8_DYNAMIC``, E4M3, compressed-tensors ``float-quantized``): Freesolo-owned checkpoints
  for the dense models (including the 27B); the 35B MoE uses Qwen's official FP8 checkpoint.

vLLM auto-detects the checkpoint's compressed-tensors quantization, so serving passes no online
``quantization`` for a pre-quant base. Adapters and routing always key off the logical ``base_model``;
only the loaded weights change.

⚠ Only point serving at a checkpoint VERIFIED to exist — a missing/renamed repo 404-crash-loops the
engine (the reason the ``serve_model_id`` mechanism was removed once). The owned repos below are
VL-preserving FP8 checkpoints published to the operator HF org and verified to load + serve in vLLM.
"""

from __future__ import annotations

# Freesolo-owned FP8_DYNAMIC checkpoints (we prequantize + publish these ourselves).
OWNED_FP8_MODEL_REPOS: dict[str, str] = {
    "Qwen/Qwen3.5-0.8B": "Freesolo-Co/Qwen3.5-0.8B-FP8",
    "Qwen/Qwen3.5-2B": "Freesolo-Co/Qwen3.5-2B-FP8",
    "Qwen/Qwen3.5-4B": "Freesolo-Co/Qwen3.5-4B-FP8",
    "Qwen/Qwen3.5-9B": "Freesolo-Co/Qwen3.5-9B-FP8",
    "Qwen/Qwen3.6-27B": "Freesolo-Co/Qwen3.6-27B-FP8",
}

# the 35B MoE uses Qwen's official FP8 checkpoint.
OFFICIAL_FP8_MODEL_REPOS: dict[str, str] = {
    "Qwen/Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B-FP8",
}

# what each base model actually loads for serving: owned FP8 for the dense models and official
# FP8 for the 35B MoE.
FP8_SERVE_MODEL_REPOS: dict[str, str] = {
    **OWNED_FP8_MODEL_REPOS,
    **OFFICIAL_FP8_MODEL_REPOS,
}


def fp8_serve_model_for(base_model: str) -> str:
    """The FP8 checkpoint the engine LOADS for ``base_model`` (owned or official Qwen)."""
    try:
        return FP8_SERVE_MODEL_REPOS[base_model]
    except KeyError as exc:
        allowed = ", ".join(FP8_SERVE_MODEL_REPOS)
        raise ValueError(
            f"Unsupported FP8 serving base model {base_model!r}. Allowed models: {allowed}"
        ) from exc
