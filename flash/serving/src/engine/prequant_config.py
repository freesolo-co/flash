"""Prequantized checkpoint mapping for active hosted serving.

The 9B loads a Freesolo-owned compressed-tensors FP8 checkpoint. The 27B loads its official FP8
checkpoint. The 35B MoE mapping remains its official FP8 default, although the validated engine
override serves the base bf16 weights so full-expert LoRA and CUDA graphs coexist.
"""

from __future__ import annotations

# freesolo-owned compressed-tensors checkpoints.
OWNED_FP8_MODEL_REPOS: dict[str, str] = {
    "Qwen/Qwen3.5-9B": "Freesolo-Co/Qwen3.5-9B-FP8",
}

# official qwen checkpoints active in hosted serving.
OFFICIAL_FP8_MODEL_REPOS: dict[str, str] = {
    "Qwen/Qwen3.8-27B": "Qwen/Qwen3.8-27B-FP8",
    "Qwen/Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B-FP8",
}

FP8_SERVE_MODEL_REPOS: dict[str, str] = {
    **OWNED_FP8_MODEL_REPOS,
    **OFFICIAL_FP8_MODEL_REPOS,
}


def fp8_serve_model_for(base_model: str) -> str:
    """Return the FP8 checkpoint loaded for ``base_model``."""
    try:
        return FP8_SERVE_MODEL_REPOS[base_model]
    except KeyError as exc:
        allowed = ", ".join(FP8_SERVE_MODEL_REPOS)
        raise ValueError(
            f"Unsupported FP8 serving base model {base_model!r}. Allowed models: {allowed}"
        ) from exc
