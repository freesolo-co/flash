"""Resolve tokenizer-only Hub snapshots without requiring unused model weights."""

from __future__ import annotations

import os

from .types import EngineConfig

TOKENIZER_WEIGHT_PATTERNS = (
    "*.safetensors",
    "pytorch_model*.bin",
    "*.gguf",
    "tf_model*.h5",
    "flax_model*.msgpack",
    "*.onnx",
    "*.pt",
    "*.pth",
    "*.ot",
)


def engine_tokenizer_path(config: EngineConfig) -> str:
    source = config.effective_tokenizer_model
    if os.environ.get("HF_HUB_OFFLINE", "").upper() not in {"1", "ON", "YES", "TRUE"}:
        return source
    if (source, config.tokenizer_revision) == (
        config.effective_served_model,
        config.model_revision,
    ):
        return source
    from huggingface_hub import snapshot_download

    # vLLM resolves repository IDs as full snapshots. Give it the exact local tokenizer
    # directory so its offline completeness check does not require unused tensor shards.
    return snapshot_download(
        repo_id=source,
        revision=config.tokenizer_revision,
        local_files_only=True,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        ignore_patterns=list(TOKENIZER_WEIGHT_PATTERNS),
    )
