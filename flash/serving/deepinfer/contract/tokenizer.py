"""The pinned model's official text encoding and context validation for generation."""

import importlib.util
from pathlib import Path

from flash.serving.deepinfer.contract.config import MAX_MODEL_LEN


def load_tokenizer(model_path):
    from tokenizers import Tokenizer

    root = Path(model_path)
    spec = importlib.util.spec_from_file_location(
        "deepseek_1017_encoding", root / "encoding" / "encoding.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Pinned DeepSeek encoding module is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Tokenizer.from_file(str(root / "tokenizer.json")), module.encode_messages


def validate_length(tokens, max_tokens):
    if not tokens or len(tokens) + max_tokens + 64 > MAX_MODEL_LEN:
        raise ValueError(f"prompt plus output exceeds the {MAX_MODEL_LEN}-token serving context")


def encode_request(tokenizer, encode_messages, request, *, prefix_cache=None):
    prompt = encode_messages(
        request.messages, thinking_mode="thinking" if request.thinking else "chat"
    )
    # The single-sequence Rust binding holds the GIL even in a worker thread.
    # A one-item batch releases it without changing rendering or token boundaries.
    tokens = (
        prefix_cache.encode(prompt, request.cache_namespace)
        if prefix_cache is not None
        else tokenizer.encode_batch([prompt], add_special_tokens=False)[0].ids
    )
    validate_length(tokens, request.max_tokens)
    return tokens


def make_prefix_cache(tokenizer):
    from flash.serve.runtime.token_prefix_cache import TokenPrefixCache

    marker = "<\uff5cUser\uff5c>"
    return TokenPrefixCache(tokenizer, marker, marker)
