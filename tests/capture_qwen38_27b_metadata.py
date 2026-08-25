"""capture compact qwen3.8 27b checkpoint metadata from immutable hub revisions."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
FP8_MODEL = "Qwen/Qwen3.8-27B-FP8"
FP8_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
OUTPUT = Path(__file__).parent / "fixtures" / "qwen38_27b_target_metadata.json"
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


def _url(model: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{model}/resolve/{revision}/{filename}"


def _read(url: str, *, end: int | None = None) -> bytes:
    headers = {"Range": f"bytes=0-{end}"} if end is not None else {}
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=60
    ) as response:
        if end is None:
            return response.read()
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise ValueError(f"range request for {url!r} returned HTTP {status}, expected 206")
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(?:\d+|\*)", content_range)
        if match is None or (int(match.group(1)), int(match.group(2))) != (0, end):
            raise ValueError(
                f"range request for {url!r} returned invalid Content-Range {content_range!r}"
            )
        requested = end + 1
        raw = response.read(requested + 1)
        if len(raw) > requested:
            raise ValueError(f"range request for {url!r} returned more than {requested} bytes")
        return raw


def _json(model: str, revision: str, filename: str) -> tuple[dict[str, Any], str]:
    raw = _read(_url(model, revision, filename))
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _safetensors_header(model: str, revision: str, filename: str) -> dict[str, Any]:
    url = _url(model, revision, filename)
    prefix = _read(url, end=7)
    if len(prefix) != 8:
        raise ValueError(f"safetensors file {filename!r} returned an incomplete header prefix")
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(
            f"safetensors file {filename!r} declares an oversized {header_length}-byte header"
        )
    raw = _read(url, end=7 + header_length)
    if len(raw) < 8 + header_length:
        raise ValueError(f"safetensors file {filename!r} returned an incomplete header")
    return json.loads(raw[8 : 8 + header_length])


def _target_shapes(index: dict[str, Any]) -> tuple[list[list[int]], int, int]:
    groups: Counter[tuple[int, int]] = Counter()
    parameter_count = 0
    for shard in sorted(set(index["weight_map"].values())):
        for name, metadata in _safetensors_header(BASE_MODEL, BASE_REVISION, shard).items():
            if name == "__metadata__":
                continue
            shape = tuple(int(value) for value in metadata["shape"])
            parameter_count += _product(shape)
            if (
                len(shape) != 2
                or name.startswith("mtp.")
                or ".mtp." in name
                or name.endswith(
                    ("embed_tokens.weight", "lm_head.weight", "visual.pos_embed.weight")
                )
            ):
                continue
            groups[(shape[1], shape[0])] += 1
    inventory = [[inputs, outputs, count] for (inputs, outputs), count in sorted(groups.items())]
    return inventory, sum(groups.values()), parameter_count


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def capture() -> dict[str, Any]:
    config, config_digest = _json(BASE_MODEL, BASE_REVISION, "config.json")
    tokenizer, tokenizer_digest = _json(BASE_MODEL, BASE_REVISION, "tokenizer_config.json")
    processor, processor_digest = _json(BASE_MODEL, BASE_REVISION, "preprocessor_config.json")
    index, index_digest = _json(BASE_MODEL, BASE_REVISION, "model.safetensors.index.json")
    fp8_config, fp8_config_digest = _json(FP8_MODEL, FP8_REVISION, "config.json")
    target_shapes, target_count, parameter_count = _target_shapes(index)
    text = config["text_config"]
    vision = config["vision_config"]
    layer_types = list(text["layer_types"])
    additions = [
        {
            "id": int(token_id),
            "content": value["content"],
            "special": bool(value["special"]),
        }
        for token_id, value in sorted(
            tokenizer["added_tokens_decoder"].items(), key=lambda item: int(item[0])
        )
    ]
    base_vocab_size = min(item["id"] for item in additions)
    tokenizer_length = max(item["id"] for item in additions) + 1
    return {
        "base": {
            "model": BASE_MODEL,
            "revision": BASE_REVISION,
            "architecture": config["architectures"][0],
            "model_type": config["model_type"],
            "parameter_count": parameter_count,
            "weight_bytes": int(index["metadata"]["total_size"]),
            "geometry": {
                "layers": int(text["num_hidden_layers"]),
                "hidden_size": int(text["hidden_size"]),
                "attention_heads": int(text["num_attention_heads"]),
                "key_value_heads": int(text["num_key_value_heads"]),
                "full_attention_layers": layer_types.count("full_attention"),
                "linear_attention_layers": layer_types.count("linear_attention"),
                "vocab_size": int(text["vocab_size"]),
                "vision_layers": int(vision["depth"]),
            },
            "target_shapes": target_shapes,
            "target_count": target_count,
        },
        "tokenizer": {
            "class": tokenizer["tokenizer_class"],
            "length": tokenizer_length,
            "vocab_size": base_vocab_size,
            "added_tokens": additions,
            "chat_template_sha256": hashlib.sha256(
                tokenizer["chat_template"].encode("utf-8")
            ).hexdigest(),
            "preserve_thinking_default": True,
        },
        "processor": {
            "class": processor["processor_class"],
            "image_processor_class": processor["image_processor_type"],
        },
        "fp8": {
            "model": FP8_MODEL,
            "revision": FP8_REVISION,
            "quant_method": fp8_config["quantization_config"]["quant_method"],
            "format": fp8_config["quantization_config"]["fmt"],
            "activation_scheme": fp8_config["quantization_config"]["activation_scheme"],
            "weight_block_size": fp8_config["quantization_config"]["weight_block_size"],
        },
        "source_sha256": {
            "config.json": config_digest,
            "tokenizer_config.json": tokenizer_digest,
            "preprocessor_config.json": processor_digest,
            "model.safetensors.index.json": index_digest,
            "fp8/config.json": fp8_config_digest,
        },
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(capture(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
