"""read adapter tensor metadata from a downloaded adapter directory.

helpers must not import ``flash.engine.worker``. heavy dependencies remain lazy so this leaf
module has no package cycle or eager gpu stack import.
"""

from __future__ import annotations

# a safetensors header is small even for huge models (a few hundred kb at most); 100 mb is a wildly
# generous ceiling that still refuses a corrupt or hostile file declaring a multi-gb header length
# before safetensors parses it.
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


def _read_adapter_tensor_metadata(adir: str) -> dict[str, tuple[int, ...]] | None:
    """return the authoritative tensor name and shape map for the representation peft loads.

    safetensors is validated through ``safe_open`` and ``get_slice`` without materializing tensor
    payloads. torch ``.bin`` files use weights-only cpu loading because their format has no separate
    metadata header. for sharded adapters, the index ``weight_map`` must agree exactly with every
    selected shard header; no duplicate, missing, extra, or misrouted key is accepted.
    """
    import os

    from flash.adapters.artifacts import loadable_adapter_weight_files

    try:
        selected = loadable_adapter_weight_files(os.listdir(adir))
    except OSError:
        return None
    if not selected:
        return None
    weight_map = _read_sharded_weight_map(adir, selected)
    tensors: dict[str, tuple[int, ...]] = {}
    for name in selected:
        path = os.path.join(adir, name)
        shard_tensors = (
            _read_safetensors_tensor_metadata(path)
            if name.endswith(".safetensors")
            else _read_bin_tensor_metadata(path)
        )
        duplicates = tensors.keys() & shard_tensors.keys()
        if duplicates:
            raise ValueError(
                f"adapter shards contain duplicate tensor keys {sorted(duplicates)[:4]}"
            )
        if weight_map is not None:
            misrouted = [key for key in shard_tensors if weight_map.get(key) != name]
            if misrouted:
                raise ValueError(
                    f"adapter shard {name} disagrees with weight_map for keys {misrouted[:4]}"
                )
        tensors.update(shard_tensors)
    if weight_map is not None and tensors.keys() != weight_map.keys():
        missing = sorted(weight_map.keys() - tensors.keys())[:4]
        extra = sorted(tensors.keys() - weight_map.keys())[:4]
        raise ValueError(
            f"adapter shard headers disagree with weight_map; missing={missing}, extra={extra}"
        )
    return tensors


def _read_adapter_tensor_keys(adir: str) -> list[str] | None:
    """return tensor key names from adapter weights without loading safetensors payloads."""
    tensors = _read_adapter_tensor_metadata(adir)
    return list(tensors) if tensors is not None else None


def _read_sharded_weight_map(adir: str, selected: list[str]) -> dict[str, str] | None:
    """return an authoritative sharded index map, or none for a selected single-file adapter."""
    import json
    import os

    from flash._internal.fileio import reject_duplicate_keys
    from flash.adapters.artifacts import ADAPTER_SHARD_PREFIX

    if not selected[0].startswith(ADAPTER_SHARD_PREFIX):
        return None
    suffix = ".safetensors" if selected[0].endswith(".safetensors") else ".bin"
    index_path = os.path.join(adir, f"adapter_model{suffix}.index.json")
    duplicate_guard = reject_duplicate_keys(
        lambda key: ValueError(f"{index_path}: duplicate JSON key {key!r}")
    )
    try:
        with open(index_path, encoding="utf-8") as index_file:
            index = json.load(index_file, object_pairs_hook=duplicate_guard)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{index_path}: unreadable adapter weight index") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or any(
        not isinstance(key, str) or not key or not isinstance(shard, str) or not shard
        for key, shard in weight_map.items()
    ):
        raise ValueError(f"{index_path}: weight_map must map non-empty tensor names to shard names")
    selected_set = set(selected)
    referenced = set(weight_map.values())
    if referenced != selected_set:
        raise ValueError(
            f"{index_path}: weight_map shards {sorted(referenced)} do not match selected shards "
            f"{sorted(selected_set)}"
        )
    return weight_map


def _read_safetensors_tensor_metadata(st_path: str) -> dict[str, tuple[int, ...]]:
    """return names and shapes after safetensors validates the complete file structure."""
    import os
    import struct

    file_size = os.path.getsize(st_path)
    with open(st_path, "rb") as tensor_file:
        length_bytes = tensor_file.read(8)
        if len(length_bytes) < 8:
            raise ValueError(f"{st_path}: too small to be a safetensors file")
        (header_length,) = struct.unpack("<Q", length_bytes)
        if header_length > file_size - 8 or header_length > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                f"{st_path}: declared safetensors header length {header_length} is implausible "
                f"(file is {file_size} bytes)"
            )

    try:
        from safetensors import safe_open

        with safe_open(st_path, framework="numpy") as tensors:
            metadata: dict[str, tuple[int, ...]] = {}
            tensor_keys = tensors.keys()
            for key in tensor_keys:
                shape = tensors.get_slice(key).get_shape()
                if not isinstance(shape, list) or any(
                    not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
                ):
                    raise ValueError(f"invalid tensor shape for {key!r}")
                metadata[key] = tuple(shape)
            return metadata
    except Exception as exc:
        raise ValueError(f"{st_path}: invalid safetensors structure ({exc})") from exc


def _read_bin_tensor_metadata(bin_path: str) -> dict[str, tuple[int, ...]]:
    """return the tensor names and shapes in one torch ``.bin`` adapter state dict."""
    import torch

    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{bin_path}: expected a tensor state dict, got {type(state).__name__}")
    bad = [
        key
        for key, value in state.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if bad:
        raise ValueError(
            f"{bin_path}: contains non-tensor entries (e.g. {bad[:4]}); "
            "expected a plain PEFT adapter state dict"
        )
    return {key: tuple(tensor.shape) for key, tensor in state.items()}
