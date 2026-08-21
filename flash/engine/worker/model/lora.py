"""read adapter tensor metadata from a downloaded adapter directory.

helpers must not import ``flash.engine.worker``. heavy dependencies remain lazy so this leaf
module has no package cycle or eager gpu stack import.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

# a safetensors header is small even for huge models (a few hundred kb at most); 100 mb is a wildly
# generous ceiling that still refuses a corrupt or hostile file declaring a multi-gb header length
# before safetensors parses it.
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class _SafetensorsTensorDescriptor:
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    start: int
    end: int


class _SafetensorsNumpyAccessor:
    def __init__(self, path: str, tensors, descriptors: dict[str, _SafetensorsTensorDescriptor]):
        self._path = path
        self._tensors = tensors
        self._descriptors = descriptors

    def keys(self) -> list[str]:
        return list(self._descriptors)

    def get_tensor(self, key: str):
        descriptor = self._descriptors[key]
        if descriptor.dtype != "BF16":
            return self._tensors.get_tensor(key)

        import numpy as np

        byte_count = descriptor.end - descriptor.start
        with open(self._path, "rb") as source:
            source.seek(descriptor.data_start + descriptor.start)
            payload = source.read(byte_count)
        if len(payload) != byte_count:
            raise ValueError(f"{self._path}: BF16 tensor {key!r} data is truncated")
        raw = np.frombuffer(payload, dtype="<u2")
        widened = raw.astype("<u4") << 16
        return widened.view("<f4").reshape(descriptor.shape)


def _read_adapter_tensor_metadata(adir: str) -> dict[str, tuple[int, ...]] | None:
    """return the authoritative tensor name and shape map for the representation peft loads.

    safetensors is validated through ``safe_open`` and ``get_slice`` without materializing tensor
    payloads. for sharded adapters, the index ``weight_map`` must agree exactly with every selected
    shard header; no duplicate, missing, extra, or misrouted key is accepted.
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
        shard_tensors = _read_safetensors_tensor_metadata(path)
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
    index_path = os.path.join(adir, "adapter_model.safetensors.index.json")
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


def _read_safetensors_tensor_descriptors(
    st_path: str,
) -> dict[str, _SafetensorsTensorDescriptor]:
    """return authenticated bounded descriptors for one safetensors shard."""
    import json
    import math
    import os
    import struct

    from safetensors import safe_open

    from flash._internal.fileio import reject_duplicate_keys

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
        header_bytes = tensor_file.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"{st_path}: safetensors header is truncated")

    duplicate_guard = reject_duplicate_keys(
        lambda key: ValueError(f"{st_path}: duplicate safetensors header key {key!r}")
    )
    try:
        header = json.loads(header_bytes, object_pairs_hook=duplicate_guard)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{st_path}: invalid safetensors structure ({exc})") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{st_path}: invalid safetensors structure (header is not an object)")

    data_start = 8 + header_length
    data_size = file_size - data_start
    descriptors: dict[str, _SafetensorsTensorDescriptor] = {}
    try:
        with safe_open(st_path, framework="numpy") as tensors:
            tensor_keys = tensors.keys()
            header_keys = [key for key in header if key != "__metadata__"]
            if set(tensor_keys) != set(header_keys):
                raise ValueError("header keys disagree with the authenticated tensor namespace")
            for key in tensor_keys:
                raw = header.get(key)
                if not isinstance(raw, dict):
                    raise ValueError(f"descriptor for {key!r} is not an object")
                dtype = raw.get("dtype")
                shape = tensors.get_slice(key).get_shape()
                offsets = raw.get("data_offsets")
                if not isinstance(dtype, str) or not dtype:
                    raise ValueError(f"descriptor dtype for {key!r} is invalid")
                if not isinstance(shape, list) or any(
                    not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
                ):
                    raise ValueError(f"invalid tensor shape for {key!r}")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(
                        not isinstance(offset, int) or isinstance(offset, bool)
                        for offset in offsets
                    )
                ):
                    raise ValueError(f"descriptor offsets for {key!r} are invalid")
                start, end = offsets
                if start < 0 or end < start or end > data_size:
                    raise ValueError(f"descriptor offsets for {key!r} are outside the data section")
                if dtype == "BF16" and end - start != math.prod(shape) * 2:
                    raise ValueError(f"BF16 descriptor byte length for {key!r} is invalid")
                descriptors[key] = _SafetensorsTensorDescriptor(
                    dtype=dtype,
                    shape=tuple(shape),
                    data_start=data_start,
                    start=start,
                    end=end,
                )
    except Exception as exc:
        raise ValueError(f"{st_path}: invalid safetensors structure ({exc})") from exc
    return descriptors


def _read_safetensors_tensor_metadata(st_path: str) -> dict[str, tuple[int, ...]]:
    """return names and shapes after safetensors validates the complete file structure."""
    return {
        key: descriptor.shape
        for key, descriptor in _read_safetensors_tensor_descriptors(st_path).items()
    }


@contextlib.contextmanager
def _open_safetensors_numpy(st_path: str):
    """open one shard with bounded BF16 materialization and normal NumPy tensors otherwise."""
    from safetensors import safe_open

    descriptors = _read_safetensors_tensor_descriptors(st_path)
    with safe_open(st_path, framework="numpy") as tensors:
        yield _SafetensorsNumpyAccessor(st_path, tensors, descriptors)
