"""Validate model snapshots written by the shared preload bootstrap."""

from __future__ import annotations

import json
import os


def _is_within(root: str, path: str) -> bool:
    try:
        return os.path.commonpath((root, path)) == root
    except ValueError:
        return False


def preload_snapshot_evidence(snapshot: object, cache_dir: str) -> str:
    """Return a normalized under-mount snapshot path only when real weights resolve."""
    if type(snapshot) is not str or not os.path.isdir(snapshot):
        raise RuntimeError("preload snapshot did not resolve to a directory")
    root = os.path.realpath(cache_dir)
    resolved = os.path.realpath(snapshot)
    if not _is_within(root, resolved):
        raise RuntimeError("preload snapshot resolved outside the mounted cache")
    snapshots_dir = os.path.dirname(resolved)
    model_root = os.path.dirname(snapshots_dir)
    model_name = os.path.basename(model_root)
    weight_root = (
        model_root
        if os.path.dirname(model_root) == root
        and os.path.basename(snapshots_dir) == "snapshots"
        and model_name.startswith("models--")
        and model_name != "models--"
        else resolved
    )
    entries = os.listdir(snapshot)
    index_names = [
        name
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        if name in entries
    ]
    weights = False
    if index_names:
        indexed_shards = set()
        for index_name in index_names:
            index_path = os.path.realpath(os.path.join(snapshot, index_name))
            if not _is_within(weight_root, index_path) or not os.path.isfile(index_path):
                raise RuntimeError("preload snapshot weight index is malformed")
            with open(index_path, encoding="utf-8") as stream:
                index = json.load(stream)
            weight_map = index.get("weight_map") if type(index) is dict else None
            if type(weight_map) is not dict or not weight_map:
                raise RuntimeError("preload snapshot weight index is malformed")
            shards = set(weight_map.values())
            if any(type(shard) is not str or not shard for shard in shards):
                raise RuntimeError("preload snapshot weight index is malformed")
            indexed_shards.update(shards)
        shard_paths = []
        for shard in indexed_shards:
            if os.path.isabs(shard) or ".." in shard.split(os.sep):
                raise RuntimeError("preload snapshot weight index is malformed")
            lexical_path = os.path.normpath(os.path.join(resolved, shard))
            if not _is_within(resolved, lexical_path):
                raise RuntimeError("preload snapshot weight index is malformed")
            shard_paths.append(os.path.realpath(lexical_path))
        weights = all(
            _is_within(weight_root, shard_path) and os.path.isfile(shard_path)
            for shard_path in shard_paths
        )
    else:
        weights = any(
            name.endswith((".safetensors", ".bin"))
            and name.startswith(("model", "pytorch_model", "tf_model", "flax_model"))
            and _is_within(weight_root, os.path.realpath(os.path.join(snapshot, name)))
            and os.path.isfile(os.path.realpath(os.path.join(snapshot, name)))
            for name in entries
        )
    if not weights:
        raise RuntimeError("preload snapshot has no complete model weights")
    return os.path.relpath(resolved, root)
