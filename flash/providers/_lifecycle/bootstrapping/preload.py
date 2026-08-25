"""Validate model snapshots written by the shared preload bootstrap."""

from __future__ import annotations

import json
import os


def preload_snapshot_evidence(snapshot: object, cache_dir: str) -> str:
    """Return a normalized under-mount snapshot path only when real weights resolve."""
    if type(snapshot) is not str or not os.path.isdir(snapshot):
        raise RuntimeError("preload snapshot did not resolve to a directory")
    root = os.path.realpath(cache_dir)
    resolved = os.path.realpath(snapshot)
    try:
        if os.path.commonpath((root, resolved)) != root:
            raise RuntimeError("preload snapshot resolved outside the mounted cache")
    except ValueError:
        raise RuntimeError("preload snapshot resolved outside the mounted cache") from None
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
            with open(os.path.join(snapshot, index_name), encoding="utf-8") as stream:
                index = json.load(stream)
            weight_map = index.get("weight_map") if type(index) is dict else None
            if type(weight_map) is not dict or not weight_map:
                raise RuntimeError("preload snapshot weight index is malformed")
            shards = set(weight_map.values())
            if any(type(shard) is not str or not shard for shard in shards):
                raise RuntimeError("preload snapshot weight index is malformed")
            indexed_shards.update(shards)
        shard_paths = [os.path.realpath(os.path.join(snapshot, shard)) for shard in indexed_shards]
        weights = all(
            os.path.commonpath((resolved, shard_path)) == resolved and os.path.isfile(shard_path)
            for shard_path in shard_paths
        )
    else:
        weights = any(
            name.endswith((".safetensors", ".bin"))
            and name.startswith(("model", "pytorch_model", "tf_model", "flax_model"))
            and os.path.commonpath((resolved, os.path.realpath(os.path.join(snapshot, name))))
            == resolved
            and os.path.isfile(os.path.realpath(os.path.join(snapshot, name)))
            for name in entries
        )
    if not weights:
        raise RuntimeError("preload snapshot has no complete model weights")
    return os.path.relpath(resolved, root)
