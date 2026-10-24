"""Dependency-free validation shared by worker and CPU-stage base-model downloads."""

import json
import os


def snapshot_has_weights(snapshot_dir: str) -> bool:
    """True when a downloaded snapshot contains resolvable model weights (all indexed shards)."""
    weight_names = ("model", "pytorch_model", "tf_model", "flax_model")

    def _resolves(name: str) -> bool:
        path = os.path.join(snapshot_dir, name)
        return os.path.isfile(os.path.realpath(path))

    try:
        entries = os.listdir(snapshot_dir)
        # sharded checkpoints: the index enumerates every required shard; a partial download with
        # SOME shards present must not pass. validate all indexed shards resolve.
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if index_name in entries:
                try:
                    with open(os.path.join(snapshot_dir, index_name)) as f:
                        index = json.load(f)
                    shards = set((index.get("weight_map") or {}).values())
                except (OSError, ValueError):
                    return False
                return bool(shards) and all(_resolves(shard) for shard in shards)
        for name in entries:
            if not name.endswith((".safetensors", ".bin")):
                continue
            # exclude non-weight .bin artifacts (tokenizer.bin, training_args.bin, adapters)
            if not name.startswith(weight_names):
                continue
            # HF cache entries are symlinks into blobs/: a dangling link (partial download) must
            # not count as weights present.
            if _resolves(name):
                return True
    except OSError:
        return False
    return False
