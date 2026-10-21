"""FSDP2 LoRA recovery checkpoints that omit immutable base parameters only.

Optimizer, scheduler, RNG, trainable parameters, and persistent buffers keep their native formats.
The pinned base is already loaded before Verl restores a checkpoint. Load hooks fill only the
explicitly omitted frozen keys from that base and validate the complete key set before native loading.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path


def _names(model) -> tuple[set[str], set[str]]:
    parameters = dict(model.named_parameters(remove_duplicate=False))
    trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("compact checkpoints require LoRA-only trainable parameters")
    frozen = set(parameters) - trainable
    # Buffers (including mutable counters) are always retained, never reconstructed from the base.
    frozen -= {name for name, _ in model.named_buffers(remove_duplicate=False)}
    return trainable, frozen


def compact_state(model, state: dict) -> list[str]:
    trainable, frozen = _names(model)
    if not trainable.issubset(state):
        raise RuntimeError("compact checkpoint parameter names do not match the FSDP2 state")
    omitted = sorted(frozen.intersection(state))
    if not omitted:
        raise RuntimeError("compact checkpoint found no immutable base parameters")
    for name in omitted:
        del state[name]
    return omitted


def restore_state(model, state: dict, manifest: dict, *, model_id: str, revision: str) -> None:
    if (
        set(manifest) != {"version", "model_id", "revision", "omitted", "retained"}
        or type(manifest["version"]) is not int
        or manifest["version"] != 1
        or manifest["model_id"] != model_id
        or manifest["revision"] != revision
    ):
        raise RuntimeError("compact checkpoint base identity or format mismatch")
    omitted, retained = manifest["omitted"], manifest["retained"]
    if not all(
        isinstance(keys, list) and all(isinstance(k, str) for k in keys)
        for keys in (omitted, retained)
    ):
        raise RuntimeError("invalid compact checkpoint key lists")
    if omitted != sorted(set(omitted)) or retained != sorted(set(retained)) or not omitted:
        raise RuntimeError("noncanonical compact checkpoint key lists")
    if set(state) != set(retained) or set(omitted).intersection(retained):
        raise RuntimeError("compact checkpoint retained state mismatch")
    trainable, frozen = _names(model)
    if not set(omitted).issubset(frozen) or not trainable.issubset(retained):
        raise RuntimeError("compact checkpoint attempted to omit mutable state")
    base = model.state_dict()
    if set(base) != set(omitted).union(retained):
        raise RuntimeError("compact checkpoint model structure changed")
    for name in omitted:
        state[name] = base[name]


def _identity() -> tuple[str, str]:
    model_id = os.environ.get("FLASH_CHECKPOINT_MODEL_ID", "")
    revision = os.environ.get("FLASH_CHECKPOINT_MODEL_REVISION", "")
    if not model_id or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("compact checkpoints require an immutable base model revision")
    return model_id, revision


def _manifest_path(manager, local_path):
    return Path(local_path) / f"flash_compact_model_rank_{manager.rank}.json"


@contextmanager
def saving(manager, local_path):
    """Filter only this model's state during the native checkpoint operation."""
    from verl.utils.fsdp_utils import fsdp_version

    model_id, revision = _identity()
    if fsdp_version(manager.model) != 2 or not manager.should_save_model:
        raise RuntimeError("compact checkpoints require the FSDP2 model save contract")
    path = _manifest_path(manager, local_path)

    def filter_state(model, state, prefix, metadata):
        if prefix:
            raise RuntimeError("compact hook must run on the FSDP2 root")
        omitted = compact_state(model, state)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": 1,
                    "model_id": model_id,
                    "revision": revision,
                    "omitted": omitted,
                    "retained": sorted(state),
                },
                sort_keys=True,
            )
        )
        os.replace(temporary, path)

    hook = manager.model.register_state_dict_post_hook(filter_state)
    try:
        yield
    finally:
        hook.remove()


@contextmanager
def loading(manager, local_path):
    """Fill omitted base keys for this load, retaining native strict validation."""
    if local_path is None or not _manifest_path(manager, local_path).exists():
        yield
        return
    from verl.utils.fsdp_utils import fsdp_version

    model_id, revision = _identity()
    if fsdp_version(manager.model) != 2:
        raise RuntimeError("compact recovery requires FSDP2")
    path = _manifest_path(manager, local_path)
    if path.is_symlink() or path.stat().st_size > 4 * 1024**2:
        raise RuntimeError("invalid compact checkpoint manifest")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise RuntimeError("invalid compact checkpoint manifest")

    def fill_base(model, state, prefix, *unused):
        if prefix:
            raise RuntimeError("compact hook must run on the FSDP2 root")
        restore_state(model, state, manifest, model_id=model_id, revision=revision)

    hook = manager.model.register_load_state_dict_pre_hook(fill_base)
    try:
        yield
    finally:
        hook.remove()
