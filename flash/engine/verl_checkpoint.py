"""Dependency-light validation for native Verl FSDP checkpoint manifests."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from flash._internal.fileio import reject_duplicate_keys
from flash.engine.verl_policy import FsdpGeneration

VERL_FSDP_CONFIG_FILE = "fsdp_config.json"
_MAX_NATIVE_FSDP_WORLD_SIZE = 8
_REQUIRED_SHARD_CLASSES = ("model", "optim", "extra_state")
_SHARD_RE = re.compile(r"(model|optim|extra_state)_world_size_(\d+)_rank_(\d+)\.pt")
_REJECT_DUPLICATE_CONFIG_KEYS = reject_duplicate_keys(
    lambda key: ValueError(f"duplicate fsdp config key: {key}")
)


@dataclass(frozen=True)
class FsdpManifestFile:
    """One direct actor-manifest member and its known size, if available."""

    name: str
    size: int | None


@dataclass(frozen=True)
class FsdpCheckpointInspection:
    """One parsed native checkpoint verdict shared by selection and diagnostics."""

    loadable: bool
    generation: int | None
    width: int | None
    rejection_reason: str | None = None
    missing_path: str | None = None

    def diagnostic(self) -> str:
        detail = self.rejection_reason or "loadable native checkpoint"
        return f"{detail}: {self.missing_path}" if self.missing_path else detail


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return None


def parse_fsdp_stamp(raw: bytes | str | None) -> tuple[int, int] | None:
    """Parse Verl's strict generation and width stamp without worker dependencies."""
    if raw is None:
        return None
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        decoded = json.loads(text, object_pairs_hook=_REJECT_DUPLICATE_CONFIG_KEYS)
    except (UnicodeDecodeError, ValueError, TypeError):
        return None
    if not isinstance(decoded, dict):
        return None
    generation = _positive_int(decoded.get("FSDP_version"))
    width = _positive_int(decoded.get("world_size"))
    if generation is None or width is None or width > _MAX_NATIVE_FSDP_WORLD_SIZE:
        return None
    return generation, width


def inspect_fsdp_checkpoint_manifest(
    stamp_raw: bytes | str | None,
    files: Iterable[FsdpManifestFile],
    *,
    expected_generation: FsdpGeneration,
    expected_world_size: int | None,
) -> FsdpCheckpointInspection:
    """Validate the exact canonical native shard set for one stamped actor directory."""
    stamp = parse_fsdp_stamp(stamp_raw)
    if stamp is None:
        return FsdpCheckpointInspection(False, None, None, "invalid or missing fsdp stamp")
    generation, width = stamp
    if generation != expected_generation:
        return FsdpCheckpointInspection(
            False,
            generation,
            width,
            f"fsdp generation mismatch (stamped {generation}, expected {expected_generation})",
        )
    if expected_world_size is not None and width != expected_world_size:
        return FsdpCheckpointInspection(
            False,
            generation,
            width,
            f"fsdp world-size mismatch (stamped {width}, expected {expected_world_size})",
        )

    actual: dict[str, int | None] = {}
    for item in files:
        name = item.name
        if name in actual:
            return FsdpCheckpointInspection(False, generation, width, "malformed shard name", name)
        match = _SHARD_RE.fullmatch(name)
        if match is None:
            if any(name.startswith(f"{kind}_world_size_") for kind in _REQUIRED_SHARD_CLASSES):
                return FsdpCheckpointInspection(
                    False, generation, width, "malformed shard name", name
                )
            continue
        _kind, width_text, rank_text = match.groups()
        parsed_width = int(width_text)
        parsed_rank = int(rank_text)
        if width_text != str(parsed_width) or rank_text != str(parsed_rank):
            return FsdpCheckpointInspection(False, generation, width, "malformed shard name", name)
        actual[name] = item.size

    expected = tuple(
        f"{kind}_world_size_{width}_rank_{rank}.pt"
        for kind in _REQUIRED_SHARD_CLASSES
        for rank in range(width)
    )
    for name in expected:
        if name not in actual:
            return FsdpCheckpointInspection(False, generation, width, "missing shard", name)
    extras = sorted(set(actual) - set(expected))
    if extras:
        return FsdpCheckpointInspection(
            False,
            generation,
            width,
            "malformed shard name",
            extras[0],
        )
    for name in expected:
        size = actual[name]
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
            return FsdpCheckpointInspection(
                False,
                generation,
                width,
                "empty or unreadable shard",
                name,
            )
    return FsdpCheckpointInspection(True, generation, width)
