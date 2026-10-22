"""Lossless transport of native checkpoint files larger than the Hub's file limit."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from flash._internal.fileio import reject_duplicate_keys

MANIFEST = "flash_checkpoint_transport.json"
PARTS = "flash_checkpoint_parts"
# Decimal bytes: comfortably below the Hub's 50 GB limit, including on single-GPU 35B saves.
PART_BYTES = 40_000_000_000
_BUFFER_BYTES = 8 * 1024 * 1024


def _validate_transport_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid checkpoint transport path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in {".", ".."} for p in path.parts):
        raise ValueError("invalid checkpoint transport path")
    return value


def parse_manifest(raw: bytes | str) -> dict[str, dict]:
    """Validate paths, order, lengths and hashes before trusting any transport metadata."""
    value = json.loads(raw, object_pairs_hook=reject_duplicate_keys(ValueError))
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "files"}
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        raise ValueError("invalid checkpoint transport version")
    files = value["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= 1024:
        raise ValueError("invalid checkpoint transport file list")
    for name, entry in files.items():
        _validate_transport_path(name)
        if name == MANIFEST or name.split("/")[0] == PARTS:
            raise ValueError("reserved checkpoint transport path")
        if not isinstance(entry, dict) or set(entry) != {"size", "parts"}:
            raise ValueError("invalid checkpoint transport file")
        size, parts = entry["size"], entry["parts"]
        if (
            type(size) is not int
            or size <= 0
            or not isinstance(parts, list)
            or not 1 <= len(parts) <= 1024
        ):
            raise ValueError("invalid checkpoint transport size")
        prefix = f"{PARTS}/{hashlib.sha256(name.encode()).hexdigest()}"
        total = 0
        for index, part in enumerate(parts):
            if not isinstance(part, dict) or set(part) != {"path", "size", "sha256"}:
                raise ValueError("invalid checkpoint transport part")
            if part["path"] != f"{prefix}/{index:05d}":
                raise ValueError("invalid checkpoint transport part order")
            if type(part["size"]) is not int or not 0 < part["size"] <= PART_BYTES:
                raise ValueError("invalid checkpoint transport part size")
            if (
                not isinstance(part["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", part["sha256"]) is None
            ):
                raise ValueError("invalid checkpoint transport checksum")
            total += part["size"]
        if total != size:
            raise ValueError("checkpoint transport length mismatch")
    return files


@contextlib.contextmanager
def upload_directory(source: str):
    """Stage large files as bounded parts; leave small checkpoint uploads unchanged."""
    root = Path(source)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not any(p.stat().st_size > PART_BYTES for p in files):
        yield source
        return
    if (root / MANIFEST).exists() or (root / PARTS).exists():
        raise ValueError("checkpoint contains reserved transport paths")
    with tempfile.TemporaryDirectory(prefix=".flash-checkpoint-", dir=root.parent) as temporary:
        stage = Path(temporary)
        manifest = {}
        for original in files:
            if original.is_symlink():
                raise ValueError("checkpoint transport does not accept symlinks")
            name = original.relative_to(root).as_posix()
            size = original.stat().st_size
            if size <= PART_BYTES:
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(original, target)
                continue
            parts = []
            prefix = f"{PARTS}/{hashlib.sha256(name.encode()).hexdigest()}"
            with original.open("rb") as reader:
                remaining = size
                while remaining:
                    relative = f"{prefix}/{len(parts):05d}"
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    length = min(remaining, PART_BYTES)
                    digest = hashlib.sha256()
                    with target.open("wb") as writer:
                        pending = length
                        while pending:
                            block = reader.read(min(pending, _BUFFER_BYTES))
                            if not block:
                                raise ValueError("checkpoint changed during transport staging")
                            writer.write(block)
                            digest.update(block)
                            pending -= len(block)
                    parts.append({"path": relative, "size": length, "sha256": digest.hexdigest()})
                    remaining -= length
                if reader.read(1):
                    raise ValueError("checkpoint changed during transport staging")
            manifest[name] = {"size": size, "parts": parts}
        (stage / MANIFEST).write_text(json.dumps({"version": 1, "files": manifest}))
        yield str(stage)


def restore_directory(directory: str) -> None:
    """Reassemble and verify transport files before native checkpoint selection/deserialization."""
    root = Path(directory)
    manifest = root / MANIFEST
    if not manifest.exists():
        return
    if manifest.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("checkpoint transport manifest is too large")
    files = parse_manifest(manifest.read_bytes())
    for name, entry in files.items():
        target = root / name
        if target.exists() or target.is_symlink():
            raise ValueError("checkpoint contains both native and transported file")
        if not target.parent.resolve().is_relative_to(root.resolve()):
            raise ValueError("checkpoint transport path escapes its directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as writer:
            temporary = Path(writer.name)
            try:
                for part in entry["parts"]:
                    source = root / part["path"]
                    if source.is_symlink() or not source.resolve().is_relative_to(root.resolve()):
                        raise ValueError("checkpoint transport part escapes its directory")
                    if source.stat().st_size != part["size"]:
                        raise ValueError("checkpoint transport part length mismatch")
                    digest = hashlib.sha256()
                    with source.open("rb") as reader:
                        while block := reader.read(_BUFFER_BYTES):
                            writer.write(block)
                            digest.update(block)
                    if digest.hexdigest() != part["sha256"]:
                        raise ValueError("checkpoint transport checksum mismatch")
                writer.close()
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    shutil.rmtree(root / PARTS)
    manifest.unlink()
