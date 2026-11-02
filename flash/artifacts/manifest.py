"""Validated immutable file tables; publication never exposes a partial checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import PurePosixPath

MAX_MANIFEST_BYTES = 32 * 1024 * 1024


class ArtifactNotFoundError(FileNotFoundError):
    status_code = 404


def validate_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(c) < 32 for c in value)
        or PurePosixPath(value).is_absolute()
        or any(p in {"", ".", ".."} for p in value.split("/"))
    ):
        raise ValueError("invalid artifact path")
    return value


def repo_prefix(repo: str) -> str:
    validate_path(repo)
    if len(repo.split("/")) != 2:
        raise ValueError("artifact repository must be namespace/name")
    return f"repos/{repo}"


def encode_manifest(repo: str, files: dict) -> bytes:
    raw = json.dumps(
        {"schema": 1, "repo": repo, "files": files}, sort_keys=True, separators=(",", ":")
    ).encode()
    decode_manifest(raw, repo)
    return raw


def decode_manifest(raw: bytes, repo: str) -> dict:
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("artifact manifest exceeds size limit")
    obj = json.loads(raw)
    if not isinstance(obj, dict) or obj.get("schema") != 1 or obj.get("repo") != repo:
        raise ValueError("artifact manifest identity mismatch")
    files = obj.get("files")
    if not isinstance(files, dict):
        raise ValueError("invalid artifact file table")
    for path, entry in files.items():
        validate_path(path)
        if not isinstance(entry, dict) or not re.fullmatch(
            r"[0-9a-f]{64}", entry.get("sha256", "")
        ):
            raise ValueError("invalid artifact digest")
        if type(entry.get("size")) is not int or entry["size"] < 0:
            raise ValueError("invalid artifact size")
        if (
            type(entry.get("modified")) not in {int, float}
            or not math.isfinite(entry["modified"])
            or entry["modified"] <= 0
        ):
            raise ValueError("invalid artifact timestamp")
    return files


def revision_for(raw: bytes) -> str:
    # Existing persisted checkpoint descriptors use 40 hex characters. File integrity
    # is always the full SHA-256; this identifier only locates an immutable manifest.
    return hashlib.sha256(raw).hexdigest()[:40]


def validate_revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("artifact revision must be an immutable 40-hex identifier")
    return value
