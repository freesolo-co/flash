"""Bounded file previews from a validated environment archive; never imports its code."""

from __future__ import annotations

import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from flash.envs.loading.pull import _extract_environment_package_archive
from flash.server.domain.registry.envs import EnvPublishError, download_package

MAX_FILES = 50
MAX_PREVIEW_BYTES = 1024 * 1024
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024


@contextmanager
def package_root(slug: str, key: dict):
    data = download_package(slug=slug, key=key)
    with tempfile.TemporaryDirectory(prefix="flash-env-preview-") as tmp:
        try:
            root = _extract_environment_package_archive(data, Path(tmp))
        except (RuntimeError, ValueError) as exc:
            raise EnvPublishError("environment archive is invalid", status=502) from exc
        yield root


def file_path(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or any(p in {"", ".", ".."} for p in relative.split("/"))
    ):
        raise EnvPublishError("invalid environment file path")
    path = root / relative
    if not path.is_file():
        raise EnvPublishError("environment file not found", status=404)
    return path


def tree(slug: str, key: dict):
    with package_root(slug, key) as root:
        files = sorted(p for p in root.rglob("*") if p.is_file())
        selected = [
            {"path": p.relative_to(root).as_posix(), "size": p.stat().st_size}
            for p in files[:MAX_FILES]
        ]
    readmes = sorted(
        (
            f["path"]
            for f in selected
            if re.fullmatch(
                r"readme(?:\.(?:md|markdown|mdx|txt|rst))?", f["path"].split("/")[-1], re.I
            )
        ),
        key=lambda p: (p.count("/"), p),
    )
    return {
        "dir": slug,
        "files": selected,
        "fileCount": len(files),
        "truncatedListing": len(files) > MAX_FILES,
        "readmePath": next(iter(readmes), None),
    }


def contents(slug: str, key: dict, paths: object):
    if (
        not isinstance(paths, list)
        or len(paths) > MAX_FILES
        or not all(isinstance(p, str) for p in paths)
    ):
        raise EnvPublishError("paths must be a list of at most 50 file paths")
    result = []
    with package_root(slug, key) as root:
        for relative in paths:
            path = file_path(root, relative)
            size = path.stat().st_size
            data = path.read_bytes() if size <= MAX_PREVIEW_BYTES else b""
            binary = b"\0" in data[:8000]
            result.append(
                {
                    "path": relative,
                    "size": size,
                    "content": None
                    if binary or size > MAX_PREVIEW_BYTES
                    else data.decode("utf-8", "replace"),
                    "binary": binary,
                    "truncated": size > MAX_PREVIEW_BYTES,
                }
            )
    return result


def raw(slug: str, key: dict, relative: str):
    with package_root(slug, key) as root:
        path = file_path(root, relative)
        if path.stat().st_size > MAX_DOWNLOAD_BYTES:
            raise EnvPublishError("environment file is too large to download", status=413)
        return path.read_bytes()
