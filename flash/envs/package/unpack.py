"""Safe extraction for Freesolo environment tar archives."""

from __future__ import annotations

import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast

from flash.envs.package.limits import (
    TAR_METADATA_TYPES,
    LimitedArchiveReader,
    tar_member_segments,
)


def extract_validated_archive_members(
    reader: LimitedArchiveReader,
    *,
    extract_base: Path,
    content_byte_limit: int,
    extracted_member_limit: int,
    scanned_member_limit: int,
    member_filter: Callable[[list[str]], bool] | None = None,
    segment_observer: Callable[[list[str]], None] | None = None,
) -> None:
    """Extract validated tar members into ``extract_base`` within explicit limits."""
    guard_root = extract_base.resolve()
    total = 0
    extracted = 0
    scanned = 0
    with tarfile.open(fileobj=cast("BinaryIO", reader), mode="r|") as tar:
        for member in tar:
            scanned += 1
            if scanned > scanned_member_limit:
                raise RuntimeError(
                    f"env package has too many entries to scan (limit {scanned_member_limit})"
                )
            if member.type in TAR_METADATA_TYPES:
                continue
            segments = tar_member_segments(
                member.name,
                unsafe_error=lambda name: RuntimeError(
                    f"unsafe path in environment archive: {name!r}"
                ),
            )
            if not segments:
                continue
            if segment_observer is not None:
                segment_observer(segments)
            if member_filter is not None and not member_filter(segments):
                continue
            normalized_name = "/".join(segments)
            target = (extract_base / normalized_name).resolve()
            if target != guard_root and guard_root not in target.parents:
                raise RuntimeError(f"unsafe path in environment archive: {member.name!r}")
            if member.islnk() or member.issym() or not (member.isreg() or member.isdir()):
                continue
            extracted += 1
            if extracted > extracted_member_limit:
                raise RuntimeError(
                    f"env package has too many members (limit {extracted_member_limit})"
                )
            total += max(0, member.size)
            if total > content_byte_limit:
                raise RuntimeError(
                    "environment archive is too large uncompressed "
                    f"({total} bytes; limit {content_byte_limit} bytes)"
                )
            member.name = normalized_name
            tar.extract(member, extract_base)
