"""Structural recognition for opaque containers the base scanner cannot inspect."""

from __future__ import annotations

import hashlib
import re
import time
import zlib
from collections.abc import Iterator
from pathlib import Path

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_ARROW_MAGIC = b"ARROW1"
_ARROW_CONTINUATION = b"\xff\xff\xff\xff"
_MAX_ARROW_METADATA_BYTES = 16 << 20
_MAX_ARROW_VECTOR_ITEMS = 100_000
# the Type union in Schema.fbs, 1-based: Null=1 through LargeListView=26. the bound only has to
# reject a byte that cannot be a type at all, so it tracks the union's current length. a file
# using a type added after this was written stops matching as Arrow and falls back to the literal
# scan, which is the safe direction: a missed refusal, never a wrong one.
_MAX_ARROW_TYPE_KIND = 26
_GIT_PACK_HEADER_BYTES = 12
_MAX_GIT_PACK_OBJECTS = 100_000
_MAX_GIT_PACK_OUTPUT_BYTES = 64 << 20
_MAX_GIT_BUNDLE_HEADER_BYTES = 4 << 20
_RPM_LEAD_BYTES = 96
_BUNDLE_UNFINISHED = object()
_BUNDLE_REF = re.compile(rb"(?:[0-9a-f]{40}|[0-9a-f]{64}) refs/[^\s]+")
_BUNDLE_PREREQUISITE = re.compile(rb"-(?:[0-9a-f]{40}|[0-9a-f]{64}) .+")


class OpaqueDeadlineExceeded(Exception):
    """An opaque-container structural check exhausted the shared scan deadline."""


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise OpaqueDeadlineExceeded


def _source_size(source: Path | bytes) -> int | None:
    if isinstance(source, bytes):
        return len(source)
    try:
        return source.stat().st_size
    except OSError:
        return None


def _read_at(source: Path | bytes, offset: int, size: int) -> bytes | None:
    if offset < 0 or size < 0:
        return None
    if isinstance(source, bytes):
        return source[offset : offset + size]
    try:
        with source.open("rb") as handle:
            handle.seek(offset)
            return handle.read(size)
    except OSError:
        return None


def _hdf5_offsets(size: int) -> Iterator[int]:
    yield 0
    offset = 512
    while offset + len(_HDF5_SIGNATURE) <= size:
        yield offset
        offset <<= 1


def _looks_like_hdf5(source: Path | bytes, deadline: float | None) -> bool:
    size = _source_size(source)
    if size is None:
        return False
    for offset in _hdf5_offsets(size):
        _check_deadline(deadline)
        if _read_at(source, offset, len(_HDF5_SIGNATURE)) == _HDF5_SIGNATURE:
            return True
    return False


def _flatbuffer_table_at(data: bytes, table: int) -> tuple[int, int, int, int] | None:
    if table < 4 or table + 4 > len(data):
        return None
    back = int.from_bytes(data[table : table + 4], "little", signed=True)
    vtable = table - back
    if back <= 0 or vtable < 4 or vtable + 4 > table:
        return None
    vtable_size = int.from_bytes(data[vtable : vtable + 2], "little")
    object_size = int.from_bytes(data[vtable + 2 : vtable + 4], "little")
    if vtable_size < 4 or vtable_size % 2 or object_size < 4:
        return None
    if vtable + vtable_size > table or table + object_size > len(data):
        return None
    return table, vtable, vtable_size, object_size


def _flatbuffer_root(data: bytes) -> tuple[int, int, int, int] | None:
    if len(data) < 12:
        return None
    return _flatbuffer_table_at(data, int.from_bytes(data[:4], "little"))


def _flatbuffer_field(
    data: bytes, table_info: tuple[int, int, int, int], slot: int, width: int = 1
) -> int | None:
    table, vtable, vtable_size, object_size = table_info
    entry = vtable + 4 + 2 * slot
    if entry + 2 > vtable + vtable_size:
        return None
    offset = int.from_bytes(data[entry : entry + 2], "little")
    if offset == 0 or offset + width > object_size or table + offset + width > len(data):
        return None
    return table + offset


def _flatbuffer_target(data: bytes, field: int | None) -> int | None:
    if field is None or field + 4 > len(data):
        return None
    target = field + int.from_bytes(data[field : field + 4], "little")
    return target if field < target < len(data) else None


def _flatbuffer_vector(data: bytes, field: int | None, width: int) -> tuple[int, int] | None:
    target = _flatbuffer_target(data, field)
    if target is None or target + 4 > len(data):
        return None
    count = int.from_bytes(data[target : target + 4], "little")
    start = target + 4
    if count > _MAX_ARROW_VECTOR_ITEMS or start + count * width > len(data):
        return None
    return start, count


def _flatbuffer_offset_vector(data: bytes, field: int | None) -> list[int] | None:
    vector = _flatbuffer_vector(data, field, 4)
    if vector is None:
        return None
    start, count = vector
    targets = []
    for index in range(count):
        at = start + 4 * index
        target = _flatbuffer_target(data, at)
        if target is None:
            return None
        targets.append(target)
    return targets


def _valid_arrow_field(data: bytes, target: int, depth: int = 0) -> bool:
    if depth > 16 or (table := _flatbuffer_table_at(data, target)) is None:
        return False
    name = _flatbuffer_target(data, _flatbuffer_field(data, table, 0, 4))
    if name is None or name + 4 > len(data):
        return False
    name_size = int.from_bytes(data[name : name + 4], "little")
    if name_size > _MAX_ARROW_METADATA_BYTES or name + 4 + name_size + 1 > len(data):
        return False
    type_kind_at = _flatbuffer_field(data, table, 2)
    type_at = _flatbuffer_target(data, _flatbuffer_field(data, table, 3, 4))
    if type_kind_at is None or not 1 <= data[type_kind_at] <= _MAX_ARROW_TYPE_KIND:
        return False
    if type_at is None or _flatbuffer_table_at(data, type_at) is None:
        return False
    children_field = _flatbuffer_field(data, table, 5, 4)
    if children_field is None:
        return True
    children = _flatbuffer_offset_vector(data, children_field)
    return children is not None and all(
        _valid_arrow_field(data, child, depth + 1) for child in children
    )


def _valid_arrow_schema(data: bytes, target: int) -> bool:
    table = _flatbuffer_table_at(data, target)
    if table is None:
        return False
    endian_at = _flatbuffer_field(data, table, 0, 2)
    if endian_at is not None and int.from_bytes(data[endian_at : endian_at + 2], "little") > 1:
        return False
    fields = _flatbuffer_offset_vector(data, _flatbuffer_field(data, table, 1, 4))
    if fields is None or not all(_valid_arrow_field(data, field) for field in fields):
        return False
    features = _flatbuffer_field(data, table, 3, 4)
    return features is None or _flatbuffer_vector(data, features, 8) is not None


def _valid_arrow_blocks(data: bytes, field: int | None) -> bool:
    if field is None:
        return True
    vector = _flatbuffer_vector(data, field, 24)
    if vector is None:
        return False
    start, count = vector
    return all(
        int.from_bytes(data[start + index * 24 : start + index * 24 + 8], "little", signed=True)
        >= 0
        for index in range(count)
    )


def _looks_like_arrow_footer(footer: bytes) -> bool:
    table = _flatbuffer_root(footer)
    if table is None:
        return False
    version_at = _flatbuffer_field(footer, table, 0, 2)
    version = int.from_bytes(footer[version_at : version_at + 2], "little") if version_at else 0
    schema = _flatbuffer_target(footer, _flatbuffer_field(footer, table, 1, 4))
    return (
        version <= 4
        and schema is not None
        and _valid_arrow_schema(footer, schema)
        and _valid_arrow_blocks(footer, _flatbuffer_field(footer, table, 2, 4))
        and _valid_arrow_blocks(footer, _flatbuffer_field(footer, table, 3, 4))
    )


def _looks_like_arrow_file(source: Path | bytes, deadline: float | None) -> bool:
    size = _source_size(source)
    if size is None or size < 20 or _read_at(source, 0, 8) != _ARROW_MAGIC + b"\x00\x00":
        return False
    trailer = _read_at(source, size - 10, 10)
    if trailer is None or trailer[-6:] != _ARROW_MAGIC:
        return False
    footer_size = int.from_bytes(trailer[:4], "little")
    footer_at = size - 10 - footer_size
    if not 12 <= footer_size <= _MAX_ARROW_METADATA_BYTES or footer_at < 8:
        return False
    _check_deadline(deadline)
    footer = _read_at(source, footer_at, footer_size)
    return footer is not None and len(footer) == footer_size and _looks_like_arrow_footer(footer)


def _looks_like_arrow_message(metadata: bytes) -> bool:
    table = _flatbuffer_root(metadata)
    if table is None:
        return False
    version_at = _flatbuffer_field(metadata, table, 0, 2)
    version = int.from_bytes(metadata[version_at : version_at + 2], "little") if version_at else 0
    header_type_at = _flatbuffer_field(metadata, table, 1)
    header = _flatbuffer_target(metadata, _flatbuffer_field(metadata, table, 2, 4))
    body_at = _flatbuffer_field(metadata, table, 3, 8)
    body_size = (
        int.from_bytes(metadata[body_at : body_at + 8], "little", signed=True) if body_at else 0
    )
    return (
        version <= 4
        and header_type_at is not None
        and metadata[header_type_at] == 1
        and header is not None
        and _valid_arrow_schema(metadata, header)
        and body_size == 0
    )


def _arrow_stream_frame(source: Path | bytes) -> tuple[int, int] | None:
    prefix = _read_at(source, 0, 8)
    if prefix is None or len(prefix) < 4:
        return None
    if prefix[:4] == _ARROW_CONTINUATION:
        if len(prefix) < 8:
            return None
        return 8, int.from_bytes(prefix[4:8], "little")
    return 4, int.from_bytes(prefix[:4], "little")


def _looks_like_arrow_stream(source: Path | bytes, deadline: float | None) -> bool:
    frame = _arrow_stream_frame(source)
    if frame is None:
        return False
    metadata_at, metadata_size = frame
    if not 12 <= metadata_size <= _MAX_ARROW_METADATA_BYTES or (metadata_at + metadata_size) % 8:
        return False
    _check_deadline(deadline)
    metadata = _read_at(source, metadata_at, metadata_size)
    return (
        metadata is not None
        and len(metadata) == metadata_size
        and _looks_like_arrow_message(metadata)
    )


class _PackReader:
    def __init__(self, source: Path | bytes, at: int, end: int):
        self.source = source
        self.at = at
        self.end = end

    def read_byte(self) -> int | None:
        value = _read_at(self.source, self.at, 1) if self.at < self.end else None
        if not value:
            return None
        self.at += 1
        return value[0]

    def skip(self, size: int) -> bool:
        if size < 0 or self.at + size > self.end:
            return False
        self.at += size
        return True

    def inflate(self, expected: int, deadline: float | None) -> bool | None:
        if expected > _MAX_GIT_PACK_OUTPUT_BYTES:
            return None
        inflate = zlib.decompressobj()
        produced = 0
        while self.at < self.end:
            _check_deadline(deadline)
            chunk = _read_at(self.source, self.at, min(1 << 20, self.end - self.at))
            if not chunk:
                return False
            limit = min(expected - produced + 1, _MAX_GIT_PACK_OUTPUT_BYTES - produced + 1)
            try:
                plain = inflate.decompress(chunk, max(1, limit))
            except zlib.error:
                return False
            produced += len(plain)
            consumed = len(chunk) - len(inflate.unconsumed_tail) - len(inflate.unused_data)
            if consumed <= 0 and not inflate.eof:
                return False
            self.at += consumed
            if produced > expected:
                return False
            if inflate.eof:
                return produced == expected
        return False


def _pack_object_size(reader: _PackReader) -> tuple[int, int] | None:
    first = reader.read_byte()
    if first is None:
        return None
    object_type = (first >> 4) & 7
    if object_type not in (1, 2, 3, 4, 6, 7):
        return None
    size, shift, byte = first & 0x0F, 4, first
    for _ in range(10):
        if not byte & 0x80:
            return object_type, size
        byte = reader.read_byte()
        if byte is None:
            return None
        size |= (byte & 0x7F) << shift
        shift += 7
    return None


def _skip_delta_base(reader: _PackReader, object_type: int, digest_size: int) -> bool:
    if object_type == 7:
        return reader.skip(digest_size)
    if object_type != 6:
        return True
    for _ in range(10):
        byte = reader.read_byte()
        if byte is None:
            return False
        if not byte & 0x80:
            return True
    return False


def _hash_matches(
    source: Path | bytes, start: int, end: int, digest_name: str, deadline: float | None
) -> bool:
    expected = _read_at(source, end, hashlib.new(digest_name).digest_size)
    if expected is None:
        return False
    digest = hashlib.new(digest_name)
    at = start
    while at < end:
        _check_deadline(deadline)
        block = _read_at(source, at, min(1 << 20, end - at))
        if not block:
            return False
        digest.update(block)
        at += len(block)
    return digest.digest() == expected


def _walk_pack(
    source: Path | bytes,
    offset: int,
    count: int,
    digest_size: int,
    end: int,
    deadline: float | None,
) -> bool | None:
    if count > _MAX_GIT_PACK_OBJECTS:
        return None
    reader = _PackReader(source, offset + _GIT_PACK_HEADER_BYTES, end)
    for _ in range(count):
        _check_deadline(deadline)
        parsed = _pack_object_size(reader)
        if parsed is None:
            return False
        object_type, expected = parsed
        if not _skip_delta_base(reader, object_type, digest_size):
            return False
        inflated = reader.inflate(expected, deadline)
        if inflated is not True:
            return inflated
    return reader.at == end


def _pack_digest(
    source: Path | bytes, offset: int, digest_name: str, deadline: float | None
) -> bool:
    size = _source_size(source)
    digest_size = hashlib.new(digest_name).digest_size
    if size is None or size - offset < _GIT_PACK_HEADER_BYTES + digest_size:
        return False
    header = _read_at(source, offset, _GIT_PACK_HEADER_BYTES)
    if header is None or header[:4] != b"PACK":
        return False
    version = int.from_bytes(header[4:8], "big")
    count = int.from_bytes(header[8:12], "big")
    end = size - digest_size
    if version not in (2, 3) or count == 0:
        return False
    if not _hash_matches(source, offset, end, digest_name, deadline):
        return False
    walked = _walk_pack(source, offset, count, digest_size, end, deadline)
    return walked is not False


def _looks_like_git_pack(source: Path | bytes, deadline: float | None) -> bool:
    return _pack_digest(source, 0, "sha1", deadline) or _pack_digest(source, 0, "sha256", deadline)


def _bundle_lines(lines: list[bytes], version: int) -> tuple[str, bool] | None:
    digest_name, saw_ref = "sha1", False
    for line in lines:
        if version == 3 and line.startswith(b"@"):
            if line == b"@object-format=sha256":
                digest_name = "sha256"
            elif line == b"@object-format=sha1" or line.startswith(b"@filter="):
                pass
            else:
                return None
        elif _BUNDLE_REF.fullmatch(line):
            saw_ref = True
        elif not _BUNDLE_PREREQUISITE.fullmatch(line):
            return None
    return digest_name, saw_ref


def _bundle_partial_line(line: bytes, version: int) -> bool:
    if not line:
        return True
    if version == 3 and line.startswith(b"@"):
        return all(32 <= byte < 127 for byte in line)
    return line[:1] in b"-0123456789abcdef" and all(32 <= byte < 127 for byte in line)


def _git_bundle_pack(
    source: Path | bytes, deadline: float | None
) -> tuple[int, str] | object | None:
    size = _source_size(source)
    if size is None:
        return None
    _check_deadline(deadline)
    prefix = _read_at(source, 0, min(size, _MAX_GIT_BUNDLE_HEADER_BYTES + 1))
    if prefix is None:
        return None
    if prefix.startswith(b"# v2 git bundle\n"):
        version = 2
    elif prefix.startswith(b"# v3 git bundle\n"):
        version = 3
    else:
        return None
    header_end = prefix.find(b"\n")
    separator = prefix.find(b"\n\n", header_end + 1)
    if separator >= 0:
        parsed = _bundle_lines(prefix[header_end + 1 : separator].splitlines(), version)
        if parsed is None or not parsed[1] or prefix[separator + 2 : separator + 6] != b"PACK":
            return None
        return separator + 2, parsed[0]
    if size <= _MAX_GIT_BUNDLE_HEADER_BYTES:
        return None
    body = prefix[header_end + 1 : _MAX_GIT_BUNDLE_HEADER_BYTES]
    lines = body.split(b"\n")
    parsed = _bundle_lines(lines[:-1], version)
    if parsed is None or not _bundle_partial_line(lines[-1], version):
        return None
    return _BUNDLE_UNFINISHED


def _looks_like_git_bundle(source: Path | bytes, deadline: float | None) -> bool:
    found = _git_bundle_pack(source, deadline)
    if found is _BUNDLE_UNFINISHED:
        return True
    return isinstance(found, tuple) and _pack_digest(source, found[0], found[1], deadline)


def _looks_like_rpm(source: Path | bytes) -> bool:
    lead = _read_at(source, 0, _RPM_LEAD_BYTES)
    if lead is None or len(lead) < _RPM_LEAD_BYTES or lead[:4] != b"\xed\xab\xee\xdb":
        return False
    package_type = int.from_bytes(lead[6:8], "big")
    name = lead[10:76].split(b"\x00", 1)[0]
    return (
        lead[4:6] == b"\x03\x00"
        and package_type in (0, 1)
        and bool(name)
        and all(32 <= byte < 127 for byte in name)
        and int.from_bytes(lead[76:78], "big") == 1
        and int.from_bytes(lead[78:80], "big") == 5
        and lead[80:96] == bytes(16)
    )


def looks_like_opaque_candidate(data: bytes) -> bool:
    """Whether a bounded stream head needs buffering for structural validation."""
    if (
        _looks_like_hdf5(data, None)
        or _looks_like_arrow_stream(data, None)
        or _looks_like_rpm(data)
    ):
        return True
    if data.startswith((_ARROW_MAGIC + b"\x00\x00", b"# v2 git bundle\n", b"# v3 git bundle\n")):
        return True
    if len(data) < _GIT_PACK_HEADER_BYTES or data[:4] != b"PACK":
        return False
    return int.from_bytes(data[4:8], "big") in (2, 3) and int.from_bytes(data[8:12], "big") > 0


def opaque_format(source: Path | bytes, *, deadline: float | None = None) -> str | None:
    """The structurally proven opaque format carried by `source`, or None."""
    checks = (
        ("HDF5", _looks_like_hdf5),
        ("Arrow IPC", _looks_like_arrow_file),
        ("Arrow IPC", _looks_like_arrow_stream),
        ("Git pack", _looks_like_git_pack),
        ("Git bundle", _looks_like_git_bundle),
    )
    for name, check in checks:
        _check_deadline(deadline)
        if check(source, deadline):
            return name
    return "RPM" if _looks_like_rpm(source) else None
