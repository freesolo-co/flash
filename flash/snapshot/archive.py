"""Deterministic managed source snapshots and trusted materialization."""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

KIND = "flash-source-snapshot"
FORMAT_VERSION = 1
ATTESTATION_KIND = "flash-source-attestation"
ATTESTATION_FORMAT_VERSION = 1
PUBLIC_PROVENANCE_KEY = "source_provenance"
TERMINAL_ATTESTATION_KEY = "_flash_source_attestation"

_DESCRIPTOR_FIELDS = frozenset(
    {"kind", "format_version", "archive_path", "sha256", "size", "revision"}
)
_ATTESTATION_FIELDS = frozenset(
    {"kind", "format_version", "sha256", "revision", "run_id", "attempt"}
)
_SHA256_ALPHABET = frozenset("0123456789abcdef")
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_FIXED_MODE = 0o644
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TRANSIENT_NETWORK_ERROR_NAMES = frozenset(
    {
        "CloseError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "NetworkError",
        "PoolTimeout",
        "ProtocolError",
        "ProxyError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "SSLError",
        "TimeoutError",
        "TimeoutException",
        "WriteError",
        "WriteTimeout",
        "gaierror",
    }
)


class SourceSnapshotError(RuntimeError):
    """A source snapshot failed strict construction or verification."""


@dataclass(frozen=True)
class SourceSnapshotDescriptor:
    archive_path: str
    sha256: str
    size: int
    revision: str

    def to_dict(self) -> dict:
        return {
            "kind": KIND,
            "format_version": FORMAT_VERSION,
            "archive_path": self.archive_path,
            "sha256": self.sha256,
            "size": self.size,
            "revision": self.revision,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def response_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def is_transient_fetch_error(exc: BaseException) -> bool:
    """Return whether a failed immutable source fetch is safe to retry."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = response_status_code(current)
        if status is not None:
            return status == 429 or 500 <= status <= 599
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if any(cls.__name__ in _TRANSIENT_NETWORK_ERROR_NAMES for cls in type(current).__mro__):
            return True
        current = current.__cause__ or current.__context__
    return False


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _SHA256_ALPHABET:
        raise SourceSnapshotError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_revision(value: object) -> str:
    if not isinstance(value, str) or len(value) != 40 or not set(value) <= _SHA256_ALPHABET:
        raise SourceSnapshotError(
            "source snapshot revision must be a 40-character lowercase commit id"
        )
    return value


def validate_member_path(path: object) -> str:
    if not isinstance(path, str) or not path:
        raise SourceSnapshotError("source snapshot member path must be a non-empty string")
    if path.startswith("/") or "\\" in path or (len(path) >= 2 and path[1] == ":"):
        raise SourceSnapshotError(f"unsafe source snapshot member path: {path!r}")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SourceSnapshotError(f"unsafe source snapshot member path: {path!r}")
    canonical = "/".join(parts)
    if canonical != path:
        raise SourceSnapshotError(f"aliased source snapshot member path: {path!r}")
    return path


def canonical_archive_path(digest: str) -> str:
    digest = _require_digest(digest, label="source snapshot digest")
    return f"source/{digest}/flash-source.zip"


def parse_descriptor(raw: object) -> SourceSnapshotDescriptor:
    if isinstance(raw, SourceSnapshotDescriptor):
        raw = raw.to_dict()
    if not isinstance(raw, dict):
        raise SourceSnapshotError("source snapshot descriptor must be an object")
    if set(raw) != _DESCRIPTOR_FIELDS:
        missing = sorted(_DESCRIPTOR_FIELDS - set(raw))
        extra = sorted(set(raw) - _DESCRIPTOR_FIELDS)
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if extra:
            detail.append(f"unknown {', '.join(extra)}")
        raise SourceSnapshotError("invalid source snapshot descriptor: " + "; ".join(detail))
    if raw["kind"] != KIND or raw["format_version"] != FORMAT_VERSION:
        raise SourceSnapshotError("unsupported source snapshot descriptor protocol")
    digest = _require_digest(raw["sha256"], label="source snapshot digest")
    size = raw["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SourceSnapshotError("source snapshot size must be a positive integer")
    archive_path = raw["archive_path"]
    if archive_path != canonical_archive_path(digest):
        raise SourceSnapshotError("source snapshot archive path does not match its digest")
    return SourceSnapshotDescriptor(
        archive_path=archive_path,
        sha256=digest,
        size=size,
        revision=_require_revision(raw["revision"]),
    )


def descriptor_for_archive(archive: bytes, revision: str) -> SourceSnapshotDescriptor:
    if not isinstance(archive, bytes) or not archive:
        raise SourceSnapshotError("source snapshot archive must be non-empty bytes")
    digest = sha256_bytes(archive)
    return SourceSnapshotDescriptor(
        archive_path=canonical_archive_path(digest),
        sha256=digest,
        size=len(archive),
        revision=_require_revision(revision),
    )


def safe_public_projection(
    descriptor: SourceSnapshotDescriptor | dict,
    *,
    verified_attempt: int | None = None,
) -> dict:
    parsed = parse_descriptor(descriptor)
    verified = (
        isinstance(verified_attempt, int)
        and not isinstance(verified_attempt, bool)
        and verified_attempt >= 0
    )
    return {
        "format_version": parsed.to_dict()["format_version"],
        "sha256": parsed.sha256,
        "verified": verified,
        "verified_attempt": verified_attempt if verified else None,
    }


def source_attestation(
    descriptor: SourceSnapshotDescriptor | dict,
    *,
    run_id: str,
    attempt: int,
) -> dict:
    parsed = parse_descriptor(descriptor)
    if not isinstance(run_id, str) or not run_id:
        raise SourceSnapshotError("source attestation run_id is invalid")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise SourceSnapshotError("source attestation attempt is invalid")
    return {
        "kind": ATTESTATION_KIND,
        "format_version": ATTESTATION_FORMAT_VERSION,
        "sha256": parsed.sha256,
        "revision": parsed.revision,
        "run_id": run_id,
        "attempt": attempt,
    }


def parse_attestation(raw: object) -> dict:
    if not isinstance(raw, dict) or set(raw) != _ATTESTATION_FIELDS:
        raise SourceSnapshotError("source attestation schema is invalid")
    if raw["kind"] != ATTESTATION_KIND or raw["format_version"] != ATTESTATION_FORMAT_VERSION:
        raise SourceSnapshotError("source attestation protocol is unsupported")
    _require_digest(raw["sha256"], label="source attestation digest")
    _require_revision(raw["revision"])
    if not isinstance(raw["run_id"], str) or not raw["run_id"]:
        raise SourceSnapshotError("source attestation run_id is invalid")
    attempt = raw["attempt"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise SourceSnapshotError("source attestation attempt is invalid")
    return dict(raw)


def validate_attestation(
    raw: object,
    descriptor: SourceSnapshotDescriptor | dict,
    *,
    run_id: str,
    attempt: int,
) -> dict:
    attestation = parse_attestation(raw)
    expected = source_attestation(descriptor, run_id=run_id, attempt=attempt)
    if attestation != expected:
        raise SourceSnapshotError("terminal source attestation does not match the managed attempt")
    return attestation


def _enumerate_package_files(package_dir: Path) -> dict[str, bytes]:
    package_dir = package_dir.resolve()
    if not package_dir.is_dir():
        raise SourceSnapshotError(f"source package directory does not exist: {package_dir}")
    contents: dict[str, bytes] = {}

    def visit(directory: Path, relative: tuple[str, ...]) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                name = entry.name
                if name == "__pycache__" or name.startswith("."):
                    continue
                member_parts = ("flash", *relative, name)
                member = validate_member_path("/".join(member_parts))
                if entry.is_symlink():
                    raise SourceSnapshotError(f"source package contains a symlink: {member!r}")
                if entry.is_dir(follow_symlinks=False):
                    visit(Path(entry.path), (*relative, name))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise SourceSnapshotError(
                        f"source package contains a non-regular file: {member!r}"
                    )
                if name.endswith((".pyc", ".pyo")):
                    continue
                with open(entry.path, "rb") as source:
                    contents[member] = source.read()

    visit(package_dir, ())
    if not contents:
        raise SourceSnapshotError("source package contains no files")
    return contents


def _zip_bytes(payload: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(payload):
            info = zipfile.ZipInfo(name, date_time=_FIXED_DATE_TIME)
            info.create_system = 3
            info.external_attr = (_FIXED_MODE | stat.S_IFREG) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload[name])
    return buffer.getvalue()


def build_source_archive(*, package_dir: Path | None = None) -> bytes:
    if package_dir is None:
        package_dir = Path(__file__).resolve().parents[1]
    return _zip_bytes(_enumerate_package_files(package_dir))


def read_verified_archive(
    archive: bytes,
    descriptor: SourceSnapshotDescriptor | dict,
) -> dict[str, bytes]:
    parsed = parse_descriptor(descriptor)
    if not isinstance(archive, bytes) or len(archive) != parsed.size:
        raise SourceSnapshotError("source snapshot archive size mismatch")
    if sha256_bytes(archive) != parsed.sha256:
        raise SourceSnapshotError("source snapshot archive digest mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as source_zip:
            infos = source_zip.infolist()
            if not infos:
                raise SourceSnapshotError("source snapshot archive contains no members")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise SourceSnapshotError("source snapshot archive contains duplicate members")
            payload: dict[str, bytes] = {}
            for info in infos:
                name = validate_member_path(info.filename)
                if not name.startswith("flash/"):
                    raise SourceSnapshotError(
                        f"source snapshot member is outside the flash package: {name!r}"
                    )
                mode = info.external_attr >> 16
                if info.is_dir():
                    raise SourceSnapshotError(
                        f"source snapshot archive contains a directory entry: {name!r}"
                    )
                if stat.S_ISLNK(mode):
                    raise SourceSnapshotError(
                        f"source snapshot archive contains a symlink: {name!r}"
                    )
                if mode and not stat.S_ISREG(mode):
                    raise SourceSnapshotError(
                        f"source snapshot archive contains a non-regular member: {name!r}"
                    )
                payload[name] = source_zip.read(info)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, SourceSnapshotError):
            raise
        raise SourceSnapshotError("source snapshot archive is not a readable zip") from exc
    return payload


def attempt_materialization_path(root: Path | str, run_id: str, attempt: int) -> Path:
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise SourceSnapshotError("source materialization run_id is invalid")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise SourceSnapshotError("source materialization attempt is invalid")
    return Path(root) / f"{run_id}-attempt-{attempt}"


def _materialize_contents(contents: dict[str, bytes], destination: Path | str) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SourceSnapshotError(f"source snapshot destination already exists: {destination}")
    temp_root = Path(tempfile.mkdtemp(prefix=".flash-source-", dir=destination.parent))
    try:
        for member, data in sorted(contents.items()):
            path = temp_root.joinpath(*member.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "xb") as output:
                output.write(data)
        os.replace(temp_root, destination)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return destination


def materialize_verified_archive_file(
    path: Path | str,
    descriptor: SourceSnapshotDescriptor | dict,
    destination: Path | str,
) -> Path:
    parsed = parse_descriptor(descriptor)
    with open(path, "rb") as source:
        archive = source.read(parsed.size + 1)
    if len(archive) != parsed.size:
        raise SourceSnapshotError("source snapshot archive size mismatch")
    return _materialize_contents(read_verified_archive(archive, parsed), destination)
