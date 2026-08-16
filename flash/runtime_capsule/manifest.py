"""The capsule manifest: what a capsule declares about itself, and how it is validated.

A capsule is a zipapp whose members are declared here by path, size, and sha256. Validation is
FAIL-CLOSED in both directions: a declared member that is absent, altered, or truncated is a
rejection, and so is an archive member nobody declared. The asymmetry matters -- checking only
that declared members match lets an attacker ADD an executable module the manifest never names,
and the loader would import it.

The manifest cannot authenticate itself. An archive cannot contain its own digest, and a manifest
sitting inside the archive can be replaced consistently with the members it describes. The
expected digest therefore lives OUTSIDE the archive, in the trusted job payload or the image
label, and `verify_archive_digest` binds the two.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# bumped only for an incompatible manifest shape. an unknown version is rejected rather than
# best-effort parsed: a reader that guesses at a shape it does not know cannot fail closed.
FORMAT_VERSION = 1

MANIFEST_NAME = "flash_capsule.json"
KIND = "flash-runtime-capsule"

_MANIFEST_FIELDS = frozenset(
    {"kind", "format_version", "profile", "profile_version", "entrypoint", "members"}
)
_MEMBER_FIELDS = frozenset({"path", "size", "sha256"})

_SHA256_LENGTH = 64
_SHA256_ALPHABET = frozenset("0123456789abcdef")


class CapsuleError(Exception):
    """A capsule failed to build or failed verification."""


@dataclass(frozen=True)
class Member:
    """One declared file inside a capsule."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    """The full declaration of a capsule's contents."""

    profile: str
    profile_version: int
    entrypoint: str
    members: tuple[Member, ...]

    def to_json(self) -> str:
        """Canonical JSON: sorted keys, compact separators, trailing newline.

        Canonical because the manifest is hashed as part of the archive. Two builds of the same
        inputs must produce identical bytes or the digest check becomes a coin flip.
        """
        payload = {
            "kind": KIND,
            "format_version": FORMAT_VERSION,
            "profile": self.profile,
            "profile_version": self.profile_version,
            "entrypoint": self.entrypoint,
            "members": [
                {"path": m.path, "size": m.size, "sha256": m.sha256}
                for m in sorted(self.members, key=lambda m: m.path)
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def validate_member_path(path: str) -> str:
    """Return `path` if it is a safe, relative, single-form member path, else raise.

    Rejects every shape that could escape the extraction root or alias another member: absolute
    paths, parent traversal, backslashes (a Windows separator that POSIX reads as a filename
    character, so the same archive means two different things), drive letters, and empty or `.`
    segments. Mirrors the tar-member rules in `flash.envs.package.limits` -- the two answer the
    same question about different archive formats.
    """
    if not isinstance(path, str) or not path:
        raise CapsuleError(f"capsule member path must be a non-empty string, got {path!r}")
    if path.startswith("/"):
        raise CapsuleError(f"capsule member path must be relative: {path!r}")
    if "\\" in path:
        raise CapsuleError(f"capsule member path must not contain a backslash: {path!r}")
    # a drive letter is only meaningful on windows, where "c:x" resolves against a per-drive cwd
    # rather than the extraction root.
    if len(path) >= 2 and path[1] == ":":
        raise CapsuleError(f"capsule member path must not carry a drive letter: {path!r}")
    for segment in path.split("/"):
        if not segment:
            raise CapsuleError(f"capsule member path has an empty segment: {path!r}")
        if segment in {".", ".."}:
            raise CapsuleError(f"capsule member path has a traversal segment: {path!r}")
    return path


def _validate_digest(value: object, *, where: str) -> str:
    """Return `value` if it is a lowercase hex sha256, else raise."""
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise CapsuleError(
            f"{where}: sha256 must be {_SHA256_LENGTH} hex characters, got {value!r}"
        )
    if not set(value) <= _SHA256_ALPHABET:
        raise CapsuleError(f"{where}: sha256 must be lowercase hex, got {value!r}")
    return value


def sha256_bytes(data: bytes) -> str:
    """The lowercase hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def parse_manifest(raw: bytes) -> Manifest:
    """Parse and fully validate manifest bytes.

    Unknown fields and unknown versions are rejected rather than ignored. Ignoring an unknown
    field silently accepts a manifest written for rules this reader does not enforce.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapsuleError(f"capsule manifest is not valid utf-8 json: {exc}") from exc
    if not isinstance(payload, dict):
        raise CapsuleError(f"capsule manifest must be a json object, got {type(payload).__name__}")

    unknown = sorted(set(payload) - _MANIFEST_FIELDS)
    if unknown:
        raise CapsuleError(f"capsule manifest has unknown field(s): {', '.join(unknown)}")
    missing = sorted(_MANIFEST_FIELDS - set(payload))
    if missing:
        raise CapsuleError(f"capsule manifest is missing field(s): {', '.join(missing)}")

    if payload["kind"] != KIND:
        raise CapsuleError(f"capsule manifest kind must be {KIND!r}, got {payload['kind']!r}")
    if payload["format_version"] != FORMAT_VERSION:
        raise CapsuleError(
            f"unsupported capsule format_version {payload['format_version']!r} "
            f"(this build understands {FORMAT_VERSION})"
        )

    profile = payload["profile"]
    if not isinstance(profile, str) or not profile:
        raise CapsuleError(f"capsule profile must be a non-empty string, got {profile!r}")
    profile_version = payload["profile_version"]
    # bool is an int subclass, and `True` as a version would compare equal to 1 while meaning
    # nothing; reject it explicitly.
    if not isinstance(profile_version, int) or isinstance(profile_version, bool):
        raise CapsuleError(f"capsule profile_version must be an int, got {profile_version!r}")

    entrypoint = validate_member_path(payload["entrypoint"])

    raw_members = payload["members"]
    if not isinstance(raw_members, list) or not raw_members:
        raise CapsuleError("capsule manifest must declare a non-empty members list")

    members: list[Member] = []
    seen: set[str] = set()
    for entry in raw_members:
        if not isinstance(entry, dict):
            raise CapsuleError(f"capsule member must be a json object, got {entry!r}")
        unknown_member = sorted(set(entry) - _MEMBER_FIELDS)
        if unknown_member:
            raise CapsuleError(f"capsule member has unknown field(s): {', '.join(unknown_member)}")
        missing_member = sorted(_MEMBER_FIELDS - set(entry))
        if missing_member:
            raise CapsuleError(f"capsule member is missing field(s): {', '.join(missing_member)}")

        path = validate_member_path(entry["path"])
        if path in seen:
            raise CapsuleError(f"capsule manifest declares {path!r} more than once")
        seen.add(path)

        size = entry["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CapsuleError(f"capsule member {path!r}: size must be a non-negative int")
        members.append(
            Member(path=path, size=size, sha256=_validate_digest(entry["sha256"], where=path))
        )

    if entrypoint not in seen:
        raise CapsuleError(f"capsule entrypoint {entrypoint!r} is not a declared member")

    return Manifest(
        profile=profile,
        profile_version=profile_version,
        entrypoint=entrypoint,
        members=tuple(members),
    )


def verify_members(manifest: Manifest, contents: dict[str, bytes]) -> None:
    """Check `contents` against `manifest`, failing closed in BOTH directions.

    `contents` maps member path to bytes and must be exactly the archive's payload members (the
    manifest itself excluded). An undeclared member is as fatal as a missing one.
    """
    declared = {m.path for m in manifest.members}
    present = set(contents)

    missing = sorted(declared - present)
    if missing:
        raise CapsuleError(f"capsule is missing declared member(s): {', '.join(missing)}")
    undeclared = sorted(present - declared)
    if undeclared:
        raise CapsuleError(f"capsule contains undeclared member(s): {', '.join(undeclared)}")

    for member in manifest.members:
        data = contents[member.path]
        if len(data) != member.size:
            raise CapsuleError(
                f"capsule member {member.path!r} size mismatch: "
                f"declared {member.size}, found {len(data)}"
            )
        actual = sha256_bytes(data)
        if actual != member.sha256:
            raise CapsuleError(
                f"capsule member {member.path!r} sha256 mismatch: "
                f"declared {member.sha256}, found {actual}"
            )


def verify_archive_digest(archive: bytes, expected_sha256: str) -> None:
    """Check the whole archive against a digest supplied from OUTSIDE it.

    This is the only check a replaced-but-internally-consistent capsule cannot satisfy, so it is
    the one that actually decides whether the code is ours.
    """
    expected = _validate_digest(expected_sha256, where="expected capsule digest")
    actual = sha256_bytes(archive)
    if actual != expected:
        raise CapsuleError(f"capsule digest mismatch: expected {expected}, found {actual}")
