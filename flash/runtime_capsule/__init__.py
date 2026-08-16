"""Versioned, checksummed runtime capsules.

Flash runs code on machines it does not own: RunPod workers, Lambda instances, Vast containers.
Historically that code travelled as SOURCE TEXT -- read off disk, docstring-stripped, embedded in
a shell heredoc, and executed on the far end. Nothing checked that what arrived was what was sent,
and a local refactor could raise NameError only once a paid GPU had already booted.

A capsule replaces that with a declared artifact: a deterministic zipapp whose every member is
named, sized, and sha256'd in a manifest, carrying a version so the far end can refuse a shape it
does not understand. The expected archive digest travels OUTSIDE the archive (job payload or image
label), which is what makes a substituted-but-self-consistent capsule detectable.

Stdlib-only on purpose: a capsule may have to be verified before any dependency is installed.
"""

from __future__ import annotations

from flash.runtime_capsule.build import (
    build_capsule,
    read_capsule,
    verify_capsule,
    write_capsule,
)
from flash.runtime_capsule.manifest import (
    FORMAT_VERSION,
    MANIFEST_NAME,
    CapsuleError,
    Manifest,
    Member,
    sha256_bytes,
)
from flash.runtime_capsule.profiles import PROFILES, Profile, get_profile

__all__ = [
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "PROFILES",
    "CapsuleError",
    "Manifest",
    "Member",
    "Profile",
    "build_capsule",
    "get_profile",
    "read_capsule",
    "sha256_bytes",
    "verify_capsule",
    "write_capsule",
]
