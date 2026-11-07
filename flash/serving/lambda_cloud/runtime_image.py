"""Reject dependency drift from the pinned runtime while allowing package version bumps."""

import sys
import tomllib
from pathlib import Path


def dependency_lock(path):
    value = tomllib.loads(Path(path).read_text())
    for package in value.get("package", []):
        if package["name"] == "freesolo-flash":
            package.pop("version", None)
    return value


def verify(image_lock, source_lock):
    if dependency_lock(image_lock) != dependency_lock(source_lock):
        raise RuntimeError(
            "Lambda runtime image dependency lock mismatch; qualify a new image digest"
        )


if __name__ == "__main__":
    verify(*sys.argv[1:])
