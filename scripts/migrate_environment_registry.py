#!/usr/bin/env python3
"""Backfill a clean environment-hub checkout into the configured private Blob container.

Dry-run is the default. Existing named heads are never overwritten. Import the exact
cutover commit to preserve its Git revision as an alias for already pinned jobs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from flash.envs.package.direct_tokens import package_contains_direct_token
from flash.envs.package.limits import ARCHIVE_MEMBER_LIMIT
from flash.envs.registry.azure import open_registry, validate_slug
from flash.envs.registry.packages import publish_directory


def snapshot(checkout: Path) -> tuple[str, list[str]]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=all", "--ignored"):
        raise RuntimeError("migration requires a clean checkout, including untracked files")
    revision = git("rev-parse", "HEAD")
    paths = git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
    slugs = sorted(
        validate_slug(path.removesuffix("/environment.py"))
        for path in paths
        if path.endswith("/environment.py") and len(path.split("/")) == 4
    )
    if not slugs:
        raise RuntimeError("checkout contains no managed environment packages")
    if any(not (checkout / slug / "environment.py").is_file() for slug in slugs):
        raise RuntimeError("migration requires a full checkout; disable sparse checkout first")
    return revision, slugs


def migrate(checkout: Path, *, apply: bool = False) -> dict:
    revision, slugs = snapshot(checkout)
    report = {
        "source_revision": revision,
        "packages": len(slugs),
        "applied": apply,
        "verified": 0,
        "failures": [],
    }
    for slug in slugs:
        try:
            root = checkout / slug
            members = list(root.rglob("*"))
            if root.is_symlink() or any(path.is_symlink() for path in members):
                raise RuntimeError("package contains symlinks")
            if (
                len(members) > ARCHIVE_MEMBER_LIMIT
                or sum(path.stat().st_size for path in members if path.is_file())
                > 256 * 1024 * 1024
            ):
                raise RuntimeError("package exceeds the runtime archive limits")
            if package_contains_direct_token(root):
                raise RuntimeError("package contains a direct access token")
            if apply:
                record = publish_directory(slug, root, create_only=True, source_revision=revision)
                with open_registry() as registry:
                    registry.download(registry.resolve(slug, record.revision))
            report["verified"] += 1
        except Exception as exc:
            # Never print SDK errors or file contents: errors may contain credentials.
            report["failures"].append({"slug": slug, "error_type": type(exc).__name__})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = migrate(args.checkout.resolve(), apply=args.apply)
    print(json.dumps(report, indent=2))
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
