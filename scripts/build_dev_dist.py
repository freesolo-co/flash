#!/usr/bin/env python3
"""Build the dev-channel distribution `freesolo-flash-dev` from this repo.

The dev channel is the *same source* as `freesolo-flash`, transformed at build time so it is a
distinct, side-by-side-installable package that defaults to the staging control plane:

  * pyproject `[project].name`    : freesolo-flash  -> freesolo-flash-dev
  * pyproject `[project].version` : [project].version -> [tool.flash-dev].version
  * pyproject `[project.scripts]` : flash -> flash-dev, flash-server -> flash-dev-server
  * flash/_channel.py             : CHANNEL = "prod" -> CHANNEL = "dev"
                                    (from which CLI_NAME=flash-dev, DIST_NAME=freesolo-flash-dev,
                                     and the flash-dev.freesolo.co default URL all derive)

It then runs `uv build` to produce sdist + wheel in ./dist.

This MUTATES the working tree in place (pyproject.toml + flash/_channel.py), so run it in a
disposable checkout — which is exactly the CI case (.github/workflows/publish-dev.yml). The prod
build path (uv build with no transform) is unaffected: the checked-in source always ships the prod
channel.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from pathlib import Path

DEV_DIST_NAME = "freesolo-flash-dev"
# pyproject [project.scripts] key renames for side-by-side install with the prod package.
SCRIPT_RENAMES = {"flash": "flash-dev", "flash-server": "flash-dev-server"}


def read_dev_version(pyproject_text: str) -> str:
    """The dev-channel version from pyproject `[tool.flash-dev].version`."""
    data = tomllib.loads(pyproject_text)
    try:
        return str(data["tool"]["flash-dev"]["version"])
    except KeyError as exc:  # pragma: no cover - config error surfaced to the operator
        raise SystemExit(
            "pyproject.toml is missing [tool.flash-dev].version (the dev-channel version)."
        ) from exc


def rewrite_pyproject(text: str, dev_version: str) -> str:
    """Rename the package to the dev distribution and retarget name/version/scripts.

    Walks the file tracking the current ``[table]`` header so it only touches `[project]` and
    `[project.scripts]` — crucially leaving the identically-named `version` line under
    `[tool.flash-dev]` (and any `flash`-prefixed strings elsewhere) untouched.
    """
    out: list[str] = []
    section: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            out.append(line)
            continue
        if section == "project":
            if re.match(r"\s*name\s*=", line):
                out.append(f'name = "{DEV_DIST_NAME}"\n')
                continue
            if re.match(r"\s*version\s*=", line):
                out.append(f'version = "{dev_version}"\n')
                continue
        elif section == "project.scripts" and "=" in line and not stripped.startswith("#"):
            key, _, value = line.partition("=")
            renamed = SCRIPT_RENAMES.get(key.strip().strip("\"'"))
            if renamed is not None:
                out.append(f"{renamed} ={value}")
                continue
        out.append(line)
    return "".join(out)


def rewrite_channel(text: str) -> str:
    """Flip the release-channel marker from prod to dev."""
    new_text, count = re.subn(
        r'^CHANNEL\s*=\s*["\']prod["\']', 'CHANNEL = "dev"', text, flags=re.MULTILINE
    )
    if count != 1:
        raise SystemExit(
            'expected exactly one `CHANNEL = "prod"` line in flash/_channel.py; '
            f"found {count}."
        )
    return new_text


def transform_tree(root: Path) -> str:
    """Apply the dev-channel rewrites in place under ``root``; return the dev version."""
    pyproject = root / "pyproject.toml"
    channel = root / "flash" / "_channel.py"
    pyproject_text = pyproject.read_text()
    dev_version = read_dev_version(pyproject_text)
    pyproject.write_text(rewrite_pyproject(pyproject_text, dev_version))
    channel.write_text(rewrite_channel(channel.read_text()))
    return dev_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the freesolo-flash-dev distribution.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repo root (default: the parent of scripts/).",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="apply the source rewrites but skip `uv build` (for inspecting the transform).",
    )
    args = parser.parse_args(argv)

    dev_version = transform_tree(args.root)
    print(f"Transformed source to dev channel: {DEV_DIST_NAME} {dev_version}")
    if args.no_build:
        return 0

    print("Running `uv build`...")
    subprocess.run(["uv", "build"], cwd=args.root, check=True)
    for dist in sorted((args.root / "dist").glob("*")):
        print(f"  built {dist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
