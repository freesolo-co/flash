"""operator-secret files that .gitignore keeps out of the TREE must also stay out of the docker
BUILD CONTEXT.

the two files are easy to drift apart because they look interchangeable and are not: .gitignore has
no effect whatsoever on what docker copies. ``Dockerfile`` is single-stage and does a bare
``COPY . .``, so any untracked operator file sitting in the working tree at build time is baked into
an image layer and shipped. that is the same leak path .env is already excluded for, and the failure
is silent -- the build succeeds and the secret rides along.

``scripts/scrub_identities.env`` is the sharp case: it holds the addresses and the internal hostname
that ``scripts/scrub_history.sh`` exists to delete from history, so a local build on the maintainer's
machine would republish in an image exactly what the scrub removes from the repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# paths .gitignore marks as operator-supplied secrets. each must ALSO be excluded from the build
# context. add to this list whenever a new operator-secret path is gitignored.
OPERATOR_SECRET_PATHS = [
    ".env",
    "scripts/scrub_identities.env",
]


def _ignore_entries(name: str) -> set[str]:
    lines = (ROOT / name).read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def test_dockerfile_still_copies_the_whole_context():
    """the reason the contract below exists. if this ever stops being true, revisit it."""
    assert "COPY . ." in (ROOT / "Dockerfile").read_text(), (
        "Dockerfile no longer does a bare `COPY . .`; the build-context exclusions below were "
        "written for that copy and should be re-derived rather than trusted"
    )


@pytest.mark.parametrize("path", OPERATOR_SECRET_PATHS)
def test_operator_secrets_are_ignored_by_both_git_and_docker(path: str):
    assert path in _ignore_entries(".gitignore"), (
        f"{path} is treated as an operator secret but is not in .gitignore"
    )
    assert path in _ignore_entries(".dockerignore"), (
        f"{path} is gitignored as an operator secret but is not in .dockerignore. .gitignore does "
        "not filter the docker build context, so `COPY . .` would bake the file into an image layer"
    )
