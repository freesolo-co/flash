#!/usr/bin/env python3
"""restore nvidia-cutlass-dsl-libs-cu13 over the -libs-base content that overwrote it.

both wheels ship into the SAME ``nvidia_cutlass_dsl/`` namespace and write many of the same paths
with different content, so whichever extracts last wins. under a parallel installer the order is
racy (NVIDIA/cutlass #3170, #3259), and when ``-libs-base`` lands last the cu13 files carry base
content at cu13 paths.

why this matters for blackwell serving: vllm's ``_resolve_gdn_prefill_backend`` selects flashinfer
gdn prefill on sm10.x ONLY when ``_is_libs_cu13_install_intact()`` passes. it fails silently -- the
branch falls through to triton after a ``warning_once`` -- so an unrepaired b200 still boots, still
serves, and still bills the b200 rate while running the slower kernel. sm90 (h100/h200) takes
flashinfer with no such condition, which is why this never bit the current tiers.

the repair reinstalls both wheels at an EXACT pinned version. an unpinned
``--force-reinstall --no-deps`` takes the newest release instead of the resolved pin, which produces
a uniformly-newer ``-libs-cu13`` against 4.5.2 siblings -- and that skew still PASSES the intactness
check, because the check re-hashes the package against its own RECORD. the check is structurally
blind to the defect the naive fix introduces, so the version is pinned here rather than discovered.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# pinned by the user in CLAUDE.md, with required sha-256 digests. only these exact artifacts are
# authorized for download and execution; anything else is a hard failure, never a fallback.
_PINNED_VERSION: Final = "4.5.2"


@dataclass(frozen=True)
class Wheel:
    name: str
    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


WHEELS: Final = (
    Wheel(
        "nvidia-cutlass-dsl-libs-cu13",
        "https://files.pythonhosted.org/packages/03/60/443e559139da15ab544761ac14f4206dffb981af48cc9856cd5b5b7cf0e7/nvidia_cutlass_dsl_libs_cu13-4.5.2-cp312-cp312-manylinux_2_28_x86_64.whl",
        "80f0cd402e0f1d1571e5aed33bfa17dbc9cb90cc5b1352f0f806b4788558e80e",
    ),
    Wheel(
        "nvidia-cutlass-dsl-libs-base",
        "https://files.pythonhosted.org/packages/97/68/c1247ab848f26c4ab56e562eea0e3f31fc14c9aaf0d883afaa92d8f05592/nvidia_cutlass_dsl_libs_base-4.5.2-cp312-cp312-manylinux_2_28_x86_64.whl",
        "15ef6a59193667e663934ef4873f8ccad37455e9b7c3c419c3072113b8aedf61",
    ),
)


class RepairError(RuntimeError):
    """the installed cutlass-dsl tree is not an approved repair state."""


def _download(wheel: Wheel, into: Path) -> Path:
    target = into / wheel.filename
    with urllib.request.urlopen(wheel.url, timeout=300) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != wheel.sha256:
        raise RepairError(
            f"{wheel.filename}: sha256 {digest} does not match the authorized {wheel.sha256}"
        )
    target.write_bytes(payload)
    return target


def _intact() -> bool:
    """run vllm's OWN intactness check, with its lru_cache stripped so it cannot answer from cache.

    importing the private helper directly keeps this honest: a reimplementation here could drift
    from the predicate that actually gates the backend choice.
    """

    # the predicate lives in the gdn module that USES it, not in a shared utils module. importing
    # it from anywhere else silently ImportErrors at build time (confirmed against vllm 0.23.0).
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _is_libs_cu13_install_intact,
    )

    check = getattr(_is_libs_cu13_install_intact, "__wrapped__", _is_libs_cu13_install_intact)
    return bool(check())


def repair(scratch: Path) -> None:
    # base first, then cu13, so cu13 content lands last and wins the shared paths.
    for wheel in reversed(WHEELS):
        path = _download(wheel, scratch)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                "--no-index",
                str(path),
            ],
            check=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only assert the tree is intact; do not reinstall",
    )
    args = parser.parse_args(argv)

    if args.verify:
        if not _intact():
            raise RepairError(
                "nvidia-cutlass-dsl-libs-cu13 is NOT intact after the repair. b200 gdn prefill "
                "would silently fall back to triton, so this build must fail rather than ship."
            )
        print(f"cutlass-dsl cu13 intact at pinned {_PINNED_VERSION}")
        return 0

    scratch = Path("/tmp/cutlass-repair")
    scratch.mkdir(parents=True, exist_ok=True)
    repair(scratch)
    if not _intact():
        raise RepairError("repair ran but the intactness check still fails")
    print(f"cutlass-dsl cu13 repaired to pinned {_PINNED_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
