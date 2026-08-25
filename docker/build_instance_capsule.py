"""Build the verified shared instance capsule during the worker image build."""

from __future__ import annotations

import sys
from pathlib import Path

from flash.runtime_capsule import build_capsule, write_capsule


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_instance_capsule.py <source-root> <destination>")
    archive, _manifest = build_capsule("instance-bootstrap", root=Path(sys.argv[1]))
    write_capsule(Path(sys.argv[2]), archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
