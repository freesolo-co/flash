"""Persist benchmark completion independently of Modal's short-lived exec status."""

import subprocess
import sys
from pathlib import Path


def main():
    code = 1
    try:
        with Path("/results/benchmark.log").open("w") as log:
            code = subprocess.run(
                [sys.executable, "-u", "/tmp/bench.py"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        return code
    finally:
        Path("/results/exit-code").write_text(str(code))


if __name__ == "__main__":
    sys.exit(main())
