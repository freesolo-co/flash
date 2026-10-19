"""readiness is published only after explicit trainer construction succeeds."""

import os
import tempfile
from pathlib import Path


def publish_ready(options) -> None:
    destination = Path(options.readiness_path)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".trainer-ready-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("runtime-ready\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
