"""Append-only driver journal and content hashes for portable benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from flash.benchmarks.schema import EVENT, Event, Span


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest(value) -> str:
    return digest_bytes(canonical_bytes(value))


def write_json(path: Path, value) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def load_events(path: Path) -> list[Event]:
    events = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                events.append(EVENT.validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid event at {path.name}:{number}: {exc}") from exc
    return events


class Recorder:
    """One process, many threads; each remote process writes a separate journal.

    All times are seconds since the request's monotonic origin. A collector must translate
    remote clock origins explicitly before merging; never compare monotonic clocks across hosts.
    """

    def __init__(self, path: Path, origin: float):
        self.path = path
        self.origin = origin
        self._lock = threading.Lock()
        path.touch(exist_ok=False)

    def now(self) -> float:
        return time.monotonic() - self.origin

    def emit(self, event: Event) -> None:
        payload = EVENT.validate_python(event).model_dump(mode="json")
        with self._lock, self.path.open("ab") as stream:
            stream.write(canonical_bytes(payload) + b"\n")
            stream.flush()

    @contextmanager
    def span(self, phase, *, source: str, attempt: int = 0):
        start = self.now()
        try:
            yield
        finally:
            self.emit(
                Span(phase=phase, start_s=start, end_s=self.now(), source=source, attempt=attempt)
            )


def seal(directory: Path) -> None:
    """Hash retained inputs, raw evidence, logs and outputs; never silently overwrite a seal."""
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("benchmark artifacts must not contain symlinks")
        if path.is_file():
            hashes[path.relative_to(directory).as_posix()] = digest_bytes(path.read_bytes())
    with (directory / "checksums.json").open("xb") as stream:
        stream.write(canonical_bytes(hashes) + b"\n")


def verify(directory: Path) -> dict:
    hashes = json.loads((directory / "checksums.json").read_text())
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("empty artifact checksums")
    observed = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("benchmark artifacts must not contain symlinks")
        if path.is_file() and path != directory / "checksums.json":
            observed[path.relative_to(directory).as_posix()] = digest_bytes(path.read_bytes())
    if hashes != observed:
        raise ValueError("benchmark artifact set or content differs from checksums.json")
    return hashes
