"""Serialize bootstrap re-entry and retain the first terminal result on the worker."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path


def _create_once(path: Path, value: dict) -> None:
    """Publish complete json without replacing a concurrent writer's result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".result-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        with contextlib.suppress(FileExistsError):
            os.link(temporary, path)
    finally:
        os.unlink(temporary)


class AttemptState:
    """Local state survives a provider restarting the same attempt's container command."""

    def __init__(self, code_dir: str, payload: dict):
        self.directory = Path(code_dir + ".bootstrap")
        self.record_path = self.directory / "result.json"
        self.marker_path = self.directory / "marker.json"
        self.identity = {
            key: payload.get(key)
            for key in (
                "run_id",
                "attempt",
                "flash_arm",
                "phase",
                "hf_repo",
                "hf_prefix",
                "source_snapshot",
            )
        }

    @contextlib.contextmanager
    def locked(self):
        import fcntl

        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "lock").open("a") as lock:
            # wait for the owner to finish; returning early lets the launch script destroy its gpu.
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("bootstrap: duplicate start waiting for attempt owner", flush=True)
                fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def read(self) -> dict | None:
        try:
            record = json.loads(self.record_path.read_text())
        except FileNotFoundError:
            return None
        if not isinstance(record, dict) or record.get("identity") != self.identity:
            raise RuntimeError("persisted bootstrap result belongs to a different attempt payload")
        marker = record.get("marker")
        if (
            not isinstance(marker, dict)
            or type(marker.get("ok")) is not bool
            or marker.get("run_id") != self.identity["run_id"]
            or marker.get("attempt") != self.identity["attempt"]
        ):
            raise RuntimeError("persisted bootstrap result is invalid")
        return marker

    def save(self, marker: dict) -> dict:
        # persist before network i/o, so a restart can retry publication without rerunning training.
        _create_once(self.record_path, {"identity": self.identity, "marker": marker})
        saved = self.read()
        assert saved is not None
        return saved

    def publication_path(self, marker: dict) -> str:
        _create_once(self.marker_path, marker)
        if json.loads(self.marker_path.read_text()) != marker:
            raise RuntimeError("persisted bootstrap marker does not match its terminal result")
        return str(self.marker_path)
