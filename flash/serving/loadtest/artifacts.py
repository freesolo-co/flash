"""exclusive, secret-safe result artifacts for hosted inference load tests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from flash.serving.loadtest.schema import ResolvedScenario, Scenario, public_scenario_dict

_ARTIFACT_FILES = (
    "scenario.authored.json",
    "scenario.resolved.json",
    "events.jsonl",
    "summary.json",
)


class ArtifactError(RuntimeError):
    pass


class ResultDirectory:
    def __init__(self, path: Path, credential: str | None = None) -> None:
        self.path = path
        self.events = EventWriter(path / "events.jsonl", credential)
        self._closed = False
        self._credential = credential

    @classmethod
    def create(
        cls, path: Path, scenario: Scenario, credential: str | None = None
    ) -> ResultDirectory:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactError(f"result directory already exists: {path}") from exc
        _atomic_json(path / "scenario.authored.json", public_scenario_dict(scenario), credential)
        return cls(path, credential)

    def write_resolved(self, resolved: ResolvedScenario) -> None:
        value = resolved.model_dump(mode="json")
        value["authored"] = public_scenario_dict(resolved.authored)
        _atomic_json(self.path / "scenario.resolved.json", value, self._credential)

    def write_summary(self, summary: dict[str, Any]) -> None:
        _atomic_json(self.path / "summary.json", summary, self._credential)

    def complete(self, summary: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise ArtifactError("result directory is already closed")
        self.events.close()
        self.write_summary(summary)
        files = {}
        for name in _ARTIFACT_FILES:
            path = self.path / name
            if not path.is_file():
                raise ArtifactError(f"required artifact is missing: {name}")
            files[name] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        completion = {
            "schema_version": 1,
            "valid": True,
            "event_rows": self.events.rows,
            "files": files,
        }
        _atomic_json(self.path / "complete.json", completion, self._credential)
        self._closed = True
        return completion

    def abort(self) -> None:
        if not self._closed:
            self.events.close()
            self._closed = True


class EventWriter:
    def __init__(self, path: Path, credential: str | None = None) -> None:
        self._path = path
        self._file = path.open("x", encoding="utf-8")
        self.rows = 0
        self._closed = False
        self._credential = credential

    def write(self, event: dict[str, Any]) -> None:
        if self._closed:
            raise ArtifactError("event writer is closed")
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        _reject_credential(line, self._credential, self._path.name)
        self._file.write(line + "\n")
        self._file.flush()
        self.rows += 1

    def close(self) -> None:
        if self._closed:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._closed = True


def verify_result_directory(path: Path) -> dict[str, Any]:
    complete_path = path / "complete.json"
    if not complete_path.is_file():
        raise ArtifactError("result is incomplete: complete.json is missing")
    completion = _read_json(complete_path)
    if completion.get("valid") is not True:
        raise ArtifactError("completion marker is invalid")
    files = completion.get("files")
    if not isinstance(files, dict) or set(files) != set(_ARTIFACT_FILES):
        raise ArtifactError("completion marker has an invalid file manifest")
    for name in _ARTIFACT_FILES:
        metadata = files[name]
        artifact_path = path / name
        if not artifact_path.is_file():
            raise ArtifactError(f"manifested artifact is missing: {name}")
        if metadata.get("sha256") != _sha256(artifact_path):
            raise ArtifactError(f"artifact hash mismatch: {name}")
        if metadata.get("bytes") != artifact_path.stat().st_size:
            raise ArtifactError(f"artifact size mismatch: {name}")
    rows = _count_jsonl(path / "events.jsonl")
    if completion.get("event_rows") != rows:
        raise ArtifactError("event row count mismatch")
    return completion


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"invalid json at events row {line_number}") from exc
            if not isinstance(value, dict):
                raise ArtifactError(f"events row {line_number} is not an object")
            events.append(value)
    return events


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for _ in file)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not read {path.name}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path.name} must contain an object")
    return value


def _reject_credential(serialized: str, credential: str | None, name: str) -> None:
    """fail closed when a serialized artifact would contain the loaded credential.

    this is the last gate before bytes reach disk, so it catches a leak introduced anywhere
    upstream rather than trusting each writer to have redacted correctly. a very short
    credential is not searched for, since a common substring would reject every artifact and
    the resulting pressure would be to remove the check.
    """
    if credential and len(credential) >= 8 and credential in serialized:
        raise ArtifactError(f"refusing to write {name}: the loaded credential appears in it")


def _atomic_json(path: Path, value: Any, credential: str | None = None) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    _reject_credential(serialized, credential, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
