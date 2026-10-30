"""Private, restart-safe request credentials; never part of specs or public status."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

from flash.runner.lifecycle import state


def _directory() -> Path:
    return Path(state.RUNS_DIR) / ".runtime-credentials"


def _path(run_id: str) -> Path:
    return _directory() / (state.require_safe_run_id(run_id) + ".enc")


def _write(path: Path, contents: bytes, *, create_only: bool = False) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        if create_only:
            # Publish a complete key exactly once, even across concurrent processes.
            with contextlib.suppress(FileExistsError):
                os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def save(run_id: str, credentials: dict[str, str]) -> None:
    """Persist before publishing a recoverable run. Key and ciphertext are mode 0600."""
    from cryptography.fernet import Fernet

    path = _path(run_id)
    if not credentials or any(
        not isinstance(k, str) or not isinstance(v, str) or not v for k, v in credentials.items()
    ):
        raise ValueError("Runtime credentials must be a nonempty string mapping")
    directory = path.parent
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    key = directory / ".key"
    if not key.exists():
        _write(key, Fernet.generate_key(), create_only=True)
    payload = json.dumps({"run_id": run_id, "credentials": credentials}).encode()
    _write(path, Fernet(key.read_bytes()).encrypt(payload))


def restore(status: state.RunStatus) -> dict[str, str]:
    """Fail closed on absent/corrupt credentials; never substitute operator environment keys."""
    if not status.requires_runtime_secrets:
        return {}
    from cryptography.fernet import Fernet

    try:
        payload = json.loads(
            Fernet((_directory() / ".key").read_bytes()).decrypt(_path(status.run_id).read_bytes())
        )
        credentials = payload["credentials"]
        if (
            payload["run_id"] != status.run_id
            or not isinstance(credentials, dict)
            or not credentials
            or any(
                not isinstance(k, str) or not isinstance(v, str) or not v
                for k, v in credentials.items()
            )
        ):
            raise ValueError("Invalid credential envelope")
        return credentials
    except Exception:
        raise ValueError(
            "recovery cannot restore request runtime secrets from the private credential store; "
            "resubmit the run"
        ) from None


def discard(run_id: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        _path(run_id).unlink()
