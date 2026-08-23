"""stdlib-only bootstrap for the packaged serving launcher."""

from __future__ import annotations

import contextlib
import os
import signal
from types import FrameType
from typing import Any, Self

_INFERENCE_TOKEN_ENV = "FLASH_INFERENCE_TOKEN"
_ARTIFACT_TOKEN_ENV = "FLASH_ARTIFACT_TOKEN"
_INFERENCE_TOKEN_FD_ENV = "FLASH_INFERENCE_TOKEN_FD"
_ARTIFACT_TOKEN_FD_ENV = "FLASH_ARTIFACT_TOKEN_FD"


class BootstrapError(RuntimeError):
    """the fixed serving bootstrap rejected its startup inputs."""


class StartupTerminated(BootstrapError):
    """the bootstrap received a startup termination signal."""

    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        super().__init__("serving startup was terminated")


class _StartupSignalGuard:
    def __init__(self) -> None:
        self._previous: dict[int, Any] = {}
        self._active = False

    def __enter__(self) -> Self:
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._terminate)
        self._active = True
        return self

    def _terminate(self, signum: int, _frame: FrameType | None) -> None:
        raise StartupTerminated(128 + signum)

    def restore(self) -> None:
        if not self._active:
            return
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._active = False

    def release_for_handoff(self) -> dict[int, Any]:
        previous = dict(self._previous)
        self._active = False
        return previous

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.restore()


def _close_raw_descriptor(raw_fd: str | None) -> None:
    if raw_fd is not None and raw_fd.isdecimal():
        with contextlib.suppress(OSError):
            os.close(int(raw_fd))


def _read_secret_descriptor(raw_fd: str | None, name: str) -> str | None:
    if raw_fd is None:
        return None
    if not raw_fd.isdecimal():
        raise BootstrapError(f"{name} descriptor is invalid")
    try:
        with os.fdopen(int(raw_fd), "r", encoding="utf-8", closefd=True) as source:
            value = source.read().strip()
    except OSError as exc:
        raise BootstrapError(f"{name} descriptor could not be read") from exc
    if not value:
        raise BootstrapError(f"{name} descriptor was empty")
    return value


def _validate_secret(value: str | None, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or not value or value != value.strip():
        raise BootstrapError(f"{name} is invalid")
    return value


def _pop_runtime_secrets() -> tuple[str, str | None]:
    inference = os.environ.pop(_INFERENCE_TOKEN_ENV, None)
    artifact = os.environ.pop(_ARTIFACT_TOKEN_ENV, None)
    inference_fd = os.environ.pop(_INFERENCE_TOKEN_FD_ENV, None)
    artifact_fd = os.environ.pop(_ARTIFACT_TOKEN_FD_ENV, None)
    try:
        if inference is not None and inference_fd is not None:
            raise BootstrapError("inference token has multiple sources")
        if artifact is not None and artifact_fd is not None:
            raise BootstrapError("artifact token has multiple sources")
        if inference is None:
            selected_fd, inference_fd = inference_fd, None
            inference = _read_secret_descriptor(selected_fd, "inference token")
        if artifact is None:
            selected_fd, artifact_fd = artifact_fd, None
            artifact = _read_secret_descriptor(selected_fd, "artifact token")
        inference = _validate_secret(inference, "inference token")
        artifact = _validate_secret(artifact, "artifact token", optional=True)
        assert inference is not None
        return inference, artifact
    finally:
        _close_raw_descriptor(inference_fd)
        _close_raw_descriptor(artifact_fd)


def _run() -> None:
    with _StartupSignalGuard() as bootstrap_signals:
        inference_token, artifact_token = _pop_runtime_secrets()
        from flash.serve.app import launch

        try:
            inference_token_holder = [inference_token]
            artifact_token_holder = [artifact_token]
            del inference_token
            del artifact_token
            launch.run_launcher_with_secrets(
                inference_token_holder,
                artifact_token_holder,
                environment=os.environ,
                previous_signal_guard=bootstrap_signals,
            )
        except launch.StartupTerminated as exc:
            raise SystemExit(exc.exit_code) from None


def main() -> None:
    try:
        _run()
    except StartupTerminated as exc:
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    main()
