"""Thin Modal SDK adapter for training Sandbox lifecycle operations."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from flash.providers.base import UnreconciledCreateError

APP_NAME = "flash-training"
PROVIDER_TAG = "flash-provider"
RUN_TAG = "flash-run"
LABEL_TAG = "flash-label"


class ModalApiError(RuntimeError):
    pass


def _modal_sdk():
    """Import the third-party SDK absolutely despite this package's ``modal`` name."""
    import modal

    return modal


@contextmanager
def _authenticated_client() -> Iterator[tuple[Any, Any]]:
    from flash.providers.modal.auth import load_credentials

    credentials = load_credentials()
    if credentials is None:
        raise ModalApiError("Modal credentials are not configured")
    sdk = _modal_sdk()
    client = sdk.Client.from_credentials(*credentials)
    try:
        yield sdk, client
    finally:
        with contextlib.suppress(Exception):
            client._close()


def _is_not_found(sdk, exc: Exception) -> bool:
    return isinstance(exc, sdk.exception.NotFoundError)


def _create_rejection_is_clean(sdk, exc: BaseException) -> bool:
    """Return whether Modal definitively refused the create before provisioning a Sandbox."""
    conflict_error = getattr(sdk.exception, "ConflictError", None)
    if isinstance(conflict_error, type) and isinstance(exc, conflict_error):
        return False
    definitive = tuple(
        cls
        for cls in (
            getattr(sdk.exception, name, None)
            for name in (
                "AuthError",
                "PermissionDeniedError",
                "NotFoundError",
                "RequestSizeError",
                "ImageBuildError",
                "InvalidError",
            )
        )
        if isinstance(cls, type)
    )
    return bool(definitive) and isinstance(exc, definitive)


def _list_with_client(sdk, client, *, tags: dict[str, str]) -> list[dict[str, Any]]:
    sandboxes: list[dict[str, Any]] = []
    for sandbox in sdk.Sandbox.list(tags=tags, client=client):
        sandbox_tags = sandbox.get_tags()
        sandboxes.append({"id": str(sandbox.object_id), "tags": dict(sandbox_tags)})
    return sandboxes


def create_sandbox(
    *args: str,
    image: str,
    gpu: str,
    env: dict[str, str],
    timeout: int,
    name: str,
    tags: dict[str, str],
) -> str:
    """Create one named training Sandbox, reconciling an ambiguous create by exact tags."""
    with _authenticated_client() as (sdk, client):
        try:
            app = sdk.App.lookup(APP_NAME, client=client, create_if_missing=True)
            sandbox = sdk.Sandbox.create(
                *args,
                app=app,
                name=name,
                tags=dict(tags),
                image=sdk.Image.from_registry(image),
                gpu=gpu,
                env=dict(env),
                timeout=int(timeout),
                client=client,
            )
            instance_id = str(sandbox.object_id).strip()
            if not instance_id:
                raise ModalApiError("Modal Sandbox create returned no object id")
            return instance_id
        # catch interrupts after create so any unrecorded sandbox still reconciles by exact label.
        except BaseException as exc:
            if _create_rejection_is_clean(sdk, exc):
                raise ModalApiError(
                    f"Modal rejected the Sandbox create request ({type(exc).__name__})"
                ) from exc
            try:
                matches = _list_with_client(sdk, client, tags={LABEL_TAG: tags[LABEL_TAG]})
            except BaseException:
                raise UnreconciledCreateError(
                    "ambiguous Modal Sandbox create could not be reconciled by its exact label"
                ) from exc
            if len(matches) == 1:
                return str(matches[0]["id"])
            raise UnreconciledCreateError(
                "ambiguous Modal Sandbox create did not resolve to one exact labelled resource"
            ) from exc


def sandbox_status(instance_id: str) -> dict[str, Any] | None:
    """Return the exact Sandbox's running or terminated state, or None when absent."""
    with _authenticated_client() as (sdk, client):
        try:
            sandbox = sdk.Sandbox.from_id(str(instance_id), client=client)
            returncode = sandbox.poll()
        except Exception as exc:
            if _is_not_found(sdk, exc):
                return None
            raise ModalApiError("Modal Sandbox status lookup failed") from exc
    if returncode is None:
        return {"status": "running"}
    return {"status": "terminated", "returncode": int(returncode)}


def sandbox_exec_succeeds(instance_id: str) -> bool:
    """Probe whether the running container can execute a trivial process."""
    with _authenticated_client() as (sdk, client):
        try:
            sandbox = sdk.Sandbox.from_id(str(instance_id), client=client)
            return sandbox.exec("true", timeout=30).wait() == 0
        except Exception as exc:
            if _is_not_found(sdk, exc):
                return False
            raise ModalApiError("Modal Sandbox exec probe failed") from exc


def sandbox_gpu_names(instance_id: str) -> list[str]:
    """Read every realized GPU product name from inside the exact Sandbox."""
    with _authenticated_client() as (sdk, client):
        try:
            sandbox = sdk.Sandbox.from_id(str(instance_id), client=client)
            process = sandbox.exec(
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                timeout=60,
            )
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            returncode = process.wait()
        except Exception as exc:
            if _is_not_found(sdk, exc):
                raise ModalApiError("Modal Sandbox disappeared before GPU attestation") from exc
            raise ModalApiError("Modal Sandbox GPU attestation failed") from exc
    if returncode != 0:
        detail = str(stderr or "").strip()
        raise ModalApiError(f"nvidia-smi GPU attestation failed: {detail[:500]}")
    names = [line.strip() for line in str(stdout or "").splitlines() if line.strip()]
    if not names:
        raise ModalApiError("nvidia-smi GPU attestation returned no devices")
    return names


def terminate_sandbox(instance_id: str) -> None:
    """Terminate one exact Sandbox; absence and prior termination are idempotent success."""
    with _authenticated_client() as (sdk, client):
        try:
            sandbox = sdk.Sandbox.from_id(str(instance_id), client=client)
            sandbox.terminate(wait=True)
        except Exception as exc:
            if _is_not_found(sdk, exc):
                return
            raise ModalApiError("Modal Sandbox termination failed") from exc


def list_sandboxes(*, tags: dict[str, str]) -> list[dict[str, Any]]:
    """List active Sandboxes carrying every requested tag, including their full tag maps."""
    with _authenticated_client() as (sdk, client):
        try:
            return _list_with_client(sdk, client, tags=tags)
        except Exception as exc:
            raise ModalApiError("Modal Sandbox listing failed") from exc
