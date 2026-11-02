"""Release the GPU, then publish a final native checkpoint on bounded CPU compute."""

from __future__ import annotations

import contextlib
import hashlib
import re
import time

from flash.engine.worker.io.cpu_handoff import MOUNT
from flash.providers.core.base import UnreconciledCreateError
from flash.providers.modal import api
from flash.providers.modal.pricing import CPU_HOURLY, MEMORY_GIB_HOURLY


def _volume_prefix(scope: str, run_id: str) -> str:
    return "flash-handoff-" + hashlib.sha256(f"{scope}:{run_id}".encode()).hexdigest()[:32] + "-a"


def volume_name(scope: str, run_id: str, attempt: int) -> str:
    from flash.providers._lifecycle.instances.instance import instance_label

    instance_label(run_id, attempt)  # enforce the same attempt bound as the sandbox name.
    return f"{_volume_prefix(scope, run_id)}{attempt}"


def run_volumes(run_id: str) -> list[str]:
    """Find exact run-owned handoffs even after their sandboxes have disappeared."""
    from flash.providers.modal.auth import scope_fingerprint

    prefix = _volume_prefix(scope_fingerprint(), run_id)

    async def listing(modal, client):
        volumes = await modal.Volume.objects.list.aio(
            environment_name=api.environment(), client=client
        )
        return [
            volume.name
            for volume in volumes
            if volume.name
            and re.fullmatch(re.escape(prefix) + r"(?:0|[1-9][0-9]{0,9})", volume.name)
        ]

    return api.call(listing)


def cleanup_run(run_id: str) -> None:
    """Delete handoffs only after the caller confirms every run sandbox has stopped."""
    names = run_volumes(run_id)

    async def remove(modal, client):
        for name in names:
            with contextlib.suppress(modal.exception.NotFoundError):
                await modal.Volume.objects.delete.aio(
                    name, environment_name=api.environment(), client=client
                )

    if names:
        api.call(remove, timeout=api.TERMINATION_TIMEOUT_SECONDS)


# Training is finished and the GPU released by the time the CPU uploader runs, and the handoff
# Volume keeps the committed checkpoint, so a failed upload is retried on a fresh uploader instead
# of failing the run (`cpu_handoff` re-validates and re-uploads idempotently). Try 0 keeps the
# original name and tags so uploads launched by an older control plane still reconcile.
UPLOAD_TRIES = 3
_UPLOAD_RETRY_DELAYS_S = (15.0, 60.0)


class CheckpointUploadError(api.ModalApiError):
    """Every CPU uploader try failed; the committed handoff Volume is preserved."""


def _uploader_tags(handle, try_index: int = 0):
    tags = {
        "flash_owner": "flash-training",
        "flash_run": handle.run_id,
        "flash_attempt": str(handle.attempt),
        "flash_parent": handle.instance_id,
        "flash_role": "checkpoint-upload",
        "flash_scope": handle.scope,
    }
    if try_index:
        tags["flash_upload_try"] = str(try_index)
    return tags


def _uploader_name(handle, try_index: int = 0):
    name = "upload-" + handle.instance_id
    return f"{name}-r{try_index}" if try_index else name


async def _find_owned_uploader(modal, client, handle, try_index: int = 0):
    try:
        sandbox = await modal.Sandbox.from_name.aio(
            api.APP_NAME,
            _uploader_name(handle, try_index),
            environment_name=api.environment(),
            client=client,
        )
    except modal.exception.NotFoundError:
        return None
    if await sandbox.get_tags.aio() != _uploader_tags(handle, try_index):
        raise UnreconciledCreateError("CPU uploader name belongs to another attempt")
    return sandbox


async def _launched_spec_sha256(modal, client, handle):
    """Bind publication to the original launch, not a reconstructed polling spec."""
    sandbox = await modal.Sandbox.from_id.aio(handle.instance_id, client=client)
    tags = await sandbox.get_tags.aio()
    expected = {
        "flash_owner": "flash-training",
        "flash_run": handle.run_id,
        "flash_attempt": str(handle.attempt),
        "flash_name": handle.name,
        "flash_handoff": volume_name(handle.scope, handle.run_id, handle.attempt),
    }
    if any(tags.get(key) != value for key, value in expected.items()):
        raise api.ModalApiError("CPU checkpoint parent launch identity mismatch")
    digest = tags.get("flash_spec")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise api.ModalApiError("CPU checkpoint parent launch has no valid specification digest")
    return digest


def finish(handle, *, deadline_at):
    """The GPU must be stopped and its Volume committed before attaching a CPU reader."""
    from flash.providers._lifecycle.net.worker import worker_image_for_gpu

    api.terminate(handle.instance_id)
    started = time.time()
    from flash.artifacts.config import runtime_environment

    storage_environment = runtime_environment()
    if not storage_environment:
        raise ValueError("Azure artifact credentials are required for CPU checkpoint publication")
    environment = {
        **storage_environment,
        "FLASH_JOB_SPEC_PATH": f"{MOUNT}/spec.json",
        "HF_REPO": handle.hf_repo,
        "RUN_ID": handle.run_id,
        "PHASE": handle.phase,
        "RUN_MODE": handle.phase,
        "ATTEMPT": str(handle.attempt),
        "FLASH_RUN_DEADLINE_AT": str(deadline_at),
        "PYTHONPATH": f"{MOUNT}/source",
    }

    def uploader(try_index: int):
        async def upload(modal, client):
            await _run_uploader(modal, client, try_index)

        return upload

    async def _run_uploader(modal, client, try_index: int):
        # a finished sandbox from an earlier try (or an earlier control plane) is not relaunched:
        # its exit code is the result for that try.
        sandbox = await _find_owned_uploader(modal, client, handle, try_index)
        if sandbox is None:
            # Recovery reduces gpu.max_wall_seconds on the poll spec. Hashing that spec
            # rejects the untouched handoff from the original GPU launch on every retry.
            environment["FLASH_EXPECTED_SPEC_SHA256"] = await _launched_spec_sha256(
                modal, client, handle
            )
            app = await modal.App.lookup.aio(
                api.APP_NAME,
                environment_name=api.environment(),
                client=client,
            )
            volume = modal.Volume.from_name(
                volume_name(handle.scope, handle.run_id, handle.attempt),
                environment_name=api.environment(),
                create_if_missing=False,
                client=client,
            )
            try:
                sandbox = await modal.Sandbox.create.aio(
                    "python",
                    "-u",
                    "-m",
                    "flash.engine.worker.io.cpu_handoff",
                    app=app,
                    name=_uploader_name(handle, try_index),
                    tags=_uploader_tags(handle, try_index),
                    image=modal.Image.from_registry(worker_image_for_gpu(handle.gpu)).pip_install(
                        "azure-storage-blob>=12.24,<13", "azure-identity>=1.19,<2"
                    ),
                    secrets=[modal.Secret.from_dict(environment)],
                    volumes={MOUNT: volume},
                    cpu=(2, 4),
                    memory=(4096, 8192),
                    timeout=min(1800, max(1, int(deadline_at - time.time()))),
                    client=client,
                )
            except Exception:
                sandbox = await _find_owned_uploader(modal, client, handle, try_index)
                if sandbox is None:
                    raise UnreconciledCreateError(
                        "CPU upload launch could not be reconciled"
                    ) from None
        await sandbox.wait.aio()
        returncode = await sandbox.poll.aio()
        if returncode != 0:
            raise api.ModalApiError(
                f"CPU checkpoint uploader {sandbox.object_id} exited with code {returncode}"
            )

    tries = 0
    for try_index in range(UPLOAD_TRIES):
        tries = try_index + 1
        try:
            api.call(uploader(try_index), deadline_at=deadline_at, timeout=1900)
            break
        except (api.ModalApiError, TimeoutError) as exc:
            remaining = deadline_at - time.time()
            if tries >= UPLOAD_TRIES or remaining <= 0:
                raise CheckpointUploadError(
                    f"CPU checkpoint upload failed after {tries} "
                    f"{'try' if tries == 1 else 'tries'}; prior durable save is preserved; {exc}"
                ) from exc
            time.sleep(
                min(
                    _UPLOAD_RETRY_DELAYS_S[min(try_index, len(_UPLOAD_RETRY_DELAYS_S) - 1)],
                    remaining,
                )
            )
    seconds = time.time() - started
    return {
        "cpu_upload_wall_seconds": seconds,
        "cpu_upload_tries": tries,
        # tries run one after another, so wall time already covers every uploader
        "cpu_upload_cost_upper_estimate_usd": seconds
        / 3600
        * (4 * CPU_HOURLY + 8 * MEMORY_GIB_HOURLY),
    }


def cleanup(handle):
    """Cancel an owned CPU child before deleting its per-attempt storage."""

    async def remove(modal, client):
        for try_index in range(UPLOAD_TRIES):
            sandbox = await _find_owned_uploader(modal, client, handle, try_index)
            if sandbox is not None:
                await sandbox.terminate.aio(wait=True)
                if await sandbox.poll.aio() is None:
                    raise api.ModalApiError("CPU uploader termination was not confirmed")
        with contextlib.suppress(modal.exception.NotFoundError):
            await modal.Volume.objects.delete.aio(
                volume_name(handle.scope, handle.run_id, handle.attempt),
                environment_name=api.environment(),
                client=client,
            )

    api.call(remove, timeout=api.TERMINATION_TIMEOUT_SECONDS)
