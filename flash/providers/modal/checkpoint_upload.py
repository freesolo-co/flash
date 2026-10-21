"""Release the GPU, then publish a final native checkpoint on bounded CPU compute."""

from __future__ import annotations

import contextlib
import hashlib
import os
import time

from flash.engine.worker.io.cpu_handoff import MOUNT
from flash.providers.core.base import UnreconciledCreateError
from flash.providers.modal import api
from flash.providers.modal.pricing import CPU_HOURLY, MEMORY_GIB_HOURLY


def volume_name(scope: str, name: str) -> str:
    return "flash-handoff-" + hashlib.sha256(f"{scope}:{name}".encode()).hexdigest()[:40]


def _uploader_tags(handle):
    return {
        "flash_owner": "flash-training",
        "flash_run": handle.run_id,
        "flash_attempt": str(handle.attempt),
        "flash_parent": handle.instance_id,
        "flash_role": "checkpoint-upload",
        "flash_scope": handle.scope,
    }


def _uploader_name(handle):
    return "upload-" + handle.instance_id


async def _find_owned_uploader(modal, client, handle):
    try:
        sandbox = await modal.Sandbox.from_name.aio(
            api.APP_NAME,
            _uploader_name(handle),
            environment_name=api.environment(),
            client=client,
        )
    except modal.exception.NotFoundError:
        return None
    if await sandbox.get_tags.aio() != _uploader_tags(handle):
        raise UnreconciledCreateError("CPU uploader name belongs to another attempt")
    return sandbox


def finish(handle, spec, *, deadline_at):
    """The GPU must be stopped and its Volume committed before attaching a CPU reader."""
    from flash.providers._lifecycle.net.worker import worker_image_for_gpu

    api.terminate(handle.instance_id)
    started = time.time()
    environment = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HF_HUB_DISABLE_XET": "1",
        "FLASH_HF_UPLOAD_THREADS": os.environ.get("FLASH_HF_UPLOAD_THREADS", "8"),
        "FLASH_JOB_SPEC_PATH": f"{MOUNT}/spec.json",
        "FLASH_EXPECTED_SPEC_SHA256": hashlib.sha256(spec.to_json().encode()).hexdigest(),
        "HF_REPO": handle.hf_repo,
        "RUN_ID": handle.run_id,
        "PHASE": handle.phase,
        "RUN_MODE": handle.phase,
        "ATTEMPT": str(handle.attempt),
        "FLASH_RUN_DEADLINE_AT": str(deadline_at),
        "PYTHONPATH": f"{MOUNT}/source",
    }
    if not environment["HF_TOKEN"]:
        raise ValueError("HF_TOKEN is required for CPU checkpoint publication")

    async def upload(modal, client):
        sandbox = await _find_owned_uploader(modal, client, handle)
        if sandbox is None:
            app = await modal.App.lookup.aio(
                api.APP_NAME,
                environment_name=api.environment(),
                client=client,
            )
            volume = modal.Volume.from_name(
                volume_name(handle.scope, handle.name),
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
                    name=_uploader_name(handle),
                    tags=_uploader_tags(handle),
                    image=modal.Image.from_registry(worker_image_for_gpu(handle.gpu)),
                    secrets=[modal.Secret.from_dict(environment)],
                    volumes={MOUNT: volume},
                    cpu=(2, 4),
                    memory=(4096, 8192),
                    timeout=min(1800, max(1, int(deadline_at - time.time()))),
                    client=client,
                )
            except Exception:
                sandbox = await _find_owned_uploader(modal, client, handle)
                if sandbox is None:
                    raise UnreconciledCreateError(
                        "CPU upload launch could not be reconciled"
                    ) from None
        await sandbox.wait.aio()
        if await sandbox.poll.aio() != 0:
            raise api.ModalApiError("CPU checkpoint upload failed; prior durable save is preserved")

    api.call(upload, deadline_at=deadline_at, timeout=1900)
    seconds = time.time() - started
    return {
        "cpu_upload_wall_seconds": seconds,
        "cpu_upload_cost_upper_estimate_usd": seconds
        / 3600
        * (4 * CPU_HOURLY + 8 * MEMORY_GIB_HOURLY),
    }


def cleanup(handle):
    """Cancel an owned CPU child before deleting its per-attempt storage."""

    async def remove(modal, client):
        sandbox = await _find_owned_uploader(modal, client, handle)
        if sandbox is not None:
            await sandbox.terminate.aio(wait=True)
            if await sandbox.poll.aio() is None:
                raise api.ModalApiError("CPU uploader termination was not confirmed")
        with contextlib.suppress(modal.exception.NotFoundError):
            await modal.Volume.objects.delete.aio(
                volume_name(handle.scope, handle.name),
                environment_name=api.environment(),
                client=client,
            )

    api.call(remove, timeout=api.TERMINATION_TIMEOUT_SECONDS)
