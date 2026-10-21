"""Bounded Modal SDK calls; no credentials or raw provider errors escape this module."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flash.providers.core.base import UnreconciledCreateError, get_gpu_info
from flash.providers.modal.auth import credentials
from flash.providers.modal.pricing import CPU_PER_GPU, MEMORY_MIB_PER_GPU

APP_NAME = "flash-training"
CACHE_NAME = "flash-weights"
CACHE_MOUNT = "/weight-cache"
MAX_SANDBOX_SECONDS = 86400
API_TIMEOUT_SECONDS = 60.0
TERMINATION_TIMEOUT_SECONDS = 180.0


class ModalApiError(RuntimeError):
    """A sanitized transport, authentication, or provider failure."""


def environment() -> str:
    return os.environ.get("MODAL_ENVIRONMENT", "main").strip() or "main"


def call(operation, *, deadline_at=None, timeout=API_TIMEOUT_SECONDS):
    """Keep SDK connections on one short-lived event loop per operation."""
    import modal

    auth = credentials()
    if auth is None:
        raise ModalApiError("Modal requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET")
    budget = min(timeout, deadline_at - time.time()) if deadline_at is not None else timeout
    if budget <= 0:
        raise TimeoutError("Modal operation exceeded the run deadline")

    async def invoke():
        client = await modal.Client.from_credentials.aio(*auth)
        try:
            return await operation(modal, client)
        finally:
            await client.__aexit__(None, None, None)

    async def bounded():
        return await asyncio.wait_for(invoke(), timeout=budget)

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(bounded())
        # Startup reconciliation also calls this synchronous API from ASGI's loop.
        # Construct the coroutine in the worker so its client opens and closes on
        # the same loop, without nesting asyncio.run or leaking an unawaited task.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(bounded())).result()
    except (UnreconciledCreateError, ModalApiError):
        raise
    except Exception:
        raise ModalApiError("Modal operation failed or timed out") from None


def create(*, name, payload, image_reference, gpu, gpu_count, capsule, deadline_at):
    """Create exactly one named sandbox; ambiguous creation never causes a blind retry."""
    import hashlib
    import json

    identity = {
        "flash_run": payload["run_id"],
        "flash_attempt": str(payload["attempt"]),
        "flash_name": name,
        "flash_owner": "flash-training",
    }
    identity["flash_spec"] = hashlib.sha256(payload["job_spec_json"].encode()).hexdigest()
    identity["flash_source"] = hashlib.sha256(
        json.dumps(payload.get("source_snapshot"), sort_keys=True).encode()
    ).hexdigest()
    if payload.get("handoff_volume"):
        identity["flash_handoff"] = payload["handoff_volume"]
    create_started = False
    with tempfile.TemporaryDirectory(prefix="flash-modal-") as directory:
        archive = Path(directory) / "capsule.pyz"
        archive.write_bytes(capsule)

        async def launch(modal, client):
            nonlocal create_started
            app = await modal.App.lookup.aio(
                APP_NAME, environment_name=environment(), create_if_missing=True, client=client
            )
            try:
                existing = await modal.Sandbox.from_name.aio(
                    APP_NAME, name, environment_name=environment(), client=client
                )
            except modal.exception.NotFoundError:
                existing = None
            if existing is not None:
                if await existing.get_tags.aio() != identity:
                    raise UnreconciledCreateError(
                        "Modal attempt name belongs to a different launch"
                    )
                return existing.object_id
            image = (
                modal.Image.from_registry(image_reference)
                .run_commands("mkdir -p /root/flash")
                .add_local_file(str(archive), "/opt/flash/capsule.pyz", copy=True)
            )
            volumes = {}
            if payload["env"].get("FLASH_WEIGHT_CACHE_DIR"):
                volumes[CACHE_MOUNT] = modal.Volume.from_name(
                    payload.get("cache_volume", CACHE_NAME),
                    environment_name=environment(),
                    create_if_missing=True,
                    client=client,
                )
            if payload.get("handoff_volume"):
                from flash.engine.worker.io.cpu_handoff import MOUNT

                volumes[MOUNT] = modal.Volume.from_name(
                    payload["handoff_volume"],
                    environment_name=environment(),
                    create_if_missing=True,
                    client=client,
                )
            secret = modal.Secret.from_dict({"FLASH_BOOTSTRAP_PAYLOAD": json.dumps(payload)})
            timeout = min(MAX_SANDBOX_SECONDS, max(1, int(deadline_at - time.time())))
            try:
                create_started = True
                sandbox = await modal.Sandbox.create.aio(
                    "python",
                    "-u",
                    "/opt/flash/capsule.pyz",
                    "bootstrap",
                    app=app,
                    name=name,
                    tags=identity,
                    image=image,
                    secrets=[secret],
                    gpu=f"{get_gpu_info(gpu).modal_name}:{gpu_count}",
                    # These are minimum reservations. Checkpoint export and compilation must
                    # be able to burst beyond them; a matching hard limit kills healthy workers.
                    cpu=CPU_PER_GPU * gpu_count,
                    memory=MEMORY_MIB_PER_GPU * gpu_count,
                    timeout=timeout,
                    workdir="/root/flash",
                    volumes=volumes,
                    client=client,
                )
                return sandbox.object_id
            except Exception:
                # a create may have landed. reattach only to the exact named and tagged attempt.
                try:
                    sandbox = await modal.Sandbox.from_name.aio(
                        APP_NAME, name, environment_name=environment(), client=client
                    )
                    if await sandbox.get_tags.aio() == identity:
                        return sandbox.object_id
                except Exception:
                    pass
                raise UnreconciledCreateError(
                    "Modal sandbox creation could not be reconciled"
                ) from None

        try:
            return call(launch, deadline_at=deadline_at, timeout=1800)
        except ModalApiError:
            if not create_started:
                raise
            raise UnreconciledCreateError(
                "Modal launch outcome unknown; reconcile before retrying"
            ) from None


def get_instance(instance_id, *, deadline_at=None):
    async def read(modal, client):
        try:
            sandbox = await modal.Sandbox.from_id.aio(instance_id, client=client)
            code = await sandbox.poll.aio()
        except modal.exception.NotFoundError:
            return None
        return {"status": "pending" if code is None else "terminated", "returncode": code}

    return call(read, deadline_at=deadline_at)


def terminate(instance_id):
    async def stop(modal, client):
        try:
            sandbox = await modal.Sandbox.from_id.aio(instance_id, client=client)
            await sandbox.terminate.aio(wait=True)
            if await sandbox.poll.aio() is None:
                raise ModalApiError("Modal sandbox termination was not confirmed")
        except modal.exception.NotFoundError:
            pass

    # Modal may spend a full minute stopping GPU descendants and committing mounted Volumes.
    # Keep ownership until that shutdown is confirmed instead of timing out at the grace boundary.
    call(stop, timeout=TERMINATION_TIMEOUT_SECONDS)


def list_instances():
    async def listing(modal, client):
        try:
            app = await modal.App.lookup.aio(
                APP_NAME, environment_name=environment(), client=client
            )
        except modal.exception.NotFoundError:
            return []
        result = []
        async for sandbox in modal.Sandbox.list.aio(
            app_id=app.app_id, tags={"flash_owner": "flash-training"}, client=client
        ):
            tags = await sandbox.get_tags.aio()
            if await sandbox.poll.aio() is None:
                result.append({"id": sandbox.object_id, "tags": tags})
        return result

    return call(listing)
