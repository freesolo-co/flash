"""Stage a pinned public SFT base model on bounded CPU compute before renting a GPU."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time

from flash.engine.support.model_snapshot import snapshot_has_weights
from flash.providers.core.base import UnreconciledCreateError
from flash.providers.modal import api
from flash.providers.modal.pricing import CPU_HOURLY, MEMORY_GIB_HOURLY

# Ship the same dependency-free validator used by the GPU worker; the CPU image has no Flash.
_SCRIPT = (
    "import json, os\nos.environ['HF_HUB_DISABLE_XET'] = '1'\n"
    "from huggingface_hub import snapshot_download\n"
    + inspect.getsource(snapshot_has_weights)
    + """
model, revision = json.loads(os.environ['FLASH_BASE_MODEL'])
for force in (False, True):
    root = snapshot_download(model, revision=revision, cache_dir='/weight-cache/hf-cache/hub',
        token=os.environ.get('HF_TOKEN'), force_download=force,
        ignore_patterns=['*.pth', '*.gguf', 'original/*', '*.onnx', '*.msgpack', '*.h5'])
    if snapshot_has_weights(root):
        break
else:
    raise RuntimeError('Pinned base snapshot has no complete model weights after forced download')
"""
)


def prepare(spec, *, attempt, payload, deadline_at):
    """Return separately attributed CPU cost; never include private adapters or environment code."""
    from flash.core.catalog import MODELS

    if (
        spec.algorithm != "sft"
        or not payload["env"].get("FLASH_WEIGHT_CACHE_DIR")
        or spec.model not in MODELS
        or not re.fullmatch(r"[0-9a-f]{40}", spec.model_revision)
    ):
        return {}
    volume_name = payload.get("cache_volume", api.CACHE_NAME)
    # independent of mutable worker code, but exact to this run/attempt/model/cache identity.
    identity = json.dumps([spec.run_id, attempt, spec.model, spec.model_revision, volume_name])
    name = "weights-" + hashlib.sha256(identity.encode()).hexdigest()[:40]
    tags = {
        "flash_owner": "flash-training",
        "flash_role": "weight-prefetch",
        "flash_run": spec.run_id,
        "flash_attempt": str(attempt),
        "flash_name": name,
    }
    started = time.time()
    create_pending = False

    async def warm(modal, client):
        nonlocal create_pending
        sandbox = await _find(modal, client, name, tags)
        if sandbox is None:
            app = await modal.App.lookup.aio(
                api.APP_NAME,
                environment_name=api.environment(),
                create_if_missing=True,
                client=client,
            )
            volume = modal.Volume.from_name(
                volume_name,
                environment_name=api.environment(),
                create_if_missing=True,
                client=client,
            )
            remaining = int(deadline_at - time.time())
            if remaining <= 0:
                raise TimeoutError("Run deadline expired before base-model staging")
            try:
                create_pending = True
                sandbox = await modal.Sandbox.create.aio(
                    "python",
                    "-c",
                    _SCRIPT,
                    app=app,
                    name=name,
                    tags=tags,
                    image=modal.Image.debian_slim(python_version="3.12").pip_install(
                        "huggingface-hub==1.27.0"
                    ),
                    timeout=min(900, remaining),
                    cpu=(2, 4),
                    memory=(4096, 8192),
                    volumes={api.CACHE_MOUNT: volume},
                    secrets=[
                        modal.Secret.from_dict(
                            {
                                "HF_TOKEN": payload["env"].get("HF_TOKEN", ""),
                                "FLASH_BASE_MODEL": json.dumps([spec.model, spec.model_revision]),
                            }
                        )
                    ],
                    client=client,
                )
            except Exception:
                try:
                    sandbox = await _find(modal, client, name, tags)
                except Exception:
                    raise UnreconciledCreateError(
                        "CPU prefetch launch could not be reconciled"
                    ) from None
                if sandbox is None:
                    raise UnreconciledCreateError(
                        "CPU prefetch launch could not be reconciled"
                    ) from None
            create_pending = False
        try:
            await sandbox.wait.aio()
            if await sandbox.poll.aio() != 0:
                raise api.ModalApiError("CPU base-model staging failed before GPU allocation")
        finally:
            # termination commits the volume before the GPU mounts it. ownership tags allow
            # run cancellation and orphan sweeping even if this control-plane process is lost.
            await sandbox.terminate.aio(wait=True)
        return sandbox.object_id

    try:
        sandbox_id = api.call(warm, deadline_at=deadline_at, timeout=1080)
    except api.ModalApiError:
        # api.call's deadline can cancel create/reconciliation before it returns an identity.
        if create_pending:
            raise UnreconciledCreateError(
                "CPU prefetch launch outcome unknown; reconcile before retrying"
            ) from None
        raise
    seconds = max(0.0, time.time() - started)
    return {
        "sandbox_id": sandbox_id,
        "wall_seconds": seconds,
        "cost_upper_estimate_usd": seconds / 3600 * (4 * CPU_HOURLY + 8 * MEMORY_GIB_HOURLY),
    }


async def _find(modal, client, name, tags):
    try:
        sandbox = await modal.Sandbox.from_name.aio(
            api.APP_NAME, name, environment_name=api.environment(), client=client
        )
    except modal.exception.NotFoundError:
        return None
    if await sandbox.get_tags.aio() != tags:
        raise UnreconciledCreateError("CPU prefetch name belongs to another run")
    return sandbox
