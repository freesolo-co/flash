"""Dedicated Modal GPU ownership for local custom-training scripts."""

from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from pathlib import Path

from flash.custom.types import integer

_log = logging.getLogger(__name__)


def create_modal_session(
    *,
    model,
    gpu,
    max_steps,
    max_wall_seconds,
    idle_seconds,
    max_context_tokens,
    lora_rank,
    revision,
    hf_token=None,
):
    import modal

    from flash.client.http import ClientError
    from flash.custom.client import TrainingClient
    from flash.providers._lifecycle.net.worker import WORKER_IMAGE

    integer(max_steps, "max_steps", 1, 1_000_000)
    integer(max_wall_seconds, "max_wall_seconds", 60, 86400)
    integer(idle_seconds, "idle_seconds", 30, max_wall_seconds)
    integer(max_context_tokens, "max_context_tokens", 2, 32768)
    integer(lora_rank, "lora_rank", 1, 128)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a Hugging Face causal language model id")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("revision must be a nonempty string or None")
    if gpu not in {"A10G", "H100"}:
        raise ValueError("custom mode currently supports the GPU-validated A10G and H100 classes")
    from huggingface_hub import get_token

    if hf_token is not None and (not isinstance(hf_token, str) or not hf_token.strip()):
        raise ValueError("hf_token must be a nonempty string or None")
    hf_token = hf_token if hf_token is not None else get_token()
    config = {
        "model": model,
        "max_steps": max_steps,
        "max_wall_seconds": max_wall_seconds,
        "idle_seconds": idle_seconds,
        "max_context_tokens": max_context_tokens,
        "lora_rank": lora_rank,
        "revision": revision,
    }
    # the existing pinned training image carries the validated torch/transformers/peft stack.
    image = (
        modal.Image.from_registry(WORKER_IMAGE)
        .pip_install(
            "fastapi>=0.136,<0.137",
            "uvicorn>=0.52,<1",
        )
        .add_local_dir(Path(__file__).resolve().parents[1], "/opt/custom/flash", copy=True)
    )
    app = modal.App.lookup("flash-custom-training", create_if_missing=True)
    name = f"flash-custom-{uuid.uuid4().hex}"
    token = secrets.token_urlsafe(48)
    worker_secrets = {"FLASH_CUSTOM_TOKEN": token}
    if hf_token:
        worker_secrets["HF_TOKEN"] = hf_token
    sandbox = None
    try:
        sandbox = modal.Sandbox.create(
            "python",
            "-m",
            "flash.custom.server",
            app=app,
            name=name,
            image=image,
            gpu=gpu,
            timeout=max_wall_seconds,
            cpu=4,
            memory=16384,
            encrypted_ports=[8000],
            secrets=[modal.Secret.from_dict(worker_secrets)],
            env={
                "PYTHONPATH": "/opt/custom",
                "PYTHONUNBUFFERED": "1",
                "FLASH_CUSTOM_CONFIG": json.dumps(config),
            },
        )
        _log.info(
            "custom GPU allocated: %s (%s, maximum %ss)", sandbox.object_id, gpu, max_wall_seconds
        )
        url = sandbox.tunnels()[8000].url
        deadline = time.monotonic() + min(1200, max_wall_seconds)
        while True:
            try:
                client = TrainingClient(url, api_key=token, timeout=10)
                break
            except ClientError:
                if sandbox.poll() is not None:
                    raise RuntimeError(
                        f"custom worker exited during startup; inspect Modal sandbox {sandbox.object_id}"
                    ) from None
                if time.monotonic() >= deadline:
                    raise TimeoutError("custom GPU session did not become ready") from None
                time.sleep(2)
        client._http.timeout = 300
        client._cleanup = sandbox.terminate
        client.sandbox_id = sandbox.object_id
        return client
    except BaseException:
        # creation can commit despite a lost response. reconcile the unique name before cleanup;
        # never launch a second paid sandbox as a retry.
        if sandbox is None:
            try:
                sandbox = modal.Sandbox.from_name("flash-custom-training", name)
            except modal.exception.NotFoundError:
                pass
            except Exception:
                _log.exception(
                    "could not reconcile custom sandbox %s; hard timeout still applies", name
                )
        if sandbox is not None:
            try:
                sandbox.terminate()
            except Exception:
                _log.exception(
                    "could not terminate custom sandbox %s; hard timeout still applies", name
                )
        raise
