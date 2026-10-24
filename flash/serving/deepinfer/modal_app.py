"""Deploy the private DeepSeek HTTP pool with Modal's request-driven autoscaler."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import modal

from flash.serving.deepinfer.config import (
    BUFFER_CONTAINERS,
    GPU,
    MAX_CONTAINERS,
    MAX_INPUTS,
    MIN_CONTAINERS,
    PORT,
    REQUEST_SECONDS,
    RUNTIME_IMAGE_ID,
    RUNTIME_PYTHON,
    SCALEDOWN_SECONDS,
    STARTUP_SECONDS,
    TARGET_INPUTS,
)

image = (
    modal.Image.from_id(RUNTIME_IMAGE_ID)
    .env(
        {
            "PYTHONPATH": "/root:/opt/DeepInfer",
            "PATH": "/opt/deepinfer-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "VIRTUAL_ENV": "/opt/deepinfer-venv",
            "DEEPINFER_ACCELERATOR": "nvidia",
            "PYTHONUNBUFFERED": "1",
            "FLASH_DEEPINFER_CACHE_ROUTING": os.environ.get("FLASH_DEEPINFER_CACHE_ROUTING", "1"),
            "FLASH_DEEPINFER_CACHE_RETENTION": os.environ.get(
                "FLASH_DEEPINFER_CACHE_RETENTION", "frequency"
            ),
        }
    )
    .add_local_python_source("flash", copy=True)
    .add_local_file(
        Path(__file__).resolve().parents[3] / "scripts" / "prepare_deepinfer_image.py",
        remote_path="/root/prepare_deepinfer_image.py",
        copy=True,
    )
    .run_commands(
        f"{RUNTIME_PYTHON} -m pip install fastapi==0.141.1 uvicorn==0.52.0",
        f"{RUNTIME_PYTHON} /root/prepare_deepinfer_image.py",
    )
)
app = modal.App("freesolo-deepseek-serving", image=image)
weights = modal.Volume.from_name("freesolo-deepseek-1017-weights", create_if_missing=True)


def _watch_server(process):
    # Retire only this four-GPU replica on startup/native worker death. Other replicas and
    # requests remain independent; ordinary client cancellation never exits the server.
    process.wait()
    os._exit(1)


@app.function(
    gpu=GPU,
    cpu=16,
    memory=262144,
    volumes={"/weights": weights},
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    buffer_containers=BUFFER_CONTAINERS,
    scaledown_window=SCALEDOWN_SECONDS,
    startup_timeout=STARTUP_SECONDS,
    timeout=REQUEST_SECONDS,
)
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
@modal.web_server(PORT, startup_timeout=STARTUP_SECONDS, requires_proxy_auth=True)
def serve():
    process = subprocess.Popen([RUNTIME_PYTHON, "-m", "flash.serving.deepinfer.http"])
    threading.Thread(target=_watch_server, args=(process,), daemon=True).start()
