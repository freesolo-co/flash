"""Deploy the private DeepSeek HTTP pool with Modal's request-driven autoscaler."""

from __future__ import annotations

import os
import subprocess
import threading

import modal

from flash.serving.deepinfer.contract.config import (
    APP_NAME,
    BUFFER_CONTAINERS,
    GPU,
    MAIN_BUFFER_CONTAINERS,
    MAIN_MAX_CONTAINERS,
    MAIN_MIN_CONTAINERS,
    MAX_CONTAINERS,
    MAX_INPUTS,
    PORT,
    RUNTIME_IMAGE_ID,
    RUNTIME_PYTHON,
    SCALEDOWN_SECONDS,
    SCALEUP_SECONDS,
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
        }
    )
    .add_local_python_source("flash", copy=True)
    .run_commands(
        f"{RUNTIME_PYTHON} -m pip install fastapi==0.141.1 uvicorn==0.52.0",
        f"{RUNTIME_PYTHON} -m flash.serving.deepinfer.native.image",
    )
)
app = modal.App(APP_NAME, image=image)
weights = modal.Volume.from_name("freesolo-deepseek-1017-weights", create_if_missing=True)
main_environment = modal.config.config.get("environment") == "main"


def _watch_server(process):
    # Retire this entire four-GPU group on native worker death.
    # Ordinary client cancellation never exits the server.
    process.wait()
    os._exit(1)


@app.server(
    gpu=GPU,
    compute_region="us",
    routing_region="us-east",
    cpu=32,
    memory=524288,  # Engram host storage and loading peaked above 300 GiB in qualification.
    volumes={"/weights": weights},
    min_containers=MAIN_MIN_CONTAINERS if main_environment else 0,
    max_containers=MAIN_MAX_CONTAINERS if main_environment else MAX_CONTAINERS,
    buffer_containers=MAIN_BUFFER_CONTAINERS if main_environment else BUFFER_CONTAINERS,
    scaledown_window=SCALEDOWN_SECONDS,
    scaleup_window=SCALEUP_SECONDS,
    startup_timeout=STARTUP_SECONDS,
    target_concurrency=TARGET_INPUTS,
    max_concurrency=MAX_INPUTS,
    port=PORT,
    unauthenticated=False,
    exit_grace_period=3600,
)
class Engine:
    @modal.enter()
    def start(self):
        process = subprocess.Popen([RUNTIME_PYTHON, "-m", "flash.serving.deepinfer.app.http"])
        threading.Thread(target=_watch_server, args=(process,), daemon=True).start()
