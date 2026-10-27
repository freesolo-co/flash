"""Deploy the private DeepSeek HTTP pool with Modal's request-driven autoscaler."""

from __future__ import annotations

import os
import subprocess
import threading

import modal

from flash.serving.deepinfer.contract.config import (
    APP_NAME,
    BUFFER_CONTAINERS,
    DRAIN_SECONDS,
    GPU,
    MAIN_BUFFER_CONTAINERS,
    MAIN_MAX_CONTAINERS,
    MAIN_MIN_CONTAINERS,
    MAIN_ROLLOUT_MAX_CONTAINERS,
    MAX_CONTAINERS,
    POOL_SAMPLE_SECONDS,
    POOL_STATE_NAME,
    POOL_STREAM_CAPACITY,
    PORT,
    RUNTIME_IMAGE_ID,
    RUNTIME_PYTHON,
    SCALEDOWN_SECONDS,
    SCALEUP_SECONDS,
    STARTUP_SECONDS,
    TARGET_INPUTS,
)

# A unique rollout identity gates both completion and removal of surge capacity.
# Normal deployments retain the steady-state cap; CI explicitly enables overlap.
deployment_id = os.environ.get("DEEPINFER_ROLLOUT_ID", "")

# The pinned image already contains the patched native extension and dependencies.
image = (
    modal.Image.from_id(RUNTIME_IMAGE_ID)
    .env(
        {
            "PYTHONPATH": "/root:/opt/DeepInfer",
            "PATH": "/opt/deepinfer-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "VIRTUAL_ENV": "/opt/deepinfer-venv",
            "DEEPINFER_ACCELERATOR": "nvidia",
            "PYTHONUNBUFFERED": "1",
            "DEEPINFER_ROLLOUT_ID": deployment_id,
        }
    )
    .add_local_python_source("flash", copy=True)
)
app = modal.App(APP_NAME, image=image)
weights = modal.Volume.from_name("freesolo-deepseek-1017-weights", create_if_missing=True)
main_environment = modal.config.config.get("environment") == "main"
pool_load = (
    modal.Dict.from_name(POOL_STATE_NAME, create_if_missing=True) if main_environment else None
)

if main_environment:
    controller_image = (
        modal.Image.debian_slim(python_version="3.12")
        .env({"DEEPINFER_ROLLOUT_ID": deployment_id})
        .add_local_python_source("flash", copy=True)
    )

    @app.function(
        image=controller_image,
        schedule=modal.Period(seconds=POOL_SAMPLE_SECONDS),
        cpu=0.25,
        memory=256,
        min_containers=1,
        max_containers=1,
        timeout=15,
    )
    def scale_pool():
        from flash.serving.deepinfer.app.autoscaling import update_pool_capacity

        return update_pool_capacity(
            modal.Server.from_name(APP_NAME, "Engine"), pool_load, deployment_id=deployment_id
        )


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
    # CI warms one replacement to verify a rolling handoff. The controller releases
    # this temporary floor after the idle window; ordinary starts remain at zero.
    min_containers=(max(1, MAIN_MIN_CONTAINERS) if deployment_id else MAIN_MIN_CONTAINERS)
    if main_environment
    else 0,
    max_containers=(MAIN_ROLLOUT_MAX_CONTAINERS if deployment_id else MAIN_MAX_CONTAINERS)
    if main_environment
    else MAX_CONTAINERS,
    buffer_containers=MAIN_BUFFER_CONTAINERS if main_environment else BUFFER_CONTAINERS,
    scaledown_window=SCALEDOWN_SECONDS,
    scaleup_window=SCALEUP_SECONDS,
    startup_timeout=STARTUP_SECONDS,
    # Native scaling is a fallback. The controller raises the floor at pool-wide
    # reserve thresholds; this higher target cannot undo that floor.
    target_concurrency=POOL_STREAM_CAPACITY if main_environment else TARGET_INPUTS,
    # Application admission caps generations at MAX_INPUTS and marks safe
    # pre-admission rejections for retries; proxy rejections cannot attest that.
    max_concurrency=0,
    port=PORT,
    unauthenticated=False,
    exit_grace_period=DRAIN_SECONDS,
)
class Engine:
    @modal.enter()
    def start(self):
        process = subprocess.Popen([RUNTIME_PYTHON, "-m", "flash.serving.deepinfer.app.http"])
        threading.Thread(target=_watch_server, args=(process,), daemon=True).start()
        if main_environment:
            from flash.serving.deepinfer.app.autoscaling import report_load

            threading.Thread(
                target=report_load,
                args=(process, pool_load, os.environ["MODAL_TASK_ID"]),
                daemon=True,
            ).start()
