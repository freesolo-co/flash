"""CPU controller only; changing this App never redeploys GPU replicas."""

import os
import time

import modal

from flash.serving.deepinfer.app.endpoints import FLEET_NAME, FLEET_STATE
from flash.serving.deepinfer.app.fleet import controller_lock, reconcile_fleet
from flash.serving.deepinfer.app.fleet_ops import FleetOps

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("modal==1.6.2", "httpx>=0.28,<1")
    .add_local_python_source("flash", copy=True)
)
app = modal.App(FLEET_NAME, image=image)
store = modal.Dict.from_name(FLEET_STATE, create_if_missing=True)


@app.function(
    schedule=modal.Period(seconds=5),
    min_containers=1,
    max_containers=1,
    cpu=0.5,
    memory=512,
    nonpreemptible=True,
    timeout=180,
    secrets=[
        modal.Secret.from_dict(
            {key: os.environ[key] for key in ("DEEPINFER_PROXY_KEY", "DEEPINFER_PROXY_SECRET")}
        )
    ],
)
def reconcile():
    with controller_lock(store) as acquired:
        if not acquired:
            return {"status": "controller_busy"}
        return reconcile_fleet(store, FleetOps(), now=time.time())
