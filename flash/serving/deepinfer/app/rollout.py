"""Wait for warmed main-pool replacements and restoration of the normal cap."""

from __future__ import annotations

import argparse
import os
import time

from flash.serving.deepinfer.app.autoscaling import rollout_ready
from flash.serving.deepinfer.contract.config import (
    APP_NAME,
    MAIN_MAX_CONTAINERS,
    POOL_SAMPLE_SECONDS,
    POOL_STATE_NAME,
    STARTUP_SECONDS,
)


def wait_for_rollout(server, store, deployment_id, *, timeout=STARTUP_SECONDS + 600):
    if not deployment_id:
        raise ValueError("DEEPINFER_ROLLOUT_ID must identify the deployment being verified")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        containers = {item.container_id for item in server._experimental_list_containers()}
        ready = rollout_ready(containers, dict(store.items()), deployment_id, time.time())
        settings = server.autoscaler.settings()
        if ready and settings.max_containers == MAIN_MAX_CONTAINERS:
            print(
                f"Rollout {deployment_id} ready: {len(containers)} warmed replicas; "
                f"maximum restored to {MAIN_MAX_CONTAINERS}.",
                flush=True,
            )
            return
        print(
            f"Waiting for rollout {deployment_id}: registered={len(containers)}, "
            f"all_replacements_warm={ready}, maximum={settings.max_containers}",
            flush=True,
        )
        time.sleep(POOL_SAMPLE_SECONDS)
    raise TimeoutError(
        "DeepSeek replacement readiness was not confirmed; old replicas were not stopped. "
        "The controller will restore the normal cap only after a successful warm handoff."
    )


def main():
    import modal

    if modal.config.config.get("environment") != "main":
        raise ValueError("Rollout verification is for the main DeepSeek pool")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--begin", action="store_true")
    args = parser.parse_args()
    deployment_id = os.environ.get("DEEPINFER_ROLLOUT_ID", "")
    if not deployment_id:
        raise ValueError("DEEPINFER_ROLLOUT_ID is required")
    store = modal.Dict.from_name(POOL_STATE_NAME, create_if_missing=args.begin)
    if args.begin:
        # Old controller containers can overlap the new deployment too. Disarm
        # their cleanup before changing the Server's capacity or application.
        store.put("rollout", {"deployment_id": deployment_id})
        return
    wait_for_rollout(
        modal.Server.from_name(APP_NAME, "Engine"),
        store,
        deployment_id,
    )


if __name__ == "__main__":
    main()
