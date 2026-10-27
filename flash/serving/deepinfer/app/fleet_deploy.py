"""Publish immutable replica source and request a drain-safe fleet rollout."""

import argparse
import os
import time

import modal

from flash.serving.deepinfer.app.endpoints import FLEET_NAME, FLEET_STATE
from flash.serving.deepinfer.contract.config import APP_NAME, REQUEST_SECONDS, STARTUP_SECONDS


def require_fleet_migration(store):
    if store.get("fleet", None) is not None:
        return
    try:
        modal.Server.from_name(APP_NAME, "Engine").get_url()
    except modal.exception.NotFoundError:
        return
    raise RuntimeError(
        "The legacy monolithic DeepSeek pool still exists. Its existing streams cannot be "
        "protected retroactively by a new deployment. Complete the documented legacy migration "
        "before enabling fleet deployments; no existing application has been modified."
    )


def wait_for_fleet(
    store, version, *, timeout=STARTUP_SECONDS + REQUEST_SECONDS + 900, staging=False
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = store.get("fleet", {"replicas": []})
        entries = state["replicas"]
        if staging:
            entries = [item for item in entries if item["phase"] != "legacy"]
        if (
            entries
            and all(item["version"] == version and item["phase"] == "active" for item in entries)
            and store.get("routes", {"endpoints": []})["endpoints"]
        ):
            print(f"Fleet {version} ready; legacy migration pending: {staging}.", flush=True)
            return
        print(
            f"Waiting for fleet {version}: {[(r['name'], r['phase']) for r in entries]}", flush=True
        )
        time.sleep(5)
    raise TimeoutError("Fleet rollout incomplete; the controller retains all undrained replicas")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-migration", action="store_true")
    args = parser.parse_args()
    if modal.config.config.get("environment") != "main":
        raise ValueError("Versioned fleets are deployed only in main")
    version = os.environ.get("DEEPINFER_ROLLOUT_ID", "")
    if not version:
        raise ValueError("DEEPINFER_ROLLOUT_ID is required")
    store = modal.Dict.from_name(FLEET_STATE, create_if_missing=True)
    require_fleet_migration(store)
    legacy = any(r["phase"] == "legacy" for r in store.get("fleet", {"replicas": []})["replicas"])
    if legacy != args.stage_migration:
        raise RuntimeError("Use --stage-migration only for a reserved, unfinished legacy migration")
    from flash.serving.deepinfer.app.fleet_app import app
    from flash.serving.deepinfer.app.modal_app import image

    # Build source once. Later scale-out reuses this exact image, not whatever
    # Python happens to be installed in the current controller deployment.
    image.build(modal.App.lookup(FLEET_NAME + "-images", create_if_missing=True))
    app.deploy(environment_name="main", strategy="rolling")
    store.put("desired", {"version": version, "image_id": image.object_id})
    wait_for_fleet(store, version, staging=args.stage_migration)


if __name__ == "__main__":
    main()
