"""Explicit, staged adoption of a legacy pool that has no admission fence."""

import argparse
import time

import modal

from flash.serving.deepinfer.app.autoscaling import pool_load
from flash.serving.deepinfer.app.endpoints import FLEET_STATE
from flash.serving.deepinfer.app.fleet import controller_lock
from flash.serving.deepinfer.contract.config import APP_NAME, MAIN_MAX_CONTAINERS, POOL_STATE_NAME


def reserve_legacy(store, server, *, app_id):
    if store.get("fleet", None) is not None:
        raise RuntimeError("Fleet already initialized; refusing to overwrite its capacity ledger")
    settings = server.autoscaler.settings()
    if settings.max_containers != MAIN_MAX_CONTAINERS:
        raise RuntimeError("Finish the existing legacy rollout at its four-container cap first")
    store.put(
        "fleet",
        {
            "replicas": [
                {
                    "name": APP_NAME,
                    "app_id": app_id,
                    "version": "legacy",
                    "phase": "legacy",
                    "url": server.get_url(),
                    "slots": MAIN_MAX_CONTAINERS,
                }
            ],
            "quiet_since": None,
        },
    )


def authorize_legacy_retirement(store, server, samples, *, now, ingress_disabled):
    if not ingress_disabled:
        raise RuntimeError("Legacy ingress must be disabled and old router containers gone first")
    state = store.get("fleet")
    legacy = next(item for item in state["replicas"] if item["phase"] == "legacy")
    if not store.get("routes", {}).get("endpoints"):
        raise RuntimeError("No warmed fleet route exists")
    containers = {item.container_id for item in server._experimental_list_containers()}
    active, complete = pool_load(containers, samples, now)
    if not complete or active:
        raise RuntimeError(
            "Every legacy replica must have fresh telemetry and zero active requests"
        )
    # Operator confirmation is essential: legacy code cannot fence admissions.
    # Once confirmed and empty, the controller may stop it and await container exit.
    legacy["phase"] = "legacy_stopping"
    store.put("fleet", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--reserve-legacy", action="store_true")
    action.add_argument("--retire-legacy", action="store_true")
    parser.add_argument("--confirm-legacy-ingress-disabled", action="store_true")
    args = parser.parse_args()
    if modal.config.config.get("environment") != "main":
        raise ValueError("Legacy migration is production-only")
    store = modal.Dict.from_name(FLEET_STATE, create_if_missing=True)
    server = modal.Server.from_name(APP_NAME, "Engine")
    with controller_lock(store) as acquired:
        if not acquired:
            raise RuntimeError("Controller busy; retry migration after the current tick")
        if args.reserve_legacy:
            reserve_legacy(store, server, app_id=modal.App.lookup(APP_NAME).app_id)
        else:
            samples = dict(modal.Dict.from_name(POOL_STATE_NAME).items())
            authorize_legacy_retirement(
                store,
                server,
                samples,
                now=time.time(),
                ingress_disabled=args.confirm_legacy_ingress_disabled,
            )


if __name__ == "__main__":
    main()
