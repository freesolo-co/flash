"""Bounded Modal and authenticated HTTP operations for the fleet controller."""

import json
import os
import subprocess
import sys

import httpx
import modal

from flash.serving.deepinfer.app.endpoints import validate_endpoint
from flash.serving.deepinfer.app.replicas import replica_app
from flash.serving.deepinfer.contract.config import APP_NAME


class FleetOps:
    def __init__(self):
        self.headers = {
            "Modal-Key": os.environ["DEEPINFER_PROXY_KEY"],
            "Modal-Secret": os.environ["DEEPINFER_PROXY_SECRET"],
        }

    def launch(self, entry):
        try:
            server = modal.Server.from_name(entry["name"], "Engine")
            endpoint = validate_endpoint(server.get_url())
            entry["app_id"] = modal.App.lookup(entry["name"]).app_id
            return endpoint
        except modal.exception.NotFoundError:
            app = replica_app(
                entry["name"], modal.Image.from_id(entry["image_id"]), entry["version"]
            )
            app.deploy(environment_name=modal.config.config.get("environment"))
            entry["app_id"] = app.app_id
            return validate_endpoint(modal.Server.from_name(entry["name"], "Engine").get_url())

    def _call(self, entry, path, *, drain=False):
        try:
            response = httpx.request(
                "POST" if drain else "GET",
                validate_endpoint(entry["url"]) + path,
                headers=self.headers,
                timeout=3,
                follow_redirects=False,
            )
            response.raise_for_status()
            sample = response.json()
            count = sample.get("active_http_requests")
            if type(count) is not int or count < 0 or type(sample.get("draining")) is not bool:
                return None
            if drain and sample["draining"] is not True:
                return None
            if not drain and (
                sample.get("status") != "ok" or sample.get("deployment_id") != entry["version"]
            ):
                return None
            return sample
        except (httpx.HTTPError, ValueError):
            return None

    def health(self, entry):
        return self._call(entry, "/health")

    def drain(self, entry):
        return self._call(entry, "/internal/drain", drain=True)

    @staticmethod
    def _cli(*args):
        result = subprocess.run(
            [sys.executable, "-m", "modal", *args, "--env", modal.config.config.get("environment")],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout

    def stop(self, entry):
        legacy = entry["name"] == APP_NAME and entry["phase"] == "legacy_stopping"
        if not legacy and not entry["name"].startswith("freesolo-deepseek-replica-"):
            raise ValueError("Refusing to retire an unmanaged application")
        try:
            # Name lookup can disappear immediately after shutdown. The immutable
            # App ID remains addressable for safe retries after an interrupted tick.
            self._cli("app", "stop", entry["app_id"], "--yes")
        except subprocess.CalledProcessError as exc:
            # CLI reports an already-stopped app with a nonzero exit status.
            if "App is already stopped." not in (exc.stdout + exc.stderr):
                raise

    def stopped(self, entry):
        # Registered HTTP endpoints omit draining/pending workers. Count actual
        # containers before reusing a slot, including stopped-app shutdown grace.
        return not any(
            row["app_id"] == entry["app_id"]
            for row in json.loads(self._cli("container", "list", "--json"))
        )
