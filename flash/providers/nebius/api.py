"""Bounded Nebius CLI calls with project-scoped, reconciled VM identity."""

import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
import time
from decimal import Decimal

from flash.providers.core.base import UnreconciledCreateError
from flash.providers.nebius.config import settings
from flash.providers.nebius.pricing import SPOT_GPUS


class NebiusApiError(RuntimeError):
    pass


def call(*args, body=None, deadline_at=None):
    budget = min(60, deadline_at - time.time()) if deadline_at is not None else 60
    if budget <= 0:
        raise TimeoutError("Nebius operation exceeded run deadline")
    command = [
        "nebius",
        "--format",
        "json",
        "--no-browser",
        "--timeout",
        f"{budget}s",
        "--retries",
        "1",
        *args,
    ]
    # Payloads contain runtime credentials: never pass them in argv or log CLI stderr.
    try:
        with tempfile.TemporaryDirectory(prefix="flash-nebius-") as directory:
            if body is not None:
                path = os.path.join(directory, "request.json")
                with open(path, "w", encoding="utf-8") as stream:
                    json.dump(body, stream)
                os.chmod(path, 0o600)
                command += ["--file", path]
            result = subprocess.run(command, capture_output=True, timeout=budget, check=True)
            return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        raise NebiusApiError("Nebius operation failed or timed out") from None


def list_instances(*, deadline_at=None):
    data = call(
        "compute",
        "instance",
        "list",
        "--parent-id",
        settings()["project_id"],
        "--all",
        deadline_at=deadline_at,
    )
    if not isinstance(data, dict) or not isinstance(data.get("items", []), list):
        raise NebiusApiError("Nebius returned an invalid instance list")
    return [
        item
        for item in data.get("items", [])
        if item.get("metadata", {}).get("labels", {}).get("flash-owner") == "flash-training"
    ]


def get_instance(instance_id, *, deadline_at=None):
    # Listing within our project also avoids trusting an id from a different account.
    for item in list_instances(deadline_at=deadline_at):
        if item["metadata"]["id"] == instance_id:
            return {"id": instance_id, "status": item["status"]["state"].lower()}
    return None


def terminate(instance_id):
    deadline = time.time() + 180
    if get_instance(instance_id, deadline_at=deadline) is None:
        return
    # Deletion can outlast the synchronous CLI request. Reconcile the owned id
    # rather than treating the timeout as proof that deletion was rejected.
    with contextlib.suppress(NebiusApiError):
        call("compute", "instance", "delete", "--id", instance_id, deadline_at=deadline)
    while time.time() < deadline:
        if get_instance(instance_id, deadline_at=deadline) is None:
            return
        time.sleep(min(2, max(0, deadline - time.time())))
    raise NebiusApiError("Nebius deletion is not confirmed")


def pricing_policy(gpu, *, deadline_at):
    """Reconcile a deterministic immutable bid policy; never silently raise the bid."""
    shape = SPOT_GPUS[gpu]
    name = (
        "flashspot"
        + hashlib.sha256(f"{shape.platform}:{shape.keep_running_usd}".encode()).hexdigest()[:20]
    )
    args = ("billing", "pricing-policy", "list", "--parent-id", settings()["project_id"], "--all")

    def find():
        for policy in call(*args, deadline_at=deadline_at).get("items", []):
            if policy["metadata"].get("name") == name:
                spec = policy["spec"]
                if (
                    spec["compute_instance_spec"]["v1"]["platform"] != shape.platform
                    or Decimal(str(spec["pricing"]["max_price_v1"]["max_price"]))
                    != shape.keep_running_usd
                ):
                    raise NebiusApiError("Nebius Spot policy differs from Flash's fixed cap")
                return policy["metadata"]["id"]
        return None

    existing = find()
    if existing:
        return existing
    try:
        result = call(
            "billing",
            "pricing-policy",
            "create",
            "--parent-id",
            settings()["project_id"],
            "--name",
            name,
            "--compute-instance-spec-v1-platform",
            shape.platform,
            "--pricing-max-price-v1-max-price",
            str(shape.keep_running_usd),
            deadline_at=deadline_at,
        )
        return result["metadata"]["id"]
    except NebiusApiError:
        existing = find()
        if existing:
            return existing
        raise


def cloud_init(payload, capsule_ref, image, *, deadline_at):
    """Use a configured GPU host image with Docker + NVIDIA container toolkit installed."""
    from flash.providers.nebius.bootstrap import DOWNLOAD

    files = [
        {
            "path": "/opt/flash/capsule.json",
            "permissions": "0600",
            "content": json.dumps(capsule_ref),
        },
        {
            "path": "/opt/flash/download.py",
            "permissions": "0600",
            "content": DOWNLOAD,
        },
        {
            "path": "/opt/flash/payload.env",
            "permissions": "0600",
            "content": "FLASH_BOOTSTRAP_PAYLOAD="
            + json.dumps(payload, separators=(",", ":"))
            + "\n",
        },
    ]
    seconds = max(1, int(deadline_at - time.time()))
    # Hard-stop the worker even if the control plane disappears. Provider recovery is FAIL.
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=30",
        str(seconds),
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--env-file",
        "/opt/flash/payload.env",
        "-v",
        "/opt/flash/capsule.pyz:/opt/flash/capsule.pyz:ro",
        "--workdir",
        "/root/flash",
        image,
        "python",
        "-u",
        "/opt/flash/capsule.pyz",
        "bootstrap",
    ]
    user_data = "#cloud-config\n" + json.dumps(
        {
            "write_files": files,
            "runcmd": [["python3", "/opt/flash/download.py"], command, ["shutdown", "-h", "now"]],
        }
    )
    if len(user_data.encode()) > 32768:
        raise ValueError("Nebius cloud-init exceeds the 32 KiB limit")
    return user_data


def create(*, name, payload, image_reference, gpu, gpu_count, capsule, disk_gb, deadline_at):
    config = settings()
    shape = SPOT_GPUS[gpu]
    labels = {
        "flash-owner": "flash-training",
        "flash-run": payload["run_id"],
        "flash-attempt": str(payload["attempt"]),
        "flash-spec": hashlib.sha256(payload["job_spec_json"].encode()).hexdigest(),
        "flash-source": hashlib.sha256(
            json.dumps(payload["source_snapshot"], sort_keys=True).encode()
        ).hexdigest(),
    }

    def reconcile():
        for item in list_instances(deadline_at=deadline_at):
            if item["metadata"].get("name") == name:
                if item["metadata"].get("labels") != labels:
                    raise UnreconciledCreateError("Nebius attempt name belongs to another launch")
                return item["metadata"]["id"]
        return None

    existing = reconcile()
    if existing:
        return existing
    from flash.providers.nebius.bootstrap import publish_capsule

    capsule_ref = publish_capsule(payload, capsule, deadline_at=deadline_at)
    user_data = cloud_init(payload, capsule_ref, image_reference, deadline_at=deadline_at)
    policy_id = pricing_policy(gpu, deadline_at=deadline_at)
    body = {
        "metadata": {"parent_id": config["project_id"], "name": name, "labels": labels},
        "spec": {
            "resources": {"platform": shape.platform, "preset": shape.preset(gpu_count)},
            "boot_disk": {
                "attach_mode": "READ_WRITE",
                "managed_disk": {
                    "name": name + "-boot",
                    "spec": {
                        "type": "NETWORK_SSD_NON_REPLICATED",
                        # Fast ephemeral scratch; NRD disks require 93-GiB units.
                        "size_gibibytes": max(279, ((int(disk_gb) + 92) // 93) * 93),
                        "source_image_id": config["image_id"],
                    },
                },
            },
            "network_interfaces": [
                {
                    "name": "eth0",
                    "subnet_id": config["subnet_id"],
                    "ip_address": {},
                    "public_ip_address": {},
                }
            ],
            "recovery_policy": "FAIL",
            "preemptible": {"on_preemption": "STOP"},
            "spot_pricing_policy": {"id": policy_id},
            "cloud_init_user_data": user_data,
        },
    }
    try:
        result = call("compute", "instance", "create", body=body, deadline_at=deadline_at)
        return result["metadata"]["id"]
    except (NebiusApiError, KeyError):
        existing = reconcile()
        if existing:
            return existing
        raise UnreconciledCreateError(
            "Nebius launch could not be reconciled; refusing duplicate allocation"
        ) from None
