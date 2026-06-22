#!/usr/bin/env python3
"""Minimal RunPod on-demand pod lifecycle helper (REST API) for benchmarking.

The flash worker image is pytorch-devel (no sshd), so we override the start command to install +
run sshd with our pubkey, exposing 22/tcp — then we scp the bench scripts and run them exactly like
on Vast. Usage:
  python runpod_pod.py create "NVIDIA H100 80GB HBM3" fla-h100 80
  python runpod_pod.py status <podId>
  python runpod_pod.py terminate <podId>
"""
import json
import os
import sys
import urllib.request

BASE = "https://rest.runpod.io/v1/pods"
KEY = os.environ["RUNPOD_API_KEY"]
IMAGE = "ghcr.io/freesolo-co/flash-worker:cu128"
HF = os.environ.get("HF_TOKEN", "")
H = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}


def call(method: str, url: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:400]}") from e


def create(gpu: str, name: str, disk: str) -> None:
    # Read the SSH pubkey lazily here (only create needs it) so status/terminate don't fail when the
    # key file is absent.
    with open(os.path.expanduser(os.environ.get("BENCH_SSH_PUBKEY", "~/.ssh/chalk_pod.pub"))) as _f:
        pubkey = _f.read().strip()
    start = (
        "apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null 2>&1; "
        "mkdir -p /run/sshd /root/.ssh; echo '" + pubkey + "' >> /root/.ssh/authorized_keys; "
        "chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd -D -p 22"
    )
    body = {
        "name": name, "imageName": IMAGE, "gpuTypeIds": [gpu], "gpuCount": 1,
        "containerDiskInGb": int(disk), "volumeInGb": 0, "ports": ["22/tcp"],
        "dockerStartCmd": ["bash", "-c", start],
        "env": {"HF_TOKEN": HF, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }
    r = call("POST", BASE, body)
    print(json.dumps({"id": r.get("id"), "status": r.get("desiredStatus")}))


def status(pod_id: str) -> None:
    r = call("GET", f"{BASE}/{pod_id}")
    print(json.dumps({"id": r.get("id"), "status": r.get("desiredStatus"),
                      "publicIp": r.get("publicIp"), "portMappings": r.get("portMappings"),
                      "ports": r.get("ports"), "machine": (r.get("machine") or {}).get("gpuTypeId")}))


def terminate(pod_id: str) -> None:
    call("DELETE", f"{BASE}/{pod_id}")
    print(json.dumps({"terminated": pod_id}))


if __name__ == "__main__":
    {"create": lambda: create(sys.argv[2], sys.argv[3], sys.argv[4]),
     "status": lambda: status(sys.argv[2]),
     "terminate": lambda: terminate(sys.argv[2])}[sys.argv[1]]()
