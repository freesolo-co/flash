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
import shlex
import sys
import urllib.request

BASE = "https://rest.runpod.io/v1/pods"
IMAGE = "ghcr.io/freesolo-co/flash-worker:cu128"
HF = os.environ.get("HF_TOKEN", "")


# Read auth + pubkey lazily (not at import) so `status`/`--help`/argparse don't crash with
# KeyError/FileNotFoundError before they even run; fail with a clear message when actually needed.
def _key() -> str:
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        raise SystemExit("RUNPOD_API_KEY is not set")
    return k


def _pubkey() -> str:
    p = os.path.expanduser(os.environ.get("BENCH_SSH_PUBKEY", "~/.ssh/chalk_pod.pub"))
    try:
        with open(p) as f:
            return f.read().strip()
    except FileNotFoundError:
        raise SystemExit(f"SSH pubkey not found at {p} (set BENCH_SSH_PUBKEY)") from None


def _headers() -> dict:
    return {"Authorization": "Bearer " + _key(), "Content-Type": "application/json"}


def call(method: str, url: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=_headers(), method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:400]}") from e


def create(gpu: str, name: str, disk: str) -> None:
    start = (
        "apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null 2>&1; "
        "mkdir -p /run/sshd /root/.ssh; echo " + shlex.quote(_pubkey()) + " >> /root/.ssh/authorized_keys; "
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


_USAGE = "usage: runpod_pod.py {create <gpu> <name> <disk> | status <id> | terminate <id>}"
_CMDS = {
    "create": lambda a: create(a[0], a[1], a[2]),
    "status": lambda a: status(a[0]),
    "terminate": lambda a: terminate(a[0]),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        raise SystemExit(_USAGE)
    try:
        _CMDS[sys.argv[1]](sys.argv[2:])
    except IndexError:
        raise SystemExit(_USAGE) from None
