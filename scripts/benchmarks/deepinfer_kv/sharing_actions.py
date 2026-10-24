"""Transfer full native snapshots between independent replicas via an HTTPS RAM relay.

This measures a CPU/network transport prototype, not GPU-direct/RDMA performance.
Credentials are provisioned into the sandbox and never included in result records.
"""

import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

import torch


def execute(native, command):
    config = json.loads(Path("/opt/DeepInfer/relay.json").read_text())
    token = Path("/opt/DeepInfer/relay-token.secret").read_text()
    url = config["url"] + "/" + command["object"] + f"-rank{native.rank}"
    headers = {"Authorization": "Bearer " + token}
    started = time.perf_counter()
    if command["operation"] == "store":
        exports = {key: native.snapshot(key) for key in command["keys"]}
        buffer = io.BytesIO()
        torch.save({key: native.snapshots[key] for key in command["keys"]}, buffer)
        data = buffer.getvalue()
        serialized = time.perf_counter()
        request = urllib.request.Request(url, data=data, headers=headers, method="PUT")
        with urllib.request.urlopen(request, timeout=120) as response:
            assert response.read().decode() == hashlib.sha256(data).hexdigest()
        return {
            "exports": exports,
            "bytes": len(data),
            "encode_export_seconds": serialized - started,
            "network_seconds": time.perf_counter() - serialized,
            "total_seconds": time.perf_counter() - started,
        }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
        assert hashlib.sha256(data).hexdigest() == response.headers["X-Content-SHA256"]
    transferred = time.perf_counter()
    snapshots = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
    restores = {key: native.restore(key, snapshot) for key, snapshot in snapshots.items()}
    return {
        "restores": restores,
        "bytes": len(data),
        "network_seconds": transferred - started,
        "decode_restore_seconds": time.perf_counter() - transferred,
        "total_seconds": time.perf_counter() - started,
    }
