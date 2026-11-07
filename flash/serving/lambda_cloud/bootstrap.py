"""Per-replica TLS and cloud-init; GPU machines never receive the Lambda API key."""

import base64
import json
import re
import secrets
import subprocess
import tempfile
from pathlib import Path

from .config import PORT, RUNTIME_IMAGE, SERVER_NAME, placement

GPU_SECRET_NAMES = (
    "HF_TOKEN",
    "FLASH_ARTIFACT_BLOB_SAS_URL",
    "FLASH_ARTIFACT_BLOB_ACCOUNT_URL",
    "FLASH_ARTIFACT_BLOB_CONTAINER",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
)


def identity():
    with tempfile.TemporaryDirectory() as directory:
        key, cert = Path(directory) / "key.pem", Path(directory) / "cert.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "365",
                "-keyout",
                str(key),
                "-out",
                str(cert),
                "-subj",
                f"/CN={SERVER_NAME}",
                "-addext",
                f"subjectAltName=DNS:{SERVER_NAME}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {
            "key": key.read_text(),
            "cert": cert.read_text(),
            "token": secrets.token_urlsafe(48),
        }


def user_data(model, revision, credentials, tls):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Lambda workers require an exact source commit")
    _, gpu = placement(model)
    env = {name: credentials[name] for name in GPU_SECRET_NAMES if credentials.get(name)}
    env.update(
        {
            "FLASH_LAMBDA_MODEL": model,
            "FLASH_LAMBDA_GPU": gpu,
            "FLASH_LAMBDA_TOKEN": tls["token"],
            "FREESOLO_DEPLOYMENT_SHA": revision,
            "HF_HOME": "/vol/hosting-cache",
            "HF_HUB_CACHE": "/vol/hosting-cache/hub",
            "TRANSFORMERS_CACHE": "/vol/hosting-cache/transformers",
            "VLLM_CACHE_ROOT": "/vol/hosting-cache/vllm",
            "FLASHINFER_WORKSPACE_BASE": "/vol/hosting-cache/flashinfer",
            "HF_HUB_DISABLE_XET": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VLLM_MOE_USE_DEEP_GEMM": "0",
            "PYTHONPATH": "/app",
        }
    )
    # Docker env files cannot encode embedded newlines safely.
    if any("\n" in value or "\r" in value for value in env.values()):
        raise ValueError("worker environment contains a newline")
    files = {
        "key.pem": tls["key"],
        "cert.pem": tls["cert"],
        "worker.env": "".join(f"{key}={value}\n" for key, value in env.items()),
    }
    setup = f"""#!/bin/bash
set -euo pipefail
mkdir -p /opt/flash-source
for attempt in $(seq 1 60); do
    mountpoint -q /vol/hosting-cache && break
    sleep 5
done
mountpoint -q /vol/hosting-cache
curl --fail --silent --show-error --location https://github.com/freesolo-co/flash/archive/{revision}.tar.gz -o /opt/flash/source.tar.gz
tar -xzf /opt/flash/source.tar.gz --strip-components=1 -C /opt/flash-source
docker pull {RUNTIME_IMAGE}
# Keep the qualified dependency stack; source overlays must use identical runtime locks.
docker run --rm --entrypoint python -v /opt/flash-source:/source:ro -v /opt/flash-source/flash:/app/flash:ro {RUNTIME_IMAGE} -m flash.serving.lambda_cloud.runtime_image /app/uv.lock /source/uv.lock
docker run -d --restart unless-stopped --name flash-worker --gpus all --ipc host --env-file /opt/flash/worker.env -p {PORT}:{PORT} -v /opt/flash-source/flash:/app/flash:ro -v /opt/flash:/run/flash:ro -v /vol/hosting-cache:/vol/hosting-cache {RUNTIME_IMAGE} python -m flash.serving.lambda_cloud.worker
"""
    files["setup.sh"] = setup
    payload = {
        "write_files": [
            {
                "path": f"/opt/flash/{name}",
                "permissions": "0600",
                "encoding": "b64",
                "content": base64.b64encode(value.encode()).decode(),
            }
            for name, value in files.items()
        ],
        "runcmd": [["bash", "/opt/flash/setup.sh"]],
    }
    return "#cloud-config\n" + json.dumps(payload)
