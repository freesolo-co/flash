"""Static Pod-side kernel-cache bake entrypoint."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _upload_marker(api, repo: str, path: str, content: str) -> None:
    api.upload_file(
        path_or_fileobj=io.BytesIO(content.encode()),
        path_in_repo=path,
        repo_id=repo,
        repo_type="dataset",
    )


def main(payload_path: str = "/root/flash/payload.json") -> int:
    from huggingface_hub import HfApi, snapshot_download

    payload = json.loads(Path(payload_path).read_text())
    if type(payload) is not dict or payload.get("mode") != "kernel_bake":
        raise RuntimeError("kernel bake payload is invalid")
    repo = str(payload["hf_repo"])
    arch = str(payload.get("arch") or "")
    token = str(payload.get("hf_token") or "")
    api = HfApi(token=token)
    _upload_marker(api, repo, "out/STARTED", f"started arch={arch}\n")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=["code/**"],
        local_dir="/runcode",
        token=token,
    )
    env = {**os.environ, "PYTHONPATH": "/runcode/code"}
    env.pop("HF_TOKEN", None)
    command = [sys.executable, "-m", "flash.engine.worker.runtime.kernel_warmup", "--out", "/out"]
    if arch:
        command += ["--arch", arch]
    os.makedirs("/out", exist_ok=True)
    with open("/out/warmup.log", "wb") as stream:
        rc = subprocess.call(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    api.upload_folder(folder_path="/out", path_in_repo="out", repo_id=repo, repo_type="dataset")
    _upload_marker(api, repo, "out/STATUS", f"rc={rc}\narch={arch}\n")
    time.sleep(90)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(*(sys.argv[1:] or [])))
