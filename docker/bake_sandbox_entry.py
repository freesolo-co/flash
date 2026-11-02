"""Warm and verify kernels inside the GPU sandbox, writing outputs for direct Modal transfer.

The CI helper mounts the Flash package at /runcode/code/flash. The worker image supplies
its dependencies. HF_TOKEN is retained only for upstream model/tokenizer downloads;
BAKE_ARCH selects the compiler architecture. /out/STATUS is written last.
"""

import json
import os
import shutil
import subprocess
import tarfile
import time

CACHE_DIR = "/opt/flash/kernelcache"
OUT_DIR = "/out"
CACHE_ARCHIVE = "/out/kernel_cache.tar.gz"


def _replace_cache_with_exported_copy(cache_dir: str, archive_path: str) -> None:
    """Archive the cache, delete the originals, and restore files at the same absolute paths."""
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        archive.add(cache_dir, arcname=".")
    shutil.rmtree(cache_dir)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(cache_dir, filter="data")


def main() -> int:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    arch = os.environ.get("BAKE_ARCH", "")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "STARTED"), "w") as marker:
        marker.write(f"started arch={arch}\n")

    env = {**os.environ, "PYTHONPATH": "/runcode/code"}
    python_bin = os.environ.get("FLASH_VERL_PYTHON", "/opt/verl-venv/bin/python")
    if not os.path.isfile(python_bin):
        raise RuntimeError(f"training interpreter missing: {python_bin}")
    cache_dir = CACHE_DIR
    # An already-baked input image must not supply stale artifacts or validation receipts.
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir)
    cmd = [
        python_bin,
        "-m",
        "flash.engine.worker.runtime.kernel_warmup",
        "--out",
        cache_dir,
    ]
    if arch:
        cmd += ["--arch", arch]
    print(f"[bake] running: {' '.join(cmd)}", flush=True)
    # capture the warmup output into /out so it ships back with the cache -- lets the CI helper show
    # which warm steps ran and why a bake failed to produce compiled kernel files.
    os.makedirs(OUT_DIR, exist_ok=True)
    started = time.monotonic()
    with open(os.path.join(OUT_DIR, "warmup.log"), "wb") as lf:
        rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    first_seconds = time.monotonic() - started
    validation = {"first_process_seconds": first_seconds, "first_exit_code": rc}
    if rc == 0:
        # Remove the compiler outputs and restore only the exported archive. A fresh process
        # alone could otherwise reuse files that never make it into the shipped artifact.
        _replace_cache_with_exported_copy(cache_dir, CACHE_ARCHIVE)
        started = time.monotonic()
        with open(os.path.join(OUT_DIR, "warmup.log"), "ab") as lf:
            lf.write(b"\n[bake] fresh-process cache reload validation\n")
            rc = subprocess.call(
                [*cmd, "--verify-cache"], env=env, stdout=lf, stderr=subprocess.STDOUT
            )
        validation.update(
            reload_process_seconds=time.monotonic() - started,
            reload_exit_code=rc,
            exported_cache_replayed=True,
        )
    with open(os.path.join(cache_dir, "cache_validation.json"), "w") as output:
        json.dump(validation, output)
    print(f"[bake] kernel_warmup rc={rc}", flush=True)
    # Preserve hidden compiler directories and the exact cache layout in one streamed archive.
    with tarfile.open(CACHE_ARCHIVE, "w:gz", compresslevel=1) as archive:
        archive.add(cache_dir, arcname=".")
    with open(os.path.join(OUT_DIR, "STATUS"), "w") as marker:
        marker.write(f"rc={rc}\narch={arch}\n")
    print(f"[bake] ENTRY_DONE rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
