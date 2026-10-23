"""Sandbox-side entry for the kernel-cache bake (runs INSIDE the cu128 worker image on a GPU sandbox).

The published worker image ships DEPS only — the flash code (incl. kernel_warmup.py) is fetched from
HF at runtime, exactly like a real worker. So this entry:
  1. snapshot_downloads the flash package (code/**) the CI helper uploaded to a temp HF dataset,
  2. warms kernels at their final /opt/flash/kernelcache path in the training interpreter,
  3. replays an exported copy, then uploads the verified archive back under out/,
  4. writes an out/STATUS marker LAST so the helper knows the run finished (and its rc).

It is invoked by the CI helper via the sandbox command. Env:
  BAKE_HF_REPO  the temp HF dataset (private) holding code/** and receiving out/
  BAKE_ARCH     TORCH_CUDA_ARCH_LIST target, e.g. "9.0"
  HF_TOKEN      write token for BAKE_HF_REPO
"""

import io
import json
import os
import shutil
import subprocess
import tarfile
import time

CACHE_DIR = "/opt/flash/kernelcache"
OUT_DIR = "/out"
CACHE_ARCHIVE = "/tmp/kernel_cache.tar.gz"


def _roundtrip_cache(cache_dir: str, archive_path: str) -> None:
    """Replay only exported files, at the absolute paths embedded by Triton."""
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        archive.add(cache_dir, arcname=".")
    shutil.rmtree(cache_dir)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(cache_dir, filter="data")


def main() -> int:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, snapshot_download

    repo = os.environ["BAKE_HF_REPO"]
    token = os.environ["HF_TOKEN"]
    arch = os.environ.get("BAKE_ARCH", "")
    api = HfApi(token=token)

    # Upload a STARTED marker FIRST -- before the code download / warmup -- so a CI-side
    # timeout can tell whether this entrypoint even ran. Present on a timeout = the warmup started (so it
    # hung or was slow); absent = the pod ran the wrong entrypoint, OR this upload itself failed while
    # the warmup is still running. Retry a few times so a transient HF blip doesn't drop the marker and
    # make the helper misread a slow run as a wrong entrypoint.
    started_uploaded = False
    for attempt in range(3):
        try:
            api.upload_file(
                path_or_fileobj=io.BytesIO(f"started arch={arch}\n".encode()),
                path_in_repo="out/STARTED",
                repo_id=repo,
                repo_type="dataset",
            )
            started_uploaded = True
            print("[bake] uploaded out/STARTED", flush=True)
            break
        except Exception as e:
            # keep the "out/STARTED" token in the line so it greps the same as the success/WARNING lines
            print(f"[bake] out/STARTED upload attempt {attempt + 1}/3 failed: {e}", flush=True)
            if attempt < 2:  # no point sleeping after the last attempt
                time.sleep(3)
    if not started_uploaded:
        # The marker never landed; warn loudly so a later "no out/STARTED" is read as upload-failed,
        # not as a wrong-entrypoint run.
        print("[bake] WARNING: out/STARTED never uploaded; warmup still proceeds", flush=True)

    # the flash code the helper uploaded (upload_code -> code/flash); run the warmup from it.
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=["code/**"],
        local_dir="/runcode",
        token=token,
    )

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
        _roundtrip_cache(cache_dir, CACHE_ARCHIVE)
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
    # Hub repositories reject nested .cache paths (FlashInfer uses one). Transport the exact
    # directory layout as an archive, compiled at its final image path, without a second tree copy.
    try:
        with tarfile.open(CACHE_ARCHIVE, "w:gz", compresslevel=1) as archive:
            archive.add(cache_dir, arcname=".")
        api.upload_file(
            path_or_fileobj=CACHE_ARCHIVE,
            path_in_repo="out/kernel_cache.tar.gz",
            repo_id=repo,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=os.path.join(OUT_DIR, "warmup.log"),
            path_in_repo="out/bake-warmup.log",
            repo_id=repo,
            repo_type="dataset",
        )
    except Exception as e:
        rc = 1
        print(f"[bake] artifact upload failed: {e}", flush=True)
    # STATUS marker LAST: its presence is the helper's completion signal.
    api.upload_file(
        path_or_fileobj=io.BytesIO(f"rc={rc}\narch={arch}\n".encode()),
        path_in_repo="out/STATUS",
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"[bake] ENTRY_DONE rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
