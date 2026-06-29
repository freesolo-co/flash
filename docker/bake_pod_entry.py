"""Pod-side entry for the kernel-cache bake (runs INSIDE the cu128 worker image on a GPU pod).

The published worker image ships DEPS only — the flash code (incl. kernel_warmup.py) is fetched from
HF at runtime, exactly like a real worker. So this entry:
  1. snapshot_downloads the flash package (code/**) the CI helper uploaded to a temp HF dataset,
  2. runs flash.engine.worker.kernel_warmup --arch <arch> --out /out on the live GPU,
  3. uploads the produced /out (mega_cache.bin + .json + triton/inductor trees) back under out/,
  4. writes an out/STATUS marker LAST so the helper knows the run finished (and its rc).

It is invoked by the CI helper via the pod's docker_args (base64-decoded to /tmp, then run). Env:
  BAKE_HF_REPO  the temp HF dataset (private) holding code/** and receiving out/
  BAKE_ARCH     TORCH_CUDA_ARCH_LIST target, e.g. "9.0"
  HF_TOKEN      write token for BAKE_HF_REPO
"""

import io
import os
import subprocess
import sys
import time


def main() -> int:
    from huggingface_hub import HfApi, snapshot_download

    repo = os.environ["BAKE_HF_REPO"]
    token = os.environ["HF_TOKEN"]
    arch = os.environ.get("BAKE_ARCH", "")
    api = HfApi(token=token)

    # Upload a STARTED marker FIRST -- before the code download / chalk install / warmup -- so a CI-side
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
    # match production: install freesolo-chalk so its default gap-filler kernels (rope, lora-delta,
    # embedding) get warmed too. best-effort — the warmup skips chalk cleanly if this fails, so a
    # chalk install hiccup just drops back to the validated 4/5 (no-chalk) bake.
    chalk_spec = os.environ.get("BAKE_CHALK_SPEC", "")
    if chalk_spec:
        print(f"[bake] installing chalk: {chalk_spec}", flush=True)
        subprocess.call([sys.executable, "-m", "pip", "install", "--no-cache-dir", chalk_spec])

    env = {**os.environ, "PYTHONPATH": "/runcode/code"}
    cmd = [sys.executable, "-m", "flash.engine.worker.kernel_warmup", "--out", "/out"]
    if arch:
        cmd += ["--arch", arch]
    print(f"[bake] running: {' '.join(cmd)}", flush=True)
    # capture the warmup output into /out so it ships back with the cache -- lets the CI helper show
    # WHICH warm steps compiled and what save_cache_artifacts returned when no mega_cache.bin lands.
    os.makedirs("/out", exist_ok=True)
    with open("/out/warmup.log", "wb") as lf:
        rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[bake] kernel_warmup rc={rc}", flush=True)

    # ship the whole cache tree back (mega blob + metadata + raw triton/inductor dirs).
    if os.path.isdir("/out"):
        try:
            api.upload_folder(
                folder_path="/out", path_in_repo="out", repo_id=repo, repo_type="dataset"
            )
        except Exception as e:
            print(f"[bake] artifact upload failed: {e}", flush=True)
    # STATUS marker LAST: its presence is the helper's completion signal.
    api.upload_file(
        path_or_fileobj=io.BytesIO(f"rc={rc}\narch={arch}\n".encode()),
        path_in_repo="out/STATUS",
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"[bake] ENTRY_DONE rc={rc}", flush=True)
    # stay alive briefly so the poller observes out/STATUS while the pod is still "up" (it polls on a
    # ~45s cadence); without this the pod exits instantly and the poller may only ever see EXITED.
    time.sleep(90)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
