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


def main() -> int:
    from huggingface_hub import HfApi, snapshot_download

    repo = os.environ["BAKE_HF_REPO"]
    token = os.environ["HF_TOKEN"]
    arch = os.environ.get("BAKE_ARCH", "")
    api = HfApi(token=token)

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
    rc = subprocess.call(cmd, env=env)
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
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
