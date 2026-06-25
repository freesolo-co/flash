"""CI helper: produce a per-arch compiled-kernel cache by offloading the warmup to a RunPod GPU.

GitHub's hosted runners have no GPU, but the kernels are content-addressed by GPU arch + toolchain,
so the warmup MUST run on a real GPU of the target arch inside the worker image's exact stack. This
helper (runs on the CPU runner) does that offload and lands the artifact in ``--out`` so the
following ``docker build --build-arg BUILD_KERNEL_CACHE=true`` bakes it into the per-sm image:

  1. upload THIS checkout's flash package to a fresh private HF dataset (code/flash), like a worker,
  2. create a RunPod pod FROM the worker image on a GPU of the target arch; its command base64-decodes
     docker/bake_pod_entry.py and runs it (download code/** -> kernel_warmup -> upload out/),
  3. poll HF for the out/STATUS marker (and pod liveness) until done or the deadline,
  4. download out/ into ``--out`` (build/kernel_cache), terminate the pod, delete the temp dataset.

Env: RUNPOD_API_KEY, HF_TOKEN (write on the Freesolo-Co namespace). Exits non-zero unless a
mega_cache.bin with matching arch metadata landed in ``--out``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import time

ARTIFACT_NAMESPACE = "Freesolo-Co"


def log(msg: str) -> None:
    print(f"[bake-ci] {msg}", flush=True)


def _upload_flash_code(api, repo: str, token: str) -> None:
    """Mirror the worker's upload_code: push the running flash package to code/flash (exact mirror)."""
    import flash

    pkg_dir = os.path.realpath(os.path.dirname(os.path.abspath(flash.__file__)))
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=pkg_dir,
        path_in_repo="code/flash",
        repo_id=repo,
        repo_type="dataset",
        ignore_patterns=["__pycache__/*", "*.pyc"],
        delete_patterns=["**"],
    )
    log(f"uploaded flash code -> {repo}:code/flash (from {pkg_dir})")


def _docker_args(entry_path: str) -> str:
    """Base64-embed the pod entry into a one-line container command (no quoting hazards)."""
    with open(entry_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (
        f"bash -lc 'echo {b64} | base64 -d > /tmp/bake_pod_entry.py "
        f"&& python /tmp/bake_pod_entry.py'"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="offload the kernel-cache warmup to a RunPod GPU")
    ap.add_argument("--arch", required=True, help="TORCH_CUDA_ARCH_LIST target, e.g. 9.0")
    ap.add_argument("--sm", required=True, help="sm tag, e.g. sm90 (must match the produced cache)")
    ap.add_argument("--gpu-type-id", required=True, help="RunPod gpuTypeId, e.g. 'NVIDIA H100 80GB HBM3'")
    ap.add_argument("--image", default="ghcr.io/freesolo-co/flash-worker:cu128")
    ap.add_argument("--out", default="build/kernel_cache")
    ap.add_argument("--container-disk-gb", type=int, default=80)
    ap.add_argument("--deadline-min", type=int, default=45)
    ap.add_argument("--run-id", default="", help="unique suffix for the temp repo (default: time-based)")
    args = ap.parse_args()

    token = os.environ["HF_TOKEN"]
    import runpod
    from huggingface_hub import HfApi, snapshot_download

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    api = HfApi(token=token)

    suffix = args.run_id or str(int(time.time()))
    repo = f"{ARTIFACT_NAMESPACE}/kernel-bake-{args.sm}-{suffix}"
    entry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bake_pod_entry.py")

    # the chalk install spec the production worker uses (so the bake warms chalk's kernels too);
    # FLASH_CHALK_SPEC overrides, else the source-of-truth DEFAULT_CHALK_SPEC, else a safe literal.
    chalk_spec = os.environ.get("FLASH_CHALK_SPEC", "").strip()
    if not chalk_spec:
        try:
            from flash.providers.runpod.train.deps import DEFAULT_CHALK_SPEC

            chalk_spec = DEFAULT_CHALK_SPEC
        except Exception:
            chalk_spec = "freesolo-chalk>=0.1.0,<0.2.0"

    _upload_flash_code(api, repo, token)

    pod = runpod.create_pod(
        name=f"kernel-bake-{args.sm}-{suffix}",
        image_name=args.image,
        gpu_type_id=args.gpu_type_id,
        cloud_type="ALL",
        container_disk_in_gb=args.container_disk_gb,
        docker_args=_docker_args(entry_path),
        env={
            "BAKE_HF_REPO": repo,
            "BAKE_ARCH": args.arch,
            "HF_TOKEN": token,
            "BAKE_CHALK_SPEC": chalk_spec,
        },
    )
    pod_id = pod["id"]
    log(f"created pod {pod_id} ({args.gpu_type_id}, {args.sm}); polling for out/STATUS")

    deadline = time.time() + args.deadline_min * 60
    outcome = "timeout"
    dead = 0
    while time.time() < deadline:
        try:
            p = runpod.get_pod(pod_id)
            ds = (p or {}).get("desiredStatus")
            up = bool((p or {}).get("runtime"))
            log(f"pod desired={ds} runtime={'up' if up else 'none'}")
            dead = dead + 1 if ds in ("TERMINATED", "EXITED") else 0
        except Exception as e:
            log(f"pod status err: {str(e)[:120]}")
        try:
            files = api.list_repo_files(repo, repo_type="dataset")
            if "out/STATUS" in files:
                outcome = "done"
                log("out/STATUS present -> warmup complete")
                break
        except Exception as e:
            log(f"hf list err: {str(e)[:120]}")
        if dead >= 2:
            outcome = "pod_died"
            log("pod terminated/exited before STATUS")
            break
        time.sleep(45)

    # always tear the pod down
    try:
        runpod.terminate_pod(pod_id)
        log(f"terminated pod {pod_id}")
    except Exception as e:
        log(f"terminate fail: {str(e)[:120]}")

    rc = 1
    if outcome == "done":
        tmp = os.path.join(args.out, ".dl")
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            allow_patterns=["out/**"],
            local_dir=tmp,
            token=token,
        )
        src = os.path.join(tmp, "out")
        os.makedirs(args.out, exist_ok=True)
        for name in os.listdir(src):
            if name == "STATUS":
                continue
            s, d = os.path.join(src, name), os.path.join(args.out, name)
            shutil.rmtree(d, ignore_errors=True) if os.path.isdir(d) else None
            shutil.move(s, d)
        shutil.rmtree(tmp, ignore_errors=True)
        rc = _verify(args.out, args.sm)

    # best-effort cleanup of the temp dataset
    try:
        api.delete_repo(repo_id=repo, repo_type="dataset")
        log(f"deleted temp dataset {repo}")
    except Exception as e:
        log(f"temp dataset delete fail (ignore): {str(e)[:120]}")

    log(f"DONE outcome={outcome} rc={rc}")
    return rc


def _verify(out: str, sm: str) -> int:
    """Confirm the artifact actually landed and its metadata matches the requested arch."""
    blob = os.path.join(out, "mega_cache.bin")
    meta = os.path.join(out, "mega_cache.json")
    if not os.path.isfile(blob):
        log(f"FAIL: no mega_cache.bin in {out}")
        return 1
    try:
        with open(meta) as f:
            got = json.load(f).get("sm")
    except Exception as e:
        log(f"FAIL: metadata unreadable ({e})")
        return 1
    if got != sm:
        log(f"FAIL: baked arch {got!r} != requested {sm!r}")
        return 1
    log(f"OK: mega_cache.bin ({os.path.getsize(blob)} bytes) for {sm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
