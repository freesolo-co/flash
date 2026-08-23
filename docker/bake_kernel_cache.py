"""CI helper: produce a per-arch compiled-kernel cache by offloading the warmup to a RunPod GPU.

GitHub's hosted runners have no GPU, but the kernels are content-addressed by GPU arch + toolchain,
so the warmup MUST run on a real GPU of the target arch inside the worker image's exact stack. This
helper (runs on the CPU runner) does that offload and lands the artifact in ``--out`` so the
following ``docker build --build-arg BUILD_KERNEL_CACHE=true`` bakes it into the per-sm image:

  1. upload THIS checkout's flash package to a fresh private HF dataset (code/flash), like a worker,
  2. create a RunPod pod FROM the worker image by walking the target arch's secure-cloud GPU types
     (and retrying the walk past transient capacity rejections); its command base64-decodes
     docker/bake_pod_entry.py and runs it
     (download code/** -> kernel_warmup -> upload out/),
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
import random
import shutil
import time
import uuid

ARTIFACT_NAMESPACE = "Freesolo-Co"

# ordered secure-cloud gpu types for each baked architecture. the first entry keeps the measured
# preferred sku; later entries prevent one empty provider pool from leaving the per-sm image stale
# and forcing ordinary worker starts back through jit. every created pod is still checked against
# --arch by kernel_warmup before any compilation, so this list cannot silently mislabel a cache.
GPU_WALK_BY_SM: dict[str, tuple[str, ...]] = {
    "sm80": (
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
    ),
    "sm86": (
        "NVIDIA RTX A6000",
        "NVIDIA A40",
        "NVIDIA RTX A5000",
        "NVIDIA GeForce RTX 3090",
    ),
    "sm89": (
        "NVIDIA L40S",
        "NVIDIA L40",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA GeForce RTX 4090",
    ),
    "sm90": (
        "NVIDIA H200",
        "NVIDIA H200 NVL",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 PCIe",
        "NVIDIA H100 NVL",
    ),
    "sm100": ("NVIDIA B200",),
    "sm120": (
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "NVIDIA GeForce RTX 5090",
    ),
}

# RunPod picks the host at create time, so a create can be rejected simply because the machine it
# picked has nothing free right now. that is transient placement, not a broken bake: the API takes
# no "not this machine" hint, but each create is placed again server-side, so retrying is what moves
# us to another host. everything else (auth, quota, an unpullable image) is a real error and must
# fail the arch immediately instead of burning attempts and GPU-queue time on it.
CAPACITY_MARKERS = (
    # "This machine does not have the resources to deploy your pod. Please try a different machine"
    "does not have the resources",
    # "There are no longer any instances available with the requested specifications." (and the
    # "...with enough disk space." variant)
    "no longer any instances",
    "no instances available",
)
CREATE_ROUNDS = 5
# a round tries every same-sm type before waiting. the last entry repeats for any further round;
# under 10 min of waiting worst case, well inside the 120-min cap.
CREATE_BACKOFF_S = (30, 60, 120, 240)


def log(msg: str) -> None:
    print(f"[bake-ci] {msg}", flush=True)


def _upload_flash_code(api, repo: str, token: str) -> None:
    """Mirror the worker's upload_code: push the running flash package to code/flash (exact mirror)."""
    import flash

    pkg_dir = os.path.realpath(os.path.dirname(os.path.abspath(flash.__file__)))
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    # mirror upload_code: force private even if the id pre-existed as a public repo, before any
    # flash source is uploaded.
    api.update_repo_settings(repo_id=repo, repo_type="dataset", private=True)
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


def _is_capacity_error(exc: BaseException) -> bool:
    """True for RunPod's transient "this host has nothing free" rejections, false for real errors."""
    msg = str(exc).lower()
    return any(marker in msg for marker in CAPACITY_MARKERS)


def _create_pod_with_gpu_walk(
    runpod,
    *,
    gpu_type_ids: tuple[str, ...],
    rounds: int = CREATE_ROUNDS,
    backoff_s: tuple[int, ...] = CREATE_BACKOFF_S,
    **kwargs,
):
    """walk same-sm gpu types by capacity; any other failure raises on the first try."""
    if not gpu_type_ids:
        raise ValueError("gpu walk must contain at least one gpu type")
    last: BaseException | None = None
    for round_index in range(rounds):
        for gpu_type_id in gpu_type_ids:
            try:
                log(f"gpu walk round {round_index + 1}/{rounds}: trying {gpu_type_id!r}")
                pod = runpod.create_pod(gpu_type_id=gpu_type_id, **kwargs)
                return pod, gpu_type_id
            except Exception as e:
                if not _is_capacity_error(e):
                    raise
                last = e
                log(f"{gpu_type_id!r} rejected for capacity: {str(e)[:160]}")
        if round_index == rounds - 1:
            break
        # jitter so the matrix legs and concurrent runs do not re-ask in lockstep.
        delay = backoff_s[min(round_index, len(backoff_s) - 1)]
        delay += random.uniform(0, 0.25 * delay)
        log(f"all {len(gpu_type_ids)} gpu types full; retrying walk in {delay:.0f}s")
        time.sleep(delay)
    raise RuntimeError(
        f"no capacity across gpu walk {gpu_type_ids!r} after {rounds} rounds"
    ) from last


def main() -> int:
    ap = argparse.ArgumentParser(description="offload the kernel-cache warmup to a RunPod GPU")
    ap.add_argument("--arch", required=True, help="TORCH_CUDA_ARCH_LIST target, e.g. 9.0")
    ap.add_argument("--sm", required=True, help="sm tag, e.g. sm90 (must match the produced cache)")
    ap.add_argument(
        "--gpu-type-id",
        action="append",
        default=[],
        help="override gpu walk with this RunPod gpuTypeId; repeat for multiple ordered choices",
    )
    ap.add_argument("--image", default="ghcr.io/freesolo-co/flash-worker:cu128")
    ap.add_argument("--out", default="build/kernel_cache")
    # the warm pod only pulls the ~20GB image + writes the cache (no model download), so keep this
    # modest -- an over-large ask shrinks the eligible host pool and trips "machine does not have the
    # resources" on scarce classes (e.g. Blackwell sm120 on secure cloud).
    ap.add_argument("--container-disk-gb", type=int, default=60)
    ap.add_argument("--deadline-min", type=int, default=90)
    ap.add_argument(
        "--run-id", default="", help="unique suffix for the temp repo (default: time+uuid)"
    )
    ap.add_argument(
        "--allowed-cuda",
        default="",
        help="comma-separated host CUDA versions to allow (e.g. 13.0 for Blackwell); empty = any",
    )
    args = ap.parse_args()
    gpu_type_ids = tuple(args.gpu_type_id) or GPU_WALK_BY_SM.get(args.sm)
    if not gpu_type_ids:
        ap.error(f"no default gpu walk for {args.sm!r}; pass --gpu-type-id")
    allowed_cuda = [v.strip() for v in args.allowed_cuda.split(",") if v.strip()] or None

    token = os.environ["HF_TOKEN"]
    import runpod
    from huggingface_hub import HfApi, snapshot_download

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    api = HfApi(token=token)

    # time + uuid so two concurrent bakes of the SAME sm never share a dataset (a second-granularity
    # timestamp alone collides under the matrix / re-runs -> corrupt/shared cache).
    suffix = args.run_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    repo = f"{ARTIFACT_NAMESPACE}/kernel-bake-{args.sm}-{suffix}"
    entry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bake_pod_entry.py")

    _upload_flash_code(api, repo, token)

    # create -> poll -> download inside try/finally so the GPU pod + temp dataset are ALWAYS released,
    # even if create_pod / polling / the download raises midway.
    pod_id = None
    outcome = "error"
    rc = 1
    try:
        pod, gpu_type_id = _create_pod_with_gpu_walk(
            runpod,
            gpu_type_ids=gpu_type_ids,
            name=f"kernel-bake-{args.sm}-{suffix}",
            image_name=args.image,
            # token-bearing pod (carries HF_TOKEN + private code/cache) -> Secure Cloud only, never a
            # community/peer-provider host.
            cloud_type="SECURE",
            container_disk_in_gb=args.container_disk_gb,
            docker_args=_docker_args(entry_path),
            # Blackwell (sm120) PTX needs CUDA-13 drivers to JIT; the matrix pins it (empty = any host).
            allowed_cuda_versions=allowed_cuda,
            env={
                "BAKE_HF_REPO": repo,
                "BAKE_ARCH": args.arch,
                "HF_TOKEN": token,
            },
        )
        pod_id = pod["id"]
        log(f"created pod {pod_id} ({gpu_type_id}, {args.sm}); polling for out/STATUS")

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
                # the pod exits right after uploading out/STATUS, so "exited" is the NORMAL success
                # tail, not a failure. re-check STATUS authoritatively (retries ride out a transient
                # HF list error) before concluding the bake actually died.
                if _status_present(api, repo, retries=3):
                    outcome = "done"
                    log("out/STATUS present (pod exited after success) -> warmup complete")
                else:
                    outcome = "pod_died"
                    log("pod terminated/exited before STATUS")
                break
            time.sleep(45)

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
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
                shutil.move(s, d)
            shutil.rmtree(tmp, ignore_errors=True)
            rc = _verify(args.out, args.sm)
        else:
            # Non-success (timeout / pod_died): the pod-side out/ (STARTED marker, warmup.log) lives in
            # the temp dataset the `finally` below deletes, so pull + print it FIRST. out/STARTED present
            # = the warmup entrypoint ran (a timeout then means it hung / was slow); STARTED absent is
            # AMBIGUOUS -- either the pod ran the wrong entrypoint, or the best-effort STARTED upload
            # failed while the warmup is still running (the pod log disambiguates). warmup.log is only
            # uploaded AFTER the warmup process returns, so it's present only if the warmup
            # finished/was killed, not on a pure mid-run hang.
            log(f"outcome={outcome}: pulling pod-side out/ for diagnostics before cleanup")
            try:
                dbg = os.path.join(args.out, ".dbg")
                snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    allow_patterns=["out/**"],
                    local_dir=dbg,
                    token=token,
                )
                src = os.path.join(dbg, "out")
                if os.path.isdir(src) and os.listdir(src):
                    started = os.path.isfile(os.path.join(src, "STARTED"))
                    log(f"warmup entrypoint ran (out/STARTED present): {started}")
                    for root, _, files in os.walk(src):
                        for f in sorted(files):
                            p = os.path.join(root, f)
                            log(f"   out/{os.path.relpath(p, src)} ({os.path.getsize(p)} b)")
                    wl = os.path.join(src, "warmup.log")
                    if os.path.isfile(wl):
                        log("--- warmup.log tail (last 60 lines) ---")
                        with open(wl, errors="replace") as wlf:
                            for line in wlf.read().splitlines()[-60:]:
                                log(f"   | {line}")
                    else:
                        log(
                            "no warmup.log (warmup never returned -> mid-run hang or still running at deadline)"
                        )
                else:
                    # Ambiguous: out/STARTED is uploaded best-effort BEFORE warmup, and the cache tree
                    # only lands AFTER warmup returns. So an empty out/ means EITHER the entrypoint
                    # never ran (wrong CMD / docker_args) OR the STARTED upload failed while warmup is
                    # still mid-run. The pod retries that upload 3x, so a dropped marker is unlikely and
                    # wrong-entrypoint is the probable cause -- but don't assert it. The only proof is
                    # the pod's OWN console (its "[bake] uploaded out/STARTED" / WARNING lines), which
                    # this helper can't fetch (pod terminated below) -> read it in the RunPod dashboard.
                    log(
                        "pod produced NO out/ -> warmup entrypoint probably never ran "
                        "(wrong CMD / docker_args); the only other cause is a failed out/STARTED "
                        "upload mid-run (unlikely, it retries 3x) -- confirm via the pod console "
                        "in the RunPod dashboard (this CI log does not capture pod stdout)"
                    )
                shutil.rmtree(dbg, ignore_errors=True)
            except Exception as e:
                log(f"diagnostic fetch failed (ignore): {str(e)[:160]}")
    finally:
        if pod_id:
            try:
                runpod.terminate_pod(pod_id)
                log(f"terminated pod {pod_id}")
            except Exception as e:
                log(f"terminate fail: {str(e)[:120]}")
        try:
            api.delete_repo(repo_id=repo, repo_type="dataset")
            log(f"deleted temp dataset {repo}")
        except Exception as e:
            log(f"temp dataset delete fail (ignore): {str(e)[:120]}")

    log(f"DONE outcome={outcome} rc={rc}")
    return rc


def _status_present(api, repo: str, retries: int = 1) -> bool:
    """Authoritative out/STATUS check, retried to ride out a transient HF list error."""
    for i in range(retries):
        try:
            if "out/STATUS" in api.list_repo_files(repo, repo_type="dataset"):
                return True
        except Exception as e:
            log(f"hf list err (status recheck {i + 1}/{retries}): {str(e)[:100]}")
        if i < retries - 1:
            time.sleep(5)
    return False


def _verify(out: str, sm: str) -> int:
    """Confirm the artifact actually landed and its metadata matches the requested arch."""
    blob = os.path.join(out, "mega_cache.bin")
    meta = os.path.join(out, "mega_cache.json")
    if not os.path.isfile(blob):
        log(f"FAIL: no mega_cache.bin in {out}; what the warmup actually produced:")
        for root, _, files in os.walk(out):
            for f in sorted(files):
                p = os.path.join(root, f)
                log(f"   present: {os.path.relpath(p, out)} ({os.path.getsize(p)} b)")
        wl = os.path.join(out, "warmup.log")
        if os.path.isfile(wl):
            log("   --- warmup.log tail ---")
            with open(wl, errors="replace") as wlf:
                tail = wlf.read().splitlines()[-40:]
            for line in tail:
                log(f"   | {line}")
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
