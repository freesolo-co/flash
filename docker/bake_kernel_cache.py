"""Bake one per-SM kernel cache through an exact short-lived RunPod Secure Pod."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time

ARTIFACT_NAMESPACE = "Freesolo-Co"
GPU_WALK_BY_SM: dict[str, tuple[str, ...]] = {
    "sm80": ("NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"),
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
CREATE_ROUNDS = 5
CREATE_BACKOFF_S = (30, 60, 120, 240)
_DEAD_STATES = frozenset({"DEAD", "EXITED", "FAILED", "STOPPED", "TERMINATED"})


def log(message: str) -> None:
    print(f"[bake-ci] {message}", flush=True)


def _upload_flash_code(api, repo: str) -> None:
    import flash

    package = os.path.realpath(os.path.dirname(os.path.abspath(flash.__file__)))
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    api.update_repo_settings(repo_id=repo, repo_type="dataset", private=True)
    api.upload_folder(
        folder_path=package,
        path_in_repo="code/flash",
        repo_id=repo,
        repo_type="dataset",
        ignore_patterns=["__pycache__/*", "*.pyc"],
        delete_patterns=["code/**", "out/**"],
    )


def _bake_spec(run_id: str, disk_gb: int, deadline_s: int):
    from flash.core.spec import JobSpec

    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "run_id": run_id,
            "train": {
                "hf_repo": f"{ARTIFACT_NAMESPACE}/kernel-bake-placeholder",
                "credit_assignment": "per_episode",
            },
            "gpu": {
                "type": "RTX 4090",
                "disk_gb": disk_gb,
                "max_wall_seconds": deadline_s,
            },
        }
    )


def _launch_payload(repo: str, arch: str, token: str) -> str:
    return json.dumps(
        {"mode": "kernel_bake", "hf_repo": repo, "arch": arch, "hf_token": token},
        sort_keys=True,
        separators=(",", ":"),
    )


def _allowed_cuda_override(value: str) -> tuple[str, ...] | None:
    return tuple(item.strip() for item in value.split(",") if item.strip()) or None


def _bake_repo(sm: str, workflow_id: str) -> str:
    digest = hashlib.sha256(workflow_id.encode("utf-8")).hexdigest()[:16]
    return f"{ARTIFACT_NAMESPACE}/kernel-bake-{sm}-{digest}"


def _cleanup_claimed_bake_intent(store, record: dict, deadline_s: int) -> None:
    from flash.providers.runpod.pod_identity import RunpodCreateAbsent, RunpodPodHandle
    from flash.providers.runpod.pods import resolve_pending_handle, terminate_handle

    handle = RunpodPodHandle.from_dict(record["handle"])
    spec = _bake_spec(record["run_id"], handle.container_disk_gb, deadline_s)
    try:
        resolved = resolve_pending_handle(
            handle,
            spec,
            record["seed"],
            deadline_at=time.time() + 120.0,
        )
    except RunpodCreateAbsent:
        resolved = handle
    else:
        if resolved.to_dict() != handle.to_dict():
            store.publish_active(record["run_id"], record["seed"], resolved.to_dict())
    store.renew()
    terminate_handle(resolved, deadline_at=time.time() + 120.0)
    store.clear()


def _recover_bake_intent(store, deadline_s: int) -> None:
    record = store.claim_expired()
    if record is not None:
        _cleanup_claimed_bake_intent(store, record, deadline_s)


def _cleanup_owned_bake_intent(store, deadline_s: int) -> None:
    if store.load() is None:
        return
    record = store.renew()
    _cleanup_claimed_bake_intent(store, record, deadline_s)


def _launch_with_gpu_walk(
    spec,
    payload: str,
    *,
    image: str,
    gpu_type_ids: tuple[str, ...],
    allowed_cuda: tuple[str, ...] | None,
    deadline_at: float,
    on_handle=None,
    cleanup_guard=None,
    after_failed_attempt=None,
    rounds: int = CREATE_ROUNDS,
    backoff_s: tuple[int, ...] = CREATE_BACKOFF_S,
):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import auth as runpod_auth
    from flash.providers.runpod.pods import launch_payload_pod

    last = None
    for round_index in range(rounds):
        for key in runpod_auth.ordered_keys():
            fingerprint = runpod_api.key_fingerprint(key)
            for gpu_type_id in gpu_type_ids:
                try:
                    log(f"gpu walk round {round_index + 1}/{rounds}: trying {gpu_type_id!r}")
                    handle = launch_payload_pod(
                        spec,
                        spec.seed,
                        serialized_payload=payload,
                        fingerprint=fingerprint,
                        data_center_id=None,
                        network_volume_id=None,
                        deadline_at=deadline_at,
                        image_name=image,
                        gpu_type_id_override=gpu_type_id,
                        allowed_cuda_versions=allowed_cuda,
                        on_handle=on_handle,
                        cleanup_guard=cleanup_guard,
                    )
                    return handle, gpu_type_id
                except runpod_api.RunpodCapacityError as exc:
                    last = exc
                    if after_failed_attempt is not None:
                        after_failed_attempt()
        if round_index + 1 == rounds:
            break
        delay = backoff_s[min(round_index, len(backoff_s) - 1)]
        delay += random.uniform(0, 0.25 * delay)
        log(f"all gpu types full; retrying walk in {delay:.0f}s")
        time.sleep(delay)
    raise runpod_api.RunpodCapacityError(
        f"no capacity across gpu walk {gpu_type_ids!r} after {rounds} rounds"
    ) from last


def _status_present(api, repo: str, retries: int = 1) -> bool:
    for index in range(retries):
        try:
            if "out/STATUS" in api.list_repo_files(repo, repo_type="dataset"):
                return True
        except Exception as exc:
            log(f"hf status check {index + 1}/{retries} failed: {str(exc)[:100]}")
        if index + 1 < retries:
            time.sleep(5)
    return False


def _poll_bake(api, repo: str, handle, deadline_at: float, *, renew_lease=None) -> str:
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod.hf_intent import INTENT_LEASE_S

    dead = 0
    next_renewal = time.time() + INTENT_LEASE_S / 3.0
    while time.time() < deadline_at:
        if renew_lease is not None and time.time() >= next_renewal:
            renew_lease()
            next_renewal = time.time() + INTENT_LEASE_S / 3.0
        if _status_present(api, repo):
            return "done"
        pod = runpod_api.get_pod_for_fingerprint(
            handle.pod_id, handle.key_fingerprint, deadline_at=deadline_at
        )
        dead = dead + 1 if pod is None or pod.desired_status in _DEAD_STATES else 0
        if dead >= 2:
            return "done" if _status_present(api, repo, retries=3) else "pod_died"
        time.sleep(min(45.0, INTENT_LEASE_S / 3.0))
    return "timeout"


def _verify(output: str, sm: str) -> int:
    blob = os.path.join(output, "mega_cache.bin")
    metadata = os.path.join(output, "mega_cache.json")
    if not os.path.isfile(blob):
        log(f"FAIL: no mega_cache.bin in {output}")
        return 1
    try:
        with open(metadata) as stream:
            actual = json.load(stream).get("sm")
    except Exception as exc:
        log(f"FAIL: metadata unreadable ({exc})")
        return 1
    if actual != sm:
        log(f"FAIL: baked arch {actual!r} != requested {sm!r}")
        return 1
    log(f"OK: mega_cache.bin ({os.path.getsize(blob)} bytes) for {sm}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="offload kernel-cache warmup to a RunPod Pod")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--sm", required=True)
    parser.add_argument("--gpu-type-id", action="append", default=[])
    parser.add_argument("--image", default="ghcr.io/freesolo-co/flash-worker:cu128")
    parser.add_argument("--out", default="build/kernel_cache")
    parser.add_argument("--container-disk-gb", type=int, default=60)
    parser.add_argument("--deadline-min", type=int, default=90)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--allowed-cuda", default="")
    args = parser.parse_args()
    gpu_types = tuple(args.gpu_type_id) or GPU_WALK_BY_SM.get(args.sm)
    if not gpu_types:
        parser.error(f"no default gpu walk for {args.sm!r}; pass --gpu-type-id")
    token = os.environ["HF_TOKEN"]
    from huggingface_hub import HfApi

    from flash.providers._lifecycle.poll import preload_instance_run_id
    from flash.providers.runpod.hf_intent import (
        HfRunpodIntentStore,
        intent_lock,
        intent_path,
        new_intent_owner,
    )

    api = HfApi(token=token)
    repo = _bake_repo(args.sm, args.workflow_id)
    identity = f"{args.workflow_id}:{args.sm}"
    owner = new_intent_owner("bake")
    deadline_at = time.time() + args.deadline_min * 60
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
    run_id = preload_instance_run_id("runpod", f"bake-{args.sm}", int(deadline_at), suffix)
    spec = _bake_spec(run_id, args.container_disk_gb, args.deadline_min * 60)
    payload = _launch_payload(repo, args.arch, token)
    allowed_cuda = _allowed_cuda_override(args.allowed_cuda)
    store = HfRunpodIntentStore(
        api,
        repo,
        intent_path("bake", identity),
        token,
        "bake",
        identity,
        owner,
    )
    outcome = "error"
    rc = 1
    cleaned = False
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    api.update_repo_settings(repo_id=repo, repo_type="dataset", private=True)
    with intent_lock(repo, store.path):
        try:
            _recover_bake_intent(store, args.deadline_min * 60)
            _upload_flash_code(api, repo)
            store.load()
            handle, selected = _launch_with_gpu_walk(
                spec,
                payload,
                image=args.image,
                gpu_type_ids=gpu_types,
                allowed_cuda=allowed_cuda,
                deadline_at=deadline_at,
                on_handle=lambda value: store.publish_active(spec.run_id, spec.seed, value),
                cleanup_guard=store.renew,
                after_failed_attempt=lambda: _cleanup_owned_bake_intent(
                    store, args.deadline_min * 60
                ),
            )
            log(f"created Pod {handle.pod_id} ({selected}, {args.sm})")
            outcome = _poll_bake(api, repo, handle, deadline_at, renew_lease=store.renew)
            if outcome == "done":
                _cleanup_owned_bake_intent(store, args.deadline_min * 60)
                from huggingface_hub import snapshot_download

                temporary = os.path.join(args.out, ".dl")
                snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    allow_patterns=["out/**"],
                    local_dir=temporary,
                    token=token,
                )
                source = os.path.join(temporary, "out")
                os.makedirs(args.out, exist_ok=True)
                for name in os.listdir(source):
                    if name == "STATUS":
                        continue
                    src, dst = os.path.join(source, name), os.path.join(args.out, name)
                    if os.path.isdir(dst):
                        shutil.rmtree(dst, ignore_errors=True)
                    shutil.move(src, dst)
                shutil.rmtree(temporary, ignore_errors=True)
                rc = _verify(args.out, args.sm)
        finally:
            _cleanup_owned_bake_intent(store, args.deadline_min * 60)
            cleaned = True
    if cleaned:
        try:
            api.delete_repo(repo_id=repo, repo_type="dataset")
        except Exception as exc:
            log(f"temp dataset delete failed: {str(exc)[:120]}")
    log(f"DONE outcome={outcome} rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
