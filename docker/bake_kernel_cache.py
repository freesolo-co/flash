"""Bake per-architecture worker kernel caches on Modal and verify the returned artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid

ARTIFACT_NAMESPACE = "Freesolo-Co"

# One Modal GPU per compiler architecture; the worker verifies the requested SM.
GPU_BY_SM = {
    "sm80": "A100-80GB",
    "sm86": "A10",
    "sm89": "L40S",
    "sm90": "H100!",
    "sm100": "B200",
    "sm120": "RTX-PRO-6000",
}


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


def main() -> int:
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    from flash.providers.modal import api as modal_api

    parser = argparse.ArgumentParser(description="Bake worker kernels on Modal")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--sm", required=True, choices=tuple(GPU_BY_SM))
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="build/kernel_cache")
    parser.add_argument("--deadline-min", type=int, default=90)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    if not 1 <= args.deadline_min <= 1440:
        parser.error("deadline must be between 1 and 1440 minutes")
    token = os.environ["HF_TOKEN"]
    hf = HfApi(token=token)
    suffix = args.run_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    repo = (
        f"{os.environ.get('FLASH_HF_NAMESPACE', ARTIFACT_NAMESPACE)}/kernel-bake-{args.sm}-{suffix}"
    )
    entry = Path(__file__).with_name("bake_sandbox_entry.py")
    _upload_flash_code(hf, repo, token)

    async def bake(modal, client):
        app = await modal.App.lookup.aio(
            "flash-kernel-bake",
            environment_name=modal_api.environment(),
            create_if_missing=True,
            client=client,
        )
        image = modal.Image.from_registry(args.image).add_local_file(
            str(entry), "/opt/flash/bake.py", copy=True
        )
        sandbox = await modal.Sandbox.create.aio(
            "python",
            "-u",
            "/opt/flash/bake.py",
            app=app,
            image=image,
            name=f"bake-{args.sm}-{suffix}",
            gpu=GPU_BY_SM[args.sm],
            timeout=args.deadline_min * 60,
            cpu=4,
            memory=32768,
            secrets=[
                modal.Secret.from_dict(
                    {"BAKE_HF_REPO": repo, "BAKE_ARCH": args.arch, "HF_TOKEN": token}
                )
            ],
            client=client,
        )
        try:
            await sandbox.wait.aio()
            if await sandbox.poll.aio() != 0:
                raise RuntimeError("Modal kernel warmup failed; inspect the bake sandbox logs")
        finally:
            await sandbox.terminate.aio(wait=True)

    try:
        modal_api.call(bake, timeout=args.deadline_min * 60 + 120)
        if not _status_present(hf, repo, retries=3):
            raise RuntimeError("kernel warmup finished without its completion marker")
        import tempfile

        with tempfile.TemporaryDirectory(prefix="flash-kernel-") as temp:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=["out/**"],
                local_dir=temp,
                token=token,
            )
            shutil.copytree(Path(temp) / "out", args.out, dirs_exist_ok=True)
        return _verify(args.out, args.sm)
    finally:
        hf.delete_repo(repo_id=repo, repo_type="dataset")


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
