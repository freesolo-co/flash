"""Bake per-architecture worker kernel caches on Modal and verify the returned artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
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


def _upload_flash_code(api, repo: str, source_root) -> None:
    """Upload the base-pinned warmup package without importing it into the transport process."""
    from pathlib import Path

    pkg_dir = Path(source_root).resolve() / "flash"
    if not (pkg_dir / "engine/worker/runtime/kernel_warmup.py").is_file():
        raise ValueError("bake source must contain the base image's Flash warmup package")
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(pkg_dir),
        path_in_repo="code/flash",
        repo_id=repo,
        repo_type="dataset",
        ignore_patterns=["__pycache__/*", "*.pyc"],
        delete_patterns=["**"],
    )
    log(f"uploaded flash code -> {repo}:code/flash (from {pkg_dir})")


def _bake_image(modal, image_ref, entry):
    """Seed the same standalone Azure client used by managed worker bootstrap."""
    from pathlib import Path

    import flash.artifacts

    image = (
        modal.Image.from_registry(image_ref)
        .pip_install("azure-storage-blob==12.31.1", "azure-identity==1.27.0")
        .add_local_file(str(entry), "/opt/flash/bake.py", copy=True)
    )
    package = Path(flash.artifacts.__file__).parent
    for module in ("api", "config", "manifest", "storage"):
        image = image.add_local_file(
            str(package / f"{module}.py"), f"/opt/flash/artifact_{module}.py", copy=True
        )
    return image


def _delete_artifacts(repo):
    from flash.artifacts.config import open_container
    from flash.artifacts.manifest import repo_prefix

    with open_container() as container:
        for blob in container.list_blobs(name_starts_with=repo_prefix(repo) + "/"):
            container.delete_blob(blob.name)


def main() -> int:
    from pathlib import Path

    from flash.artifacts.api import ArtifactStore, download_snapshot
    from flash.artifacts.config import runtime_environment
    from flash.providers.modal import api as modal_api

    parser = argparse.ArgumentParser(description="Bake worker kernels on Modal")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--sm", required=True, choices=tuple(GPU_BY_SM))
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="build/kernel_cache")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Checkout matching the base image; transport runs from this helper's checkout",
    )
    parser.add_argument("--deadline-min", type=int, default=90)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    if not 1 <= args.deadline_min <= 1440:
        parser.error("deadline must be between 1 and 1440 minutes")
    artifacts = ArtifactStore()
    worker_secrets = runtime_environment()
    if not worker_secrets:
        raise RuntimeError("Azure build artifact storage is not configured")
    suffix = args.run_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    repo = f"{os.environ.get('FLASH_ARTIFACT_NAMESPACE', ARTIFACT_NAMESPACE)}/kernel-bake-{args.sm}-{suffix}"
    entry = Path(__file__).with_name("bake_sandbox_entry.py")

    async def bake(modal, client):
        app = await modal.App.lookup.aio(
            "flash-kernel-bake",
            environment_name=modal_api.environment(),
            create_if_missing=True,
            client=client,
        )
        image = _bake_image(modal, args.image, entry)
        sandbox = await modal.Sandbox.create.aio(
            "python",
            "-u",
            "/opt/flash/bake.py",
            app=app,
            image=image,
            name=f"bake-{args.sm}-{suffix}",
            gpu=GPU_BY_SM[args.sm],
            timeout=args.deadline_min * 60,
            cpu=(4, 8),
            # Keep the reservation at 32 GiB, but allow compiler memory to burst. The
            # H100 warmup was evicted for thrashing at the previous 64 GiB limit.
            memory=(32768, 131072),
            secrets=[
                modal.Secret.from_dict(
                    {"BAKE_ARTIFACT_REPO": repo, "BAKE_ARCH": args.arch, **worker_secrets}
                )
            ],
            client=client,
        )
        try:
            await sandbox.wait.aio()
            returncode = await sandbox.poll.aio()
            if returncode != 0:
                # The provider wrapper sanitizes ordinary exceptions. Preserve this
                # credential-free diagnostic so CI names the failed sandbox and exit.
                raise modal_api.ModalApiError(
                    f"Kernel warmup exited with code {returncode}; "
                    f"inspect Modal sandbox {sandbox.object_id}"
                )
        finally:
            await sandbox.terminate.aio(wait=True)

    try:
        _upload_flash_code(artifacts, repo, args.source_root)
        modal_api.call(bake, timeout=args.deadline_min * 60 + 120)
        if not _status_present(artifacts, repo, retries=3):
            raise RuntimeError("kernel warmup finished without its completion marker")
        import tempfile

        with tempfile.TemporaryDirectory(prefix="flash-kernel-") as temp:
            download_snapshot(
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=["out/**"],
                local_dir=temp,
            )
            staging = Path(temp) / "out"
            status = (staging / "STATUS").read_text()
            if status != f"rc=0\narch={args.arch}\n":
                raise RuntimeError("kernel warmup returned an invalid completion marker")
            archive = staging / "kernel_cache.tar.gz"
            current_log = (staging / "bake-warmup.log").read_bytes()
            _extract_cache_archive(archive, staging)
            archive.unlink()
            # A cache may contain an older log. Keep this run's subprocess log authoritative.
            (staging / "bake-warmup.log").write_bytes(current_log)
            (staging / "warmup.log").write_bytes(current_log)
            result = _verify(str(staging), args.sm)
            if result == 0:
                shutil.copytree(staging, args.out, dirs_exist_ok=True)
            return result
    finally:
        try:
            _delete_artifacts(repo)
        except Exception:
            log(
                f"WARNING: could not delete temporary Azure artifacts for {repo}; remove it manually"
            )


def _extract_cache_archive(archive_path, destination) -> None:
    """Restore cache layout without allowing archive links or paths outside the output tree."""
    from pathlib import Path

    root = Path(destination).resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        total = 0
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise ValueError("invalid kernel cache archive member")
            total += member.size
            if total > 32 * 1024**3:
                raise ValueError("kernel cache archive exceeds 32 GiB")
        archive.extractall(root, members=members, filter="data")


def _status_present(api, repo: str, retries: int = 1) -> bool:
    """Authoritative out/STATUS check, retried to ride out a transient Azure read error."""
    for i in range(retries):
        try:
            if "out/STATUS" in api.list_repo_files(repo, repo_type="dataset"):
                return True
        except Exception as e:
            log(f"artifact status recheck {i + 1}/{retries}: {type(e).__name__}")
        if i < retries - 1:
            time.sleep(5)
    return False


def _verify(out: str, sm: str) -> int:
    """Confirm the artifact actually landed and its metadata matches the requested arch."""
    from flash.engine.worker.runtime.kernel_warmup import CACHE_META_FILENAME, has_raw_kernel_cache

    meta = os.path.join(out, CACHE_META_FILENAME)
    if not has_raw_kernel_cache(out):
        log(f"FAIL: no compiled kernel artifacts in {out}; what the warmup actually produced:")
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
    try:
        with open(os.path.join(out, "cache_validation.json")) as file:
            validation = json.load(file)
        with open(os.path.join(out, "cache_reload.json")) as file:
            reload = json.load(file)
        if (
            validation.get("first_exit_code") != 0
            or validation.get("reload_exit_code") != 0
            or validation.get("exported_cache_replayed") is not True
            or reload.get("cached_groups_compilation_forbidden") is not True
            or reload.get("cached_groups") != ["flash_attn", "fla_gdn"]
            or reload.get("startup_jit_groups") != (["flashinfer_gdn"] if sm == "sm90" else [])
            or reload.get("compile_attempts") != 0
            or reload.get("passed") is not True
        ):
            raise ValueError("fresh-process cache reuse was not proven")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        log(f"FAIL: cache reload validation failed ({error})")
        return 1
    log(f"OK: compiled kernel artifacts for {sm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
