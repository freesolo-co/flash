"""Bake per-architecture worker kernel caches on Modal and verify the returned artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
import uuid

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


async def _collect_bake(sandbox, out, arch: str, deadline: int) -> None:
    """Run warmup while the sandbox stays alive, then stream its outputs before teardown."""
    from flash.providers.modal import api as modal_api

    process = await sandbox.exec.aio("python", "-u", "/opt/flash/bake.py", timeout=deadline)
    returncode = await process.wait.aio()
    if returncode != 0:
        raise modal_api.ModalApiError(
            f"Kernel warmup exited with code {returncode}; "
            f"inspect Modal sandbox {sandbox.object_id}"
        )
    status = await sandbox.filesystem.read_text.aio("/out/STATUS")
    if status != f"rc=0\narch={arch}\n":
        raise modal_api.ModalApiError("kernel warmup finished without a valid completion marker")
    await sandbox.filesystem.copy_to_local.aio(
        "/out/kernel_cache.tar.gz", out / "kernel_cache.tar.gz"
    )
    await sandbox.filesystem.copy_to_local.aio("/out/warmup.log", out / "bake-warmup.log")


def main() -> int:
    from pathlib import Path

    import flash
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
    suffix = args.run_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    entry = Path(__file__).with_name("bake_sandbox_entry.py")

    async def bake(modal, client):
        app = await modal.App.lookup.aio(
            "flash-kernel-bake",
            environment_name=modal_api.environment(),
            create_if_missing=True,
            client=client,
        )
        image = (
            modal.Image.from_registry(args.image)
            .add_local_dir(
                Path(flash.__file__).parent,
                "/runcode/code/flash",
                copy=True,
                ignore=["**/__pycache__/**", "**/*.pyc"],
            )
            .add_local_file(str(entry), "/opt/flash/bake.py", copy=True)
        )
        sandbox = await modal.Sandbox.create.aio(
            "sleep",
            "infinity",
            app=app,
            image=image,
            name=f"bake-{args.sm}-{suffix}",
            gpu=GPU_BY_SM[args.sm],
            timeout=min(86400, args.deadline_min * 60 + 600),
            cpu=(4, 8),
            # Keep the reservation at 32 GiB, but allow compiler memory to burst. The
            # H100 warmup was evicted for thrashing at the previous 64 GiB limit.
            memory=(32768, 131072),
            secrets=[modal.Secret.from_dict({"BAKE_ARCH": args.arch, "HF_TOKEN": token})],
            client=client,
        )
        try:
            await _collect_bake(sandbox, staging, args.arch, args.deadline_min * 60)
        finally:
            await sandbox.terminate.aio(wait=True)

    # Never let leftovers from a previous bake satisfy this run's verification.
    with tempfile.TemporaryDirectory(prefix="flash-kernel-") as temp:
        staging = Path(temp)
        modal_api.call(bake, timeout=args.deadline_min * 60 + 600)
        archive = staging / "kernel_cache.tar.gz"
        # Preserve the current log even if an older log exists inside the exported cache.
        current_log = (staging / "bake-warmup.log").read_bytes()
        _extract_cache_archive(archive, staging)
        archive.unlink()
        (staging / "bake-warmup.log").write_bytes(current_log)
        (staging / "warmup.log").write_bytes(current_log)
        result = _verify(str(staging), args.sm)
        if result == 0:
            shutil.copytree(staging, args.out, dirs_exist_ok=True)
        return result


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
