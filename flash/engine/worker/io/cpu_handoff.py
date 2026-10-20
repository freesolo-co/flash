"""Final-checkpoint handoff through a private, per-attempt Modal Volume."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

MOUNT = "/checkpoint-handoff"


def handoff_root() -> Path | None:
    return Path(MOUNT) if os.environ.get("FLASH_MODAL_CPU_UPLOADS") == "1" else None


def stage_final_checkpoint(step: int, directory: str) -> bool:
    """Copy only an explicit final save; intermediate saves keep normal durability."""
    from flash.engine.worker.runtime import state

    root = handoff_root()
    if root is None or state.JOB_SPEC is None or step != state.JOB_SPEC.train.max_steps:
        return False
    source = Path(directory)
    if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("checkpoint handoff does not accept symlinks")
    descriptor = {"version": 1, "run_id": state.RUN_ID, "attempt": state.ATTEMPT, "step": step}
    # A complete checkpoint is published by rename; interrupted copies cannot be consumed.
    with tempfile.TemporaryDirectory(prefix="staging-", dir=root) as temporary:
        staged = Path(temporary) / "checkpoint"
        shutil.copytree(source, staged)
        shutil.rmtree(root / "checkpoint", ignore_errors=True)
        staged.rename(root / "checkpoint")
    code = Path(__file__).resolve().parents[3]
    if not (root / "source" / "flash").exists():
        shutil.copytree(
            code, root / "source" / "flash", ignore=shutil.ignore_patterns("__pycache__")
        )
    target = root / "handoff.json"
    temporary = root / "handoff.json.tmp"
    temporary.write_text(json.dumps(descriptor))
    temporary.replace(target)
    return True


def defer_completion(done_path: str) -> bool:
    """Keep DONE private until the CPU has committed the native checkpoint."""
    root = handoff_root()
    if root is None or not (root / "handoff.json").is_file():
        return False
    shutil.copyfile(done_path, root / "DONE")
    shutil.copyfile("/tmp/metrics.json", root / "metrics.json")
    return True


def main() -> None:
    from flash.engine.worker.io import hf
    from flash.engine.worker.runtime import state

    root = Path(MOUNT)
    descriptor = root / "handoff.json"
    if descriptor.is_symlink() or descriptor.stat().st_size > 4096:
        raise ValueError("invalid checkpoint handoff")
    expected = {
        "version": 1,
        "run_id": state.RUN_ID,
        "attempt": state.ATTEMPT,
        "step": state.JOB_SPEC.train.max_steps,
    }
    if json.loads(descriptor.read_text()) != expected:
        raise ValueError("checkpoint handoff belongs to a different attempt")
    checkpoint = root / "checkpoint"
    if checkpoint.is_symlink() or any(p.is_symlink() for p in checkpoint.rglob("*")):
        raise ValueError("invalid checkpoint handoff tree")
    if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        raise ValueError("empty checkpoint handoff")
    os.environ.pop("FLASH_MODAL_CPU_UPLOADS", None)
    started = time.time()
    if not hf.upload_resume_checkpoint(expected["step"], str(checkpoint), emit_heartbeat=False):
        raise RuntimeError("CPU checkpoint publication failed")
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({**expected, "upload_seconds": time.time() - started}))
    hf.hf_upload_file(str(receipt), f"cpu_upload_attempt{state.ATTEMPT}.json", required=True)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.setdefault("notes", {})["cpu_upload_seconds"] = time.time() - started
    metrics_path.write_text(json.dumps(metrics))
    hf.hf_upload_file(str(metrics_path), "metrics.json", required=True)
    hf.hf_upload_file(str(root / "DONE"), "DONE", required=True)
    # The original, source-attested terminal marker becomes public only after every artifact.
    hf.hf_upload_file(
        str(root / "terminal.json"), f"modal_attempt{state.ATTEMPT}.json", required=True
    )


if __name__ == "__main__":
    main()
