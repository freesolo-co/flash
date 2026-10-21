"""Final-checkpoint handoff through a private, per-attempt Modal Volume."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

MOUNT = "/checkpoint-handoff"


def handoff_root() -> Path | None:
    root = Path(MOUNT)
    return root if root.is_dir() else None


def stage_final_checkpoint(step: int, directory: str, *, final_step: int | None) -> bool:
    """Copy the trainer's resolved final save; intermediate saves upload normally."""
    from flash.engine.worker.runtime import state

    root = handoff_root()
    if root is None or state.JOB_SPEC is None or final_step is None or step != final_step:
        return False
    if type(final_step) is not int or final_step < 1:
        raise ValueError("invalid final checkpoint step")
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
    (root / "spec.json").write_text(state.JOB_SPEC.to_json())
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
    spec_path = Path(MOUNT) / "spec.json"
    if spec_path.is_symlink() or hashlib.sha256(
        spec_path.read_bytes()
    ).hexdigest() != os.environ.get("FLASH_EXPECTED_SPEC_SHA256"):
        raise ValueError("CPU checkpoint job specification mismatch")
    from flash.engine.worker.io import hf
    from flash.engine.worker.runtime import state

    root = Path(MOUNT)
    descriptor = root / "handoff.json"
    if descriptor.is_symlink() or descriptor.stat().st_size > 4096:
        raise ValueError("invalid checkpoint handoff")
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    final_step = metrics.get("step")
    if type(final_step) is not int or final_step < 1:
        raise ValueError("missing final training step in checkpoint handoff")
    expected = {
        "version": 1,
        "run_id": state.RUN_ID,
        "attempt": state.ATTEMPT,
        "step": final_step,
    }
    if json.loads(descriptor.read_text()) != expected:
        raise ValueError("checkpoint handoff belongs to a different attempt")
    checkpoint = root / "checkpoint"
    if checkpoint.is_symlink() or any(p.is_symlink() for p in checkpoint.rglob("*")):
        raise ValueError("invalid checkpoint handoff tree")
    if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        raise ValueError("empty checkpoint handoff")
    started = time.time()
    if not hf.upload_resume_checkpoint(expected["step"], str(checkpoint), emit_heartbeat=False):
        raise RuntimeError("CPU checkpoint publication failed")
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({**expected, "upload_seconds": time.time() - started}))
    hf.hf_upload_file(str(receipt), f"cpu_upload_attempt{state.ATTEMPT}.json", required=True)
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
