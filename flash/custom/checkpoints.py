"""Durable custom state in the same run artifact tree as SFT/GRPO adapters."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


class RunCheckpoints:
    def __init__(self, context):
        self.context = context
        self.published = {}
        self.by_name = {}
        self.cached = None

    def publish(self, directory, metadata):
        from flash.engine.worker.io import hf

        step = metadata["step"]
        # adapter and full optimizer state commit together. a listed adapter therefore always
        # has its matching resume state, unlike separate best-effort uploads.
        metadata = {**metadata, "format": "flash-custom-v1"}
        (directory / "metadata.json").write_text(json.dumps(metadata))
        receipt = hf.publish_deployable_checkpoint(
            str(directory / "adapter"),
            step,
            checkpoint_dir=str(directory),
            required=True,
            context=self.context,
        )
        if receipt.resume is None or receipt.deployable is None:
            raise RuntimeError("checkpoint publication did not commit training state and adapter")
        reference = f"{self.context.config.run_id}/step-{step}"
        self.published[step] = directory
        self.by_name[directory.name] = step
        previous = self.cached
        self.cached = directory
        if previous is not None and previous != directory:
            shutil.rmtree(previous)
        return {"reference": reference, "durable": True}

    def restore(self, backend, resolved):
        from huggingface_hub import snapshot_download

        from flash.engine.support.checkpoint_transport import restore_directory

        with tempfile.TemporaryDirectory(prefix="flash-custom-resume-") as workspace:
            root = snapshot_download(
                local_dir=workspace,
                repo_id=resolved["repo"],
                repo_type="dataset",
                revision=resolved["revision"],
                allow_patterns=[f"{resolved['path']}/**"],
                token=os.environ.get("HF_TOKEN"),
            )
            source = Path(root) / resolved["path"]
            restore_directory(str(source))
            metadata = json.loads((source / "metadata.json").read_text())
            if metadata.get("format") != "flash-custom-v1":
                raise ValueError("checkpoint does not contain custom training state")
            return backend.restore_checkpoint(source)

    def finish(self, directory):
        from flash.engine.worker.io import hf

        # final adapters use the existing run/final deploy path as well as step-N listing.
        hf.hf_upload_folder(
            str(directory / "adapter"), "adapter", required=True, context=self.context
        )


def validate_algorithm_state(value):
    if not isinstance(value, dict):
        raise ValueError("algorithm_state must be a JSON object")
    raw = json.dumps(value, allow_nan=False)
    if len(raw.encode()) > 1024 * 1024:
        raise ValueError("algorithm_state exceeds 1 MiB")
    return json.loads(raw)
