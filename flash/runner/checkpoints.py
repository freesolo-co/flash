"""List a run's deployable per-step RL checkpoints from its HF artifact repo."""

from __future__ import annotations

import os
import re

from flash.runner import adapter_prefix
from flash.spec import JobSpec

_ADAPTER_WEIGHT_FILES = frozenset({"adapter_model.safetensors", "adapter_model.bin"})


def checkpoint_adapter_prefix(spec: JobSpec, step: int, seed: int | None = None) -> str:
    """Return the per-step adapter prefix; deploy_adapter appends /adapter to this."""
    return f"{adapter_prefix(spec, seed)}/checkpoints/step-{step}"


def _adapter_file_re(base: str) -> re.Pattern[str]:
    """Matches ``<base>/checkpoints/step-<N>/adapter/<filename>`` and captures (step, filename)."""
    return re.compile(re.escape(base) + r"/checkpoints/step-(\d+)/adapter/([^/]+)$")


def list_checkpoints(spec: JobSpec, seed: int | None = None) -> list[dict]:
    """Deployable per-step adapter snapshots for ``spec``, ascending by step."""
    repo = spec.train.hf_repo
    if not repo:
        return []
    base = adapter_prefix(spec, seed)
    pattern = _adapter_file_re(base)
    try:
        from huggingface_hub import HfApi

        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            repo, repo_type="dataset"
        )
    except Exception as exc:  # listing is best-effort; never raise into a run/route
        print(f"[ckpt] list warn for {spec.run_id}: {exc}")
        return []
    by_step: dict[int, set[str]] = {}
    for path in files:
        match = pattern.search(path)
        if match:
            by_step.setdefault(int(match.group(1)), set()).add(match.group(2))
    out: list[dict] = []
    for step in sorted(by_step):
        names = by_step[step]
        if "adapter_config.json" not in names or names.isdisjoint(_ADAPTER_WEIGHT_FILES):
            continue
        prefix = checkpoint_adapter_prefix(spec, step, seed)
        out.append(
            {
                "step": step,
                "adapter_prefix": prefix,
                "subfolder": f"{prefix}/adapter",
                "repo_id": repo,
                "repo_type": "dataset",
            }
        )
    return out
