"""List a run's deployable per-step RL checkpoints from its HF artifact repo.

The GPU worker publishes each trainer save's LoRA adapter to a stable, NON-pruned path
(``<adapter_prefix>/checkpoints/step-<N>/adapter``; see
``flash.engine.worker.publish_deployable_checkpoint``). This module is the control-plane
reader: it enumerates those snapshots so ``flash checkpoints`` can list them and
``flash deploy --step N`` can serve a specific one — including for a run that was cancelled or
failed mid-RL and so never sealed a final adapter. HF (not the backend DB) is the source of
truth for what's deployable; backend persistence is a mirror (see
``flash.server.checkpoints``)."""

from __future__ import annotations

import os
import re

from flash.runner import adapter_prefix
from flash.spec import JobSpec

# The PEFT weights file a step must carry (alongside adapter_config.json) to be servable.
_ADAPTER_WEIGHT_FILES = frozenset({"adapter_model.safetensors"})


def checkpoint_adapter_prefix(spec: JobSpec, step: int) -> str:
    """The ``adapter_prefix`` that serves checkpoint ``step``.

    ``deploy_adapter`` appends ``/adapter`` to whatever prefix it's given, so this returns the
    per-step root (``<run prefix>/checkpoints/step-<N>``) — matching the worker's upload path —
    and the existing deploy path needs no special-casing for checkpoints."""
    return f"{adapter_prefix(spec)}/checkpoints/step-{step}"


def _adapter_file_re(base: str) -> re.Pattern[str]:
    """Matches ``<base>/checkpoints/step-<N>/adapter/<filename>`` and captures (step, filename)."""
    return re.compile(re.escape(base) + r"/checkpoints/step-(\d+)/adapter/([^/]+)$")


def list_checkpoints(spec: JobSpec) -> list[dict]:
    """Deployable per-step adapter snapshots for ``spec``, ascending by step.

    A step is included only if its adapter folder carries BOTH ``adapter_config.json`` AND a
    weights file (so ``/deploy --step`` can never target a half-uploaded, unloadable step). Each
    entry: ``{"step", "adapter_prefix", "subfolder", "repo_id", "repo_type"}`` where
    ``adapter_prefix`` is the value to hand ``deploy_adapter`` to serve that exact step and
    ``subfolder`` is the full path of the adapter folder in the repo. Returns ``[]`` when the
    run has no HF repo or no published snapshots (older runs, or none saved yet)."""
    repo = spec.train.hf_repo
    if not repo:
        return []
    base = adapter_prefix(spec)
    pattern = _adapter_file_re(base)
    try:
        from huggingface_hub import HfApi

        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            repo, repo_type="dataset"
        )
    except Exception as exc:  # listing is best-effort; never raise into a run/route
        print(f"[ckpt] list warn for {spec.run_id}: {exc}")
        return []
    # Collect each step's adapter-folder filenames, then keep only steps with config + weights.
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
        prefix = checkpoint_adapter_prefix(spec, step)
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
