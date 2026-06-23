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


def checkpoint_adapter_prefix(spec: JobSpec, step: int, seed: int | None = None) -> str:
    """The ``adapter_prefix`` that serves checkpoint ``step``.

    ``deploy_adapter`` appends ``/adapter`` to whatever prefix it's given, so this returns the
    per-step root (``<run prefix>/checkpoints/step-<N>``) — matching the worker's upload path —
    and the existing deploy path needs no special-casing for checkpoints."""
    return f"{adapter_prefix(spec, seed)}/checkpoints/step-{step}"


def _adapter_config_re(base: str) -> re.Pattern[str]:
    """Matches ``<base>/checkpoints/step-<N>/adapter/adapter_config.json`` — the presence of the
    PEFT config is what marks a step as actually loadable/deployable."""
    return re.compile(
        re.escape(base) + r"/checkpoints/step-(\d+)/adapter/adapter_config\.json$"
    )


def list_checkpoints(spec: JobSpec, seed: int | None = None) -> list[dict]:
    """Deployable per-step adapter snapshots for ``spec``, ascending by step.

    Each entry: ``{"step", "adapter_prefix", "subfolder", "repo_id", "repo_type"}`` where
    ``adapter_prefix`` is the value to hand ``deploy_adapter`` to serve that exact step and
    ``subfolder`` is the full path of the adapter folder in the repo. Returns ``[]`` when the
    run has no HF repo or no published snapshots (older runs, or none saved yet)."""
    repo = spec.train.hf_repo
    if not repo:
        return []
    base = adapter_prefix(spec, seed)
    pattern = _adapter_config_re(base)
    try:
        from huggingface_hub import HfApi

        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            repo, repo_type="dataset"
        )
    except Exception as exc:  # listing is best-effort; never raise into a run/route
        print(f"[ckpt] list warn for {spec.run_id}: {exc}")
        return []
    out: list[dict] = []
    for path in files:
        match = pattern.search(path)
        if not match:
            continue
        step = int(match.group(1))
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
    out.sort(key=lambda c: c["step"])
    return out
