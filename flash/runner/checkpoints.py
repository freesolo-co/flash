"""List a run's deployable per-step RL checkpoints from its HF artifact repo.

The GPU worker publishes each trainer save's LoRA adapter to a stable, NON-pruned path
(``<adapter_prefix>/checkpoints/step-<N>/adapter``; see
``flash.engine.worker.publish_deployable_checkpoint``). This module is the control-plane
reader: it enumerates those snapshots so ``flash checkpoints`` can list them and
``flash deploy <run-id>/step-N`` can serve a specific one — including for a run that was cancelled or
failed mid-RL and so never sealed a final adapter. HF (not the backend DB) is the source of
truth for what's deployable; backend persistence is a mirror (see
``flash.server.checkpoints``)."""

from __future__ import annotations

import os
import re

from flash.adapter_artifacts import ADAPTER_WEIGHT_FILES
from flash.runner import adapter_prefix
from flash.spec import JobSpec


class CheckpointListingError(RuntimeError):
    """Raised when checkpoint listing must be authoritative but HF cannot be queried."""


def checkpoint_adapter_prefix(spec: JobSpec, step: int) -> str:
    """The ``adapter_prefix`` that serves checkpoint ``step``.

    ``deploy_adapter`` appends ``/adapter`` to whatever prefix it's given, so this returns the
    per-step root (``<run prefix>/checkpoints/step-<N>``) — matching the worker's upload path —
    and the existing deploy path needs no special-casing for checkpoints."""
    return f"{adapter_prefix(spec)}/checkpoints/step-{step}"


def _adapter_file_re(base: str) -> re.Pattern[str]:
    """Matches ``<base>/checkpoints/step-<N>/adapter/<filename>`` and captures (step, filename)."""
    return re.compile(re.escape(base) + r"/checkpoints/step-(\d+)/adapter/([^/]+)$")


def _adapter_folder_has_required_files(files: list[str], prefix: str) -> bool:
    names = {
        path.removeprefix(f"{prefix}/adapter/")
        for path in files
        if path.startswith(f"{prefix}/adapter/")
        and "/" not in path.removeprefix(f"{prefix}/adapter/")
    }
    return "adapter_config.json" in names and not names.isdisjoint(ADAPTER_WEIGHT_FILES)


def _repo_files(spec: JobSpec, *, strict: bool) -> list[str]:
    repo = spec.train.hf_repo
    if not repo:
        return []
    try:
        from huggingface_hub import HfApi

        return HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo, repo_type="dataset")
    except Exception as exc:
        if strict:
            raise CheckpointListingError(
                f"could not verify deployable checkpoints for {spec.run_id}: {exc}"
            ) from exc
        print(f"[ckpt] list warn for {spec.run_id}: {exc}")
        return []


def _checkpoints_from_files(spec: JobSpec, files: list[str]) -> list[dict]:
    repo = spec.train.hf_repo
    if not repo:
        return []
    base = adapter_prefix(spec)
    pattern = _adapter_file_re(base)
    # Collect each step's adapter-folder filenames, then keep only steps with config + weights.
    by_step: dict[int, set[str]] = {}
    for path in files:
        match = pattern.search(path)
        if match:
            by_step.setdefault(int(match.group(1)), set()).add(match.group(2))
    out: list[dict] = []
    for step in sorted(by_step):
        names = by_step[step]
        if "adapter_config.json" not in names or names.isdisjoint(ADAPTER_WEIGHT_FILES):
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


def list_checkpoints(spec: JobSpec) -> list[dict]:
    """Deployable per-step adapter snapshots for ``spec``, ascending by step.

    A step is included only if its adapter folder carries BOTH ``adapter_config.json`` AND a
    weights file (so checkpoint deploy can never target a half-uploaded, unloadable step). Each
    entry: ``{"step", "adapter_prefix", "subfolder", "repo_id", "repo_type"}`` where
    ``adapter_prefix`` is the value to hand ``deploy_adapter`` to serve that exact step and
    ``subfolder`` is the full path of the adapter folder in the repo. Returns ``[]`` when the
    run has no HF repo or no published snapshots (older runs, or none saved yet)."""
    return _checkpoints_from_files(spec, _repo_files(spec, strict=False))


def checkpoint_step_exists(spec: JobSpec, step: int) -> bool:
    """Authoritatively verify a warm-start checkpoint step before provisioning a worker.

    Unlike ``list_checkpoints()``, this raises ``CheckpointListingError`` when HF listing fails
    so transient/auth errors are not misreported as "step missing".
    """
    return any(
        int(item.get("step", -1)) == int(step)
        for item in _checkpoints_from_files(spec, _repo_files(spec, strict=True))
    )


def final_adapter_exists(spec: JobSpec) -> bool:
    """Authoritatively verify that the run-level final adapter exists in the artifact repo."""
    try:
        files = _repo_files(spec, strict=True)
    except CheckpointListingError as exc:
        cause = exc.__cause__ or exc
        raise CheckpointListingError(
            f"could not verify final adapter for {spec.run_id}: {cause}"
        ) from cause
    return _adapter_folder_has_required_files(files, adapter_prefix(spec))
