"""W&B logging setup for the fine-tuning worker.

Named ``wandb_log`` (not ``wandb``) so ``import wandb`` always resolves to the real package.
"""

from __future__ import annotations

from flash.engine.worker.runtime.state import WorkerContext


def wandb_run_name(*, context: WorkerContext) -> str:
    """W&B run name, from the typed ``[wandb] run_name`` config (``JOB_SPEC.wandb.run_name``) only —
    no WANDB_NAME environment variable. An explicit name is used verbatim (the user owns the
    naming); otherwise a stable id tying the dashboard run to the Flash run
    (``flash-<phase>-<run_id>``). Passed into the verl config; the child process runs
    ``wandb.init`` and reports the run link back over the marker channel."""
    configured = context.config.spec.wandb.run_name if context.config.spec else None
    if configured and configured.strip():
        return configured.strip()
    return f"flash-{context.config.phase}-{context.config.run_id}"
