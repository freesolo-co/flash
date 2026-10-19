"""W&B logging setup for the fine-tuning worker.

Named ``wandb_log`` (not ``wandb``) so ``import wandb`` always resolves to the real package.
"""

from __future__ import annotations

import flash.engine.worker.runtime.state as _worker_state


def wandb_run_name() -> str:
    """W&B run name, from the typed ``[wandb] run_name`` config (``JOB_SPEC.wandb.run_name``) only —
    no WANDB_NAME environment variable. An explicit name is used verbatim (the user owns the
    naming); otherwise a stable id tying the dashboard run to the Flash run
    (``flash-<phase>-<run_id>``). Passed into the verl config; the child process runs
    ``wandb.init`` and reports the run link back over the marker channel."""
    configured = _worker_state.JOB_SPEC.wandb.run_name if _worker_state.JOB_SPEC else None
    if configured and configured.strip():
        return configured.strip()
    return f"flash-{_worker_state.PHASE}-{_worker_state.RUN_ID}"
