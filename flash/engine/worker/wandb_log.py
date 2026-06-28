"""W&B logging setup for the fine-tuning worker.

Named ``wandb_log`` (not ``wandb``) so ``import wandb`` always resolves to the real package.
"""

from __future__ import annotations

import os
import threading

from flash.engine.worker._pkg import W as _w


def _wandb_importable() -> bool:
    """Return True if wandb appears importable. find_spec can RAISE on partial installs; treat that as True."""
    import importlib.util

    try:
        return importlib.util.find_spec("wandb") is not None
    except Exception:
        return True  # ambiguous probe -> "present enough to try"


def wandb_report_to() -> list[str]:
    """TRL/HF ``report_to`` targets. Returns ["wandb"] when WANDB_API_KEY is set, else [].

    Initializes the run directly (project from JOB_SPEC.wandb, not WANDB_PROJECT env) so the
    Trainer's WandbCallback reuses the already-initialized run. Best-effort; never aborts training."""
    if not os.environ.get("WANDB_API_KEY"):
        return []
    if not _wandb_importable():
        print("[wandb] WANDB_API_KEY set but the wandb package is missing; skipping W&B logging")
        return []
    try:
        import wandb

        if wandb.run is None:
            project = (_w.JOB_SPEC.wandb.project if _w.JOB_SPEC else None) or "flash"
            wandb.init(project=project, name=wandb_run_name())
    except Exception as e:
        print(
            f"[wandb] W&B init failed ({e}); skipping W&B logging (metrics.json is still written)"
        )
        return []
    return ["wandb"]


def wandb_run_name() -> str:
    """W&B run name, from the typed ``[wandb] run_name`` config (``JOB_SPEC.wandb.run_name``) only —
    no WANDB_NAME environment variable. An explicit name is used verbatim (the user owns the
    naming); otherwise a stable id tying the dashboard run to the Flash run
    (``flash-<phase>-<run_id>``). Passed to the Trainer via ``TrainingArguments.run_name``
    and to ``wandb.init`` above."""
    configured = _w.JOB_SPEC.wandb.run_name if _w.JOB_SPEC else None
    if configured and configured.strip():
        return configured.strip()
    return f"flash-{_w.PHASE}-{_w.RUN_ID}"


def wandb_run_info() -> dict:
    """Return live W&B run's {url, id, project}, or {} if not active. Never raises."""
    try:
        import wandb

        run = getattr(wandb, "run", None)
        if run is None:
            return {}
        return {
            "wandb_url": getattr(run, "url", None),
            "wandb_id": getattr(run, "id", None),
            "wandb_project": getattr(run, "project", None),
        }
    except Exception:
        return {}


def wandb_finish(exit_code: int = 0) -> None:
    """Finalize the W&B run before the worker's hard os._exit().

    Hard-exit skips wandb's atexit sync, leaving the run dangling (W&B marks it ``crashed``).
    Explicit finish here prevents that. Best-effort; never raises."""
    if not os.environ.get("WANDB_API_KEY"):
        return
    if not _wandb_importable():
        return
    try:
        import wandb

        if getattr(wandb, "run", None) is None:
            return

        errs: list[Exception] = []

        def _finish() -> None:
            try:
                wandb.finish(exit_code=exit_code)
            except Exception as e:
                errs.append(e)

        t = threading.Thread(target=_finish, daemon=True)
        t.start()
        # Longer wait on success to avoid cutting off the flush that causes the "crashed" mark.
        wait_s = _w._WANDB_FINISH_WAIT_S if exit_code == 0 else _w._WANDB_FINISH_FAIL_WAIT_S
        t.join(timeout=wait_s)
        if t.is_alive():
            print(f"[wandb] finish() did not complete within {wait_s}s; continuing with hard exit")
        elif errs:
            print(f"[wandb] finish() warning: {errs[0]}")
    except Exception as e:  # pragma: no cover - logging-only path
        print(f"[wandb] finish() warning: {e}")
