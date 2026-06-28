"""W&B (Weights & Biases) logging setup for the fine-tuning worker.

Project + run name come ONLY from the typed ``[wandb]`` config (``JOB_SPEC.wandb``); there is
no WANDB_PROJECT / WANDB_NAME env var. Run-scoped state (``JOB_SPEC``/``PHASE``/``RUN_ID``/
``SEED``) is read through the worker package at CALL time so a test's
``monkeypatch.setattr(worker, "JOB_SPEC", ...)`` reaches these readers.

The module name is ``wandb_log`` (not ``wandb``) so ``import wandb`` inside these functions
always resolves to the real top-level W&B package, never this sibling module.
"""

from __future__ import annotations

import os
import threading

from flash.engine.worker._pkg import W as _w


def wandb_report_to() -> list[str]:
    """TRL/HF ``report_to`` targets. Restores the W&B logging the legacy freesolo training path had
    but the flash migration dropped: report to W&B whenever WANDB_API_KEY is present. No key -> []
    (silent, the metrics.json artifact is still the source of truth).

    Project + run name come ONLY from the typed ``[wandb]`` config (``JOB_SPEC.wandb``) — there is
    NO WANDB_PROJECT / WANDB_NAME environment variable. HF's WandbCallback has no project argument
    and would read WANDB_PROJECT from the env, so we initialize the run directly via the wandb SDK
    here (``wandb.init(project=..., name=...)``); the Trainer's callback then reuses that run. The
    run's entity is the API key's default account/team (we don't pass ``entity=``), so the only
    W&B env var is the WANDB_API_KEY credential."""
    if not os.environ.get("WANDB_API_KEY"):
        return []
    import importlib.util

    # find_spec can RAISE (not just return None) when wandb is in sys.modules with an absent/partial
    # __spec__ (a partially-initialized import / namespace package) — that would crash the worker even
    # though W&B logging is best-effort (mirrors the guard in wandb_finish). Treat an ambiguous probe
    # as "present enough to try" and let the guarded wandb.init below decide; only a definitive None
    # (probe succeeded, module truly absent) skips logging here.
    try:
        if importlib.util.find_spec("wandb") is None:
            print("[wandb] WANDB_API_KEY set but the wandb package is missing; skipping W&B logging")
            return []
    except Exception:
        pass  # ambiguous probe -> fall through to the guarded wandb.init below
    # Best-effort, like the bitsandbytes import above: a partial/broken wandb install or an
    # init failure (auth, network, runtime import error) must NOT abort training — W&B logging is
    # optional and metrics.json is the source of truth. Any failure -> no W&B logging ([]).
    try:
        import wandb

        if wandb.run is None:  # init from the spec so the project needs no WANDB_PROJECT env
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
    """The live W&B run's {url, id, project} if W&B is active, else {}. Recorded in metrics.json so
    the W&B run is verifiable + the freesolo agent's `wandb_runs` / the SDK's link_wandb can point at
    the real dashboard URL — the link the flash migration otherwise dropped. Never raises."""
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
    """Finalize the W&B run before the worker's hard ``os._exit()``.

    The worker hard-exits to dodge the colocated-vLLM teardown deadlock (see main),
    which skips wandb's atexit sync — so a *successfully completed* run was left
    dangling and W&B eventually marked it ``crashed`` even though all metrics were
    logged. Explicitly finish the run (we own it: we called ``wandb.init`` in
    ``wandb_report_to``) so it shows ``finished``. Best-effort; never raises (W&B is
    optional, metrics.json is the source of truth)."""
    if not os.environ.get("WANDB_API_KEY"):
        return
    import importlib.util

    # find_spec can RAISE (not just return None) when wandb is already in sys.modules with an
    # absent/partial __spec__ (e.g. a namespace-package or a partially-initialized import) — that
    # would propagate out of the shutdown path and skip the hard exit. Keep it best-effort: treat any
    # probe failure as "wandb present enough to try", and let the import + finish below (already
    # wrapped) decide. Only a definitive None (probe succeeded, module truly absent) returns early.
    try:
        if importlib.util.find_spec("wandb") is None:
            return
    except Exception:
        pass  # ambiguous probe -> fall through and try to finish (still fully guarded below)
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
        # On SUCCESS (exit_code == 0) wandb.finish() must flush the full run; a slow network / large
        # run can take well over 5s, and cutting it off there is what leaves the run dangling ->
        # "crashed". Allow a longer, still-bounded wait on success; keep the short cut-off on the
        # FAILURE path (exit_code != 0) where we want to abort fast and the run is failing anyway.
        # Read THROUGH the worker package so a test's monkeypatch of the wait knobs is honored.
        wait_s = _w._WANDB_FINISH_WAIT_S if exit_code == 0 else _w._WANDB_FINISH_FAIL_WAIT_S
        t.join(timeout=wait_s)
        if t.is_alive():
            print(f"[wandb] finish() did not complete within {wait_s}s; continuing with hard exit")
        elif errs:
            print(f"[wandb] finish() warning: {errs[0]}")
    except Exception as e:  # pragma: no cover - logging-only path
        print(f"[wandb] finish() warning: {e}")
