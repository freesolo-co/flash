"""Shared ``fresh_runner`` helper.

Reloads ``flash.runner`` and redirects RUNS_DIR/RESULTS_DIR under a tmp dir — the
reload+monkeypatch dance the audit found duplicated across the orchestrator/jobs tests.
"""

from __future__ import annotations

import importlib
import os


def fresh_runner(tmp, monkeypatch):
    """Reload ``flash.runner`` for the network-shaped submit/poll path with mocks.

    Redirects the fixed RUNS_DIR/RESULTS_DIR module constants under ``tmp`` via
    monkeypatch so they're restored after the test (the module object is shared, so a
    bare assignment leaks).
    """
    import flash.runner as runner

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(str(tmp), "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", os.path.join(str(tmp), "results"))
    return runner


def provisioned_status(runner, spec, *, state="running", **kwargs):
    """A ``RunStatus`` as the control plane persists it for a *provisioned* run.

    Mirrors the submit path: ``spec`` is the public (managed-field-stripped) view for display,
    and the complete internal worker spec is recorded under
    ``effective_preparation["worker_spec"]`` -- the authoritative carrier the runner recovers
    run identity and platform-managed fields from during attach/recovery (see
    ``_internal_spec_from_status`` / ``effective_spec_from_status``: run_id, train.hf_repo,
    gpu.max_wall_seconds, gpu.max_retries, ...). A run only acquires a durable provider handle
    *after* this snapshot is persisted (``_persist_effective_worker_spec`` runs before
    ``submit_run``), so a fixture with ``remote=`` must carry it too -- otherwise recovery falls
    back to the lossy public spec and reconstructs a ``run_id="local"`` placeholder.

    Pass the per-test runner module (imported as ``runner`` / ``orch`` / ``R``) as the first
    argument. ``run_id`` / ``spec`` / ``effective_preparation`` default to the provisioned shape
    but stay overridable through ``kwargs`` for the rare fixture that needs to.
    """
    kwargs.setdefault("run_id", spec.run_id)
    kwargs.setdefault("spec", spec.to_dict())
    kwargs.setdefault("effective_preparation", {"worker_spec": spec.to_internal_dict()})
    return runner.RunStatus(state=state, **kwargs)
