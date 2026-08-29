"""Shared runner state helpers for isolated tests."""

from __future__ import annotations

import os

import flash.core.spec as runner_spec
import flash.runner.lifecycle.preparation as runner_preparation
import flash.runner.lifecycle.state as runner_state


def fresh_runner(tmp, monkeypatch) -> None:
    """Redirect the canonical runner state roots under ``tmp`` for a test."""
    monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(str(tmp), "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", os.path.join(str(tmp), "results"))


def provisioned_status(spec, *, state="running", **kwargs):
    """A ``RunStatus`` as the control plane persists it for a *provisioned* run.

    Mirrors the submit path: ``spec`` is the public (managed-field-stripped) view for display,
    and the complete internal worker spec is recorded under
    ``effective_preparation["worker_spec"]`` -- the authoritative carrier the runner recovers
    run identity and platform-managed fields from during attach/recovery (see
    ``_internal_spec_from_status`` / ``effective_spec_from_status``: run_id, train.hf_repo,
    gpu.max_wall_seconds, gpu.max_retries, ...). A run only acquires a durable provider handle
    *after* this snapshot is persisted (``_persist_effective_worker_spec`` runs before
    ``submit_attempt``), so a fixture with ``remote=`` must carry it too -- otherwise recovery falls
    back to the lossy public spec and reconstructs a ``run_id="local"`` placeholder.

    ``run_id`` / ``spec`` / ``effective_preparation`` default to the provisioned shape but stay
    overridable through ``kwargs`` for the rare fixture that needs to.
    """
    kwargs.setdefault("run_id", spec.run_id)
    public = spec.to_dict()
    kwargs.setdefault("spec", public)
    # both production writers (``_persist_effective_worker_spec`` and the create path in
    # ``runner.submit``) always store a ``preparation_digest`` alongside the worker spec, and the
    # validating loader now requires one for any auto-pinned or profiled run. A fixture without it
    # is a shape production cannot produce, so omitting it would fail those runs here for a reason
    # no real run can hit -- and, worse, would let a test "detect tampering" via the missing digest
    # rather than via the tampering itself.
    kwargs.setdefault(
        "effective_preparation",
        {
            "worker_spec": spec.to_internal_dict(),
            "version": 1,
            "preparation_digest": runner_preparation._preparation_digest(
                runner_spec.JobSpec.from_dict(public), spec, None
            ),
        },
    )
    return runner_state.RunStatus(state=state, **kwargs)


def save_provisioned_status(status, **save_kwargs) -> None:
    """Persist a provisioned fixture with the attempt bookkeeping its remote implies.

    A run only holds a provider handle after that attempt was durably reserved, so the persisted
    next-attempt counter is always ``remote["attempt"] + 1``. Saving a fixture with a remote but a
    default counter of 0 produces a shape production cannot write, and every ownership check --
    which requires ``next_attempt - 1 == attempt`` -- then rejects the run for a reason no real
    record can hit.
    """
    remote = status.remote if isinstance(status.remote, dict) else None
    attempt = remote.get("attempt") if remote else None
    if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0:
        save_kwargs.setdefault("_next_attempt", attempt + 1)
    runner_state._save_status(status, **save_kwargs)
