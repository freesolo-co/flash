"""Orchestrate one DRIVE attempt: scaffold -> agent -> validate/cost-gate -> (submit/poll).

The dry-run path is fully local and offline — it validates the agent's config with the same
``spec_from_file`` that ``flash train --dry-run`` runs, catching config errors for free before
any spend. The real path publishes the env and submits through the Flash control plane
(``ApiClient``); it is gated behind ``--dry-run`` first and a preflight cost check, and is the
subject of plan milestone M2 (it needs credentials + GPU, so it is not exercised in CI).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from autoenv.drive.agent_backend import AgentBackend, AgentOutcome
from autoenv.drive.workspace import Workspace, scaffold
from autoenv.manifest import PaperCase
from flash.cost.spec import estimate_for_spec
from flash.schema import spec_from_file

# Client-side terminal states (mirror flash.runner.TERMINAL_STATES without importing the
# operator-side runner module, which would pull server/provider dependencies onto the client).
TERMINAL_STATES = frozenset({"done", "failed", "cancelled", "dry_run"})


@dataclass
class DriveResult:
    """The outcome of driving one case through Flash."""

    case_id: str
    state: str  # "dry_run" | "done" | "failed" | "cancelled" | "error"
    run_id: str | None = None
    env_id: str | None = None
    config_path: str | None = None
    cost_usd: float | None = None
    estimated_usd: float | None = None
    workspace: str | None = None
    agent: AgentOutcome | None = None
    spec: dict | None = None
    notes: str = ""
    diagnostics: dict = field(default_factory=dict)


def _validate_and_cost(config_path: Path) -> tuple[dict, float]:
    """Local dry-run validation + preflight cost — exactly what ``flash train --dry-run/--cost`` do."""
    spec = spec_from_file(str(config_path))
    estimate = estimate_for_spec(spec)
    return spec.to_dict(), estimate.total_usd


def drive(
    case: PaperCase,
    *,
    backend: AgentBackend,
    model_id: str,
    train_rows: list[dict],
    dest: str | Path,
    dry_run: bool = True,
    client=None,
    poll_interval_s: float = 10.0,
    max_wait_s: float = 3 * 3600,
) -> DriveResult:
    """Scaffold, run the agent, validate the config, then (real mode) submit and poll."""
    ws = scaffold(case, dest, model_id, train_rows=train_rows)
    outcome = backend.run(ws, case)
    base = DriveResult(
        case_id=case.id,
        state="error",
        env_id=outcome.env_id,
        config_path=outcome.config_path,
        workspace=str(ws.root),
        agent=outcome,
    )
    if not outcome.ok:
        base.notes = f"agent backend {backend.name!r} did not finish: {outcome.notes}"
        return base

    # Validate + cost-gate (local, free) before any spend.
    try:
        spec_dict, est_usd = _validate_and_cost(Path(outcome.config_path))
    except Exception as exc:
        base.notes = f"config failed dry-run validation: {exc}"
        return base
    base.spec = spec_dict
    base.estimated_usd = est_usd

    if est_usd > case.max_usd:
        base.notes = (
            f"preflight ${est_usd:.2f} exceeds case budget ${case.max_usd:.2f}; not submitting"
        )
        return base

    if dry_run:
        base.state = "dry_run"
        base.run_id = f"autoenv-dryrun-{case.id}"
        base.notes = "validated locally (dry-run); no run submitted"
        return base

    # --- Real submit path (milestone M2; needs an ApiClient + credentials, not run in CI) ---
    if client is None:
        base.notes = "real run requested but no Flash client provided"
        return base
    return _submit_real(case, ws, outcome, base, client, poll_interval_s, max_wait_s)


def _submit_real(
    case: PaperCase,
    ws: Workspace,
    outcome: AgentOutcome,
    base: DriveResult,
    client,
    poll_interval_s: float,
    max_wait_s: float,
) -> DriveResult:
    """Publish the env and submit the run through the control plane, then poll to terminal.

    Exercised in milestone M2 (live credentials + GPU). Kept thin and explicit so the live
    bring-up has a single place to harden.
    """
    published = client.publish_env(name=case.id, path=str(ws.root))
    env_id = published.get("id") or published.get("env_id") or outcome.env_id
    base.env_id = env_id

    # Re-pin the published env id on the parsed spec (not via text replace on the config —
    # that would silently no-op if the agent reformatted the [environment] id line).
    spec_dict = spec_from_file(str(outcome.config_path)).to_dict()
    spec_dict["environment"]["id"] = env_id
    created = client.create_run(spec_dict)
    run_id = created.get("run_id") or created.get("id")
    base.run_id = run_id

    deadline = time.monotonic() + max_wait_s
    state = created.get("state", "queued")
    while state not in TERMINAL_STATES and time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        status = client.get_run(run_id)
        state = status.get("state", state)
        base.cost_usd = status.get("cost_usd")
    base.state = state if state in TERMINAL_STATES else "error"
    base.notes = f"run {run_id} finished in state {base.state}"
    return base
