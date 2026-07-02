"""Cross-folder contract: the freesolo Codex trainer agent (../agent) drives the CLI
and consumes its run-id / run-state / metrics outputs. This asserts that the flash
side provides what the agent depends on.

The agent package lives in a sibling folder (../agent/src) that is NOT installed in
the flash venv, so we add it to sys.path here (mirroring how the existing
end-to-end tests import across packages). When that import can't resolve, the
agent-side assertions are skipped rather than failing the flash suite.

Run: cd flash && .venv/bin/python -m pytest \
        tests/test_agent_flash_cli_contract.py -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# --- cross-folder sys.path shim: make ../agent/src importable -----------------
_AGENT_SRC = Path(__file__).resolve().parents[2] / "agent" / "src"
if _AGENT_SRC.is_dir() and str(_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(_AGENT_SRC))


def _import_agent(modpath: str):
    try:
        mod = __import__(modpath, fromlist=["*"])
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"agent module {modpath!r} not importable in this runner: {exc}")
    return mod


# --- flash side (always importable in the flash venv) ---------------------
from flash.cli import main
from flash.runner import (
    TERMINAL_STATES,
    RunStatus,
    get_status,
    new_run_id,
    require_safe_run_id,
)
from flash.spec import JobSpec


@pytest.mark.parametrize(
    "subcommand",
    [
        "train",
        "status",
        "runs",
        "cancel",
        "env",
    ],
)
def test_agent_required_subcommands_exist(subcommand: str) -> None:
    """The agent's worker drives the CLI's `train/status/runs/cancel/env/...`
    subcommands.

    argparse exits with code 2 and an 'invalid choice' message when a subcommand
    does not exist. We invoke each with `--help`, which exits 0 for a real
    subcommand (and never touches the network/credentials). A removed/renamed
    subcommand would make this raise SystemExit(2) instead.
    """
    with pytest.raises(SystemExit) as excinfo:
        main([subcommand, "--help"])
    # --help exits 0 for an existing subcommand; an unknown subcommand exits 2.
    assert excinfo.value.code == 0, f"`flash {subcommand}` is missing from the CLI"


def test_env_push_subcommand_exists() -> None:
    """The agent publishes a locally-authored Freesolo env via `flash env push`."""
    with pytest.raises(SystemExit) as excinfo:
        main(["env", "push", "--help"])
    assert excinfo.value.code == 0, "`flash env push` is missing from the CLI"


def test_unknown_subcommand_is_rejected() -> None:
    """Sanity that the --help probe above actually distinguishes existing commands:
    a bogus subcommand must exit 2 (argparse 'invalid choice')."""
    with pytest.raises(SystemExit) as excinfo:
        main(["definitely-not-a-real-subcommand", "--help"])
    assert excinfo.value.code == 2


def test_train_dry_run_emits_run_id_and_state(tmp_path: Path, capsys) -> None:
    """`flash train --dry-run` is the fully-local path the agent uses to validate a
    config without a server/GPU. The agent reads `run_id` + `state` from the JSON it
    prints (see codex/outputs.py run_id field), so this asserts those keys exist with
    a real run id and a `dry_run` state — fully offline.
    """
    config = tmp_path / "rl.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-4B"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "owner/env"\n'
        "[train]\n"
        'hf_repo = "owner/runs"\n'
        "steps = 10\n"
        "[gpu]\n"
        'type = "RTX 5090"\n'
    )
    rc = main(["train", str(config), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    import json

    payload = json.loads(out)
    assert payload["state"] == "dry_run"
    assert isinstance(payload["run_id"], str)
    assert payload["run_id"]
    assert "spec" in payload


# --- new_run_id format vs the agent's run-id expectations ---------------------


def test_new_run_id_format_is_filesystem_safe_and_stable() -> None:
    """flash.runner.new_run_id() must stay within the safe run-id alphabet (it flows
    into filesystem paths AND into the agent's tracker as best_run_id/run_id). The
    agent resolves the flash run id from the tracker (workflows.common
    .training_run_id_from_tracker) and threads it back into its output schema, so the
    id has to survive a string round-trip and the path-containment check.
    """
    rid = new_run_id()
    # default prefix + epoch + 8 hex chars
    assert re.fullmatch(r"flash-\d+-[0-9a-f]{8}", rid), rid
    # The runner's own path-safety gate must accept it (no traversal characters).
    assert require_safe_run_id(rid) == rid


def test_agent_tracker_run_id_round_trips_through_resolver(tmp_path: Path) -> None:
    """End-to-end of the run-id seam: a flash run id written into the agent's
    progress.json tracker (as best_run_id) must be read back unchanged by the agent's
    resolve_training_run_id, and reflected into the summary JSON. This is exactly the
    flow `flash train` -> tracker -> agent output uses, so a change to either the id
    format or the tracker field name breaks it.
    """
    common = _import_agent("workflows.common")
    import json

    rid = new_run_id()
    tracker_path = tmp_path / common.AGENT_CHECKPOINT_TRACKER_FILE
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_path.write_text(
        json.dumps({common.TRAINING_RESUME_STATE_FIELD: {"best_run_id": rid}}),
        encoding="utf-8",
    )

    # The tracker is canonical: its best_run_id wins over the summary's run_id.
    summary_in = json.dumps({"status": "succeeded", "run_id": "stale-id"})
    summary_out, resolved = common.resolve_training_run_id(
        summary=summary_in, worktree_path=tmp_path
    )
    assert resolved == rid
    assert json.loads(summary_out)["run_id"] == rid


def test_agent_terminal_states_subset_of_flash_terminal_states() -> None:
    """The agent waits for a run to reach a terminal state before reading final
    metrics. flash.runner.TERMINAL_STATES is the source of truth; the CLI's
    _CLI_DONE_STATES (what `flash`/the agent polls until) must be a superset and must
    include every flash terminal state, or the agent could poll forever on a state
    the CLI never treats as done.
    """
    import flash.cli as cli_main

    assert TERMINAL_STATES <= cli_main._CLI_DONE_STATES
    # "done" is the success terminal the agent's verifier gates run_completed on.
    assert "done" in TERMINAL_STATES
    assert {"done", "failed", "cancelled"} <= TERMINAL_STATES


def test_get_status_returns_fields_the_agent_reads(tmp_path, monkeypatch) -> None:
    """`flash status <run_id>` returns the run's status JSON; the agent reads `state`
    and `run_id` from it (and the CLI's status/runs commands read cost_usd/spec). Assert a
    persisted RunStatus exposes those keys so a field rename in RunStatus can't
    silently strip what the agent/CLI consume.
    """
    from flash import runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    rid = new_run_id()
    status = RunStatus(run_id=rid, state="done", spec={"model": "m"})
    runner._save_status(status)

    loaded = get_status(rid).to_dict()
    for key in ("run_id", "state", "spec", "cost_usd", "updated_at", "created_at", "adapter_ref"):
        assert key in loaded, f"RunStatus dropped {key!r} that the CLI/agent reads"
    assert loaded["state"] in (TERMINAL_STATES | {"queued", "running", "provisioning", "deployed"})
    assert loaded["adapter_ref"] is None


def test_done_status_exposes_adapter_ref(tmp_path, monkeypatch) -> None:
    from flash import runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    rid = "flash-status-adapter-ref"
    spec = JobSpec.from_dict(
        {
            "run_id": rid,
            "algorithm": "sft",
            "model": "Qwen/Qwen3.5-2B",
            "train": {
                "epochs": 1,
                "hf_repo": f"Freesolo-Co/flashrun-{rid}",
            },
        }
    )
    runner._save_status(RunStatus(run_id=rid, state="running", spec=spec.to_dict()))
    assert get_status(rid).to_dict()["adapter_ref"] is None

    runner._save_status(RunStatus(run_id=rid, state="done", spec=spec.to_dict()))
    # the public short ref: exactly what train.init_from_adapter accepts
    assert get_status(rid).to_dict()["adapter_ref"] == rid
