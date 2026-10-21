"""Cross-folder contract: the freesolo Codex trainer agent (../agent) drives the CLI
and consumes its run-id / run-state / metrics outputs. This asserts that the flash
side provides what the agent depends on, and pins the one surface deliberately taken
away from it (the root run-management aliases).

The agent package lives in a sibling folder (../agent/src) that is NOT installed in
the flash venv, so we add it to sys.path here (mirroring how the existing
end-to-end tests import across packages). When that import can't resolve, the
agent-side assertions are skipped rather than failing the flash suite.

Run: cd flash && .venv/bin/python -m pytest \
        tests/test_agent_flash_cli_contract.py -q
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
from pathlib import Path

import pytest

import flash.runner.lifecycle.state as runner_state

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
from flash.cli.parsing.main import main
from flash.core.spec import JobSpec
from flash.runner.lifecycle.state import TERMINAL_STATES, RunStatus, new_run_id, require_safe_run_id
from flash.runner.lifecycle.status import get_status


@pytest.mark.parametrize(
    "command",
    [
        ("train",),
        ("runs", "status"),
        ("runs", "list"),
        ("runs", "cancel"),
        ("env",),
    ],
)
def test_agent_required_subcommands_exist(command: tuple[str, ...]) -> None:
    """The agent contract uses the canonical grouped run-management commands."""
    with pytest.raises(SystemExit) as excinfo:
        main([*command, "--help"])
    rendered = " ".join(command)
    assert excinfo.value.code == 0, f"`flash {rendered}` is missing from the CLI"


@pytest.mark.parametrize("command", ["status", "log", "cancel", "checkpoints"])
def test_deployed_agent_root_run_shims_are_not_registered(command: str, capsys) -> None:
    """The root run-management forms are gone; only the grouped `flash runs ...` forms remain.

    these were kept registered-but-hidden so deployed agents could migrate. that grace period is
    over, so pin the removal to keep the shims from creeping back. agent-side prompts still naming
    them are tracked in freesolo-co/freesolo#618 -- note `checkpoints` became `runs checkpoint`
    (singular), so a plain `flash ` -> `flash runs ` rewrite gets that one wrong.
    """
    with pytest.raises(SystemExit) as excinfo:
        main([command, "--help"])
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


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


def test_train_dry_run_emits_run_id_and_state(tmp_path: Path, capsys, monkeypatch) -> None:
    """`flash train --dry-run` is the path the agent uses to validate a config without a
    GPU. Dry-run now routes through the server (`create_run(dry_run=True)`) so it runs the
    real submit-time preflights, but the JSON contract the agent consumes is unchanged: it
    reads `run_id` + `state` from the printed payload (see codex/outputs.py run_id field),
    so this asserts those keys exist with a real run id and a `dry_run` state, and that the
    CLI passed `dry_run=True` (never allocating a GPU).
    """
    config = tmp_path / "rl.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "owner/project/env"\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 10\n"
        "[gpu]\n"
        ""
    )

    seen: dict = {}

    class _FakeClient:
        def create_run(self, spec, runtime_secrets=None, dry_run=False, client_train_schema=None):
            seen["dry_run"] = dry_run
            seen["runtime_secrets"] = runtime_secrets
            seen["client_train_schema"] = client_train_schema
            return {"run_id": new_run_id(), "state": "dry_run", "spec": spec}

    monkeypatch.setattr("flash.cli.commands.ops.train.client_from_config", lambda: _FakeClient())

    rc = main(["train", str(config), "--dry-run"])
    assert rc == 0
    # The CLI must route the dry-run through the server as a dry-run — never as a real submit.
    assert seen["dry_run"] is True
    out = capsys.readouterr().out
    import json

    payload = json.loads(out)
    assert payload["state"] == "dry_run"
    assert isinstance(payload["run_id"], str)
    assert payload["run_id"]
    assert "spec" in payload


# --- new_run_id format vs the agent's run-id expectations ---------------------


def test_new_run_id_format_is_filesystem_safe_and_stable() -> None:
    """flash.runner_state.new_run_id() must stay within the safe run-id alphabet (it flows
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
    metrics. flash.runner_state.TERMINAL_STATES is the source of truth; the CLI's
    _CLI_DONE_STATES (what `flash`/the agent polls until) must be a superset and must
    include every flash terminal state, or the agent could poll forever on a state
    the CLI never treats as done.
    """
    import flash.cli.commands.ops.runs as cli_runs

    assert TERMINAL_STATES <= cli_runs._CLI_DONE_STATES
    # "done" is the success terminal the agent's verifier gates run_completed on.
    assert "done" in TERMINAL_STATES
    assert {"done", "failed", "cancelled"} <= TERMINAL_STATES


def test_get_status_returns_fields_the_agent_reads(tmp_path, monkeypatch) -> None:
    """`flash runs status <run_id>` returns the run's status JSON; the agent reads `state`
    and `run_id` from it (and the CLI's status/runs commands read cost_usd/spec). Assert a
    persisted RunStatus exposes those keys so a field rename in RunStatus can't
    silently strip what the agent/CLI consume.
    """

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    rid = new_run_id()
    status = RunStatus(
        run_id=rid,
        state="done",
        spec={"model": "m", "project": "11111111-1111-4111-8111-111111111111"},
    )
    runner_state._save_status(status)

    loaded = get_status(rid).to_dict()
    for key in ("run_id", "state", "spec", "cost_usd", "updated_at", "created_at", "adapter_ref"):
        assert key in loaded, f"RunStatus dropped {key!r} that the CLI/agent reads"
    assert loaded["state"] in (TERMINAL_STATES | {"queued", "running", "provisioning", "deployed"})
    assert loaded["adapter_ref"] is None


def test_done_status_exposes_adapter_ref(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    rid = "flash-status-adapter-ref"
    spec = JobSpec.from_dict(
        {
            "run_id": rid,
            "project": "11111111-1111-4111-8111-111111111111",
            "algorithm": "sft",
            "model": "Qwen/Qwen3.5-9B",
            "train": {
                "epochs": 1,
                "max_examples": 8,
                "hf_repo": f"Freesolo-Co/flashrun-{rid}",
            },
        }
    )
    # hf_repo is platform-managed and stripped from the public spec, so the adapter ref is resolved
    # from the internal worker spec persisted in effective_preparation (present for every
    # provisioned run), exactly as submit_job records it.
    worker_prep = {"worker_spec": spec.to_internal_dict()}
    runner_state._save_status(
        RunStatus(
            run_id=rid, state="running", spec=spec.to_dict(), effective_preparation=worker_prep
        )
    )
    # a non-terminal run does not expose the adapter ref even once its worker spec is prepared
    assert get_status(rid).to_dict()["adapter_ref"] is None

    runner_state._save_status(
        RunStatus(run_id=rid, state="done", spec=spec.to_dict(), effective_preparation=worker_prep)
    )
    # the public final checkpoint: exactly what train.init_from_adapter accepts
    assert get_status(rid).to_dict()["adapter_ref"] == f"{rid}/final"


def test_done_status_with_removed_spec_key_serializes_without_adapter_ref(
    tmp_path, monkeypatch
) -> None:
    # a record written by an OLDER plane can carry a since-removed spec key (gpu.exact_type,
    # pre-#670); strict JobSpec parsing raises on it. the record must still serialize (one bad
    # record must not 500 the whole runs list) -- it just resolves no adapter ref.

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    rid = "flash-status-legacy-spec-key"
    legacy_spec = {
        "run_id": rid,
        "algorithm": "sft",
        "model": "Qwen/Qwen3.5-2B",
        "gpu": {"type": "H100", "exact_type": "H100 SXM"},
        "train": {"epochs": 1},
    }
    # written directly: these records were created by the older plane and sit on disk as-is
    # (the current plane's _save_status could never produce one).
    os.makedirs(runner_state.RUNS_DIR, exist_ok=True)
    record = RunStatus(
        run_id=rid,
        state="done",
        spec=legacy_spec,
        effective_preparation={"worker_spec": dict(legacy_spec)},
    )
    with open(os.path.join(runner_state.RUNS_DIR, f"{rid}.json"), "w") as f:
        json.dump(dataclasses.asdict(record), f)
    loaded = get_status(rid).to_dict()
    assert loaded["state"] == "done"
    assert loaded["adapter_ref"] is None
