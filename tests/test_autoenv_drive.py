"""DRIVE plumbing: scaffold a workspace and validate the agent's config via dry-run (offline)."""

from __future__ import annotations

from pathlib import Path

import autoenv
from autoenv.drive import drive, scaffold
from autoenv.drive.agent_backend import ScriptedBaseline, get_backend
from autoenv.ingest import fetch_rows
from autoenv.manifest import PaperCase

SMOKE_CASE = Path(autoenv.__file__).parent / "cases" / "arithmetic_smoke_sft.toml"


def _rows(case: PaperCase) -> list[dict]:
    return fetch_rows(
        case.resolved_train(),
        input_field=case.dataset.input_field,
        output_field=case.dataset.output_field,
    )


def test_scaffold_produces_flash_layout_and_freezes_reward(tmp_path):
    case = PaperCase.load(SMOKE_CASE)
    ws = scaffold(case, tmp_path / "ws", "Qwen/Qwen3.5-0.8B", train_rows=_rows(case))

    # Flash's own scaffold landed (incl. the agent playbook).
    assert (ws.root / "TRAINING.md").is_file()
    assert (ws.root / "configs").is_dir()
    assert ws.environment_py.is_file()
    assert ws.train_jsonl.is_file()

    # Easy mode froze the supplied reward next to the env and imports it.
    assert (ws.root / "reward.py").is_file()
    env_src = ws.environment_py.read_text()
    assert "from reward import score" in env_src

    # Train rows were written in canonical shape.
    first = ws.train_jsonl.read_text().splitlines()[0]
    assert '"input"' in first
    assert '"output"' in first

    # The config is pinned to the resolved model + algorithm + a valid env-id placeholder.
    cfg = ws.config_path.read_text()
    assert 'model = "Qwen/Qwen3.5-0.8B"' in cfg
    assert 'algorithm = "sft"' in cfg
    assert "epochs = 1" in cfg
    assert "max_steps = 20" in cfg


def test_dry_run_drive_validates_and_costs(tmp_path):
    case = PaperCase.load(SMOKE_CASE)
    result = drive(
        case,
        backend=ScriptedBaseline(),
        model_id="Qwen/Qwen3.5-0.8B",
        train_rows=_rows(case),
        dest=tmp_path / "ws",
        dry_run=True,
    )
    assert result.state == "dry_run"
    assert result.run_id == "autoenv-dryrun-arithmetic-smoke-sft"
    assert result.estimated_usd is not None
    assert result.estimated_usd >= 0
    # spec round-tripped through flash's real parser.
    assert result.spec["model"] == "Qwen/Qwen3.5-0.8B"
    assert result.spec["algorithm"] == "sft"


def test_drive_blocks_when_over_budget(tmp_path):
    import dataclasses

    case = dataclasses.replace(PaperCase.load(SMOKE_CASE), max_usd=0.0)
    result = drive(
        case,
        backend=ScriptedBaseline(),
        model_id="Qwen/Qwen3.5-0.8B",
        train_rows=_rows(case),
        dest=tmp_path / "ws",
        dry_run=True,
    )
    assert result.state == "error"
    assert "budget" in result.notes


def test_get_backend_resolves_and_rejects():
    import pytest

    assert get_backend("scripted").name == "scripted"
    with pytest.raises(ValueError, match="unknown agent backend"):
        get_backend("nope")
