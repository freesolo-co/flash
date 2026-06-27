"""autoenv CLI dispatch — gate / drive (dry-run) / run on the bundled smoke case, offline."""

from __future__ import annotations

from pathlib import Path

import autoenv
from autoenv.cli import build_parser, main

SMOKE_CASE = str(Path(autoenv.__file__).parent / "cases" / "arithmetic_smoke_sft.toml")


def test_parser_builds_with_expected_subcommands():
    parser = build_parser()
    # Every subcommand wires a handler.
    actions = [a for a in parser._actions if a.dest == "cmd"]
    assert actions, "expected a subcommand action"
    names = set(actions[0].choices)
    assert {"gate", "drive", "run", "eval", "score", "report"} <= names


def test_gate_offline_exits_zero():
    assert main(["gate", SMOKE_CASE, "--offline"]) == 0


def test_gate_with_probe_exits_zero():
    assert main(["gate", SMOKE_CASE]) == 0


def test_drive_dry_run_exits_zero(tmp_path, capsys):
    rc = main(["drive", SMOKE_CASE, "--dest", str(tmp_path / "ws")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[dry_run]" in out
    assert "Qwen/Qwen3.5-0.8B" in out


def test_run_gates_then_drives(tmp_path, capsys):
    rc = main(["run", SMOKE_CASE, "--offline", "--dest", str(tmp_path / "ws")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ELIGIBLE" in out
    assert "[dry_run]" in out


def test_eval_stub_reports_not_yet():
    assert main(["eval", SMOKE_CASE]) == 2
