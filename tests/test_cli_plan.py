"""`flash plan` + the `--dry-run` warnings seam (CPU-only, fully offline).

`flash plan` is the pre-flight planner the agent/user runs before paying for a GPU: it parses a
config locally and prints the resolved spec, effective knobs, and config advice. These tests
drive it in-process (like test_cli_commands.py) against catalog models (offline) and assert the
human + JSON surfaces, plus that the linter advice now rides along in `flash train --dry-run`'s
JSON without disturbing the run_id/state/spec contract the agent depends on, and that JSON
stdout stays parseable even when config parsing prints policy warnings.
"""

from __future__ import annotations

import json
from pathlib import Path

from flash.cli.main import main

# A config that trips several linter checks: 9B + rank 8, group_size 2, temp 0, thinking with a
# 320-token budget, GRPO lr 1e-4, eval cadence past the run length.
_BAD_CONFIG = (
    'model = "Qwen/Qwen3.5-9B"\n'
    'algorithm = "grpo"\n'
    "thinking = true\n"
    "[environment]\n"
    'id = "primeintellect/gsm8k"\n'
    "[train]\n"
    'hf_repo = "owner/runs"\n'
    "steps = 50\n"
    "lora_rank = 8\n"
    "group_size = 2\n"
    "temperature = 0\n"
    "max_tokens = 320\n"
    "learning_rate = 1e-4\n"
    "eval_every_steps = 200\n"
    "[gpu]\n"
    'type = "RTX 5090"\n'
)

# A healthy config: defaults everywhere the worker fills in, sane rank for a 4B model.
_CLEAN_CONFIG = (
    'model = "Qwen/Qwen3.5-4B"\n'
    'algorithm = "grpo"\n'
    "[environment]\n"
    'id = "primeintellect/gsm8k"\n'
    "[train]\n"
    'hf_repo = "owner/runs"\n'
    "steps = 150\n"
    "lora_rank = 32\n"
    "[gpu]\n"
    'type = "RTX 5090"\n'
)

# An open-model ("allow" policy) config: parsing it prints policy warnings to stdout
# (a thinking-on-unknown-template note, from flash.schema). Used to prove --json stdout
# stays pure JSON.
_OPEN_MODEL_CONFIG = (
    'model = "acme/mystery-2b"\n'
    'algorithm = "grpo"\n'
    'model_policy = "allow"\n'
    "thinking = true\n"
    "[environment]\n"
    'id = "owner/env"\n'
    "[train]\n"
    'hf_repo = "owner/runs"\n'
    "steps = 50\n"
    "[gpu]\n"
    'type = "RTX 5090"\n'
    "allow_unvalidated = true\n"
)

# A second open-model config whose warning comes from the OTHER stdout source: the VRAM-fit
# estimate in flash.catalog._resolve_open_model. A 6B model on a 32 GB RTX 5090 (GRPO) is a
# "tight" fit, so resolve_model prints a VRAM-policy warning (no thinking note here —
# thinking is off — so this isolates the catalog warning path the schema-note config doesn't
# exercise). Both JSON commands must keep this warning off stdout too.
_OPEN_MODEL_TIGHT_VRAM_CONFIG = (
    'model = "acme/bigmystery-6b"\n'
    'algorithm = "grpo"\n'
    'model_policy = "allow"\n'
    "[environment]\n"
    'id = "owner/env"\n'
    "[train]\n"
    'hf_repo = "owner/runs"\n'
    "steps = 50\n"
    "[gpu]\n"
    'type = "RTX 5090"\n'
    "allow_unvalidated = true\n"
)


def _write(tmp_path: Path, body: str) -> str:
    cfg = tmp_path / "config.toml"
    cfg.write_text(body)
    return str(cfg)


def test_plan_human_output_lists_identity_and_advice(tmp_path: Path, capsys) -> None:
    rc = main(["plan", _write(tmp_path, _BAD_CONFIG)])
    assert rc == 0
    out = capsys.readouterr().out
    # identity + effective knobs are reported
    assert "Qwen/Qwen3.5-9B" in out
    assert "grpo" in out
    assert "effective training knobs" in out
    assert "lora_rank" in out
    # advice section is present and names the offending knobs
    assert "advice (" in out
    assert "group_size" in out
    assert "temperature" in out


def test_plan_clean_config_reports_no_advice(tmp_path: Path, capsys) -> None:
    rc = main(["plan", _write(tmp_path, _CLEAN_CONFIG)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "advice: none" in out


def test_plan_json_output_is_machine_readable(tmp_path: Path, capsys) -> None:
    rc = main(["plan", _write(tmp_path, _BAD_CONFIG), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "plan"
    assert payload["model"]["id"] == "Qwen/Qwen3.5-9B"
    assert payload["algorithm"] == "grpo"
    assert payload["effective_knobs"]  # non-empty
    fields = {w["field"] for w in payload["warnings"]}
    assert {"train.lora_rank", "train.group_size", "train.temperature", "train.max_tokens"} <= fields


def test_plan_effective_knobs_tag_set_vs_default(tmp_path: Path, capsys) -> None:
    # group_size unset -> shows the recipe default tagged "default"; lora_rank set -> "set".
    main(["plan", _write(tmp_path, _CLEAN_CONFIG), "--json"])
    payload = json.loads(capsys.readouterr().out)
    knobs = {k["knob"]: k for k in payload["effective_knobs"]}
    assert knobs["group_size"]["source"] == "default"
    assert knobs["group_size"]["value"] == 8  # RECIPE.rl.group_size
    assert knobs["lora_rank"]["source"] == "set"
    assert knobs["lora_rank"]["value"] == 32


def test_plan_overrides_apply(tmp_path: Path, capsys) -> None:
    # --set flows through spec_from_file, so a bad override surfaces advice on a clean config.
    rc = main(["plan", _write(tmp_path, _CLEAN_CONFIG), "--set", "train.group_size=2", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "train.group_size" in {w["field"] for w in payload["warnings"]}


def test_plan_json_stdout_is_pure_despite_policy_warnings(tmp_path: Path, capsys, monkeypatch) -> None:
    # The open-model "allow" policy + thinking=true makes config parsing print policy warnings.
    # Those must land on stderr, not corrupt the --json stdout payload.
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id: 2.0, raising=True)
    rc = main(["plan", _write(tmp_path, _OPEN_MODEL_CONFIG), "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)  # raises if a warning line leaked onto stdout
    assert payload["state"] == "plan"
    # the policy warning text went to stderr, not the JSON stdout
    assert "open-model policy" not in captured.out
    assert "open-model policy" in captured.err


def test_dry_run_json_carries_warnings_and_keeps_contract(tmp_path: Path, capsys) -> None:
    rc = main(["train", _write(tmp_path, _BAD_CONFIG), "--dry-run"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # the agent contract: run_id / state / spec are untouched
    assert payload["state"] == "dry_run"
    assert isinstance(payload["run_id"], str)
    assert payload["run_id"]
    assert "spec" in payload
    # additive: advice now rides along
    assert "warnings" in payload
    assert {"train.group_size", "train.temperature"} <= {w["field"] for w in payload["warnings"]}


def test_dry_run_clean_config_has_empty_warnings(tmp_path: Path, capsys) -> None:
    main(["train", _write(tmp_path, _CLEAN_CONFIG), "--dry-run"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["warnings"] == []


def test_dry_run_json_pure_despite_policy_warnings(tmp_path: Path, capsys, monkeypatch) -> None:
    # Same stdout-purity guarantee for `flash train --dry-run` (always JSON).
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id: 2.0, raising=True)
    rc = main(["train", _write(tmp_path, _OPEN_MODEL_CONFIG), "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["state"] == "dry_run"
    assert "open-model policy" not in captured.out
    assert "open-model policy" in captured.err


def test_plan_json_pure_despite_vram_fit_warning(tmp_path: Path, capsys, monkeypatch) -> None:
    # The other stdout-warning source: flash.catalog._resolve_open_model prints a VRAM-fit
    # warning on a "tight" open-model estimate (here a 6B model on a 32 GB RTX 5090). It must
    # land on stderr, leaving the --json stdout payload a clean, parseable JSON document.
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id: 6.0, raising=True)
    rc = main(["plan", _write(tmp_path, _OPEN_MODEL_TIGHT_VRAM_CONFIG), "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)  # raises if the VRAM warning leaked onto stdout
    assert payload["state"] == "plan"
    assert "open-model policy" not in captured.out
    # the VRAM-fit warning ("tight") went to stderr, not the JSON stdout
    assert "open-model policy" in captured.err
    assert "tight" in captured.err


def test_dry_run_json_pure_despite_vram_fit_warning(tmp_path: Path, capsys, monkeypatch) -> None:
    # Same VRAM-fit warning source through `flash train --dry-run` (always JSON): a "tight"
    # open-model estimate must not corrupt the dry-run JSON the agent parses from stdout.
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id: 6.0, raising=True)
    rc = main(["train", _write(tmp_path, _OPEN_MODEL_TIGHT_VRAM_CONFIG), "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["state"] == "dry_run"
    assert isinstance(payload["run_id"], str)
    assert payload["run_id"]
    assert "open-model policy" not in captured.out
    assert "open-model policy" in captured.err
