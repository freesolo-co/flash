import json
import subprocess
import sys
from pathlib import Path

import pytest

from flash.benchmarks.evidence import load_events, seal, verify, write_json
from flash.benchmarks.runner import _execute, _read_trial, replay, run_suite, suite_summary
from flash.benchmarks.schema import Case, Suite

ROOT = Path(__file__).resolve().parents[2]


def test_full_offline_matrix_replays_and_records_cold_warm_pairs(tmp_path):
    suite = Suite.model_validate_json(
        (ROOT / "examples/benchmarks/quality-economics.json").read_text()
    )
    suite.seeds = [17]
    output = tmp_path / "results"
    result = run_suite(suite, output, ROOT, smoke=True)
    assert result["all_trials_valid"]
    assert not result["production_claims_eligible"]
    assert not result["missing_workloads"]
    assert len(result["trials"]) == 8
    assert replay(output) == result
    for case in suite.cases:
        cold = output / f"{case.name}-17-cold"
        warm_request = json.loads((output / f"{case.name}-17-warm/request.json").read_text())
        assert warm_request["previous_trial"] == str(cold)
        assert (
            warm_request["configuration_digest"]
            != json.loads((cold / "request.json").read_text())["configuration_digest"]
        )
    assert (output / "source.tar").stat().st_size > 0


@pytest.mark.parametrize("mutation", ["edit", "add", "remove", "symlink"])
def test_artifact_verification_rejects_changed_set_or_bytes(tmp_path, mutation):
    write_json(tmp_path / "input.json", {"one": 1})
    seal(tmp_path)
    if mutation == "edit":
        write_json(tmp_path / "input.json", {"one": 2})
    elif mutation == "remove":
        (tmp_path / "input.json").unlink()
    elif mutation == "add":
        (tmp_path / "new.txt").write_text("extra")
    else:
        (tmp_path / "link").symlink_to(tmp_path / "input.json")
    with pytest.raises(ValueError, match=r"differs|symlink"):
        verify(tmp_path)


def test_live_requires_clean_source_before_running_a_driver(tmp_path, case):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.email=test@example.org",
            "-c",
            "user.name=test",
            "commit",
            "--allow-empty",
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
    )
    (source / "untracked.py").write_text("raise RuntimeError('must not run')")
    with pytest.raises(ValueError, match="clean source checkout"):
        run_suite(Suite(name="test", seeds=[17], cases=[case]), tmp_path / "results", source)


def test_timeout_fails_and_retains_logs(tmp_path):
    result = _execute(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
        tmp_path,
        ROOT,
        0.5,
        0.1,
    )
    assert result["timed_out"]
    assert result["returncode"] != 0
    assert (tmp_path / "stdout.log").read_text().strip() == "started"


def test_invalid_or_truncated_evidence_is_a_retained_failed_trial(tmp_path, case, request_data):
    write_json(tmp_path / "request.json", request_data)
    write_json(tmp_path / "execution.json", {"returncode": 0, "timed_out": False, "elapsed_s": 1.0})
    (tmp_path / "events.jsonl").write_text('{"kind":"quality"')
    result = _read_trial(tmp_path, case, False)
    assert not result["valid"]
    assert "invalid_evidence" in result["issues"][0]


def test_missed_targets_stay_in_aggregate_denominator(case):
    rows = [
        {
            "case": case.name,
            "cache_requested": "cold",
            "valid": True,
            "target_reached": True,
            "time_to_target_s": 10.0,
        },
        {"case": case.name, "cache_requested": "cold", "valid": True, "target_reached": False},
    ]
    result = suite_summary(Suite(name="test", cases=[case]), rows, smoke=False)
    assert result["groups"][0]["target_attainment_fraction"] == 0.5
    assert result["groups"][0]["median_time_to_target_s_among_reached"] == 10.0
    assert not result["production_claims_eligible"]


def test_pin_and_case_validation(case):
    raw = case.model_dump()
    raw["revisions"]["image"] = "ghcr.io/example:latest"
    with pytest.raises(ValueError, match="image"):
        Case.model_validate(raw)
    with pytest.raises(ValueError, match="case names"):
        Suite(name="test", cases=[case, case])
    with pytest.raises(ValueError, match="seeds must be unique"):
        Suite(name="test", seeds=[17, 17], cases=[case])


def test_empty_or_non_json_event_lines_do_not_silently_disappear(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="invalid event"):
        load_events(path)
