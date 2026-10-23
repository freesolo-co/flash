import copy
import json
from pathlib import Path

import pytest

from flash.benchmarks.comparison import _pair, compare_runs
from flash.benchmarks.evidence import seal
from flash.benchmarks.runner import run_suite
from flash.benchmarks.schema import LEDGERS, Suite


def trial(*, reached=True, settled=True):
    return {
        "case": "sft",
        "seed": 17,
        "cache_requested": "cold",
        "valid": True,
        "target_reached": reached,
        "time_to_target_s": 10.0 if reached else None,
        "quality_curve": [{"value": 0.9 if reached else 0.7}],
        "timing": {"wall_s": 20.0},
        "money": {
            ledger: {
                "complete": settled,
                "known_usd": 2.0,
                "total_usd": 2.0 if settled else None,
                "to_target": {"lower_usd": 0.5, "upper_usd": 1.5},
            }
            for ledger in LEDGERS
        },
    }


def test_failed_target_keeps_horizon_comparison_separate():
    left, right = trial(), trial(reached=False)
    right["timing"]["wall_s"] = 5.0
    result = _pair(left, right)
    assert result["horizon_wall_ratio"] == 0.25
    assert result["time_to_target_ratio"] is None
    assert result["candidate_target_reached"] is False
    assert result["money"]["infrastructure_actual"]["candidate_to_target"] is None
    assert result["last_observed_quality_delta"] == pytest.approx(-0.2)


def test_partial_costs_never_become_actual_cost_ratios():
    result = _pair(trial(), trial(settled=False))
    assert result["time_to_target_ratio"] == 1.0
    for ledger in LEDGERS:
        assert result["money"][ledger]["candidate_known_usd"] == 2.0
        assert result["money"][ledger]["total_cost_ratio"] is None
        assert result["money"][ledger]["candidate_to_target"] is None


def test_invalid_trial_cannot_produce_a_speedup():
    right = trial()
    right["valid"] = False
    result = _pair(trial(), right)
    assert result["time_to_target_ratio"] is None
    assert result["horizon_wall_ratio"] is None
    assert result["last_observed_quality_delta"] is None


def test_target_cost_preserves_receipt_bounds():
    result = _pair(trial(), trial())
    assert result["money"]["infrastructure_actual"]["candidate_to_target"] == {
        "lower_usd": 0.5,
        "upper_usd": 1.5,
    }


def test_sealed_comparison_requires_matched_inputs_and_explicit_treatment(tmp_path, case):
    source = Path(__file__).resolve().parents[2]
    left, right = tmp_path / "baseline", tmp_path / "candidate"
    suite = Suite(name="test", seeds=[17], cases=[case])
    run_suite(suite, left, source, smoke=True)
    candidate = suite.model_copy(deep=True)
    candidate.cases[0].configuration["gpu"] = "B200"
    run_suite(candidate, right, source, smoke=True)
    original = json.loads((right / "suite.json").read_text())
    assert compare_runs(left, right, allow_changes=("configuration.gpu",))["synthetic"]
    assert not compare_runs(left, right, allow_changes=("configuration.gpu",))[
        "production_claims_eligible"
    ]

    def replace_suite(value):
        (right / "checksums.json").unlink()
        (right / "suite.json").write_text(json.dumps(value))
        seal(right)

    changed = copy.deepcopy(original)
    changed["cases"][0]["configuration"]["gpu"] = "B200"
    replace_suite(changed)
    with pytest.raises(ValueError, match="unmatched inputs"):
        compare_runs(left, right)
    result = compare_runs(left, right, allow_changes=("configuration.gpu",))
    assert result["changed_inputs"]["sft"] == ["configuration.gpu"]
    assert result["pairs"][0]["both_valid"] is True
    request_path = right / "sft-17-cold/request.json"
    request = json.loads(request_path.read_text())
    request["configuration"]["parameters"]["gpu"] = "H100"
    request_path.write_text(json.dumps(request))
    replace_suite(changed)
    assert not compare_runs(left, right, allow_changes=("configuration.gpu",))["pairs"][0][
        "both_valid"
    ]
    changed["cases"][0]["target"]["value"] = 0.5
    replace_suite(changed)
    with pytest.raises(ValueError, match="cannot allow"):
        compare_runs(left, right, allow_changes=("target.value",))
    with pytest.raises(ValueError, match=r"target\.value"):
        compare_runs(left, right, allow_changes=("configuration.gpu",))
    (right / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        compare_runs(left, right)
