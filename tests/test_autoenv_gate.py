"""Eligibility gate — runs fully offline on the bundled smoke case (local data, catalog model)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import autoenv
from autoenv.gate import gate_case
from autoenv.gate.model_match import params_b_from_name, resolve_flash_model
from autoenv.manifest import PaperCase

SMOKE_CASE = Path(autoenv.__file__).parent / "cases" / "arithmetic_smoke_sft.toml"


def test_smoke_case_is_eligible_with_dataset_probe():
    # The bundled dataset is local, so the probe runs offline.
    report = gate_case(PaperCase.load(SMOKE_CASE), probe_dataset=True)
    assert report.eligible, report.summary()
    assert report.flash_model == "Qwen/Qwen3.5-0.8B"
    assert not report.model_substituted
    assert report.estimated_usd is not None
    names = {r.name for r in report.rules}
    assert {
        "model_supported",
        "text_only",
        "public_dataset",
        "automatable_metric",
        "cost_within_budget",
    } <= names


def test_offline_skips_dataset_probe_but_stays_eligible():
    report = gate_case(PaperCase.load(SMOKE_CASE), probe_dataset=False)
    assert report.eligible
    probe = next(r for r in report.rules if r.name == "public_dataset")
    assert probe.status == "skipped"


def test_vlm_task_is_ineligible():
    case = dataclasses.replace(
        PaperCase.load(SMOKE_CASE),
        goal="Answer questions about the input image and describe the visual scene.",
    )
    report = gate_case(case, probe_dataset=False)
    assert not report.eligible
    text_rule = next(r for r in report.rules if r.name == "text_only")
    assert not text_rule.passed


def test_unknown_metric_is_ineligible():
    case = dataclasses.replace(
        PaperCase.load(SMOKE_CASE),
        metric=dataclasses.replace(PaperCase.load(SMOKE_CASE).metric, name="bleu_made_up"),
    )
    report = gate_case(case, probe_dataset=False)
    assert not report.eligible


def test_tiny_budget_fails_cost_gate():
    case = dataclasses.replace(PaperCase.load(SMOKE_CASE), max_usd=0.0001)
    report = gate_case(case, probe_dataset=False)
    cost_rule = next(r for r in report.rules if r.name == "cost_within_budget")
    assert not cost_rule.passed
    assert not report.eligible


def test_params_b_from_name():
    assert params_b_from_name("Qwen2.5-1.5B-Instruct") == 1.5
    assert params_b_from_name("Llama-3.2-3B") == 3.0
    assert params_b_from_name("pythia-410m") == 0.41
    assert params_b_from_name("no-size-here") is None


def test_noncatalog_model_substitutes_to_nearest_size():
    match = resolve_flash_model("SomeOrg/Mystery-7B", "Mystery-7B", "grpo")
    assert match.substituted
    assert match.model_id in {"Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"}  # nearest catalog sizes to 7B


def test_catalog_model_is_exact():
    match = resolve_flash_model("Qwen/Qwen3.5-0.8B", "Qwen2.5-0.5B", "sft")
    assert match.exact
    assert not match.substituted
