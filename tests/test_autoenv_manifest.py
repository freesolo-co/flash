"""PaperCase manifest loading + validation."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import autoenv
from autoenv.manifest import ManifestError, PaperCase

CASES_DIR = Path(autoenv.__file__).parent / "cases"
SMOKE_CASE = CASES_DIR / "arithmetic_smoke_sft.toml"


def test_loads_bundled_smoke_case():
    case = PaperCase.load(SMOKE_CASE)
    assert case.id == "arithmetic-smoke-sft"
    assert case.algorithm == "sft"
    assert case.difficulty == "easy"
    assert case.flash_model == "Qwen/Qwen3.5-0.8B"
    assert case.metric.name == "numeric_match"
    assert case.metric.reported == pytest.approx(0.85)


def test_relative_paths_resolve_against_manifest_dir():
    case = PaperCase.load(SMOKE_CASE)
    train = Path(case.resolved_train())
    eval_ = Path(case.resolved_eval())
    assert train.is_absolute()
    assert train.is_file()
    assert eval_.is_absolute()
    assert eval_.is_file()
    # supplied reward resolved to an absolute existing file
    assert case.supplied_reward_path is not None
    assert Path(case.supplied_reward_path).is_file()


def test_easy_mode_requires_supplied_reward():
    case = PaperCase.load(SMOKE_CASE)
    with pytest.raises(ManifestError, match="supplied_reward_path"):
        dataclasses.replace(case, supplied_reward_path=None)


def test_rejects_unknown_algorithm():
    case = PaperCase.load(SMOKE_CASE)
    with pytest.raises(ManifestError, match="algorithm"):
        dataclasses.replace(case, algorithm="ppo")


def test_rejects_bad_difficulty():
    case = PaperCase.load(SMOKE_CASE)
    with pytest.raises(ManifestError, match="difficulty"):
        dataclasses.replace(case, difficulty="medium")


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('id = "x"\n', encoding="utf-8")
    with pytest.raises(ManifestError):
        PaperCase.load(bad)


def test_remote_refs_pass_through():
    case = PaperCase.load(SMOKE_CASE)
    remote = dataclasses.replace(
        case,
        dataset=dataclasses.replace(case.dataset, train="hf:openai/gsm8k", eval="org/name"),
    )
    assert remote.resolved_train() == "hf:openai/gsm8k"
    assert remote.resolved_eval() == "org/name"
