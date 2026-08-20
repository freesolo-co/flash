"""`resolve_adapter` must read the adapter's own config, not trust the caller's flags.

The resolver downloads `adapter_config.json` to digest it, then used to discard it and write the
caller-supplied `--lora-rank` / `--model` straight into the immutable manifest. Every one of these
mismatches was therefore caught only by `_validate_adapter_config` *inside the paid GPU container*,
so `--dry-run` reported success and a real deployment could leave billable resources in a failed or
outcome-unknown state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flash.serve import resolve as resolve_module
from flash.serve.resolve import ADAPTER_CONFIG, ADAPTER_WEIGHTS, ResolveError, resolve_adapter

BASE = "Qwen/Qwen3.5-4B"
BASE_REVISION = "b" * 40
ARTIFACT_REVISION = "a" * 40


def _install_hub(monkeypatch, tmp_path: Path, config: dict) -> None:
    """Stand up the two hub reads the resolver makes, backed by real files on disk."""
    (tmp_path / ADAPTER_CONFIG).write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / ADAPTER_WEIGHTS).write_bytes(b"weights-bytes")

    class _Info:
        sha = ARTIFACT_REVISION

    class _Api:
        def repo_info(self, **_kwargs):
            return _Info()

    def _download(*, filename: str, **_kwargs) -> str:
        return str(tmp_path / Path(filename).name)

    monkeypatch.setattr(resolve_module, "_hub_api", lambda: _Api())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)


def _resolve(**overrides):
    kwargs = {
        "run_id": "run1",
        "artifact_repo_id": "Freesolo-Co/artifacts",
        "artifact_subfolder": "rl/run1/seed0/adapter",
        "base_model": BASE,
        "base_model_revision": BASE_REVISION,
        "lora_rank": 32,
    }
    kwargs.update(overrides)
    return resolve_adapter(**kwargs)


def test_a_rank_that_disagrees_with_the_config_is_rejected(monkeypatch, tmp_path) -> None:
    _install_hub(monkeypatch, tmp_path, {"peft_type": "LORA", "r": 16})

    with pytest.raises(ResolveError, match="disagrees"):
        _resolve(lora_rank=32)


def test_a_base_model_that_disagrees_with_the_config_is_rejected(monkeypatch, tmp_path) -> None:
    _install_hub(
        monkeypatch,
        tmp_path,
        {"peft_type": "LORA", "r": 32, "base_model_name_or_path": "Qwen/Qwen3.5-9B"},
    )

    with pytest.raises(ResolveError, match="trained against"):
        _resolve(base_model=BASE)


def test_the_manifest_binds_the_revision_the_adapter_was_trained_against(
    monkeypatch, tmp_path
) -> None:
    # the model repo is mutable, so resolving its tip at deploy time can pair the adapter with
    # weights it never saw. the config's own revision wins.
    trained_against = "c" * 40
    _install_hub(
        monkeypatch,
        tmp_path,
        {
            "peft_type": "LORA",
            "r": 32,
            "base_model_name_or_path": BASE,
            "revision": trained_against,
        },
    )

    resolved = _resolve(base_model_revision=BASE_REVISION)
    assert resolved.adapter.base_model_revision == trained_against


def test_an_agreeing_config_resolves(monkeypatch, tmp_path) -> None:
    _install_hub(
        monkeypatch,
        tmp_path,
        {"peft_type": "LORA", "r": 32, "base_model_name_or_path": BASE},
    )

    resolved = _resolve()
    assert resolved.adapter.lora_rank == 32
    assert resolved.adapter.base_model == BASE
    assert resolved.adapter.artifact_revision == ARTIFACT_REVISION


def test_an_unreadable_config_is_a_resolve_error(monkeypatch, tmp_path) -> None:
    (tmp_path / ADAPTER_CONFIG).write_text("{not json", encoding="utf-8")
    (tmp_path / ADAPTER_WEIGHTS).write_bytes(b"weights-bytes")

    class _Info:
        sha = ARTIFACT_REVISION

    class _Api:
        def repo_info(self, **_kwargs):
            return _Info()

    monkeypatch.setattr(resolve_module, "_hub_api", lambda: _Api())
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *, filename, **_kwargs: str(tmp_path / Path(filename).name),
    )

    with pytest.raises(ResolveError):
        _resolve()
