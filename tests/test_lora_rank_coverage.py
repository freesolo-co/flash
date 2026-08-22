"""Hermetic branch coverage for adapter identity and metadata loading failures."""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

import flash.adapters.lora_rank as lora_rank


def test_adapter_config_path_rejects_an_invalid_storage_reference(monkeypatch) -> None:
    """Invalid warm-start references must fail before constructing a remote config path."""
    monkeypatch.setattr(lora_rank, "resolve_adapter_ref", lambda ref: None)

    with pytest.raises(ValueError, match="could not be resolved"):
        lora_rank.adapter_config_path_from_ref("invalid")


def test_file_identity_accepts_attribute_style_lfs_metadata() -> None:
    """Hugging Face attribute-style LFS metadata must produce a stable immutable identity."""
    info = SimpleNamespace(
        lfs=SimpleNamespace(sha256="sha256:weight", oid=None, size=321),
        blob_id=None,
        size=None,
    )

    assert lora_rank._file_identity(info) == "sha256:weight:321"


def test_file_identity_rejects_metadata_without_immutable_content() -> None:
    """Weight metadata without an LFS oid or blob id must never be treated as pinned."""
    info = SimpleNamespace(lfs=None, blob_id=None, size=12)

    with pytest.raises(ValueError, match="no immutable content identity"):
        lora_rank._file_identity(info)


@pytest.mark.parametrize("mode", ["api-error", "invalid-sha"])
def test_resolve_hf_dataset_revision_fails_closed(monkeypatch, mode) -> None:
    """Dataset revision pinning must reject both API failures and mutable revision values."""
    import huggingface_hub

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def repo_info(self, repo, repo_type):
            if mode == "api-error":
                raise RuntimeError("hub unavailable")
            return SimpleNamespace(sha="main")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    message = "could not pin" if mode == "api-error" else "immutable commit SHA"
    with pytest.raises(ValueError, match=message):
        lora_rank.resolve_hf_dataset_revision("owner/data", token="token")


def test_typed_canonical_tree_encodes_all_json_scalar_and_array_types() -> None:
    """Decimal-aware hashing must preserve scalar and array type distinctions."""
    assert lora_rank._typed_canonical_tree(None) == ["null"]
    assert lora_rank._typed_canonical_tree(True) == ["bool", True]
    assert lora_rank._typed_canonical_tree(3) == ["int", "3"]
    assert lora_rank._typed_canonical_tree(1.5) == ["float", 1.5]
    assert lora_rank._typed_canonical_tree([None, Decimal("1.0")]) == [
        "array",
        [["null"], ["decimal", "1.0"]],
    ]

    with pytest.raises(TypeError, match="not JSON serializable"):
        lora_rank._typed_canonical_tree({1, 2})


def test_adapter_artifact_identity_rejects_invalid_reference(monkeypatch) -> None:
    """Artifact identity must reject invalid references before querying Hugging Face."""
    monkeypatch.setattr(lora_rank, "resolve_adapter_ref", lambda ref: None)

    with pytest.raises(ValueError, match="reference is invalid"):
        lora_rank.adapter_artifact_identity("invalid", {})


@pytest.mark.parametrize("mode", ["api-error", "no-weight"])
def test_adapter_artifact_identity_fails_closed_on_remote_metadata(monkeypatch, mode) -> None:
    """Artifact identity must reject unavailable metadata and repositories without required weights."""
    import huggingface_hub

    monkeypatch.setattr(lora_rank, "resolve_adapter_ref", lambda ref: ("owner/runs", "sft/run"))

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_tree(self, *args, **kwargs):
            if mode == "api-error":
                raise RuntimeError("hub unavailable")
            return [SimpleNamespace(path="sft/run/adapter/README.md")]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    message = "could not verify" if mode == "api-error" else "no required weight file"
    with pytest.raises(ValueError, match=message):
        lora_rank.adapter_artifact_identity("owner/runs:sft/run", {"r": 8})


@pytest.mark.parametrize(
    "function",
    [
        lora_rank.rank_from_adapter_config,
        lora_rank.alpha_from_adapter_config,
    ],
)
def test_rank_and_alpha_reject_non_mapping_configs(function) -> None:
    """Rank metadata helpers must reject top-level JSON arrays with a source-specific error."""
    with pytest.raises(ValueError, match="is not a JSON object"):
        function([], source="adapter")


def test_inspect_adapter_config_rejects_non_mapping_config() -> None:
    """Full adapter inspection must reject non-object JSON before reading PEFT fields."""
    with pytest.raises(ValueError, match="is not a JSON object"):
        lora_rank.inspect_adapter_config([], source="adapter", target_model="model")


@pytest.mark.parametrize("mode", ["download", "invalid-json", "non-object"])
def test_load_hf_adapter_config_reports_remote_and_json_failures(
    tmp_path, monkeypatch, mode
) -> None:
    """Adapter config loading must classify download, parse, and top-level shape failures."""
    import huggingface_hub

    monkeypatch.setattr(
        lora_rank,
        "adapter_config_path_from_ref",
        lambda ref: ("owner/runs", "sft/run/adapter/adapter_config.json"),
    )
    path = tmp_path / "adapter_config.json"
    if mode == "invalid-json":
        path.write_text("{broken", encoding="utf-8")
    elif mode == "non-object":
        path.write_text(json.dumps([1, 2]), encoding="utf-8")

    def download(**kwargs):
        if mode == "download":
            raise RuntimeError("hub unavailable")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    message = {
        "download": "failed to read",
        "invalid-json": "invalid JSON",
        "non-object": "is not a JSON object",
    }[mode]
    with pytest.raises(ValueError, match=message):
        lora_rank.load_hf_adapter_config("owner/runs:sft/run")


def test_preflight_without_warm_start_returns_none() -> None:
    """Runs without an adapter warm start must skip all remote metadata inspection."""
    spec = SimpleNamespace(
        train=SimpleNamespace(init_from_adapter="   ", init_from_adapter_revision=""),
        model="model",
    )

    assert lora_rank.preflight_init_adapter_lora_rank(spec) is None


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1"])
def test_strict_declared_ranks_reject_invalid_scalar_r(value) -> None:
    with pytest.raises(ValueError, match="r must be a positive integer"):
        lora_rank.strict_declared_lora_ranks({"r": value})


@pytest.mark.parametrize("value", [None, [], "q_proj"])
def test_strict_declared_ranks_reject_invalid_pattern_container(value) -> None:
    with pytest.raises(ValueError, match="rank_pattern must be an object"):
        lora_rank.strict_declared_lora_ranks({"r": 1, "rank_pattern": value})


@pytest.mark.parametrize("key", ["", " ", 1])
def test_strict_declared_ranks_reject_invalid_pattern_key(key) -> None:
    with pytest.raises(ValueError, match="keys must be non-empty strings"):
        lora_rank.strict_declared_lora_ranks({"r": 1, "rank_pattern": {key: 2}})


@pytest.mark.parametrize("value", [0, -1, True, 2.0, "2"])
def test_strict_declared_ranks_reject_invalid_pattern_value(value) -> None:
    with pytest.raises(ValueError, match="values must be positive integers"):
        lora_rank.strict_declared_lora_ranks({"r": 1, "rank_pattern": {"q_proj": value}})


def test_strict_declared_ranks_reject_malformed_pattern_regex() -> None:
    with pytest.raises(ValueError, match="invalid regex"):
        lora_rank.strict_declared_lora_ranks({"r": 1, "rank_pattern": {"(": 2}})


def test_strict_declared_ranks_preserve_ordered_first_match_resolution() -> None:
    declared = lora_rank.strict_declared_lora_ranks(
        {"r": 1, "rank_pattern": {"q_proj": 2, ".*q_proj": 4}}
    )

    assert lora_rank._rank_for_module("model.layers.0.self_attn.q_proj", declared) == 2


def test_strict_declared_ranks_use_scalar_for_unmatched_valid_override() -> None:
    declared = lora_rank.strict_declared_lora_ranks({"r": 3, "rank_pattern": {"v_proj": 5}})

    assert lora_rank._rank_for_module("model.layers.0.self_attn.q_proj", declared) == 3
