"""Hosted admission must read `adapter_config.json` under the rules the engine will apply.

`adapter_artifact_metadata` is the last check before a hosted deployment allocates provider
resources, and it read the config permissively: a plain `json.load` for the rank, and nothing else.
Every other field the GPU container validates in `_validate_adapter_config` -- `peft_type`,
`task_type`, `modules_to_save`, and above all `base_model_name_or_path` -- went unread, so a config
the container was certain to refuse was admitted at registration and the deployment failed only
after a provider had started billing.

The customer-owned resolver already enforced all of it. These tests pin the hosted path to the same
verdicts, from the same bytes.
"""

from __future__ import annotations

import json
import re
import types

import pytest

from flash.serve.contract.errors import AdapterTensorMissing
from flash.serve.deployment.adapter_check import adapter_artifact_metadata

BASE = "Qwen/Qwen3.5-9B"
REVISION = "a" * 40
SUBFOLDER = "sft/r-1/seed0/adapter"


def _valid_config() -> dict:
    """exactly what flash's own exporter stamps: rank, peft type, and the base it trained on."""
    return {"r": 32, "peft_type": "LORA", "base_model_name_or_path": BASE}


def _install_hub(monkeypatch, tmp_path, *, raw: bytes) -> None:
    """back the two hub reads with real bytes on disk, so the reader sees the real decode path."""
    cfg = tmp_path / "adapter_config.json"
    cfg.write_bytes(raw)

    class _HfApi:
        def list_repo_tree(self, **kwargs):
            prefix = str(kwargs.get("path_in_repo") or "").rstrip("/")
            return [
                types.SimpleNamespace(path=f"{prefix}/adapter_model.safetensors", size=4096),
            ]

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **_kwargs: str(cfg), HfApi=_HfApi),
    )


def _admit(monkeypatch, tmp_path, config, *, expected_base: str = BASE, raw: bytes | None = None):
    _install_hub(monkeypatch, tmp_path, raw=raw if raw is not None else json.dumps(config).encode())
    return adapter_artifact_metadata(
        "org/repo", SUBFOLDER, artifact_revision=REVISION, expected_base_model=expected_base
    )


def test_valid_config_is_admitted_with_its_declared_rank(monkeypatch, tmp_path):
    metadata = _admit(monkeypatch, tmp_path, _valid_config())

    assert metadata.lora_rank == 32
    assert metadata.targets_images is False
    assert len(metadata.artifact_digest) == 64


def test_declared_base_model_mismatch_is_rejected_before_tensor_listing(monkeypatch, tmp_path):
    """The exact failure the container raises as `adapter logical base model does not match`."""
    listed: list[str] = []

    class _HfApi:
        def list_repo_tree(self, **kwargs):
            listed.append(str(kwargs.get("path_in_repo")))
            return []

    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({**_valid_config(), "base_model_name_or_path": "Qwen/Qwen3.8-27B"}))
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **_kwargs: str(cfg), HfApi=_HfApi),
    )

    with pytest.raises(ValueError, match=re.escape("trained against 'Qwen/Qwen3.8-27B'")):
        adapter_artifact_metadata(
            "org/repo", SUBFOLDER, artifact_revision=REVISION, expected_base_model=BASE
        )
    # a doomed config must not even cost the tree listing, let alone a provider allocation.
    assert listed == []


def test_missing_base_model_is_rejected(monkeypatch, tmp_path):
    """Absent is not `no opinion`: the container compares the field, and an absent one cannot match."""
    with pytest.raises(ValueError, match="declares no base_model_name_or_path"):
        _admit(monkeypatch, tmp_path, {"r": 32, "peft_type": "LORA"})


def test_non_lora_peft_type_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="peft_type must be LORA"):
        _admit(monkeypatch, tmp_path, {**_valid_config(), "peft_type": "IA3"})


def test_wrong_task_type_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="task_type must be absent or CAUSAL_LM"):
        _admit(monkeypatch, tmp_path, {**_valid_config(), "task_type": "SEQ_CLS"})


def test_modules_to_save_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="modules_to_save adapters are not supported"):
        _admit(monkeypatch, tmp_path, {**_valid_config(), "modules_to_save": ["lm_head"]})


def test_non_string_revision_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="revision must be a string"):
        _admit(monkeypatch, tmp_path, {**_valid_config(), "revision": 7})


def test_surrounding_whitespace_in_base_model_is_rejected(monkeypatch, tmp_path):
    """Stripping to match would admit a config the container -- which compares raw -- refuses."""
    with pytest.raises(ValueError, match="surrounding whitespace"):
        _admit(monkeypatch, tmp_path, {**_valid_config(), "base_model_name_or_path": f" {BASE} "})


def test_duplicate_key_is_rejected(monkeypatch, tmp_path):
    """Last-value-wins would admit against one rank and load against the other."""
    raw = b'{"r": 8, "r": 32, "peft_type": "LORA", "base_model_name_or_path": "%s"}' % BASE.encode()

    with pytest.raises(ValueError, match="duplicate key"):
        _admit(monkeypatch, tmp_path, None, raw=raw)


def test_non_finite_constant_is_rejected(monkeypatch, tmp_path):
    raw = b'{"r": NaN, "peft_type": "LORA", "base_model_name_or_path": "%s"}' % BASE.encode()

    with pytest.raises(ValueError, match="not readable json"):
        _admit(monkeypatch, tmp_path, None, raw=raw)


def test_utf16_config_is_rejected(monkeypatch, tmp_path):
    """`json.load` auto-detects utf-16 per rfc 4627; the container's strict decode does not."""
    raw = json.dumps(_valid_config()).encode("utf-16")

    with pytest.raises(ValueError, match="not readable json"):
        _admit(monkeypatch, tmp_path, None, raw=raw)


def test_json_array_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="must be a json object"):
        _admit(monkeypatch, tmp_path, None, raw=b"[]")


def test_config_is_validated_before_tensors_are_listed(monkeypatch, tmp_path):
    """Ordering is the whole point: the cheap local verdict must precede the remote tree walk."""
    listed: list[str] = []

    class _HfApi:
        def list_repo_tree(self, **kwargs):
            listed.append(str(kwargs.get("path_in_repo")))
            return []

    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"r": 32, "peft_type": "IA3"}))
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **_kwargs: str(cfg), HfApi=_HfApi),
    )

    with pytest.raises(ValueError, match="peft_type must be LORA"):
        adapter_artifact_metadata(
            "org/repo", SUBFOLDER, artifact_revision=REVISION, expected_base_model=BASE
        )
    assert listed == []


def test_valid_config_still_reaches_the_tensor_check(monkeypatch, tmp_path):
    """The new gates must not swallow the tensor verification that already existed."""

    class _HfApi:
        def list_repo_tree(self, **kwargs):
            prefix = str(kwargs.get("path_in_repo") or "").rstrip("/")
            return [types.SimpleNamespace(path=f"{prefix}/README.md", size=12)]

    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps(_valid_config()))
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **_kwargs: str(cfg), HfApi=_HfApi),
    )

    with pytest.raises(AdapterTensorMissing, match="no adapter_model tensor file"):
        adapter_artifact_metadata(
            "org/repo", SUBFOLDER, artifact_revision=REVISION, expected_base_model=BASE
        )
