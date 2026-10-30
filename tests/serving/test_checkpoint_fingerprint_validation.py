from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from flash.serve.contract.provenance import immutable_binding_projection
from flash.serving.src.io.schemas import PersistedAdapterRecord
from flash.serving.src.store.persistence import _row_to_adapter
from tests.serving.test_persistence import _record, _row


def _legacy_row(**overrides):
    binding = immutable_binding_projection(_record())
    binding["adapter_id"] = binding.pop("model_id")
    old = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _row(artifact_fingerprint=old, **overrides)


def test_runtime_accepts_canonical_fingerprint() -> None:
    row = _row()
    assert (
        PersistedAdapterRecord.model_validate(_row_to_adapter(row)).model_id == row["checkpoint_id"]
    )


def test_runtime_rejects_unmigrated_fingerprint() -> None:
    with pytest.raises(ValidationError, match="checkpoint-field cutover"):
        PersistedAdapterRecord.model_validate(_row_to_adapter(_legacy_row()))


def test_runtime_rejects_corrupted_binding() -> None:
    with pytest.raises(ValidationError, match="fingerprint"):
        PersistedAdapterRecord.model_validate(_row_to_adapter(_row(artifact_fingerprint="f" * 64)))


def test_runtime_rejects_inconsistent_checkpoint_identity() -> None:
    with pytest.raises(RuntimeError, match="inconsistent checkpoint identity"):
        _row_to_adapter(_legacy_row(checkpoint_id="other/step-20"))
