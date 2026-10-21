from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from flash.serve.contract.provenance import immutable_binding_projection
from flash.serving.src.io.schemas import PersistedAdapterRecord
from flash.serving.src.store.persistence import _row_to_adapter
from scripts.migrate_checkpoint_fingerprints import fingerprint_change
from tests.serving.test_persistence import _record, _row


def _legacy_row(**overrides):
    binding = immutable_binding_projection(_record())
    binding["adapter_id"] = binding.pop("model_id")
    old = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _row(artifact_fingerprint=old, **overrides)


def test_migration_changes_only_the_fingerprint_encoding() -> None:
    row = _legacy_row()
    before = row.copy()
    old, new = fingerprint_change(row)
    assert old == row["artifact_fingerprint"]
    assert new == _record().artifact_fingerprint
    assert old != new
    assert row == before
    migrated = {**row, "artifact_fingerprint": new}
    assert fingerprint_change(migrated) is None
    assert (
        PersistedAdapterRecord.model_validate(_row_to_adapter(migrated)).model_id
        == row["checkpoint_id"]
    )


def test_runtime_rejects_unmigrated_fingerprint() -> None:
    with pytest.raises(ValidationError, match="checkpoint-field cutover"):
        PersistedAdapterRecord.model_validate(_row_to_adapter(_legacy_row()))


def test_migration_rejects_corrupted_or_changed_binding() -> None:
    with pytest.raises(ValueError, match="migration refused"):
        fingerprint_change(_row(artifact_fingerprint="f" * 64))
    with pytest.raises(ValueError, match="migration refused"):
        fingerprint_change(_legacy_row(source_revision="c" * 40))


def test_migration_rejects_inconsistent_checkpoint_identity() -> None:
    with pytest.raises(RuntimeError, match="inconsistent checkpoint identity"):
        fingerprint_change(_legacy_row(checkpoint_id="other/step-20"))
