"""One-time, offline migration from adapter_id to model_id fingerprint inputs.

Run from this checkout after installing scripts/checkpoint-migration-requirements.txt.
Requires FLASH_CHECKPOINT_DATABASE_URL and defaults to a read-only dry run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

from flash.serve.contract.provenance import (
    immutable_binding_fingerprint,
    immutable_binding_projection,
)
from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.store.persistence import _row_to_adapter


def fingerprint_change(row: dict[str, Any]) -> tuple[str, str] | None:
    """Validate the full binding before returning its old and canonical fingerprints."""
    record = AdapterRecord.model_validate(_row_to_adapter(row))
    canonical = immutable_binding_fingerprint(record)
    if record.artifact_fingerprint == canonical:
        return None
    # Legacy encoding exists only in this offline migration, never in serving acceptance.
    legacy = immutable_binding_projection(record)
    legacy["adapter_id"] = legacy.pop("model_id")
    previous = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if record.artifact_fingerprint != previous:
        raise ValueError("stored fingerprint does not match either encoding; migration refused")
    return previous, canonical


def migrate(connection: Any, *, apply: bool = False) -> int:
    """Change binding fingerprints atomically; never rewrite historical usage evidence."""
    with connection.transaction(), connection.cursor() as cursor:
        if apply:
            cursor.execute("SET LOCAL lock_timeout = '5s'")
            cursor.execute(
                "LOCK TABLE internal.serving_usage_outbox, flash.hosted_lora_adapters "
                "IN ACCESS EXCLUSIVE MODE"
            )
        else:
            cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute("SELECT row_to_json(binding) FROM flash.hosted_lora_adapters AS binding")
        rows = [item[0] for item in cursor.fetchall()]
        changes = [(row, change) for row in rows if (change := fingerprint_change(row)) is not None]
        if not apply or not changes:
            return len(changes)
        if any(row["status"] != "disabled" for row in rows):
            raise ValueError("disable all hosted checkpoints before migrating fingerprints")
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM internal.serving_usage_outbox WHERE state <> 'delivered')"
        )
        if cursor.fetchone()[0]:
            raise ValueError("resolve and drain all usage outbox events before migrating")
        # The table lock excludes concurrent writers. Rollback restores this transactional DDL
        # if any statement fails; the immutable binding trigger is never left disabled.
        cursor.execute(
            "ALTER TABLE flash.hosted_lora_adapters "
            "DISABLE TRIGGER hosted_lora_adapters_immutable_update"
        )
        for row, (previous, canonical) in changes:
            cursor.execute(
                "UPDATE flash.hosted_lora_adapters "
                "SET artifact_fingerprint = %s, deployment_generation = NULL, "
                "url = NULL, updated_at = clock_timestamp() "
                "WHERE org_id = %s AND checkpoint_id = %s AND artifact_fingerprint = %s",
                (canonical, row["org_id"], row["checkpoint_id"], previous),
            )
            if cursor.rowcount != 1:
                raise ValueError("checkpoint binding changed during migration")
        cursor.execute(
            "ALTER TABLE flash.hosted_lora_adapters "
            "ENABLE TRIGGER hosted_lora_adapters_immutable_update"
        )
        return len(changes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="apply during the offline cutover")
    args = parser.parse_args()
    database_url = os.environ.get("FLASH_CHECKPOINT_DATABASE_URL")
    if not database_url:
        parser.error("FLASH_CHECKPOINT_DATABASE_URL is required")
    import psycopg

    with psycopg.connect(database_url) as connection:
        count = migrate(connection, apply=args.apply)
    print(f"{'Migrated' if args.apply else 'Would migrate'} {count} checkpoint fingerprints")


if __name__ == "__main__":
    main()
