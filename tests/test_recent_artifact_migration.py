"""The one-time migration is strictly bounded to a frozen three-day inventory."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.artifacts.api import download_file
from scripts.migrate_recent_hf_artifacts import (
    apply_plan,
    make_plan,
    recent_files,
    state_references,
    validate_plan,
)
from tests.test_azure_artifacts import REPO, azure  # noqa: F401

NOW = datetime(2026, 11, 2, tzinfo=UTC)


def entry(path, age, content=b"test"):
    return SimpleNamespace(
        path=path,
        size=len(content),
        blob_id=hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
        lfs=None,
        last_commit=SimpleNamespace(date=NOW - timedelta(days=age)),
    )


def test_cutoff_is_inclusive_and_excludes_old_checkpoints_and_future_files():
    selected = recent_files(
        [
            entry("sft/run/checkpoints/step-1/adapter/config", 4),
            entry("sft/run/checkpoints/step-2/adapter/config", 3),
            entry("sft/run/checkpoints/step-2/adapter/weights", 2.9),
            entry("sft/run/heartbeat.json", 1),
            entry("future", -1),
        ],
        cutoff=NOW - timedelta(days=3),
        end=NOW,
    )
    assert {file["path"] for file in selected} == {
        "sft/run/checkpoints/step-2/adapter/config",
        "sft/run/checkpoints/step-2/adapter/weights",
        "sft/run/heartbeat.json",
    }


def test_checkpoint_upload_crossing_cutoff_is_copied_as_one_unit():
    selected = recent_files(
        [
            entry("sft/run/checkpoints/step-1/adapter/config", 3.01),
            entry("sft/run/checkpoints/step-1/adapter/weights", 2.99),
        ],
        cutoff=NOW - timedelta(days=3),
        end=NOW,
    )
    assert len(selected) == 2


def test_unknown_file_dates_fail_closed():
    item = entry("file", 1)
    item.last_commit = None
    with pytest.raises(ValueError, match="timestamp"):
        recent_files([item], cutoff=NOW - timedelta(days=3), end=NOW)


def plan_for(content=b"test"):
    source = SimpleNamespace(
        list_datasets=lambda **kwargs: [
            SimpleNamespace(id=REPO),
            SimpleNamespace(id="Freesolo-Co/unrelated"),
        ],
        repo_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40),
        list_repo_tree=lambda *args, **kwargs: [entry("sft/run/adapter/weights", 1, content)],
    )
    return make_plan(source, "Freesolo-Co", now=NOW)


def test_plan_freezes_exact_window_and_source_revision():
    plan = plan_for()
    validate_plan(plan)
    assert plan["cutoff"] == "2026-10-30T00:00:00+00:00"
    assert len(plan["snapshots"]) == 1
    assert plan["snapshots"][0]["revision"] == "a" * 40


def test_rejects_expanded_or_tampered_inventory():
    plan = plan_for()
    plan["snapshots"][0]["files"][0]["modified"] = (NOW - timedelta(days=4)).timestamp()
    with pytest.raises(ValueError, match="outside"):
        validate_plan(plan)
    plan["cutoff"] = (NOW - timedelta(days=4)).isoformat()
    with pytest.raises(ValueError, match="exactly three days"):
        validate_plan(plan)


def source_download(content):
    def download(**kwargs):
        assert kwargs["revision"] == "a" * 40
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        return str(path)

    return download


def test_import_preserves_pinned_revision_content_identity_and_is_idempotent(azure, tmp_path):  # noqa: F811
    _, client, store = azure
    plan = plan_for()
    first = apply_plan(plan, store, source_download(b"test"))
    assert apply_plan(plan, store, source_download(b"test")) == first
    path = download_file(REPO, "sft/run/adapter/weights", revision="a" * 40, local_dir=tmp_path)
    assert Path(path).read_bytes() == b"test"
    assert (
        client.list_repo_tree(REPO, recursive=True)[0].blob_id
        == plan["snapshots"][0]["files"][0]["source_blob_id"]
    )


def test_import_does_not_overwrite_an_existing_azure_head(azure):  # noqa: F811
    _, client, store = azure
    revision = client.upload_file(repo_id=REPO, path_in_repo="new", path_or_fileobj=b"new").oid
    apply_plan(plan_for(), store, source_download(b"test"))
    assert store.manifest(REPO)[1] == revision
    assert client.list_repo_files(REPO) == ["new"]


def test_bad_source_bytes_publish_neither_alias_nor_head(azure):  # noqa: F811
    container, _, store = azure
    with pytest.raises(ValueError, match="content identity"):
        apply_plan(plan_for(), store, source_download(b"evil"))
    assert not any("/aliases/" in name or name.endswith("/HEAD") for name in container.blobs)


def test_recent_heartbeat_does_not_hide_an_out_of_window_serving_pin():
    source = SimpleNamespace(
        list_datasets=lambda **kwargs: [],
        repo_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40),
        list_repo_tree=lambda *args, **kwargs: [
            entry("sft/run/heartbeat.json", 1),
            entry("sft/run/checkpoints/step-1/adapter/weights", 4),
        ],
    )
    with pytest.raises(ValueError, match="required artifact path"):
        make_plan(
            source,
            "Freesolo-Co",
            now=NOW,
            references=[
                {
                    "repo": REPO,
                    "revision": "a" * 40,
                    "path": "sft/run/checkpoints/step-1/adapter",
                }
            ],
        )


@pytest.mark.parametrize("terminal", ["done", "deployed", "failed", "cancelled", "dry_run"])
def test_private_run_dependencies_use_effective_spec_and_skip_finished_runs(tmp_path, terminal):
    spec = {
        "train": {
            "hf_repo": REPO,
            "init_from_adapter": f"{REPO}:sft/source/adapter",
            "init_from_adapter_revision": "b" * 40,
        },
        "custom": {
            "resolved_resume": {
                "repo": REPO,
                "revision": "c" * 40,
                "path": "custom/source/checkpoints/step-1",
            }
        },
    }
    record = {
        "updated_at": NOW.timestamp(),
        "state": "queued",
        "spec": {},
        "effective_preparation": {"worker_spec": spec},
        "source_snapshot": {"revision": "a" * 40, "archive_path": "source/code.zip"},
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(record))
    references = state_references(tmp_path, now=NOW)
    assert {ref["revision"] for ref in references} == {"a" * 40, "b" * 40, "c" * 40}
    assert all(ref["path"] for ref in references)
    record["state"] = terminal
    path.write_text(json.dumps(record))
    assert state_references(tmp_path, now=NOW) == []
