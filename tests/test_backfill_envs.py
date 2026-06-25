"""Backfill of the GitHub env-hub into Azure Blob + Postgres."""

from __future__ import annotations

import io
import tarfile

import pytest

from flash.server import azure_blob, environment_store
from flash.server.scripts import backfill_envs_to_azure as backfill


def _make_hub(tmp_path):
    """A fake environment-hub checkout: two envs + a noise dir without environment.py."""
    hub = tmp_path / "environment-hub"
    (hub / "alice" / "math").mkdir(parents=True)
    (hub / "alice" / "math" / "environment.py").write_text("def load_environment(**k): return None\n")
    (hub / "alice" / "math" / "datasets").mkdir()
    (hub / "alice" / "math" / "datasets" / "train.jsonl").write_text('{"x": 1}\n')
    (hub / "bob" / "code").mkdir(parents=True)
    (hub / "bob" / "code" / "environment.py").write_text("def load_environment(**k): return None\n")
    # A dir without environment.py is skipped.
    (hub / "bob" / "notanenv").mkdir(parents=True)
    (hub / "bob" / "notanenv" / "readme.md").write_text("nope\n")
    # A repo-control dir is skipped.
    (hub / ".git").mkdir()
    return hub


def test_iter_env_dirs_finds_only_real_envs(tmp_path):
    hub = _make_hub(tmp_path)
    found = {slug for slug, *_ in backfill._iter_env_dirs(hub)}
    assert found == {"alice/math", "bob/code"}


def test_package_dir_puts_environment_py_at_root(tmp_path):
    hub = _make_hub(tmp_path)
    data = backfill._package_dir(hub / "alice" / "math")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = set(tar.getnames())
    assert "environment.py" in names
    assert "datasets/train.jsonl" in names


def test_backfill_uploads_and_indexes(tmp_path, monkeypatch):
    hub = _make_hub(tmp_path)
    uploads: dict[str, bytes] = {}
    upserts: list[dict] = []
    monkeypatch.setattr(azure_blob, "upload_package", lambda key, data: uploads.__setitem__(key, data))
    monkeypatch.setattr(azure_blob, "container_name", lambda: "flash-environments")
    monkeypatch.setattr(environment_store, "lookup", lambda slug: None)
    monkeypatch.setattr(environment_store, "upsert", lambda **kw: upserts.append(kw))

    count = backfill.backfill(hub)
    assert count == 2
    assert set(uploads) == {
        "flash-envs/alice/math/package.tar.gz",
        "flash-envs/bob/code/package.tar.gz",
    }
    slugs = {u["slug"] for u in upserts}
    assert slugs == {"alice/math", "bob/code"}
    for u in upserts:
        assert len(u["package_sha256"]) == 64
        assert u["blob_container"] == "flash-environments"


def test_backfill_skips_already_indexed(tmp_path, monkeypatch):
    hub = _make_hub(tmp_path)
    uploads: list[str] = []
    monkeypatch.setattr(azure_blob, "upload_package", lambda key, data: uploads.append(key))
    monkeypatch.setattr(azure_blob, "container_name", lambda: "flash-environments")
    monkeypatch.setattr(environment_store, "upsert", lambda **kw: None)

    def already_indexed(slug):
        return environment_store.EnvironmentRecord(
            slug=slug,
            namespace=slug.split("/")[0],
            name=slug.split("/")[1],
            blob_container="flash-environments",
            blob_key=f"flash-envs/{slug}/package.tar.gz",
            package_sha256="a" * 64,
            size_bytes=1,
            version=1,
        )

    monkeypatch.setattr(environment_store, "lookup", already_indexed)
    count = backfill.backfill(hub)
    assert count == 0
    assert uploads == []  # nothing re-uploaded


def test_backfill_dry_run_writes_nothing(tmp_path, monkeypatch):
    hub = _make_hub(tmp_path)

    def boom(*a, **k):
        raise AssertionError("dry-run must not write")

    monkeypatch.setattr(azure_blob, "upload_package", boom)
    monkeypatch.setattr(environment_store, "upsert", boom)
    monkeypatch.setattr(environment_store, "lookup", lambda slug: None)
    count = backfill.backfill(hub, dry_run=True)
    assert count == 2  # counts what it WOULD upload


def test_backfill_rejects_missing_hub(tmp_path):
    with pytest.raises(SystemExit):
        backfill.backfill(tmp_path / "does-not-exist")
