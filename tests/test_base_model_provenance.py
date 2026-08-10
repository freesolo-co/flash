from __future__ import annotations

import json

from flash.engine.worker.io import hf


def test_is_commit_sha_accepts_only_full_hex():
    assert hf.is_commit_sha("a" * 40)
    assert hf.is_commit_sha("0123456789abcdef0123456789abcdef01234567")
    assert not hf.is_commit_sha("a" * 39)  # too short
    assert not hf.is_commit_sha("main")  # a ref, not a commit
    assert not hf.is_commit_sha("g" * 40)  # non-hex
    assert not hf.is_commit_sha("")


def test_resolve_cached_model_commit_returns_snapshot_basename(monkeypatch):
    commit = "b" * 40
    seen = {}

    def fake_snapshot_download(*, repo_id, revision, local_files_only, cache_dir):
        seen["revision"] = revision
        seen["local_files_only"] = local_files_only
        return f"/hf-cache/models--org--m/snapshots/{commit}"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert hf.resolve_cached_model_commit("org/m", "main") == commit
    # a mutable ref is passed through to the offline resolver, never the network.
    assert seen["revision"] == "main"
    assert seen["local_files_only"] is True


def test_resolve_cached_model_commit_empty_when_uncached(monkeypatch):
    def fake_snapshot_download(**_kwargs):
        raise FileNotFoundError("not in local cache")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert hf.resolve_cached_model_commit("org/m", "deadbeef") == ""


def test_resolve_cached_model_commit_empty_when_basename_not_a_commit(monkeypatch):
    def fake_snapshot_download(**_kwargs):
        return "/hf-cache/models--org--m/snapshots/main"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert hf.resolve_cached_model_commit("org/m", "") == ""


def test_resolve_cached_model_commit_falls_back_to_shared_cache(monkeypatch):
    # ephemeral (default) cache misses; the shared weight mount resolves the commit.
    commit = "d" * 40
    monkeypatch.setattr(hf, "_shared_weight_cache_dir", lambda: "/shared-hub")

    def fake_snapshot_download(*, repo_id, revision, local_files_only, cache_dir):
        if cache_dir is None:
            raise FileNotFoundError("not in ephemeral cache")
        assert cache_dir == "/shared-hub"
        return f"/shared-hub/models--org--m/snapshots/{commit}"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert hf.resolve_cached_model_commit("org/m", "main") == commit


def test_write_base_model_provenance_records_resolved_commit(monkeypatch, tmp_path):
    commit = "c" * 40
    monkeypatch.setattr(hf, "resolve_cached_model_commit", lambda model_id, revision: commit)

    adapter_dir = tmp_path / "adapter"
    hf.write_base_model_provenance(str(adapter_dir), "org/m", "main")

    payload = json.loads((adapter_dir / "base_model_provenance.json").read_text())
    assert payload == {
        "model_id": "org/m",
        "requested_revision": "main",
        "resolved_commit": commit,
    }


def test_write_base_model_provenance_records_null_when_unresolved(monkeypatch, tmp_path):
    monkeypatch.setattr(hf, "resolve_cached_model_commit", lambda model_id, revision: "")

    adapter_dir = tmp_path / "adapter"
    hf.write_base_model_provenance(str(adapter_dir), "org/m", "")

    payload = json.loads((adapter_dir / "base_model_provenance.json").read_text())
    assert payload == {
        "model_id": "org/m",
        "requested_revision": None,
        "resolved_commit": None,
    }
