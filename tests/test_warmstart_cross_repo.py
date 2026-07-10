"""Worker adapter downloads consume internal storage refs, not public init_from_adapter refs."""

import os
import shutil

import flash.engine.worker as W


def _capture(monkeypatch, prefix, hf_repo="Freesolo-Co/flashrun-self"):
    """Call _download_adapter with snapshot_download stubbed; return its kwargs + the result.

    _download_adapter targets an absolute local_dir (/tmp/evdl), so no chdir is needed; the
    stub just materializes the expected adapter dir there (cleaned up by the caller).
    """
    calls = {}

    def fake_snapshot_download(**kw):
        calls.update(kw)
        adapter_prefix = kw["allow_patterns"][0].removesuffix("/adapter/*")
        os.makedirs(os.path.join(kw["local_dir"], adapter_prefix, "adapter"), exist_ok=True)

    monkeypatch.setattr(W, "HF_REPO", hf_repo, raising=False)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    try:
        return calls, W._download_adapter(prefix)
    finally:
        shutil.rmtree("/tmp/evdl", ignore_errors=True)


def test_storage_ref_downloads_from_other_repo(monkeypatch):
    calls, out = _capture(monkeypatch, "Freesolo-Co/flashrun-sftX:sft/flash-sftX")
    assert calls["repo_id"] == "Freesolo-Co/flashrun-sftX"
    assert calls["allow_patterns"] == ["sft/flash-sftX/adapter/*"]
    assert out is not None
    assert out.endswith("sft/flash-sftX/adapter")


def test_checkpoint_step_adapter_ref_downloads_that_step(monkeypatch):
    """The resolved storage ref can target a deployable per-step adapter snapshot."""
    calls, out = _capture(monkeypatch, "Freesolo-Co/flashrun-rlX:rl/flash-rlX/checkpoints/step-40")
    assert calls["repo_id"] == "Freesolo-Co/flashrun-rlX"
    assert calls["allow_patterns"] == ["rl/flash-rlX/checkpoints/step-40/adapter/*"]
    assert out is not None
    assert out.endswith("rl/flash-rlX/checkpoints/step-40/adapter")


def test_checkpoint_ref_with_trailing_path_is_rejected(monkeypatch):
    calls, out = _capture(
        monkeypatch, "Freesolo-Co/flashrun-rlX:rl/flash-rlX/checkpoints/step-40/adapter"
    )
    assert calls == {}
    assert out is None


def test_bare_prefix_is_not_an_internal_storage_ref(monkeypatch):
    calls, out = _capture(monkeypatch, "sft/flash-self")
    assert calls == {}
    assert out is None


def test_managed_repo_without_prefix_is_not_an_internal_storage_ref(monkeypatch):
    calls, out = _capture(monkeypatch, "Freesolo-Co/flashrun-flash-1782194170-ce1cfcff")
    assert calls == {}
    assert out is None


# ---- warm-start source marker (control-plane producer for the artifact GC) --------------------


def _mark(monkeypatch, ref, child_run_id):
    """Call _mark_warmstart_source with HfApi.upload_file stubbed; return captured upload kwargs."""
    import types

    import huggingface_hub

    import flash.runner as R

    captured = {}

    class _FakeApi:
        def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None, repo_type=None):
            captured.update(path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    worker_spec = types.SimpleNamespace(train=types.SimpleNamespace(init_from_adapter=ref))
    R._mark_warmstart_source(worker_spec, child_run_id)
    return captured


def test_mark_warmstart_source_writes_marker_into_source_repo(monkeypatch):
    captured = _mark(monkeypatch, "Freesolo-Co/flashrun-src:sft/flash-src", "flash-child-1")
    assert captured == {
        "path_in_repo": "referenced_by/flash-child-1",
        "repo_id": "Freesolo-Co/flashrun-src",
        "repo_type": "dataset",
    }


def test_mark_warmstart_source_noops_without_a_real_dependency(monkeypatch):
    assert _mark(monkeypatch, None, "flash-child-1") == {}  # no warm-start ref
    assert _mark(monkeypatch, "flash-src/step-5", "flash-child-1") == {}  # public form, no ":" repo
    assert _mark(monkeypatch, "Freesolo-Co/flashrun-src:sft/flash-src", "local") == {}  # dry/local child
