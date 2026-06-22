"""init_from_adapter cross-repo warm-start: '<owner>/<repo>:<prefix>' downloads from a
sibling managed artifact repo; a bare '<prefix>' uses the run's own HF_REPO."""

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
        os.makedirs(os.path.join(kw["local_dir"], prefix.split(":", 1)[-1], "adapter"), exist_ok=True)

    monkeypatch.setattr(W, "HF_REPO", hf_repo, raising=False)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    try:
        return calls, W._download_adapter(prefix)
    finally:
        shutil.rmtree("/tmp/evdl", ignore_errors=True)


def test_cross_repo_prefix_downloads_from_other_repo(monkeypatch):
    calls, out = _capture(monkeypatch, "Freesolo-Co/flashrun-sftX:sft/flash-sftX/seed0")
    assert calls["repo_id"] == "Freesolo-Co/flashrun-sftX"
    assert calls["allow_patterns"] == ["sft/flash-sftX/seed0/adapter/*"]
    assert out is not None
    assert out.endswith("sft/flash-sftX/seed0/adapter")


def test_bare_prefix_uses_own_repo(monkeypatch):
    calls, _out = _capture(monkeypatch, "sft/flash-self/seed0")
    assert calls["repo_id"] == "Freesolo-Co/flashrun-self"
    assert calls["allow_patterns"] == ["sft/flash-self/seed0/adapter/*"]
