"""init_from_adapter cross-repo warm-start: '<owner>/<repo>:<prefix>' downloads from a
sibling managed artifact repo; a bare '<prefix>' uses the run's own HF_REPO."""
import os
import flash.engine.worker as W


def _capture(monkeypatch, prefix, hf_repo="Freesolo-Co/flashrun-self"):
    calls = {}

    def fake_snapshot_download(**kw):
        calls.update(kw)
        # materialize the expected adapter dir so the function returns it
        d = os.path.join(kw["local_dir"], prefix.split(":", 1)[-1], "adapter")
        os.makedirs(d, exist_ok=True)

    monkeypatch.setattr(W, "HF_REPO", hf_repo, raising=False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    out = W._download_adapter(prefix)
    return calls, out


def test_cross_repo_prefix_downloads_from_other_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls, out = _capture(monkeypatch, "Freesolo-Co/flashrun-sftX:sft/flash-sftX/seed0")
    assert calls["repo_id"] == "Freesolo-Co/flashrun-sftX"
    assert calls["allow_patterns"] == ["sft/flash-sftX/seed0/adapter/*"]
    assert out and out.endswith("sft/flash-sftX/seed0/adapter")


def test_bare_prefix_uses_own_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls, out = _capture(monkeypatch, "sft/flash-self/seed0")
    assert calls["repo_id"] == "Freesolo-Co/flashrun-self"
    assert calls["allow_patterns"] == ["sft/flash-self/seed0/adapter/*"]
