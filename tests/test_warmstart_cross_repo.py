"""init_from_adapter uses the adapter_ref form emitted by flash status."""

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


def test_status_adapter_ref_downloads_from_other_repo(monkeypatch):
    calls, out = _capture(monkeypatch, "Freesolo-Co/flashrun-sftX:sft/flash-sftX")
    assert calls["repo_id"] == "Freesolo-Co/flashrun-sftX"
    assert calls["allow_patterns"] == ["sft/flash-sftX/adapter/*"]
    assert out is not None
    assert out.endswith("sft/flash-sftX/adapter")


def test_checkpoint_step_adapter_ref_downloads_that_step(monkeypatch):
    """`<ref>/checkpoints/step-N` resolves to the per-step deployable adapter path, so a run can
    warm-start from a specific saved checkpoint instead of the run-level adapter."""
    calls, out = _capture(
        monkeypatch, "Freesolo-Co/flashrun-rlX:rl/flash-rlX/checkpoints/step-40"
    )
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


def test_bare_prefix_is_not_a_public_init_adapter_ref(monkeypatch):
    calls, out = _capture(monkeypatch, "sft/flash-self")
    assert calls == {}
    assert out is None


def test_managed_repo_without_prefix_is_not_a_public_init_adapter_ref(monkeypatch):
    calls, out = _capture(monkeypatch, "Freesolo-Co/flashrun-flash-1782194170-ce1cfcff")
    assert calls == {}
    assert out is None
