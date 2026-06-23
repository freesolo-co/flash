"""Deployable RL checkpoints (CPU-only).

Covers the three control-plane-adjacent pieces of the feature:
  - the worker publishing each save's LoRA adapter to a stable per-step path,
  - the lister that enumerates those snapshots from HF,
  - the backend client that mirrors them to the freesolo run_checkpoints store.

All HF/network boundaries are stubbed; nothing here touches a GPU or the network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.runner.checkpoints import checkpoint_adapter_prefix, list_checkpoints
from flash.spec import JobSpec

SPEC_DICT = {
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {"steps": 1, "seeds": [0], "hf_repo": "org/test-runs"},
    "gpu": {"type": "RTX 5090"},
    "run_id": "flash-ckpt-1",
}


def _spec() -> JobSpec:
    return JobSpec.from_dict(SPEC_DICT)


# --------------------------------------------------------------------------------------------
# Worker: publish_deployable_checkpoint
# --------------------------------------------------------------------------------------------
class _RecordingHfApi:
    def __init__(self):
        self.uploads: list[dict] = []

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)


def _prime_worker(monkeypatch, recorder, *, repo="org/test-runs", phase="rl", run="flash-ckpt-1"):
    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "HF_REPO", repo)
    monkeypatch.setattr(worker, "PHASE", phase)
    monkeypatch.setattr(worker, "RUN_ID", run)
    monkeypatch.setattr(worker, "SEED", 0)
    monkeypatch.setattr(worker, "hf_api", lambda: recorder)
    # heartbeat would otherwise commit to HF; silence it for the unit test.
    monkeypatch.setattr(worker, "heartbeat", lambda *a, **k: None)
    return worker


def test_publish_deployable_checkpoint_uploads_adapter_only(tmp_path, monkeypatch):
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
    (ckpt / "optimizer.pt").write_bytes(b"opt")

    subfolder = worker.publish_deployable_checkpoint(str(ckpt), 80)

    assert subfolder == "rl/flash-ckpt-1/seed0/checkpoints/step-80/adapter"
    assert len(rec.uploads) == 1
    up = rec.uploads[0]
    assert up["path_in_repo"] == "rl/flash-ckpt-1/seed0/checkpoints/step-80/adapter"
    assert up["repo_type"] == "dataset"
    # Trainer-state files are excluded so each per-step snapshot is just the small LoRA adapter.
    assert "optimizer.pt" in up["ignore_patterns"]
    # The deployable path must NOT prune older steps (every step stays deployable).
    assert "delete_patterns" not in up


def test_publish_deployable_checkpoint_skips_without_adapter(tmp_path, monkeypatch):
    """A checkpoint that carries no PEFT adapter is never advertised as deployable."""
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "optimizer.pt").write_bytes(b"opt")  # no adapter_config.json

    assert worker.publish_deployable_checkpoint(str(ckpt), 10) is None
    assert rec.uploads == []


def test_publish_deployable_checkpoint_no_repo_is_noop(tmp_path, monkeypatch):
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, repo="")  # local run, no HF repo
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")

    assert worker.publish_deployable_checkpoint(str(ckpt), 5) is None
    assert rec.uploads == []


# --------------------------------------------------------------------------------------------
# Control plane: list_checkpoints
# --------------------------------------------------------------------------------------------
class _FakeHfApiFiles:
    def __init__(self, files):
        self._files = files

    def __call__(self, *a, **k):  # stand in for HfApi(...)
        return self

    def list_repo_files(self, repo, repo_type=None):
        return self._files


def _patch_hf_files(monkeypatch, files):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApiFiles(files))


def test_checkpoint_adapter_prefix():
    assert (
        checkpoint_adapter_prefix(_spec(), 60)
        == "rl/flash-ckpt-1/seed0/checkpoints/step-60"
    )


def test_list_checkpoints_parses_and_sorts(monkeypatch):
    base = "rl/flash-ckpt-1/seed0"
    files = [
        f"{base}/checkpoints/step-80/adapter/adapter_config.json",
        f"{base}/checkpoints/step-80/adapter/adapter_model.safetensors",
        f"{base}/checkpoints/step-40/adapter/adapter_config.json",
        # noise that must NOT be picked up:
        f"{base}/checkpoint/checkpoint-90/optimizer.pt",
        f"{base}/adapter/adapter_config.json",
        f"{base}/heartbeat.json",
    ]
    _patch_hf_files(monkeypatch, files)

    out = list_checkpoints(_spec())

    assert [c["step"] for c in out] == [40, 80]
    assert out[0]["adapter_prefix"] == f"{base}/checkpoints/step-40"
    assert out[0]["subfolder"] == f"{base}/checkpoints/step-40/adapter"
    assert out[0]["repo_id"] == "org/test-runs"
    assert out[1]["step"] == 80


def test_list_checkpoints_no_repo(monkeypatch):
    spec = JobSpec.from_dict({**SPEC_DICT, "train": {"steps": 1, "seeds": [0], "hf_repo": ""}})
    assert list_checkpoints(spec) == []


def test_list_checkpoints_swallows_hf_error(monkeypatch):
    import huggingface_hub

    class _Boom:
        def __call__(self, *a, **k):
            return self

        def list_repo_files(self, *a, **k):
            raise RuntimeError("hf down")

    monkeypatch.setattr(huggingface_hub, "HfApi", _Boom())
    assert list_checkpoints(_spec()) == []  # best-effort: never raises into a run/route


# --------------------------------------------------------------------------------------------
# Backend client: register_run_checkpoints / register_checkpoints_best_effort
# --------------------------------------------------------------------------------------------
def _status(**kw):
    base = {
        "run_id": "flash-ckpt-1",
        "spec": SPEC_DICT,
        "billing_context": {"org_id": "org-xyz"},
    }
    base.update(kw)
    return SimpleNamespace(**base)


_CKPTS = [
    {"step": 40, "subfolder": "rl/flash-ckpt-1/seed0/checkpoints/step-40/adapter",
     "repo_id": "org/test-runs", "repo_type": "dataset"},
    {"step": 80, "subfolder": "rl/flash-ckpt-1/seed0/checkpoints/step-80/adapter",
     "repo_id": "org/test-runs", "repo_type": "dataset"},
]


def test_register_run_checkpoints_body_shape(monkeypatch):
    import flash.server.checkpoints as ck

    captured = {}
    monkeypatch.setattr(
        ck, "_post_checkpoints", lambda *, token, body: captured.update(token=token, body=body) or {}
    )
    ck.register_run_checkpoints(internal_key="int-key", status=_status(), checkpoints=_CKPTS)

    assert captured["token"] == "int-key"
    body = captured["body"]
    assert body["orgId"] == "org-xyz"
    assert body["runId"] == "flash-ckpt-1"
    assert body["baseModel"] == "Qwen/Qwen3.5-4B"
    assert body["repoId"] == "org/test-runs"
    assert body["repoType"] == "dataset"
    assert body["checkpoints"] == [
        {"step": 40, "subfolder": "rl/flash-ckpt-1/seed0/checkpoints/step-40/adapter"},
        {"step": 80, "subfolder": "rl/flash-ckpt-1/seed0/checkpoints/step-80/adapter"},
    ]


def test_register_run_checkpoints_requires_org():
    import flash.server.checkpoints as ck

    with pytest.raises(ValueError, match="org id"):
        ck.register_run_checkpoints(
            internal_key="k", status=_status(billing_context={}), checkpoints=_CKPTS
        )


def test_best_effort_noop_without_internal_key(monkeypatch):
    import flash.server.checkpoints as ck

    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    # Even if HF had checkpoints, no internal key => skip persistence (HF stays source of truth).
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: _CKPTS)
    assert ck.register_checkpoints_best_effort(_status()) == 0


def test_best_effort_registers(monkeypatch):
    import flash.server.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: _CKPTS)
    posted = {}
    monkeypatch.setattr(ck, "_post_checkpoints", lambda *, token, body: posted.update(body=body) or {})

    assert ck.register_checkpoints_best_effort(_status()) == 2
    assert posted["body"]["runId"] == "flash-ckpt-1"


def test_best_effort_swallows_backend_failure(monkeypatch):
    import urllib.error

    import flash.server.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: _CKPTS)

    def boom(*, token, body):
        raise urllib.error.URLError("backend down")

    monkeypatch.setattr(ck, "_post_checkpoints", boom)
    assert ck.register_checkpoints_best_effort(_status()) == 0  # never raises


def test_best_effort_no_checkpoints(monkeypatch):
    import flash.server.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: [])
    assert ck.register_checkpoints_best_effort(_status()) == 0
