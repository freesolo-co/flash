"""Deployable RL checkpoints (CPU-only).

Covers worker publication, HF listing, and backend mirroring. All network boundaries are stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import flash.engine.worker.io.hf as worker_hf
from flash.core.spec import JobSpec
from flash.runner.results.checkpoints import (
    CheckpointListingError,
    adapter_artifact_exists,
    checkpoint_adapter_prefix,
    list_checkpoints,
)

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"

SPEC_DICT = {
    "model": "Qwen/Qwen3.5-9B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {"epochs": 1, "max_examples": 1, "hf_repo": "org/test-runs"},
    "gpu": {"type": "RTX 5090"},
    "run_id": "flash-ckpt-1",
    "project": _PROJECT_ID,
}


def _spec() -> JobSpec:
    return JobSpec.from_dict(SPEC_DICT)


# --------------------------------------------------------------------------------------------
# Worker: publish_deployable_checkpoint
# --------------------------------------------------------------------------------------------
class _RecordingHfApi:
    def __init__(self, files: list[str] | None = None):
        self.uploads: list[dict] = []
        self.deleted: list[str] = []
        self._files = files or []

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)

    def list_repo_files(self, repo_id, repo_type):
        return self._files

    def delete_folder(self, path_in_repo, repo_id, repo_type):
        self.deleted.append(path_in_repo)


def _prime_worker(monkeypatch, recorder, *, repo="org/test-runs", phase="rl", run="flash-ckpt-1"):
    import flash.engine.worker.io.hf as worker

    monkeypatch.setattr(worker._worker_state, "HF_REPO", repo)
    monkeypatch.setattr(worker._worker_state, "PHASE", phase)
    monkeypatch.setattr(worker._worker_state, "RUN_ID", run)
    monkeypatch.setattr(worker._worker_state, "SEED", 0)
    monkeypatch.setattr(worker_hf, "hf_api", lambda: recorder)
    # heartbeat would otherwise commit to hf; silence it for the unit test.
    monkeypatch.setattr(worker._worker_heartbeat, "heartbeat", lambda *a, **k: None)
    return worker


def test_publish_deployable_checkpoint_uploads_adapter_only(tmp_path, monkeypatch):
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
    (ckpt / "optimizer.pt").write_bytes(b"opt")

    subfolder = worker.publish_deployable_checkpoint(str(ckpt), 80)

    assert subfolder == "rl/flash-ckpt-1/checkpoints/step-80/adapter"
    assert len(rec.uploads) == 1
    up = rec.uploads[0]
    assert up["path_in_repo"] == "rl/flash-ckpt-1/checkpoints/step-80/adapter"
    assert up["repo_type"] == "dataset"
    # Trainer-state files are excluded so each per-step snapshot is just the small LoRA adapter.
    assert "optimizer.pt" in up["ignore_patterns"]
    # The deployable path must NOT prune older steps (every step stays deployable).
    assert "delete_patterns" not in up


def test_publish_deployable_checkpoint_writes_base_model_provenance(tmp_path, monkeypatch):
    # regression (#538 finding 6): a per-step / opd-reconcile deployable is published straight from a
    # trainer dir that never passed through the final _save_adapter path, so publish itself stamps the
    # base-model provenance sidecar, sourced from the job spec's pinned base model, before uploading.
    import json

    import flash.engine.worker.io.hf as hf
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    commit = "e" * 40
    monkeypatch.setattr(
        worker._worker_state, "JOB_SPEC", SimpleNamespace(model="org/base", model_revision="main")
    )
    monkeypatch.setattr(hf, "resolve_cached_model_commit", lambda model_id, revision: commit)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")

    worker.publish_deployable_checkpoint(str(ckpt), 80)

    payload = json.loads((ckpt / "base_model_provenance.json").read_text())
    assert payload == {
        "model_id": "org/base",
        "requested_revision": "main",
        "resolved_commit": commit,
    }
    # the sidecar rides inside the same atomic upload; it must not be stripped as trainer state.
    assert len(rec.uploads) == 1
    assert "base_model_provenance.json" not in rec.uploads[0]["ignore_patterns"]


def test_publish_deployable_checkpoint_without_job_spec_writes_no_provenance(tmp_path, monkeypatch):
    # back-compat: with no JOB_SPEC (e.g. local recipe runs) publish writes no base_model_provenance.json
    # rather than a misleading empty record, and still publishes the deployable (provenance is additive).
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    monkeypatch.setattr(worker._worker_state, "JOB_SPEC", None, raising=False)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")

    worker.publish_deployable_checkpoint(str(ckpt), 80)

    assert not (ckpt / "base_model_provenance.json").exists()
    assert len(rec.uploads) == 1


def test_publish_deployable_checkpoint_with_empty_model_writes_no_provenance(tmp_path, monkeypatch):
    # #542 finding: the guard mirrors the final-save path (write only for a non-empty base model), so a
    # JOB_SPEC with no model stamps no sidecar rather than a misleading empty-model_id record, and still
    # publishes the deployable.
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    monkeypatch.setattr(
        worker._worker_state,
        "JOB_SPEC",
        SimpleNamespace(model="", model_revision=""),
        raising=False,
    )
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")

    worker.publish_deployable_checkpoint(str(ckpt), 80)

    assert not (ckpt / "base_model_provenance.json").exists()
    assert len(rec.uploads) == 1


def test_publish_deployable_checkpoint_rejects_bin_weights(tmp_path, monkeypatch):
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.bin").write_bytes(b"weights")

    assert worker.publish_deployable_checkpoint(str(ckpt), 80) is None
    assert rec.uploads == []


def test_publish_deployable_checkpoint_skips_without_adapter(tmp_path, monkeypatch):
    """A checkpoint that carries no PEFT adapter is never advertised as deployable."""
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "optimizer.pt").write_bytes(b"opt")  # no adapter_config.json

    assert worker.publish_deployable_checkpoint(str(ckpt), 10) is None
    assert rec.uploads == []


def test_publish_deployable_checkpoint_skips_config_without_weights(tmp_path, monkeypatch):
    """A checkpoint with adapter_config.json but no weights file isn't loadable -> not published."""
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-20"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")  # config only, no adapter_model.*

    assert worker.publish_deployable_checkpoint(str(ckpt), 20) is None
    assert rec.uploads == []


def test_publish_deployable_checkpoint_no_repo_is_noop(tmp_path, monkeypatch):
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, repo="")  # local run, no HF repo
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")

    assert worker.publish_deployable_checkpoint(str(ckpt), 5) is None
    assert rec.uploads == []


def test_publish_deployable_checkpoint_starts_no_upload_at_deadline(tmp_path, monkeypatch):
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    monkeypatch.setattr(worker._worker_state, "_remaining_worker_wall_seconds", lambda: 0.0)
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")

    assert worker.publish_deployable_checkpoint(str(ckpt), 5) is None
    assert rec.uploads == []


def test_prune_stale_resume_checkpoints_deletes_only_older_steps(monkeypatch):
    """Pruning removes older steps without deleting a newer concurrently published checkpoint."""
    import flash.engine.worker.io.hf as worker_hf

    prefix = "rl/flash-ckpt-1"
    files = [
        f"{prefix}/checkpoint/checkpoint-20/optimizer.pt",
        f"{prefix}/checkpoint/checkpoint-20/adapter_model.safetensors",
        f"{prefix}/checkpoint/checkpoint-40/optimizer.pt",
        f"{prefix}/checkpoint/checkpoint-60/optimizer.pt",
        f"{prefix}/checkpoint/checkpoint-80/optimizer.pt",
        f"{prefix}/checkpoints/step-60/adapter/adapter_model.safetensors",  # deployable (plural) -> keep
        f"{prefix}/metrics.json",
    ]
    rec = _RecordingHfApi(files)
    _prime_worker(monkeypatch, rec)

    worker_hf._prune_stale_resume_checkpoints(60)

    assert sorted(rec.deleted) == [
        f"{prefix}/checkpoint/checkpoint-20",
        f"{prefix}/checkpoint/checkpoint-40",
    ]
    assert f"{prefix}/checkpoint/checkpoint-60" not in rec.deleted
    assert f"{prefix}/checkpoint/checkpoint-80" not in rec.deleted
    assert all("checkpoints/" not in d for d in rec.deleted)  # deployable tree untouched


def test_prune_stale_resume_checkpoints_no_repo_is_noop(monkeypatch):
    import flash.engine.worker.io.hf as worker_hf

    rec = _RecordingHfApi(["rl/r/checkpoint/checkpoint-1/optimizer.pt"])
    _prime_worker(monkeypatch, rec, repo="")  # local run, no HF repo
    worker_hf._prune_stale_resume_checkpoints(5)
    assert rec.deleted == []


# --------------------------------------------------------------------------------------------
# Control plane: list_checkpoints
# --------------------------------------------------------------------------------------------
class _FakeHfApiFiles:
    def __init__(self, files):
        self._files = files

    def __call__(self, *a, **k):  # stand in for HfApi(...)
        return self

    def list_repo_files(self, repo, repo_type=None, revision=None):
        return self._files


def _patch_hf_files(monkeypatch, files):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApiFiles(files))


def test_checkpoint_adapter_prefix():
    assert checkpoint_adapter_prefix(_spec(), 60) == "rl/flash-ckpt-1/checkpoints/step-60"


def test_list_checkpoints_parses_and_sorts(monkeypatch):
    base = "rl/flash-ckpt-1"
    files = [
        f"{base}/checkpoints/step-80/adapter/adapter_config.json",
        f"{base}/checkpoints/step-80/adapter/adapter_model.safetensors",
        f"{base}/checkpoints/step-40/adapter/adapter_config.json",
        f"{base}/checkpoints/step-40/adapter/adapter_model.safetensors",
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


def test_list_checkpoints_rejects_bin_weights(monkeypatch):
    base = "rl/flash-ckpt-1"
    files = [
        f"{base}/checkpoints/step-40/adapter/adapter_config.json",
        f"{base}/checkpoints/step-40/adapter/adapter_model.bin",
    ]
    _patch_hf_files(monkeypatch, files)

    assert list_checkpoints(_spec()) == []


def test_list_checkpoints_skips_step_without_weights(monkeypatch):
    """A step with adapter_config.json but no weights file is NOT advertised as deployable."""
    base = "rl/flash-ckpt-1"
    files = [
        # step-40 is complete; step-60 has config only (half-uploaded) and must be excluded.
        f"{base}/checkpoints/step-40/adapter/adapter_config.json",
        f"{base}/checkpoints/step-40/adapter/adapter_model.safetensors",
        f"{base}/checkpoints/step-60/adapter/adapter_config.json",
    ]
    _patch_hf_files(monkeypatch, files)
    assert [c["step"] for c in list_checkpoints(_spec())] == [40]


def test_list_checkpoints_no_repo(monkeypatch):
    spec = JobSpec.from_dict(
        {**SPEC_DICT, "train": {"epochs": 1, "max_examples": 1, "hf_repo": ""}}
    )
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


def test_final_adapter_artifact_exists_requires_config_and_weights(monkeypatch):
    base = "rl/flash-ckpt-1"
    _patch_hf_files(
        monkeypatch,
        [
            f"{base}/adapter/adapter_config.json",
            f"{base}/adapter/adapter_model.safetensors",
        ],
    )

    assert adapter_artifact_exists(_spec(), step=None) is True


def test_final_adapter_artifact_exists_rejects_bin_weights(monkeypatch):
    base = "rl/flash-ckpt-1"
    _patch_hf_files(
        monkeypatch,
        [
            f"{base}/adapter/adapter_config.json",
            f"{base}/adapter/adapter_model.bin",
        ],
    )

    assert adapter_artifact_exists(_spec(), step=None) is False


def test_final_adapter_artifact_exists_rejects_incomplete_or_nested_files(monkeypatch):
    base = "rl/flash-ckpt-1"
    _patch_hf_files(
        monkeypatch,
        [
            f"{base}/adapter/adapter_config.json",
            f"{base}/adapter/nested/adapter_model.safetensors",
        ],
    )

    assert adapter_artifact_exists(_spec(), step=None) is False


def test_final_adapter_artifact_exists_raises_listing_error(monkeypatch):
    import huggingface_hub

    class _Boom:
        def __call__(self, *a, **k):
            return self

        def list_repo_files(self, *a, **k):
            raise RuntimeError("hf down")

    monkeypatch.setattr(huggingface_hub, "HfApi", _Boom())

    with pytest.raises(
        CheckpointListingError,
        match="could not verify adapter artifacts for flash-ckpt-1: hf down",
    ):
        adapter_artifact_exists(_spec(), step=None)


# --------------------------------------------------------------------------------------------
# Backend client: register_run_checkpoints / register_checkpoints_best_effort
# --------------------------------------------------------------------------------------------
def _status(**kw):
    base = {
        "run_id": "flash-ckpt-1",
        "spec": SPEC_DICT,
        "billing_context": {"org_id": "org-xyz"},
        # a provisioned run always carries the internal worker-spec carrier; hf_repo + run_id are
        # platform-managed and read from it (see _internal_spec_from_status).
        "effective_preparation": {"worker_spec": SPEC_DICT},
    }
    base.update(kw)
    return SimpleNamespace(**base)


_CKPTS = [
    {
        "step": 40,
        "subfolder": "rl/flash-ckpt-1/checkpoints/step-40/adapter",
        "repo_id": "org/test-runs",
        "repo_type": "dataset",
    },
    {
        "step": 80,
        "subfolder": "rl/flash-ckpt-1/checkpoints/step-80/adapter",
        "repo_id": "org/test-runs",
        "repo_type": "dataset",
    },
]


def test_register_run_checkpoints_body_shape(monkeypatch):
    import flash.server.domain.registry.checkpoints as ck

    captured = {}
    monkeypatch.setattr(
        ck,
        "_post_checkpoints",
        lambda *, token, body: captured.update(token=token, body=body) or {},
    )
    ck.register_run_checkpoints(internal_key="int-key", status=_status(), checkpoints=_CKPTS)

    assert captured["token"] == "int-key"
    body = captured["body"]
    assert body["orgId"] == "org-xyz"
    assert body["runId"] == "flash-ckpt-1"
    assert body["baseModel"] == "Qwen/Qwen3.5-9B"
    assert body["repoId"] == "org/test-runs"
    assert body["repoType"] == "dataset"
    # the receiver requires an explicit project and 422s the batch without one.
    assert body["projectId"] == _PROJECT_ID
    assert body["checkpoints"] == [
        {"step": 40, "subfolder": "rl/flash-ckpt-1/checkpoints/step-40/adapter"},
        {"step": 80, "subfolder": "rl/flash-ckpt-1/checkpoints/step-80/adapter"},
    ]


def test_register_run_checkpoints_requires_a_project(monkeypatch):
    # a run whose spec carries no project must fail loudly here rather than posting a body the
    # backend rejects as a 422 that the best-effort wrapper would swallow.
    import flash.server.domain.registry.checkpoints as ck

    posted = []
    monkeypatch.setattr(ck, "_post_checkpoints", lambda **kw: posted.append(kw) or {})
    spec_without_project = {k: v for k, v in SPEC_DICT.items() if k != "project"}
    with pytest.raises(ValueError, match="project id"):
        ck.register_run_checkpoints(
            internal_key="k",
            status=_status(spec=spec_without_project),
            checkpoints=_CKPTS,
        )
    assert posted == []


def test_register_run_checkpoints_requires_org():
    import flash.server.domain.registry.checkpoints as ck

    with pytest.raises(ValueError, match="org id"):
        ck.register_run_checkpoints(
            internal_key="k",
            status=_status(billing_context={}, platform_context=None),
            checkpoints=_CKPTS,
        )


def test_register_run_checkpoints_falls_back_to_platform_context(monkeypatch):
    # Internal/operator runs carry org only in platform_context (billing_context is None):
    # registration must still scope rows to that org. _run_org_id falls back to billing-then-platform
    # (same order as routes/serving.py::_run_org; NOT runs, which is platform-first).
    import flash.server.domain.registry.checkpoints as ck

    captured = {}
    monkeypatch.setattr(
        ck, "_post_checkpoints", lambda *, token, body: captured.update(body=body) or {}
    )
    ck.register_run_checkpoints(
        internal_key="int-key",
        status=_status(billing_context=None, platform_context={"org_id": "org-plat"}),
        checkpoints=_CKPTS,
    )
    assert captured["body"]["orgId"] == "org-plat"


def test_best_effort_noop_without_internal_key(monkeypatch):
    import flash.server.domain.registry.checkpoints as ck

    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    # Even if HF had checkpoints, no internal key => skip persistence (HF stays source of truth).
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: _CKPTS)
    assert ck.register_checkpoints_best_effort(_status()) == 0


def test_best_effort_registers(monkeypatch):
    import flash.server.domain.registry.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: _CKPTS)
    posted = {}
    monkeypatch.setattr(
        ck, "_post_checkpoints", lambda *, token, body: posted.update(body=body) or {}
    )

    assert ck.register_checkpoints_best_effort(_status()) == 2
    assert posted["body"]["runId"] == "flash-ckpt-1"


def test_best_effort_swallows_backend_failure(monkeypatch):
    import io
    import urllib.error

    import flash.server.domain.registry.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: _CKPTS)

    def boom(*, token, body):
        raise urllib.error.URLError("backend down")

    monkeypatch.setattr(ck, "_post_checkpoints", boom)
    log = io.StringIO()
    assert ck.register_checkpoints_best_effort(_status(), log=log) == 0  # never raises
    # A genuine backend failure MUST stay visible.
    assert "warn" in log.getvalue()


def test_best_effort_skips_silently_when_no_org(monkeypatch):
    # Internal/operator run with no org in either context: skip quietly, do NOT warn, do NOT hit the
    # backend, and do NOT perform the HF checkpoint listing — the org check short-circuits BEFORE
    # the network call. Regression guard for the noisy "missing org id" log and the wasted HF
    # listing on an expected-skip run.
    import io

    import flash.server.domain.registry.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    listed = {"called": False}

    def fake_list(spec):
        listed["called"] = True
        return _CKPTS

    monkeypatch.setattr(ck, "list_checkpoints", fake_list)

    def fail(*, token, body):  # pragma: no cover - must never be called
        raise AssertionError("_post_checkpoints must not be called without an org")

    monkeypatch.setattr(ck, "_post_checkpoints", fail)
    log = io.StringIO()
    status = _status(billing_context={}, platform_context=None)
    assert ck.register_checkpoints_best_effort(status, log=log) == 0
    assert "warn" not in log.getvalue()
    assert listed["called"] is False  # org check short-circuits before the HF listing


def test_best_effort_no_checkpoints(monkeypatch):
    import flash.server.domain.registry.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: [])
    assert ck.register_checkpoints_best_effort(_status()) == 0


# ---------------------------------------------------------------------------
# finalization must publish an unaligned last step as `RUN_ID/step-N`; source-wiring tests avoid a
# full trainer runtime.
# ---------------------------------------------------------------------------
def _finalize_src(module_name: str, fn_name: str) -> str:
    import importlib
    import inspect

    mod = importlib.import_module(module_name)
    return inspect.getsource(getattr(mod, fn_name))


def test_run_rl_publishes_final_step_as_deployable_checkpoint():
    src = _finalize_src("flash.engine.worker.train.entry.rl_train", "run_rl_train")
    # Warm-start CONTINUES the one adapter in place, so the saved adapter already carries SFT+GRPO on
    # the original catalog base and deploys as-is: the final adapter is uploaded as the default and
    # published as the deployable checkpoint directly — no recombine step.
    assert "save_pretrained(adapter_dir)" in src
    assert '_worker_hf.hf_upload_folder(adapter_dir, "adapter", required=True)' in src
    assert "_worker_hf.publish_deployable_checkpoint(adapter_dir, steps_run)" in src


def test_run_sft_publishes_final_step_as_deployable_checkpoint():
    # The invariant lives in the verl worker, which owns the whole SFT path; `run_sft` is a one-line
    # delegation to it. Asserting on the delegator's source would pass on any body at all, so this
    # follows the wiring to where the adapter is actually exported and published.
    src = _finalize_src("flash.engine.worker.train.entry.sft_train", "run_sft_train")
    assert "_export_checkpoint_adapter(" in src
    assert '_worker_hf.hf_upload_folder(adapter_dir, "adapter", required=True)' in src
    assert "_worker_hf.publish_deployable_checkpoint(adapter_dir, final_step)" in src


def test_worker_io_modules_only_reference_attributes_their_siblings_define():
    """A cross-module `_hf.<name>` reference must name something that exists.

    `flash/engine/worker/io/checkpoint_upload.py` shipped calling five `_hf.` attributes that were
    defined nowhere in the repo (`_stage_optional_directory`, `_OPTIONAL_CHECKPOINT_UPLOADER`,
    `_OPTIONAL_DEPLOYABLE_UPLOADER`, `_checkpoint_upload_lock_timeout`, `_latest_checkpoint_dir`).
    Nothing imported the module, so import-time checks and the linter stayed quiet and only a real
    save would have raised AttributeError. This walks the io package's own `module.attr` references
    and fails on any that its sibling does not actually define.
    """
    import ast
    import importlib
    import pathlib

    io_dir = pathlib.Path(importlib.import_module("flash.engine.worker.io").__file__).parent
    # module-alias -> real module, as bound by `from ... import x as _y` inside this package.
    missing: list[str] = []
    for path in sorted(io_dir.glob("*.py")):
        tree = ast.parse(path.read_text())
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "flash.engine.worker.io"
            ):
                for alias in node.names:
                    target = f"{node.module}.{alias.name}"
                    if alias.asname and _module_or_none(target) is not None:
                        aliases[alias.asname] = target
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            ):
                mod = _module_or_none(aliases[node.value.id])
                if mod is not None and not hasattr(mod, node.attr):
                    missing.append(f"{path.name}:{node.lineno} -> {node.value.id}.{node.attr}")
    assert not missing, "references to undefined sibling attributes: " + "; ".join(missing)


def _module_or_none(dotted: str):
    import importlib

    try:
        return importlib.import_module(dotted)
    except Exception:
        return None
