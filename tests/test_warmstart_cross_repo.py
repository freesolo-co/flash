"""Worker adapter downloads consume internal storage refs, not public init_from_adapter refs."""

import json
import os
import shutil
from types import SimpleNamespace

import pytest

import flash.engine.worker as W

_REVISION = "a" * 40


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


def test_worker_download_uses_pinned_source_revision(monkeypatch):
    monkeypatch.setattr(
        W,
        "JOB_SPEC",
        SimpleNamespace(train=SimpleNamespace(init_from_adapter_revision=_REVISION)),
    )
    calls, _out = _capture(monkeypatch, "Freesolo-Co/flashrun-sftX:sft/flash-sftX")
    assert calls["revision"] == _REVISION


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


def test_worker_download_failure_redacts_internal_storage_ref(monkeypatch, capsys):
    import huggingface_hub

    internal_ref = "private-owner/private-repo:sft/source-run/checkpoints/step-20"

    def fail_download(**kwargs):
        raise RuntimeError(f"failed to fetch {kwargs['repo_id']} {kwargs['allow_patterns']}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_download)
    with pytest.raises(RuntimeError) as exc_info:
        W._download_adapter(internal_ref)

    captured = capsys.readouterr()
    exposed = f"{exc_info.value}\n{captured.out}\n{captured.err}"
    assert "private-owner" not in exposed
    assert "private-repo" not in exposed
    assert "sft/source-run" not in exposed
    assert "prepared warm-start source adapter could not be downloaded" in str(exc_info.value)


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
        def upload_file(
            self, path_or_fileobj=None, path_in_repo=None, repo_id=None, repo_type=None
        ):
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
    assert (
        _mark(monkeypatch, "Freesolo-Co/flashrun-src:sft/flash-src", "local") == {}
    )  # dry/local child


def test_prepare_init_adapter_preserves_public_ref_and_loads_config_once(monkeypatch):
    import flash.lora_rank as rank_mod
    import flash.runner as R
    import flash.runner.checkpoints as checkpoints
    from flash.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = R.RunStatus(run_id="source-run", state="done", spec=source.to_dict())
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {
                "init_from_adapter": "source-run",
                "lora_rank": 8,
                "lora_alpha": 16,
            },
        }
    )
    calls = []

    monkeypatch.setattr(R, "get_status", lambda run_id: source_status)
    monkeypatch.setattr(rank_mod, "resolve_hf_dataset_revision", lambda repo, token: _REVISION)
    monkeypatch.setattr(
        checkpoints, "adapter_artifact_exists", lambda spec, *, step, revision=None: True
    )

    def load_config(adapter_ref, token, revision):
        calls.append((adapter_ref, token, revision))
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-4B",
            "r": 32,
            "rank_pattern": {"module": 64},
            "lora_alpha": 64,
            "alpha_pattern": {"module": 128},
        }

    monkeypatch.setattr(rank_mod, "load_hf_adapter_config", load_config)
    monkeypatch.setattr(
        rank_mod,
        "adapter_artifact_identity",
        lambda *a, **k: rank_mod.AdapterArtifactIdentity(
            "digest", "config", "adapter_model.safetensors", "weight:1"
        ),
    )
    public_spec, worker_spec, identity = R._prepare_init_from_adapter(child, token="token")

    assert calls == [("owner/source-runs:sft/source-run", "token", _REVISION)]
    assert public_spec.train.init_from_adapter == "source-run"
    assert worker_spec.train.init_from_adapter == "owner/source-runs:sft/source-run"
    assert worker_spec.train.init_from_adapter_revision == _REVISION
    assert (public_spec.train.lora_rank, public_spec.train.lora_alpha) == (8, 16)
    assert (worker_spec.train.lora_rank, worker_spec.train.lora_alpha) == (64, 128)
    assert identity == {
        "digest": "digest",
        "config_sha256": "config",
        "weight_filename": "adapter_model.safetensors",
        "weight_identity": "weight:1",
    }


def test_prepare_job_estimates_from_source_effective_worker_spec(monkeypatch):
    import types

    import flash.cost.spec as cost_spec
    import flash.lora_rank as rank_mod
    import flash.runner as R
    import flash.runner.checkpoints as checkpoints
    from flash.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = R.RunStatus(run_id="source-run", state="done", spec=source.to_dict())
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {
                "init_from_adapter": "source-run/step-20",
                "lora_rank": 8,
                "lora_alpha": 16,
            },
        }
    )
    estimated_specs = []

    monkeypatch.setattr(R, "get_status", lambda run_id: source_status)
    monkeypatch.setattr(rank_mod, "resolve_hf_dataset_revision", lambda repo, token: _REVISION)
    monkeypatch.setattr(
        checkpoints, "adapter_artifact_exists", lambda spec, *, step, revision=None: True
    )
    monkeypatch.setattr(
        rank_mod,
        "load_hf_adapter_config",
        lambda adapter_ref, token, revision: {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-4B",
            "r": 64,
            "lora_alpha": 128,
        },
    )
    monkeypatch.setattr(
        rank_mod,
        "adapter_artifact_identity",
        lambda *a, **k: rank_mod.AdapterArtifactIdentity(
            "digest", "config", "adapter_model.safetensors", "weight:1"
        ),
    )

    def estimate(spec):
        estimated_specs.append(spec)
        return types.SimpleNamespace(total_usd=12.5)

    monkeypatch.setattr(cost_spec, "estimate_for_spec", estimate)

    prepared = R.prepare_job(child)

    assert prepared.public_spec.train.init_from_adapter == "source-run/step-20"
    assert (prepared.public_spec.train.lora_rank, prepared.public_spec.train.lora_alpha) == (8, 16)
    assert prepared.worker_spec.train.init_from_adapter == (
        "owner/source-runs:sft/source-run/checkpoints/step-20"
    )
    assert (prepared.worker_spec.train.lora_rank, prepared.worker_spec.train.lora_alpha) == (
        64,
        128,
    )
    assert estimated_specs == [prepared.worker_spec]
    assert prepared.estimated_cost_usd == 12.5


def test_effective_preparation_persists_but_is_not_public(monkeypatch, tmp_path):
    import flash.runner as R
    from flash.spec import JobSpec

    monkeypatch.setattr(R, "RUNS_DIR", str(tmp_path / "runs"))
    public = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run", "lora_rank": 8},
        }
    )
    public_dict = public.to_dict()
    worker = JobSpec.from_dict(
        {
            **public_dict,
            "train": {
                **public_dict["train"],
                "init_from_adapter": "owner/source:sft/source-run",
                "init_from_adapter_revision": _REVISION,
                "lora_rank": 64,
            },
        }
    )
    identity = {"digest": "immutable-v1"}
    status = R.RunStatus(
        run_id="child-run",
        state="queued",
        spec=public_dict,
        effective_preparation={
            "worker_spec": worker.to_dict(),
            "adapter_identity": identity,
            "preparation_digest": R._preparation_digest(public, worker, identity),
        },
    )
    R._save_status(status)

    assert "effective_preparation" not in status.to_dict()
    with open(R.runs_file_path("child-run", ".json")) as f:
        stored = json.load(f)
    assert stored["effective_preparation"]["worker_spec"]["train"]["lora_rank"] == 64
    loaded = R.get_status("child-run")
    assert loaded.spec["train"]["lora_rank"] == 8
    assert loaded.effective_preparation == stored["effective_preparation"]


def test_recovery_revalidates_pinned_revision_after_default_branch_moves(monkeypatch):
    import flash.lora_rank as rank_mod
    import flash.runner as R
    from flash.spec import JobSpec

    public = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run", "lora_rank": 8},
        }
    )
    worker_dict = public.to_dict()
    worker_dict["train"].update(
        {
            "init_from_adapter": "private-owner/private-source:rl/source-run",
            "init_from_adapter_revision": _REVISION,
            "lora_rank": 64,
            "lora_alpha": 128,
        }
    )
    worker = JobSpec.from_dict(worker_dict)
    identity = rank_mod.AdapterArtifactIdentity(
        "artifact-v1", "config-v1", "adapter_model.safetensors", "weights-v1:1"
    )
    status = R.RunStatus(
        run_id=public.run_id,
        state="running",
        spec=public.to_dict(),
        effective_preparation={
            "worker_spec": worker.to_dict(),
            "adapter_identity": identity.to_dict(),
            "preparation_digest": R._preparation_digest(public, worker, identity.to_dict()),
        },
    )
    seen = []

    def load_config(adapter_ref, token, revision):
        seen.append(revision)
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": public.model,
            "r": 64 if revision == _REVISION else 8,
            "lora_alpha": 128 if revision == _REVISION else 16,
        }

    monkeypatch.setattr(rank_mod, "load_hf_adapter_config", load_config)
    monkeypatch.setattr(rank_mod, "adapter_artifact_identity", lambda *a, **k: identity)

    recovered = R.effective_spec_from_status(status, verify_source=True)

    assert recovered == worker
    assert seen == [_REVISION]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", 32),
        ("alpha", 128),
        ("algorithm", "opd"),
        ("environment", "other/environment"),
        ("gpu", "H200 SXM"),
        ("hf_repo", "private-owner/other-child"),
        ("source", "private-owner/private-source:rl/other-source"),
    ],
)
def test_effective_snapshot_rejects_tampering(field, value):
    import copy

    import flash.runner as R
    from flash.spec import JobSpec

    public = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "environment": {"id": "owner/environment"},
            "train": {"init_from_adapter": "source-run", "lora_rank": 8, "lora_alpha": 16},
        }
    )
    worker_dict = public.to_dict()
    worker_dict["train"].update(
        {
            "init_from_adapter": "private-owner/private-source:rl/source-run",
            "init_from_adapter_revision": _REVISION,
            "lora_rank": 64,
            "lora_alpha": 64,
        }
    )
    worker = JobSpec.from_dict(worker_dict)
    identity = {"digest": "artifact-v1"}
    snapshot = {
        "worker_spec": copy.deepcopy(worker.to_dict()),
        "adapter_identity": identity,
        "preparation_digest": R._preparation_digest(public, worker, identity),
    }
    target = snapshot["worker_spec"]
    if field == "rank":
        target["train"]["lora_rank"] = value
    elif field == "alpha":
        target["train"]["lora_alpha"] = value
    elif field == "algorithm":
        target["algorithm"] = value
    elif field == "environment":
        target["environment"]["id"] = value
    elif field == "gpu":
        target["gpu"]["type"] = value
    elif field == "hf_repo":
        target["train"]["hf_repo"] = value
    else:
        target["train"]["init_from_adapter"] = value
    status = R.RunStatus(
        run_id=public.run_id,
        state="running",
        spec=public.to_dict(),
        effective_preparation=snapshot,
    )
    with pytest.raises(ValueError, match="effective preparation"):
        R.effective_spec_from_status(status)


def test_worker_metrics_sanitizer_redacts_nested_and_direct_private_refs():
    from flash.engine.accounting import RunMetrics, sanitize_worker_metrics

    private_ref = "private-owner/private-repo:rl/source-run/checkpoints/step-20"
    payload = {
        "init_from_adapter": private_ref,
        "notes": {
            "init_from_adapter": private_ref,
            "job_spec": {
                "train": {
                    "init_from_adapter": private_ref,
                    "init_from_adapter_revision": _REVISION,
                    "hf_repo": "private-owner/private-child",
                }
            },
        },
    }
    sanitized = sanitize_worker_metrics(payload)
    encoded = json.dumps(sanitized)
    assert "private-owner" not in encoded
    assert "private-repo" not in encoded
    assert _REVISION not in encoded
    metrics = RunMetrics(notes=payload)
    assert "private-owner" not in metrics.to_json()


def test_persist_metrics_reports_only_sanitized_worker_metrics(monkeypatch, tmp_path):
    import flash.runner as R
    import flash.server.run_registry as registry
    from flash.spec import JobSpec

    monkeypatch.setattr(R, "RESULTS_DIR", str(tmp_path / "results"))
    private_ref = "private-owner/private-repo:rl/source-run"
    spec = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
        }
    )
    reported = []
    monkeypatch.setattr(
        registry,
        "record_training_checkpoint",
        lambda **kwargs: reported.append(kwargs["metrics"]),
    )
    R._persist_metrics(
        spec,
        {
            "wall_seconds": 1,
            "notes": {
                "init_from_adapter": private_ref,
                "job_spec": {
                    "train": {"init_from_adapter": private_ref, "hf_repo": "private/child"}
                },
            },
        },
    )
    with open(os.path.join(R.artifacts_dir(spec), "metrics.json")) as f:
        persisted = json.load(f)
    assert reported == [persisted]
    assert "private-owner" not in json.dumps(persisted)
    assert "private-repo" not in json.dumps(persisted)
    assert "private/child" not in json.dumps(persisted)


def test_legacy_warmstart_status_fails_closed_without_private_snapshot():
    import flash.runner as R
    from flash.spec import JobSpec

    public = JobSpec.from_dict(
        {
            "run_id": "legacy-child",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run"},
        }
    )
    status = R.RunStatus(run_id="legacy-child", state="running", spec=public.to_dict())
    with pytest.raises(ValueError, match="original preparation snapshot is unavailable"):
        R.effective_spec_from_status(status, verify_source=True)
