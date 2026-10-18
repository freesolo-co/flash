"""Worker adapter downloads consume internal storage refs, not public init_from_adapter refs."""

import json
import os
import shutil
from types import SimpleNamespace

import pytest

import flash.core.catalog as catalog
import flash.engine.worker.model.adapter as worker_adapter
import flash.engine.worker.runtime.state as worker_state
import flash.runner.lifecycle.preparation as runner_preparation
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
import flash.runner.supervise.recovery as runner_recovery
from tests._helpers.runner import provisioned_status

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
        adapter_dir = os.path.join(kw["local_dir"], adapter_prefix, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        # a real download lands a loadable adapter (config + weights); the worker now treats a dir
        # without them as an incomplete transfer, so the stub materializes both.
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w", encoding="utf-8") as fh:
            json.dump({"peft_type": "LORA"}, fh)
        with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "wb") as fh:
            fh.write(b"fixture")

    monkeypatch.setattr(worker_state, "HF_REPO", hf_repo)
    monkeypatch.setattr(
        "flash.engine.worker.model.adapter._warmstart_adapter_is_loadable",
        lambda adapter_dir: (
            os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))
            and os.path.isfile(os.path.join(adapter_dir, "adapter_model.safetensors"))
        ),
    )
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    try:
        return calls, worker_adapter._download_adapter(prefix)
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
        worker_state,
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
        worker_adapter._download_adapter(internal_ref)

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

    captured = {}

    class _FakeApi:
        def upload_file(
            self, path_or_fileobj=None, path_in_repo=None, repo_id=None, repo_type=None
        ):
            captured.update(path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    worker_spec = types.SimpleNamespace(train=types.SimpleNamespace(init_from_adapter=ref))
    runner_preparation._mark_warmstart_source(worker_spec, child_run_id)
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
    import flash.adapters.lora_rank as rank_mod
    import flash.runner.results.checkpoints as checkpoints
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = provisioned_status(source, state="done")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {
                "init_from_adapter": "source-run/final",
                "lora_rank": 8,
                "lora_alpha": 16,
            },
        }
    )
    calls = []

    monkeypatch.setattr(runner_status, "get_status", lambda run_id: source_status)
    monkeypatch.setattr(rank_mod, "resolve_hf_dataset_revision", lambda repo, token: _REVISION)
    monkeypatch.setattr(
        checkpoints, "adapter_artifact_exists", lambda spec, *, step, revision=None: True
    )

    def load_config(adapter_ref, token, revision):
        calls.append((adapter_ref, token, revision))
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-9B",
            "r": 32,
            "rank_pattern": {"module": 64},
            "lora_alpha": 64,
            "alpha_pattern": {"module": 128},
            # peft>=0.19 writes this on every save; submit rejects an unmarked adapter, and
            # these cases are about rank/revision/identity, not modality.
            "exclude_modules": None,
        }

    monkeypatch.setattr(rank_mod, "load_hf_adapter_config", load_config)
    monkeypatch.setattr(
        rank_mod,
        "adapter_artifact_identity",
        lambda *a, **k: rank_mod.AdapterArtifactIdentity(
            "digest", "config", "adapter_model.safetensors", "weight:1"
        ),
    )
    public_spec, worker_spec, identity, source_context = (
        runner_preparation._prepare_init_from_adapter(child, token="token")
    )

    assert calls == [("owner/source-runs:sft/source-run", "token", _REVISION)]
    assert public_spec.train.init_from_adapter == "source-run/final"
    assert worker_spec.train.init_from_adapter == "owner/source-runs:sft/source-run"
    assert worker_spec.train.init_from_adapter_revision == _REVISION
    assert (public_spec.train.lora_rank, public_spec.train.lora_alpha) == (8, 128)
    assert (worker_spec.train.lora_rank, worker_spec.train.lora_alpha) == (64, 128)
    assert identity == {
        "digest": "digest",
        "config_sha256": "config",
        "weight_filename": "adapter_model.safetensors",
        "weight_identity": "weight:1",
    }
    assert source_context is None


def test_qwen36_adapter_is_never_reinterpreted_as_qwen38(monkeypatch):
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.6-27B",
            "model_revision": "a" * 40,
            "model_revision_auto": True,
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = provisioned_status(source, state="done")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.8-27B",
            "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "model_revision_auto": True,
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run/final"},
        }
    )
    monkeypatch.setattr(runner_status, "get_status", lambda run_id: source_status)

    with pytest.raises(ValueError, match=r"unsupported model 'Qwen/Qwen3\.6-27B'"):
        runner_preparation._prepare_init_from_adapter(child, token="token")


def test_prepare_init_adapter_requires_exact_model_revision_match(monkeypatch):
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-9B",
            "model_revision": "a" * 40,
            "model_revision_auto": True,
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = provisioned_status(source, state="done")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "model_revision": "b" * 40,
            "model_revision_auto": True,
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run/final"},
        }
    )
    monkeypatch.setattr(runner_status, "get_status", lambda run_id: source_status)

    with pytest.raises(ValueError, match=r"source model_revision.*does not match target"):
        runner_preparation._prepare_init_from_adapter(child, token="token")


class _ReachedArtifactResolution(Exception):
    """Sentinel: execution got past the warm-start revision check."""


def test_unmanaged_source_revision_is_rejected_during_decode():
    from flash.core.spec import JobSpec

    with pytest.raises(ValueError, match="model_revision requires model_revision_auto=True"):
        JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-9B",
                "model_revision": _REVISION,
                "algorithm": "sft",
                "train": {"hf_repo": "owner/source-runs"},
            }
        )


def test_warm_start_inherits_a_runner_assigned_source_revision(monkeypatch):
    """A GRPO child warm-starting from SFT inherits the source's runner-managed pin."""
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-9B",
            "model_revision": _REVISION,
            "model_revision_auto": True,
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = provisioned_status(source, state="done")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run/final"},
        }
    )
    monkeypatch.setattr(runner_status, "get_status", lambda run_id: source_status)
    monkeypatch.setattr(
        runner_state, "_internal_spec_from_status", lambda status: source, raising=False
    )

    # stop right after the revision reconciliation with a sentinel: the artifact and rank
    # preflights below it need real HF state, and this test is only about which revision the child
    # ends up carrying. Reaching the sentinel IS the pass condition -- it means execution got past
    # the equality check, which is what used to raise.
    monkeypatch.setattr(
        "flash.adapters.lora_rank.resolve_hf_dataset_revision",
        lambda *_a, **_kw: "rev",
    )
    monkeypatch.setattr(
        "flash.runner.results.checkpoints.adapter_artifact_exists",
        lambda *_a, **_kw: (_ for _ in ()).throw(_ReachedArtifactResolution()),
        raising=False,
    )

    # `_inner`, not the public wrapper: the wrapper re-raises everything as
    # WarmStartPreparationError, which would swallow the sentinel and the mismatch alike and leave
    # this test unable to tell them apart. A bare `raises(Exception)` has the same problem -- it
    # would pass on the very mismatch this test exists to rule out.
    with pytest.raises(_ReachedArtifactResolution):
        runner_preparation._prepare_init_from_adapter_inner(child, token="token")


def test_warm_start_pin_is_inherited_before_the_spec_is_sized_against_it(monkeypatch):
    """The inherited pin must be on the spec BEFORE `resolve_model` sizes the run.

    Sizing reads the revision: `resolve_model` re-derives params and vocab from the pinned commit,
    validates them against the catalog row, and applies the revision-specific disk floor. Inheriting
    inside `_prepare_init_from_adapter`, which runs after `resolve_model`, `_with_model_disk`, and
    `_assign_weight_cache_volume`, provisions the child as if unpinned while training it pinned and
    skips the geometry validation the pin exists to enforce.

    So this asserts the ORDER, not just the final value: what `resolve_model` was handed. The
    sibling test above covers the value; only this one fails if the inheritance moves back down.
    """
    import flash.adapters.lora_rank as rank_mod
    import flash.cost.spec as cost_spec
    import flash.runner.results.checkpoints as checkpoints
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-9B",
            "model_revision": _REVISION,
            "model_revision_auto": True,
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = provisioned_status(source, state="done")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run/final", "lora_rank": 8, "lora_alpha": 16},
        }
    )
    seen_revisions = []

    monkeypatch.setattr(runner_status, "get_status", lambda run_id: source_status)
    # the resolver would hit the HF API for the sha; the pin is already immutable, so echo it back
    monkeypatch.setattr(
        runner_preparation, "_resolve_model_revision", lambda spec, *, required=False: spec
    )

    real_resolve = catalog.resolve_model

    def spy(model_id, algorithm, model_revision=""):
        seen_revisions.append(model_revision)
        return real_resolve(model_id, algorithm)  # unpinned: no HF geometry fetch in a unit test

    monkeypatch.setattr(catalog, "resolve_model", spy)
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
            "base_model_name_or_path": "Qwen/Qwen3.5-9B",
            "r": 64,
            "lora_alpha": 128,
            # peft>=0.19 writes this on every save; submit rejects an unmarked adapter, and
            # these cases are about rank/revision/identity, not modality.
            "exclude_modules": None,
        },
    )
    monkeypatch.setattr(
        rank_mod,
        "adapter_artifact_identity",
        lambda *a, **k: rank_mod.AdapterArtifactIdentity(
            "digest", "config", "adapter_model.safetensors", "weight:1"
        ),
    )
    monkeypatch.setattr(cost_spec, "estimate_for_spec", lambda spec: SimpleNamespace(total_usd=1.0))

    prepared = runner_submit.prepare_job(child)

    assert seen_revisions == [_REVISION], seen_revisions
    assert prepared.worker_spec.model_revision == _REVISION
    # and the provenance survives, or deploy refuses the child for a pin it never wrote
    assert prepared.worker_spec.model_revision_auto is True


def _auto_pinned_source(*, org_id="org-a"):
    """A completed, auto-pinned SFT source run owned by ``org_id``."""
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-9B",
            "model_revision": _REVISION,
            "model_revision_auto": True,
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    status = provisioned_status(
        source, state="done", billing_context={"org_id": org_id} if org_id else None
    )
    return source, status


def _unpinned_child():
    from flash.core.spec import JobSpec

    return JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run/final"},
        }
    )


def test_revision_inheritance_refuses_a_source_the_submitter_cannot_access(monkeypatch):
    """Inheritance must authorize the source BEFORE reading its internals.

    The hoisted adoption runs before `_prepare_init_from_adapter`, which is where the same-org
    check used to gate every read of a source run. Without its own check, a child naming another
    org's auto-pinned run would adopt that run's internal revision, and `prepare_job` would then
    do revision-specific `resolve_model` and operator-token HF work against a run the submitter
    cannot access -- surfacing revision-dependent behaviour as ordinary preparation errors rather
    than the deliberately redacted warm-start error.

    Not inheriting is the whole assertion: the submission still fails, but through
    `_prepare_init_from_adapter`'s same-org error, which is the redacted path.
    """

    _, status = _auto_pinned_source(org_id="org-a")
    monkeypatch.setattr(runner_status, "get_status", lambda run_id: status)

    child = _unpinned_child()
    # same org -> inherited
    assert (
        runner_preparation._inherit_warmstart_revision(child, owner_org_id="org-a").model_revision
        == _REVISION
    )
    # different org -> untouched, and the pin never reaches sizing
    foreign = runner_preparation._inherit_warmstart_revision(child, owner_org_id="org-b")
    assert foreign.model_revision == ""
    assert foreign.model_revision_auto is False


def test_revision_inheritance_refuses_a_tampered_source_snapshot(monkeypatch):
    """Inheritance must read through the VALIDATING loader, not the raw internal one.

    `_internal_spec_from_status` returns `snapshot["worker_spec"]` unverified.
    `effective_spec_from_status` checks public/worker equality and the preparation digest first.
    Adopting an unverified internal revision would make the child match a tampered source, so the
    later revision-mismatch guard -- which exists to catch exactly this -- would compare two equal
    values and pass. The child would then train the source adapter against base weights it was
    never created for.
    """

    _, status = _auto_pinned_source(org_id="org-a")
    # tamper: the worker half claims a different base commit than the public half
    status.effective_preparation["worker_spec"]["model_revision"] = "b" * 40
    monkeypatch.setattr(runner_status, "get_status", lambda run_id: status)

    inherited = runner_preparation._inherit_warmstart_revision(
        _unpinned_child(), owner_org_id="org-a"
    )

    assert inherited.model_revision == ""  # nothing adopted from an unverified snapshot
    assert inherited.model_revision_auto is False


def test_warm_start_preparation_refuses_a_tampered_source_snapshot(monkeypatch):
    """The SECOND adoption site must refuse a tampered source too.

    `_prepare_init_from_adapter_inner` re-runs `_adopted_warmstart_revision` so the equality check
    below it cannot depend on which entry point was used. That repeat is a second place a revision
    is copied off the source, and it read the unvalidated `_internal_spec_from_status`. Hardening
    only the hoisted path leaves this one adopting a tampered worker-half revision, after which the
    equality check compares the child against the value it just copied -- two equal values, so it
    passes.

    The sibling `test_revision_inheritance_refuses_a_tampered_source_snapshot` covers the hoisted
    function and never enters this one, which is exactly why this path needs its own test.
    """

    _, status = _auto_pinned_source(org_id="org-a")
    status.effective_preparation["worker_spec"]["model_revision"] = "b" * 40
    monkeypatch.setattr(runner_status, "get_status", lambda run_id: status)
    # would be reached only if the tampered pin were adopted; getting here at all is the failure
    monkeypatch.setattr(
        "flash.adapters.lora_rank.resolve_hf_dataset_revision", lambda *_a, **_kw: "rev"
    )
    monkeypatch.setattr(
        "flash.runner.results.checkpoints.adapter_artifact_exists",
        lambda *_a, **_kw: (_ for _ in ()).throw(_ReachedArtifactResolution()),
        raising=False,
    )

    # either guard inside the validating loader is a correct refusal: this source is auto-pinned,
    # so the digest check now fires before the structural compare can. Matching one specific
    # message would pin WHICH guard catches it rather than THAT it is caught, so this asserts the
    # refusal and rules out the sentinel -- reaching artifact resolution means the tampered pin was
    # adopted, which is the actual regression.
    with pytest.raises(ValueError, match=r"preparation failed integrity|match the public run"):
        runner_preparation._prepare_init_from_adapter_inner(
            _unpinned_child(), owner_org_id="org-a", token="token"
        )


def test_prepare_job_estimates_from_source_effective_worker_spec(monkeypatch):
    import types

    import flash.adapters.lora_rank as rank_mod
    import flash.cost.spec as cost_spec
    import flash.runner.results.checkpoints as checkpoints
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs", "max_context_tokens": 8192},
        }
    )
    source_status = provisioned_status(source, state="done")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {
                "init_from_adapter": "source-run/step-20",
                "lora_rank": 8,
                "lora_alpha": 16,
            },
        }
    )
    estimated_specs = []
    source_reads = []

    def get_source(run_id):
        source_reads.append(run_id)
        return source_status

    monkeypatch.setattr(runner_status, "get_status", get_source)
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
            "base_model_name_or_path": "Qwen/Qwen3.5-9B",
            "r": 64,
            "lora_alpha": 128,
            # peft>=0.19 writes this on every save; submit rejects an unmarked adapter, and
            # these cases are about rank/revision/identity, not modality.
            "exclude_modules": None,
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

    prepared = runner_submit.prepare_job(child)

    assert prepared.public_spec.train.init_from_adapter == "source-run/step-20"
    assert (prepared.public_spec.train.lora_rank, prepared.public_spec.train.lora_alpha) == (8, 128)
    assert prepared.worker_spec.train.init_from_adapter == (
        "owner/source-runs:sft/source-run/checkpoints/step-20"
    )
    assert (prepared.worker_spec.train.lora_rank, prepared.worker_spec.train.lora_alpha) == (
        64,
        128,
    )
    assert estimated_specs == [prepared.worker_spec]
    assert prepared.estimated_cost_usd == 12.5
    assert prepared.prompt_budget["warm_start_context"] == 8192
    assert source_reads == ["source-run", "source-run"]


def test_effective_preparation_persists_but_is_not_public(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    public = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run/final", "lora_rank": 8},
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
    status = runner_state.RunStatus(
        run_id="child-run",
        state="queued",
        spec=public_dict,
        effective_preparation={
            "worker_spec": worker.to_internal_dict(),
            "adapter_identity": identity,
            "version": 1,
            "preparation_digest": runner_preparation._preparation_digest(public, worker, identity),
        },
    )
    runner_state._save_status(status)

    assert "effective_preparation" not in status.to_dict()
    with open(runner_state.runs_file_path("child-run", ".json")) as f:
        stored = json.load(f)
    assert stored["effective_preparation"]["worker_spec"]["train"]["lora_rank"] == 64
    loaded = runner_status.get_status("child-run")
    assert "lora_rank" not in loaded.spec["train"]
    assert loaded.effective_preparation == stored["effective_preparation"]


@pytest.mark.parametrize(
    ("stored_ref", "expected_ref"),
    [
        ("private-owner/private-source:rl/source-run", "source-run/final"),
        ("private-owner/private-source:rl/source-run/checkpoints/step-20", "source-run/step-20"),
    ],
)
def test_public_status_redacts_internal_storage_ref_on_valid_spec(stored_ref, expected_ref):
    # A worker spec persists the internal storage locator (which embeds the private HF repo) as a
    # valid spec; the public status must rewrite it back to
    # the user-facing checkpoint ref instead of leaking the repo.

    raw_spec = {
        "run_id": "child-run",
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "gpu": {"type": "RTX 4090"},
        "train": {"init_from_adapter": stored_ref, "init_from_adapter_revision": _REVISION},
    }
    status = runner_state.RunStatus(run_id="child-run", state="running", spec=raw_spec)

    public_spec = status.to_dict()["spec"]

    assert public_spec["train"]["init_from_adapter"] == expected_ref
    assert "private-source" not in json.dumps(public_spec)
    assert "init_from_adapter_revision" not in public_spec["train"]


@pytest.mark.parametrize("snapshot", [None, []])
def test_persist_effective_warmstart_requires_valid_snapshot(monkeypatch, tmp_path, snapshot):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    public = JobSpec.from_dict(
        {
            "run_id": "legacy-child",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run/final"},
        }
    )
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=public.run_id,
            state="provisioning",
            spec=public.to_dict(),
            effective_preparation=snapshot,
        )
    )

    with pytest.raises(ValueError, match="effective preparation"):
        runner_submit._persist_effective_worker_spec(public)


def test_selected_gpu_is_persisted_for_handleless_cleanup(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    public = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "gpu": {"type": "RTX 4090"},
            "train": {"init_from_adapter": "source-run/final", "lora_rank": 8},
        }
    )
    worker_dict = public.to_internal_dict()
    worker_dict["train"].update(
        {
            "init_from_adapter": "owner/source:rl/source-run",
            "init_from_adapter_revision": _REVISION,
            "lora_rank": 64,
        }
    )
    worker = JobSpec.from_dict(worker_dict)
    identity = {"digest": "immutable-v1"}
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=public.run_id,
            state="provisioning",
            spec=public.to_dict(),
            effective_preparation={
                "worker_spec": worker.to_internal_dict(),
                "adapter_identity": identity,
                "version": 1,
                "preparation_digest": runner_preparation._preparation_digest(
                    public, worker, identity
                ),
            },
        )
    )
    selected_dict = worker.to_internal_dict()
    selected_dict["gpu"]["type"] = "RTX 5090"
    selected = JobSpec.from_dict(selected_dict)

    assert runner_submit._persist_effective_worker_spec(selected)
    stored = runner_status.get_status(public.run_id)
    assert stored.effective_preparation["worker_spec"]["gpu"]["type"] == "RTX 5090"
    assert runner_status.effective_spec_from_status(stored).gpu.type == "RTX 5090"

    cleaned = []

    class Provider:
        def gc(self, spec):
            cleaned.append(spec)

    monkeypatch.setattr(providers, "get_provider", lambda name: Provider())
    # runpod configured: its gc is the one reaping the rN-suffixed endpoints this test is about.
    # (only runpod is available, so the assertion counts one gc, not three.)
    monkeypatch.setattr(providers, "available_providers", lambda: ("runpod",))
    runner_recovery._gc_run_endpoints(public)

    assert [spec.gpu.type for spec in cleaned] == ["RTX 5090"]


def test_recovery_revalidates_pinned_revision_after_default_branch_moves(monkeypatch):
    import flash.adapters.lora_rank as rank_mod
    from flash.core.spec import JobSpec

    public = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run/final", "lora_rank": 8},
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
    status = runner_state.RunStatus(
        run_id=public.run_id,
        state="running",
        spec=public.to_dict(),
        effective_preparation={
            "worker_spec": worker.to_internal_dict(),
            "adapter_identity": identity.to_dict(),
            "version": 1,
            "preparation_digest": runner_preparation._preparation_digest(
                public, worker, identity.to_dict()
            ),
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
            # peft>=0.19 writes this on every save; submit rejects an unmarked adapter, and
            # these cases are about rank/revision/identity, not modality.
            "exclude_modules": None,
        }

    monkeypatch.setattr(rank_mod, "load_hf_adapter_config", load_config)
    monkeypatch.setattr(rank_mod, "adapter_artifact_identity", lambda *a, **k: identity)

    recovered = runner_status.effective_spec_from_status(status, verify_source=True)

    assert recovered == worker
    assert seen == [_REVISION]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", 32),
        ("alpha", 128),
        ("algorithm", "opd"),
        ("environment", "other/environment"),
        ("gpu", "H200"),
        ("hf_repo", "private-owner/other-child"),
        ("source", "private-owner/private-source:rl/other-source"),
    ],
)
def test_effective_snapshot_rejects_tampering(field, value):
    import copy

    from flash.core.spec import JobSpec

    public = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "environment": {"id": "owner/environment"},
            "train": {"init_from_adapter": "source-run/final", "lora_rank": 8, "lora_alpha": 16},
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
        "worker_spec": copy.deepcopy(worker.to_internal_dict()),
        "adapter_identity": identity,
        "version": 1,
        "preparation_digest": runner_preparation._preparation_digest(public, worker, identity),
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
    status = runner_state.RunStatus(
        run_id=public.run_id,
        state="running",
        spec=public.to_dict(),
        effective_preparation=snapshot,
    )
    with pytest.raises(ValueError, match="effective preparation"):
        runner_status.effective_spec_from_status(status)


@pytest.mark.parametrize(
    "retired_model",
    ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"],
)
def test_effective_spec_rejects_persisted_removed_model_before_activation(retired_model):
    from flash.core.spec import JobSpec

    current = JobSpec.from_dict(
        {"run_id": "retired", "model": "Qwen/Qwen3.5-9B", "algorithm": "sft"}
    )
    stored = current.to_dict()
    stored["model"] = retired_model
    status = runner_state.RunStatus(run_id="retired", state="running", spec=stored)

    with pytest.raises(ValueError, match="unsupported model"):
        runner_status.effective_spec_from_status(status)


def test_worker_metrics_sanitizer_redacts_nested_and_direct_private_refs():
    from flash.engine.result.accounting import RunMetrics, sanitize_worker_metrics

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
    import flash.server.domain.registry.runs as registry
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    private_ref = "private-owner/private-repo:rl/source-run"
    spec = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
        }
    )
    reported = []
    monkeypatch.setattr(
        registry,
        "record_training_checkpoint",
        lambda **kwargs: reported.append(kwargs["metrics"]),
    )
    runner_status._persist_metrics(
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
    with open(os.path.join(runner_state.artifacts_dir(spec), "metrics.json")) as f:
        persisted = json.load(f)
    assert reported == [persisted]
    assert "private-owner" not in json.dumps(persisted)
    assert "private-repo" not in json.dumps(persisted)
    assert "private/child" not in json.dumps(persisted)


@pytest.mark.parametrize("source_algorithm", ["sft", "grpo", "opd"])
@pytest.mark.parametrize("target_algorithm", ["sft", "grpo", "opd"])
def test_every_algorithm_pair_prepares_a_warm_start(
    monkeypatch, source_algorithm, target_algorithm
):
    """All nine source/target combinations resolve a warm start identically.

    Warm start never compared the two algorithms; what it had instead was a blanket refusal of SFT
    as the TARGET, which removed three of the nine cells. This walks the whole matrix so a
    re-introduced asymmetry -- in either direction -- fails here rather than in a paid run.

    Each cell asserts the two things preparation owes the child regardless of algorithm: it adopts
    the source's pin (with its provenance, so the child stays deployable), and it reaches artifact
    resolution instead of stopping at a policy check. The sentinel IS the pass condition, exactly as
    in the sibling inheritance tests above.
    """
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-9B",
            "model_revision": _REVISION,
            # runner-assigned, which is the pin shape a child must be able to inherit AND deploy.
            "model_revision_auto": True,
            "algorithm": source_algorithm,
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = provisioned_status(source, state="done")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": source.model,
            "algorithm": target_algorithm,
            "train": {"init_from_adapter": f"{source.run_id}/final"},
        }
    )
    monkeypatch.setattr(runner_status, "get_status", lambda run_id: source_status)

    inherited = runner_preparation._inherit_warmstart_revision(child)
    assert inherited.model_revision == _REVISION
    # carried, not re-derived: a self-resolved pin reads as author-supplied and deploy refuses it.
    assert inherited.model_revision_auto is True

    monkeypatch.setattr(
        "flash.adapters.lora_rank.resolve_hf_dataset_revision",
        lambda *_a, **_kw: "rev",
    )
    monkeypatch.setattr(
        "flash.runner.results.checkpoints.adapter_artifact_exists",
        lambda *_a, **_kw: (_ for _ in ()).throw(_ReachedArtifactResolution()),
        raising=False,
    )

    # `_inner`, not the public wrapper: the wrapper flattens everything into
    # WarmStartPreparationError, which would make a real rejection indistinguishable from the
    # sentinel and let this test pass on the very block it exists to rule out.
    with pytest.raises(_ReachedArtifactResolution):
        runner_preparation._prepare_init_from_adapter_inner(inherited, token="token")


def test_sft_child_prepares_against_the_inherited_source_pin(monkeypatch):
    """A warm-started SFT run profiles and sizes against the SOURCE's pin, not a fresh one.

    SFT is the only algorithm `prepare_job` force-pins, and it does so immediately after the
    inheritance. If the child resolved its own pin instead, it would take the base model's CURRENT
    hub tip -- so `_prepare_init_from_adapter_inner`'s equality check would reject every source
    trained before the tip last moved, and SFT continuation would work only by luck. The profile
    digest is keyed on the same revision (`_require_sft_workload_profile`), so the tokenizer would
    also disagree with the weights actually being continued.

    Asserts what the resolver was HANDED, not just the end state: a `required=True` call arriving
    with an empty revision is the regression, even when the final spec happens to look right.
    """
    import flash.adapters.lora_rank as rank_mod
    import flash.cost.spec as cost_spec
    import flash.runner.results.checkpoints as checkpoints
    from flash.core.spec import JobSpec

    source, source_status = _auto_pinned_source(org_id="")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": source.model,
            "algorithm": "sft",
            "environment": {"id": "freesolo/math-agent/gsm8k"},
            "train": {"init_from_adapter": "source-run/final", "epochs": 1, "max_examples": 8},
        }
    )
    resolver_calls = []

    monkeypatch.setattr(runner_status, "get_status", lambda run_id: source_status)

    def fake_resolve(spec, *, required=False):
        resolver_calls.append((spec.model_revision, required))
        return spec

    monkeypatch.setattr(runner_preparation, "_resolve_model_revision", fake_resolve)
    # a pinned spec makes `resolve_model` re-derive geometry from the commit, which is a live HF
    # read; size unpinned so this stays a unit test of the ordering.
    real_resolve = catalog.resolve_model
    monkeypatch.setattr(
        catalog,
        "resolve_model",
        lambda model_id, algorithm, model_revision="": real_resolve(model_id, algorithm),
    )
    # sft-only preparation: the packaged dataset and its pinned env are a separate contract from
    # warm start, and profiling one here would test the profiler rather than the inheritance.
    profiled = []
    monkeypatch.setattr(
        runner_preparation, "_require_pinned_profile_environment", lambda spec: spec
    )
    monkeypatch.setattr(
        runner_preparation,
        "_require_sft_workload_profile",
        lambda spec: profiled.append(spec.model_revision) or spec,
    )
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
            "base_model_name_or_path": source.model,
            "r": 64,
            "lora_alpha": 128,
            # peft>=0.19 writes this on every save; submit rejects an unmarked adapter, and
            # these cases are about rank/revision/identity, not modality.
            "exclude_modules": None,
        },
    )
    monkeypatch.setattr(
        rank_mod,
        "adapter_artifact_identity",
        lambda *a, **k: rank_mod.AdapterArtifactIdentity(
            "digest", "config", "adapter_model.safetensors", "weight:1"
        ),
    )
    monkeypatch.setattr(cost_spec, "estimate_for_spec", lambda spec: SimpleNamespace(total_usd=1.0))

    prepared = runner_submit.prepare_job(child)

    # the force-pin ran, and it was already holding the source's revision when it did.
    assert resolver_calls == [(_REVISION, True)], resolver_calls
    # the tokenizer the profile digest keys on is the base the adapter was actually trained against
    assert profiled == [_REVISION], profiled
    assert prepared.worker_spec.model_revision == _REVISION
    assert prepared.worker_spec.model_revision_auto is True
    # source metadata stays authoritative for rank/alpha on this path too
    assert prepared.worker_spec.train.lora_rank == 64
    assert prepared.worker_spec.train.lora_alpha == 128


def test_inherited_sft_pin_survives_the_force_pin_when_the_hub_tip_moved(monkeypatch):
    """The source's sha must outlive `_resolve_model_revision`, not just `_inherit_warmstart_revision`.

    sft is force-pinned (`required=True`), but an inherited runner-managed pin is already resolved.
    if the resolver overwrites that sha with the current hub tip, warm start breaks when the base model
    moves, and `_adopted_warmstart_revision` cannot repair it because the pin is already set. Stubbing
    the resolver would hide exactly that, so this test drives the real one over a moved hub.
    """
    from flash.core.spec import JobSpec

    moved_tip = "b" * 40

    class _Info:
        sha = moved_tip

    class _Api:
        def __init__(self, *_a, **_kw):
            pass

        def model_info(self, _model, revision=None):
            # a real hub resolves None to whatever the tip is now, which is no longer the source's
            return _Info() if revision is None else SimpleNamespace(sha=revision)

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)

    inherited = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "model_revision": _REVISION,
            "model_revision_auto": True,
            "train": {"init_from_adapter": "source-run/final", "hf_repo": "org/repo"},
            "environment": {"id": "org/env"},
        }
    )

    resolved = runner_preparation._resolve_model_revision(inherited, required=True)

    assert resolved.model_revision == _REVISION, (
        f"force-pin overwrote the inherited source pin with {resolved.model_revision!r}"
    )
    assert resolved.model_revision_auto is True


def test_force_pin_still_pins_a_fresh_sft_run_to_the_hub_tip(monkeypatch):
    """The short-circuit above must not disarm the force-pin it sits in front of."""
    from flash.core.spec import JobSpec

    tip = "b" * 40

    class _Api:
        def __init__(self, *_a, **_kw):
            pass

        def model_info(self, _model, revision=None):
            return SimpleNamespace(sha=tip if revision is None else revision)

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)

    fresh = JobSpec.from_dict(
        {
            "run_id": "fresh-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "train": {"hf_repo": "org/repo"},
            "environment": {"id": "org/env"},
        }
    )

    resolved = runner_preparation._resolve_model_revision(fresh, required=True)

    assert resolved.model_revision == tip
    assert resolved.model_revision_auto is True


def test_recovery_reproduces_a_digest_taken_before_lora_alpha_was_public(monkeypatch, tmp_path):
    """A run prepared before 1.1.35 stored no public `lora_alpha`; its digest must still verify.

    `to_dict()` now MATERIALIZES alpha (defaulting to 2 * lora_rank), so rehashing such a run with
    today's serialization hashes bytes it never had. `workload_profile` predates the change
    (1.1.32), so the digest branch is genuinely reachable for runs in that window -- and
    `reallocation_spec_from_status` is what the retry path calls, so a mismatch marks a live run
    `unrecoverable` rather than retrying it.
    """
    import hashlib

    from flash.core.spec import JobSpec
    from flash.core.spec_persistence import PREPARATION_ENVELOPE_VERSION

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    public = JobSpec.from_dict(
        {
            "run_id": "pre-alpha-run",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "gpu": {"type": "RTX 4090"},
            "train": {"lora_rank": 32},
        }
    )
    # the stored public spec as an older build wrote it: no `lora_alpha` key at all.
    stored_public = public.to_dict()
    stored_public["train"].pop("lora_alpha", None)
    assert "lora_alpha" not in stored_public["train"]

    worker_dict = public.to_internal_dict()
    worker_dict["workload_profile"] = {"tokens": 128}
    worker = JobSpec.from_dict(worker_dict)
    # the digest the ORIGINAL build computed, hashed HERE rather than through the function under
    # test: deriving it from `_preparation_digest` would move with the code and pass either way.
    # that build normalized too -- falsy managed keys and an empty `environment.pip` were omitted --
    # so the fixture has to reproduce THAT shape, not this build's raw serialization. `pip` is
    # already absent from `stored_public` because `to_dict()` emits an empty tuple here.
    original_worker = worker.to_internal_dict()
    for key in (
        "model_revision_auto",
        "model_revision_force_pin",
        "gpu_count_auto",
        "workload_profile_input_digest",
        "workload_profile_producer_version",
        "workload_profile",
    ):
        if not original_worker.get(key):
            original_worker.pop(key, None)
    original_public = {**stored_public, "environment": dict(stored_public["environment"])}
    if not original_public["environment"].get("pip"):
        original_public["environment"].pop("pip", None)
    original_digest = hashlib.sha256(
        json.dumps(
            {
                "version": PREPARATION_ENVELOPE_VERSION,
                "public_spec": original_public,
                "worker_spec": original_worker,
                "adapter_identity": None,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    runner_state._save_status(
        runner_state.RunStatus(
            run_id=public.run_id,
            state="provisioning",
            spec=stored_public,
            effective_preparation={
                "worker_spec": worker.to_internal_dict(),
                "adapter_identity": None,
                "version": 1,
                "workload_profile": worker.workload_profile or None,
                "preparation_digest": original_digest,
            },
        )
    )

    recovered = runner_status.effective_spec_from_status(runner_status.get_status(public.run_id))
    assert recovered.train.lora_rank == 32


def test_persisting_a_pre_alpha_run_does_not_rewrite_its_digest_out_of_reach(monkeypatch, tmp_path):
    """Rewriting the snapshot must replay the same `lora_alpha` omission recovery replays.

    `_persist_effective_worker_spec` runs on the ATTACH and RESUBMIT paths (`supervise/attach.py`,
    `supervise/attempt_supervision.py`), rebuilding `public_spec` from `status.spec` -- whose
    `to_dict()` re-materializes an alpha the stored record never had -- and it never updates
    `status.spec` to match. So a pre-1.1.35 run that recovers once is retired on its NEXT attach:
    `server/platform/runtime.py` marks it `unrecoverable` when `reallocation_spec_from_status`
    raises. The trigger gate is `workload_profile or model_revision_auto`, and auto-pinning is the
    default, so this is the ordinary case rather than a corner.
    """
    from dataclasses import replace

    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    public = replace(
        JobSpec.from_dict(
            {
                "run_id": "pre-alpha-persist",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "sft",
                "gpu": {"type": "RTX 4090"},
                "train": {"lora_rank": 32},
            }
        ),
        model_revision="a" * 40,
        model_revision_auto=True,
    )
    stored_public = public.to_dict()
    stored_public["train"].pop("lora_alpha", None)
    assert "lora_alpha" not in stored_public["train"]

    runner_state._save_status(
        runner_state.RunStatus(
            run_id=public.run_id,
            state="provisioning",
            spec=stored_public,
            effective_preparation={
                "worker_spec": public.to_internal_dict(),
                "adapter_identity": None,
                "version": 1,
                "workload_profile": None,
                "preparation_digest": runner_preparation._preparation_digest(
                    public, public, None, stored_public=stored_public
                ),
            },
        )
    )
    # the record recovers BEFORE the rewrite -- so a failure after it is caused by the rewrite
    # itself, not by a fixture that was never valid.
    assert (
        runner_status.reallocation_spec_from_status(
            runner_status.get_status(public.run_id)
        ).train.lora_rank
        == 32
    )

    assert runner_submit._persist_effective_worker_spec(public)

    # the run survives its own attach: the rewritten digest is still reproducible from the stored
    # public spec, which the rewrite left untouched.
    recovered = runner_status.reallocation_spec_from_status(runner_status.get_status(public.run_id))
    assert recovered.train.lora_rank == 32


def test_digest_matches_the_shape_the_deployed_release_writes(monkeypatch, tmp_path):
    """The digest must hash the canonical shape `dev` writes TODAY, not this build's raw dict.

    `_preparation_digest` omits falsy managed keys and an empty `environment.pip`. That is
    NORMALIZATION, not a legacy replay: it is unconditional, reads nothing off the stored record,
    and defines the digest the currently deployed release (1.2.88) computes for every run it
    prepares. Dropping it would make the digest depend on which build wrote the record, so a
    snapshot written by production stops verifying here -- and `reallocation_spec_from_status` is
    the retry path, which `server/platform/runtime.py` turns into `unrecoverable`.

    The expected digest is hashed HERE over dev's canonical bytes rather than taken from the
    function under test, so it cannot move with the code.
    """
    import hashlib
    from dataclasses import replace

    from flash.core.spec import JobSpec
    from flash.core.spec_persistence import PREPARATION_ENVELOPE_VERSION

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = replace(
        JobSpec.from_dict(
            {
                "run_id": "canonical-digest-run",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "sft",
                "gpu": {"type": "RTX 4090"},
                "train": {"lora_rank": 32},
            }
        ),
        model_revision="a" * 40,
        model_revision_auto=True,
    )

    public_payload = spec.to_dict()
    worker_payload = spec.to_internal_dict()
    if not public_payload["environment"].get("pip"):
        public_payload["environment"].pop("pip", None)
    stripped = []
    for key in (
        "model_revision_auto",
        "model_revision_force_pin",
        "gpu_count_auto",
        "workload_profile_input_digest",
        "workload_profile_producer_version",
        "workload_profile",
    ):
        if not worker_payload.get(key):
            worker_payload.pop(key, None)
            stripped.append(key)
    # the fixture is only meaningful if this spec actually carries keys the normalization drops.
    assert stripped, "spec must exercise at least one omitted managed key"

    canonical_digest = hashlib.sha256(
        json.dumps(
            {
                "version": PREPARATION_ENVELOPE_VERSION,
                "public_spec": public_payload,
                "worker_spec": worker_payload,
                "adapter_identity": None,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    assert (
        runner_preparation._preparation_digest(spec, spec, None, stored_public=spec.to_dict())
        == canonical_digest
    )

    # and a record carrying that digest -- the shape production writes -- recovers on the retry path.
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            effective_preparation={
                "worker_spec": spec.to_internal_dict(),
                "adapter_identity": None,
                "version": PREPARATION_ENVELOPE_VERSION,
                "workload_profile": None,
                "preparation_digest": canonical_digest,
            },
        )
    )
    assert (
        runner_status.reallocation_spec_from_status(
            runner_status.get_status(spec.run_id)
        ).train.lora_rank
        == 32
    )


def test_a_snapshot_written_before_the_version_key_still_recovers(monkeypatch, tmp_path):
    """The `version` stamp landed in 1.2.59; runs prepared by an older build are still in flight.

    Rejecting an ABSENT version makes `reallocation_spec_from_status` raise on the retry path,
    which marks a live run `unrecoverable` instead of retrying it. A malformed value is still
    rejected -- absence is a known shape, a bad type is not.
    """
    from flash.core.spec_persistence import (
        PREPARATION_ENVELOPE_VERSION,
        validate_persisted_spec_envelope,
    )

    assert (
        validate_persisted_spec_envelope({"worker_spec": {}, "preparation_digest": "d"})
        == PREPARATION_ENVELOPE_VERSION
    )
    with pytest.raises(ValueError, match="positive integer"):
        validate_persisted_spec_envelope({"version": "1"})
    with pytest.raises(ValueError, match="unsupported persisted preparation envelope version"):
        validate_persisted_spec_envelope({"version": PREPARATION_ENVELOPE_VERSION + 1})
