"""Worker adapter downloads consume internal storage refs, not public init_from_adapter refs."""

import json
import os
import shutil
import struct
from types import SimpleNamespace

import pytest

# The LIVE package proxy, not `import flash.engine.worker as W`. The code under test reads its
# run-scoped globals through this same proxy (see flash/engine/worker/model/adapter.py), which
# resolves `sys.modules['flash.engine.worker']` on every access. A module-level `import ... as W`
# binds one package OBJECT at collection time; any earlier test that drops the package from
# sys.modules and re-imports it (tests/test_worker_stack.py does, to re-run the import scope that
# captures RUN_MODE/JOB_SPEC) replaces that object, and this alias keeps pointing at the dead one.
# Patching the stale alias then sets an attribute nothing reads -- `revision` comes back None and
# the failure is attributed to this file rather than the re-import that caused it. Patching through
# the proxy targets whatever package the code will actually read.
from flash.engine.worker.runtime.pkg_proxy import W
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
        tensor_name = "base_model.model.layers.0.q_proj.lora_A.default.weight"
        header = json.dumps(
            {tensor_name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        ).encode()
        with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            fh.write(struct.pack("<f", 1.0))

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
    import flash.adapters.lora_rank as rank_mod
    import flash.runner as R
    import flash.runner.results.checkpoints as checkpoints
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = provisioned_status(R, source, state="done")
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
    public_spec, worker_spec, identity, source_context = R._prepare_init_from_adapter(
        child, token="token"
    )

    assert calls == [("owner/source-runs:sft/source-run", "token", _REVISION)]
    assert public_spec.train.init_from_adapter == "source-run"
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


def test_prepare_init_adapter_requires_exact_model_revision_match(monkeypatch):
    import flash.runner as R
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": "source-revision",
            "model_revision_auto": True,
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = provisioned_status(R, source, state="done")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": "target-revision",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run"},
        }
    )
    monkeypatch.setattr(R, "get_status", lambda run_id: source_status)

    with pytest.raises(ValueError, match=r"source model_revision.*does not match target"):
        R._prepare_init_from_adapter(child, token="token")


class _ReachedArtifactResolution(Exception):
    """Sentinel: execution got past the warm-start revision check."""


def test_warm_start_inherits_a_legacy_authored_source_revision(monkeypatch):
    """A child inherits an authored source pin without laundering its deploy provenance."""
    from fastapi import HTTPException

    import flash.runner as R
    import flash.server.routes.serving as serving
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": _REVISION,
            "algorithm": "grpo",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    stored_public = {**source.to_dict(), "model_revision": _REVISION}
    source_status = R.RunStatus(
        state="done",
        run_id=source.run_id,
        spec=stored_public,
        effective_preparation={
            "worker_spec": source.to_internal_dict(),
            "preparation_digest": R._preparation_digest(
                JobSpec.from_dict(stored_public),
                source,
                None,
                legacy_public_keys={"model_revision": _REVISION},
            ),
        },
    )
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": source.model,
            "algorithm": "opd",
            "train": {"init_from_adapter": source.run_id},
        }
    )
    monkeypatch.setattr(R, "get_status", lambda run_id: source_status)

    inherited = R._inherit_warmstart_revision(child)
    assert inherited.model_revision == _REVISION
    assert inherited.model_revision_auto is False

    monkeypatch.setattr(
        "flash.adapters.lora_rank.resolve_hf_dataset_revision",
        lambda *_a, **_kw: "rev",
    )
    monkeypatch.setattr(
        "flash.runner.results.checkpoints.adapter_artifact_exists",
        lambda *_a, **_kw: (_ for _ in ()).throw(_ReachedArtifactResolution()),
        raising=False,
    )
    with pytest.raises(_ReachedArtifactResolution):
        R._prepare_init_from_adapter_inner(child, token="token")

    deploy_status = R.RunStatus(
        state="done",
        run_id=inherited.run_id,
        spec=inherited.to_dict(),
        effective_preparation={"worker_spec": inherited.to_internal_dict()},
    )
    with pytest.raises(HTTPException, match="legacy revision-pinned base model"):
        serving._validate_deploy_request(
            inherited.run_id,
            deploy_status,
            JobSpec.from_dict(deploy_status.spec),
            {},
            True,
        )


def test_warm_start_inherits_a_runner_assigned_source_revision(monkeypatch):
    """A GRPO child warm-starting off SFT inherits the parent's auto pin AND its provenance.

    SFT is always force-pinned by the runner, and this check demands the child's revision equal the
    source's. Before this, satisfying it meant the AUTHOR writing the sha into rl.toml -- which made
    the child's pin author-supplied, which deploy refuses. So a warm start off SFT could pass this
    check or be deployable, never both.

    The paired mismatch control is `test_prepare_init_adapter_requires_exact_model_revision_match`
    above: an already-pinned child is never overwritten, so a different target revision still raises.
    """
    import flash.runner as R
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": _REVISION,
            "model_revision_auto": True,
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = provisioned_status(R, source, state="done")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run"},
        }
    )
    monkeypatch.setattr(R, "get_status", lambda run_id: source_status)
    monkeypatch.setattr(R, "_internal_spec_from_status", lambda status: source, raising=False)

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
        R._prepare_init_from_adapter_inner(child, token="token")


def test_warm_start_pin_is_inherited_before_the_spec_is_sized_against_it(monkeypatch):
    """The inherited pin must be on the spec BEFORE `resolve_model` sizes the run.

    Sizing reads the revision: `resolve_model` re-derives params/vocab from the pinned commit and
    raises `min_disk_gb` to `ceil(2 * params_b) + 64`, which for half of today's catalog exceeds the
    catalog default (Qwen3.5-4B: 0 -> 73). Inheriting inside `_prepare_init_from_adapter`, which
    runs after `resolve_model`, `_with_model_disk`, and `_assign_weight_cache_volume`, provisions
    the child as if unpinned while training it pinned, and skips the geometry validation the pin
    exists to enforce.

    So this asserts the ORDER, not just the final value: what `resolve_model` was handed. The
    sibling test above covers the value; only this one fails if the inheritance moves back down.
    """
    import flash.adapters.lora_rank as rank_mod
    import flash.cost.spec as cost_spec
    import flash.runner as R
    import flash.runner.results.checkpoints as checkpoints
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": _REVISION,
            "model_revision_auto": True,
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    source_status = provisioned_status(R, source, state="done")
    child = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run", "lora_rank": 8, "lora_alpha": 16},
        }
    )
    seen_revisions = []

    monkeypatch.setattr(R, "get_status", lambda run_id: source_status)
    # the resolver would hit the HF API for the sha; the pin is already immutable, so echo it back
    monkeypatch.setattr(R, "_resolve_model_revision", lambda spec, *, required=False: spec)

    real_resolve = R.resolve_model

    def spy(model_id, algorithm, model_revision=""):
        seen_revisions.append(model_revision)
        return real_resolve(model_id, algorithm)  # unpinned: no HF geometry fetch in a unit test

    monkeypatch.setattr(R, "resolve_model", spy)
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
    monkeypatch.setattr(cost_spec, "estimate_for_spec", lambda spec: SimpleNamespace(total_usd=1.0))

    prepared = R.prepare_job(child)

    assert seen_revisions == [_REVISION], seen_revisions
    assert prepared.worker_spec.model_revision == _REVISION
    # and the provenance survives, or deploy refuses the child for a pin it never wrote
    assert prepared.worker_spec.model_revision_auto is True


def _auto_pinned_source(R, *, org_id="org-a"):
    """A completed, auto-pinned SFT source run owned by ``org_id``."""
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": _REVISION,
            "model_revision_auto": True,
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs"},
        }
    )
    status = provisioned_status(
        R, source, state="done", billing_context={"org_id": org_id} if org_id else None
    )
    return source, status


def _unpinned_child():
    from flash.core.spec import JobSpec

    return JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run"},
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
    import flash.runner as R

    _, status = _auto_pinned_source(R, org_id="org-a")
    monkeypatch.setattr(R, "get_status", lambda run_id: status)

    child = _unpinned_child()
    # same org -> inherited
    assert R._inherit_warmstart_revision(child, owner_org_id="org-a").model_revision == _REVISION
    # different org -> untouched, and the pin never reaches sizing
    foreign = R._inherit_warmstart_revision(child, owner_org_id="org-b")
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
    import flash.runner as R

    _, status = _auto_pinned_source(R, org_id="org-a")
    # tamper: the worker half claims a different base commit than the public half
    status.effective_preparation["worker_spec"]["model_revision"] = "b" * 40
    monkeypatch.setattr(R, "get_status", lambda run_id: status)

    inherited = R._inherit_warmstart_revision(_unpinned_child(), owner_org_id="org-a")

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
    import flash.runner as R

    _, status = _auto_pinned_source(R, org_id="org-a")
    status.effective_preparation["worker_spec"]["model_revision"] = "b" * 40
    monkeypatch.setattr(R, "get_status", lambda run_id: status)
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
        R._prepare_init_from_adapter_inner(_unpinned_child(), owner_org_id="org-a", token="token")


def test_prepare_job_estimates_from_source_effective_worker_spec(monkeypatch):
    import types

    import flash.adapters.lora_rank as rank_mod
    import flash.cost.spec as cost_spec
    import flash.runner as R
    import flash.runner.results.checkpoints as checkpoints
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "train": {"hf_repo": "owner/source-runs", "max_context_tokens": 8192},
        }
    )
    source_status = provisioned_status(R, source, state="done")
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
    source_reads = []

    def get_source(run_id):
        source_reads.append(run_id)
        return source_status

    monkeypatch.setattr(R, "get_status", get_source)
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
    import flash.runner as R
    from flash.core.spec import JobSpec

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
            "worker_spec": worker.to_internal_dict(),
            "adapter_identity": identity,
            "preparation_digest": R._preparation_digest(public, worker, identity),
        },
    )
    R._save_status(status)

    assert "effective_preparation" not in status.to_dict()
    legacy_status = R.RunStatus(
        run_id="legacy-child",
        state="queued",
        spec=public.to_internal_dict(),
    )
    public_status_spec = legacy_status.to_dict()["spec"]
    assert "init_from_adapter_revision" not in public_status_spec["train"]
    assert "lora_rank" not in public_status_spec["train"]
    with open(R.runs_file_path("child-run", ".json")) as f:
        stored = json.load(f)
    assert stored["effective_preparation"]["worker_spec"]["train"]["lora_rank"] == 64
    loaded = R.get_status("child-run")
    assert "lora_rank" not in loaded.spec["train"]
    assert loaded.effective_preparation == stored["effective_preparation"]


@pytest.mark.parametrize(
    "raw_spec",
    [
        None,
        "legacy-spec",
        {"train": "legacy-train", "gpu": {}},
        {"train": {}, "gpu": "legacy-gpu"},
    ],
)
def test_public_status_tolerates_malformed_legacy_spec_shapes(raw_spec):
    import flash.runner as R

    status = R.RunStatus(
        run_id="legacy-run",
        state="running",
        spec=raw_spec,
        effective_preparation={"private": "snapshot"},
    )

    public = status.to_dict()

    assert public["spec"] == raw_spec
    assert "effective_preparation" not in public


@pytest.mark.parametrize("init_ref", ["source-run", 123, 0, {}])
def test_public_status_redacts_identifiable_warmstart_fields_from_malformed_spec(init_ref):
    import flash.runner as R

    raw_spec = {
        "train": {
            "init_from_adapter": init_ref,
            "init_from_adapter_revision": _REVISION,
            "lora_rank": 64,
        },
        "gpu": "legacy-gpu",
    }
    status = R.RunStatus(run_id="legacy-run", state="running", spec=raw_spec)

    public_spec = status.to_dict()["spec"]

    assert public_spec["train"] == {"init_from_adapter": init_ref}
    assert raw_spec["train"]["init_from_adapter_revision"] == _REVISION
    assert raw_spec["train"]["lora_rank"] == 64


@pytest.mark.parametrize(
    ("stored_ref", "expected_ref"),
    [
        ("private-owner/private-source:rl/source-run", "source-run"),
        ("private-owner/private-source:rl/source-run/checkpoints/step-20", "source-run/step-20"),
    ],
)
def test_public_status_redacts_internal_storage_ref_on_valid_spec(stored_ref, expected_ref):
    # A worker/effective or legacy record can persist the internal storage locator (which embeds the
    # private HF repo) as a VALID spec that parses cleanly; the public status must rewrite it back to
    # the user-facing checkpoint ref instead of leaking the repo.
    import flash.runner as R

    raw_spec = {
        "run_id": "child-run",
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "gpu": {"type": "RTX 4090"},
        "train": {"init_from_adapter": stored_ref, "init_from_adapter_revision": _REVISION},
    }
    status = R.RunStatus(run_id="child-run", state="running", spec=raw_spec)

    public_spec = status.to_dict()["spec"]

    assert public_spec["train"]["init_from_adapter"] == expected_ref
    assert "private-source" not in json.dumps(public_spec)
    assert "init_from_adapter_revision" not in public_spec["train"]


@pytest.mark.parametrize("snapshot", [None, []])
def test_persist_effective_warmstart_requires_valid_snapshot(monkeypatch, tmp_path, snapshot):
    import flash.runner as R
    from flash.core.spec import JobSpec

    monkeypatch.setattr(R, "RUNS_DIR", str(tmp_path / "runs"))
    public = JobSpec.from_dict(
        {
            "run_id": "legacy-child",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {"init_from_adapter": "source-run"},
        }
    )
    R._save_status(
        R.RunStatus(
            run_id=public.run_id,
            state="provisioning",
            spec=public.to_dict(),
            effective_preparation=snapshot,
        )
    )

    with pytest.raises(ValueError, match="effective preparation"):
        R._persist_effective_worker_spec(public)


def test_selected_gpu_is_persisted_for_handleless_cleanup(monkeypatch, tmp_path):
    import flash.providers as providers
    import flash.runner as R
    from flash.core.spec import JobSpec

    monkeypatch.setattr(R, "RUNS_DIR", str(tmp_path / "runs"))
    public = JobSpec.from_dict(
        {
            "run_id": "child-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "gpu": {"type": "RTX 4090"},
            "train": {"init_from_adapter": "source-run", "lora_rank": 8},
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
    R._save_status(
        R.RunStatus(
            run_id=public.run_id,
            state="provisioning",
            spec=public.to_dict(),
            effective_preparation={
                "worker_spec": worker.to_internal_dict(),
                "adapter_identity": identity,
                "preparation_digest": R._preparation_digest(public, worker, identity),
            },
        )
    )
    selected_dict = worker.to_internal_dict()
    selected_dict["gpu"]["type"] = "RTX 5090"
    selected = JobSpec.from_dict(selected_dict)

    assert R._persist_effective_worker_spec(selected)
    stored = R.get_status(public.run_id)
    assert stored.effective_preparation["worker_spec"]["gpu"]["type"] == "RTX 5090"
    assert R.effective_spec_from_status(stored).gpu.type == "RTX 5090"

    cleaned = []

    class Provider:
        def gc(self, spec):
            cleaned.append(spec)

    monkeypatch.setattr(providers, "get_provider", lambda name: Provider())
    # runpod configured: its gc is the one reaping the rN-suffixed endpoints this test is about.
    # (only runpod is available, so the assertion counts one gc, not three.)
    monkeypatch.setattr(providers, "available_providers", lambda: ("runpod",))
    R._gc_run_endpoints(public)

    assert [spec.gpu.type for spec in cleaned] == ["RTX 5090"]


def test_recovery_revalidates_pinned_revision_after_default_branch_moves(monkeypatch):
    import flash.adapters.lora_rank as rank_mod
    import flash.runner as R
    from flash.core.spec import JobSpec

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
            "worker_spec": worker.to_internal_dict(),
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
        ("gpu", "H200"),
        ("hf_repo", "private-owner/other-child"),
        ("source", "private-owner/private-source:rl/other-source"),
    ],
)
def test_effective_snapshot_rejects_tampering(field, value):
    import copy

    import flash.runner as R
    from flash.core.spec import JobSpec

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
        "worker_spec": copy.deepcopy(worker.to_internal_dict()),
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
    import flash.runner as R
    import flash.server.domain.run_registry as registry
    from flash.core.spec import JobSpec

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
    from flash.core.spec import JobSpec

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
