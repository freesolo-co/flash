"""submission preparation for warm starts, validation, and sft dataset estimates.

continuing from an existing adapter requires verifying the parent run's identity, ownership, and
preparation digest before reusing its weights. sft preparation separately resolves the pinned
environment package, reads its dataset file without importing user code, and attaches the raw-record
token estimate required for quoting and training preparation.

split out of `flash.runner` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import replace

from flash.core.spec import MANAGED_GPU_KEYS, JobSpec
from flash.core.spec_persistence import PREPARATION_ENVELOPE_VERSION
from flash.runner.lifecycle import status as status_ops


class WarmStartPreparationError(ValueError):
    """a submit failed while preparing the source adapter."""


class WorkloadProfileUnavailable(ValueError):
    """the sft packaged-dataset estimate failed or cannot be trusted."""


class EnvironmentRefNotFound(ValueError):
    """the environment ref cannot be resolved by the control plane."""


def _context_org_id(context: dict | None) -> str:
    if not isinstance(context, dict):
        return ""
    return str(context.get("org_id") or "").strip()


def _status_org_id(status) -> str:
    return _context_org_id(status.billing_context) or _context_org_id(status.platform_context)


def _source_owned_by_key(src_run_id: str, owner_key_id: int | None) -> bool:
    if owner_key_id is None:
        return False
    try:
        from flash.server.platform import db

        return db.run_owner(src_run_id) == owner_key_id
    except Exception:
        return False


def _adopted_warmstart_revision(spec: JobSpec, src_spec: JobSpec) -> JobSpec:
    """Take the warm-start source's runner-managed immutable base-model pin."""
    if spec.model_revision or not src_spec.model_revision:
        return spec
    if not src_spec.model_revision_auto:
        raise ValueError("warm-start source has an unsupported unmanaged model revision")
    return replace(
        spec,
        model_revision=src_spec.model_revision,
        model_revision_auto=True,
    )


def _warmstart_source_is_authorized(
    src_status,
    src_run_id: str,
    *,
    owner_org_id: str = "",
    owner_key_id: int | None = None,
) -> bool:
    """Whether the submitter may read the warm-start source run's internals.

    Shared by the two places that read a source run, so neither can drift into a weaker rule than
    the other. An empty ``owner_org_id`` means the caller has no org context (local/self-hosted
    submission), which is the pre-existing contract for this check.
    """
    owner_org_id = owner_org_id.strip()
    if not owner_org_id:
        return True
    src_org_id = _status_org_id(src_status)
    if src_org_id:
        return src_org_id == owner_org_id
    return _source_owned_by_key(src_run_id, owner_key_id)


def _inherit_warmstart_revision(
    spec: JobSpec,
    *,
    owner_org_id: str = "",
    owner_key_id: int | None = None,
) -> JobSpec:
    """Adopt a warm-start source's pin BEFORE the spec is sized against it.

    Sizing reads the revision: ``resolve_model`` re-derives params/vocab/disk from the pinned
    commit's geometry, and ``min_disk_gb`` becomes ``ceil(2 * params_b) + 64``, which for half of today's
    catalog is strictly larger than the catalog default. Adopting the pin only inside
    ``_prepare_init_from_adapter`` -- which runs after ``resolve_model``, ``_with_model_disk``, and
    ``_assign_weight_cache_volume`` -- would provision the child as if unpinned while training it
    pinned, and skip the geometry validation the pin exists to enforce.

    Best-effort by design: every way the source can be unusable (unknown run, wrong org, wrong
    model, missing artifacts) is diagnosed by ``_prepare_init_from_adapter`` with its own message
    and its own error type. Raising here would report those as generic submission failures instead,
    so an unreadable source simply leaves the spec untouched and the real check speaks.

    Applies to an SFT target too, and that case is the one with no second chance. SFT is the only
    algorithm ``prepare_job`` force-pins (``_resolve_model_revision(required=True)``), and that call
    runs immediately after this one. Skipping the inheritance here would let the child resolve its
    own pin to whatever the base model's hub tip is now, so a source trained before the tip moved
    fails the equality check in ``_prepare_init_from_adapter_inner``. inheriting first also preserves
    the source pin's runner-managed provenance.

    Two ordering rules this function must not relax, because it now runs BEFORE the code that used
    to enforce them:

    * authorize before reading. Adopting a pin off a run the submitter cannot access would leak
      revision-dependent behaviour through ordinary preparation errors and would do operator-token
      HF work on an unauthorized run's commit.
    * read through ``effective_spec_from_status``, not ``_internal_spec_from_status``. The latter
      returns ``snapshot["worker_spec"]`` unverified; the former checks public/worker equality and
      the preparation digest first. Adopting an unverified internal revision would make the child
      match a tampered source, which is precisely what the later mismatch guard exists to catch --
      and inheriting it would silence that guard rather than trip it.
    """
    ref = spec.train.init_from_adapter
    if spec.model_revision or not ref:
        return spec
    from flash.schema import parse_checkpoint_ref

    parsed = parse_checkpoint_ref(ref)
    if parsed is None:
        return spec
    try:
        src_status = status_ops.get_status(parsed[0])
        if not _warmstart_source_is_authorized(
            src_status, parsed[0], owner_org_id=owner_org_id, owner_key_id=owner_key_id
        ):
            return spec  # _prepare_init_from_adapter raises the same-org error
        src_spec = status_ops.effective_spec_from_status(src_status)
    except Exception:
        return spec
    if src_spec.model != spec.model:
        return spec  # _prepare_init_from_adapter raises on this with the specific message
    return _adopted_warmstart_revision(spec, src_spec)


def _prepare_init_from_adapter(
    spec: JobSpec,
    *,
    owner_org_id: str = "",
    owner_key_id: int | None = None,
    token: str | None = None,
) -> tuple[JobSpec, JobSpec, dict | None, int | None]:
    """prepare public and worker specs with source-authoritative adapter metadata.

    Failures here are genuinely about the warm-start source, so they are tagged
    ``WarmStartPreparationError`` for the submit route. Everything else in ``prepare_job`` (gpu
    sizing, budget, environment resolution) must keep its own message rather than be reported as a
    bad adapter, since those run for non-warm-start runs too and have nothing to do with the
    adapter.
    """
    try:
        return _prepare_init_from_adapter_inner(
            spec, owner_org_id=owner_org_id, owner_key_id=owner_key_id, token=token
        )
    except WarmStartPreparationError:
        raise
    except Exception as exc:
        raise WarmStartPreparationError(str(exc)) from exc


def _prepare_init_from_adapter_inner(
    spec: JobSpec,
    *,
    owner_org_id: str = "",
    owner_key_id: int | None = None,
    token: str | None = None,
) -> tuple[JobSpec, JobSpec, dict | None, int | None]:
    ref = spec.train.init_from_adapter
    if not ref:
        return spec, spec, None, None
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config
    from flash.adapters.lora_rank import (
        adapter_artifact_identity,
        load_hf_adapter_config,
        preflight_init_adapter_lora_rank,
        resolve_hf_dataset_revision,
    )
    from flash.adapters.targets import require_modality_marker
    from flash.runner.results.checkpoints import CheckpointListingError, adapter_artifact_exists
    from flash.schema import checkpoint_storage_ref, format_checkpoint_ref, parse_checkpoint_ref

    parsed = parse_checkpoint_ref(ref)
    if parsed is None:
        raise ValueError(
            "train.init_from_adapter must be `<run_id>/final` or `<run_id>/step-N` "
            f"(a checkpoint listed by `flash runs checkpoint`); got {ref!r}"
        )
    src_run_id, step = parsed
    try:
        src_status = status_ops.get_status(src_run_id)
    except FileNotFoundError:
        raise ValueError(f"train.init_from_adapter references unknown run {src_run_id!r}") from None
    owner_org_id = owner_org_id.strip()
    if not _warmstart_source_is_authorized(
        src_status, src_run_id, owner_org_id=owner_org_id, owner_key_id=owner_key_id
    ):
        raise ValueError("train.init_from_adapter source run must belong to the same Freesolo org")
    # hf_repo is platform-managed and stripped from the source run's public spec; its authoritative
    # value lives in that run's internal worker spec, which the warm-start needs to locate the
    # source adapter artifacts.
    #
    # read through the VALIDATING loader, not `_internal_spec_from_status`: the latter returns
    # `snapshot["worker_spec"]` unverified, and `_adopted_warmstart_revision` below copies a
    # revision off it. A tampered worker half would be adopted onto the child, which then EQUALS
    # it, so the mismatch check below compares two equal values and passes -- the adoption
    # silences the guard instead of tripping it. `effective_spec_from_status` verifies public/
    # worker equality and the preparation digest first.
    src_spec = status_ops.effective_spec_from_status(src_status)
    if src_spec.model != spec.model:
        raise ValueError(
            f"train.init_from_adapter source model {src_spec.model!r} does not match target model "
            f"{spec.model!r}"
        )
    # normally a no-op: `prepare_job` adopts the pin before it sizes anything. this repeats the
    # decision for callers that reach the warm-start path directly, so the equality check below
    # cannot depend on which entry point was used.
    spec = _adopted_warmstart_revision(spec, src_spec)
    if src_spec.model_revision != spec.model_revision:
        raise ValueError(
            "train.init_from_adapter source model_revision "
            f"{src_spec.model_revision!r} does not match target model_revision "
            f"{spec.model_revision!r}"
        )
    if not src_spec.train.hf_repo:
        raise ValueError(
            f"train.init_from_adapter run {src_run_id!r} has no stored adapter artifacts"
        )
    if step is None and src_status.state not in {"done", "deployed"}:
        raise ValueError(
            f"train.init_from_adapter references run {src_run_id!r}, but that run is "
            f"{src_status.state!r}; use a completed source run or a concrete "
            f"{src_run_id}/step-N checkpoint"
        )
    storage = checkpoint_storage_ref(src_spec.train.hf_repo, src_spec.phase, src_run_id, step)
    revision = resolve_hf_dataset_revision(src_spec.train.hf_repo, token)
    try:
        exists = adapter_artifact_exists(src_spec, step=step, revision=revision)
    except CheckpointListingError as exc:
        raise ValueError(str(exc)) from exc
    if not exists:
        target = format_checkpoint_ref(src_run_id, step)
        raise ValueError(
            f"train.init_from_adapter references {target!r}, but its complete adapter artifact "
            "was not found"
        )
    worker_spec = replace(
        spec,
        train=replace(
            spec.train,
            init_from_adapter=storage,
            init_from_adapter_revision=revision,
        ),
    )
    config = load_hf_adapter_config(storage, token, revision)
    # here rather than only in the worker: this is the same config the worker will re-download and
    # re-check, and an unmarked adapter is unusable by every algorithm, so deciding it now turns a
    # post-allocation worker failure into a submit-time error and spends no GPU on a doomed run.
    require_modality_marker(config, source=f"train.init_from_adapter source {ref!r}")
    validate_fused_expert_adapter_config(config, spec.model)
    metadata = preflight_init_adapter_lora_rank(
        worker_spec, token=token, config_loader=lambda _ref, _token, _revision: config
    )
    assert metadata is not None
    identity = adapter_artifact_identity(storage, config, token, revision).to_dict()
    public_spec = replace(spec, train=replace(spec.train, lora_alpha=metadata.alpha))
    worker_spec = replace(
        worker_spec,
        train=replace(worker_spec.train, lora_rank=metadata.rank, lora_alpha=metadata.alpha),
    )
    source_context = int(getattr(src_spec.train, "max_context_tokens", 0) or 0) or None
    return public_spec, worker_spec, identity, source_context


def _mark_warmstart_source(worker_spec: JobSpec, child_run_id: str) -> None:
    """Drop a 0-byte ``referenced_by/<child_run_id>`` marker into the warm-start SOURCE run's HF repo.

    ``worker_spec`` is post-resolution, so its ``init_from_adapter`` is the internal
    ``<repo>:<phase>/<run_id>...`` storage ref whose repo is the source. Best-effort: a failed
    marker never blocks submission -- it only forfeits the GC grace (the source can still be spared
    by being deployed or recently written).
    """
    import io

    ref = worker_spec.train.init_from_adapter
    if not ref or ":" not in ref or not child_run_id or child_run_id == "local":
        return
    source_repo = ref.split(":", 1)[0].strip()
    if not source_repo:
        return
    with contextlib.suppress(Exception):
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=io.BytesIO(b""),
            path_in_repo=f"referenced_by/{child_run_id}",
            repo_id=source_repo,
            repo_type="dataset",
        )


def _preparation_digest(
    public_spec: JobSpec,
    worker_spec: JobSpec,
    adapter_identity: dict | None,
    *,
    stored_public: object = None,
) -> str:
    public_payload = public_spec.to_dict()
    # `lora_alpha` became user-authorable in 1.1.35, so `to_dict()` now MATERIALIZES it (defaulting
    # to 2 * lora_rank) where a spec prepared before that stored no key at all. rehashing such a
    # run with today's serialization hashes bytes it never had, so its digest can only be
    # reproduced by replaying the omission. `workload_profile` predates the change (1.1.32), so the
    # digest branch is genuinely reachable for runs in that window.
    #
    # this cannot forge a value: the omission is read from the STORED public spec, and the digest
    # it is compared against was computed over the original. a warm start is excluded because
    # `to_dict()` strips rank and alpha for those in EVERY version, so absence there dates nothing
    # and dropping the binding would lose real cover -- `_validate_effective_spec` excludes alpha
    # from its structural comparison, leaving the digest as the only thing tying the two halves.
    if isinstance(stored_public, Mapping):
        stored_train = stored_public.get("train")
        if (
            isinstance(stored_train, Mapping)
            and "lora_alpha" not in stored_train
            and not stored_train.get("init_from_adapter")
        ):
            public_payload["train"].pop("lora_alpha", None)
    worker_payload = worker_spec.to_internal_dict()
    # NORMALIZATION, not legacy replay: this is the canonical shape `dev` hashes TODAY (1.2.88), and
    # it is unconditional -- it reads nothing off the stored record and applies to every run this
    # build prepares. Deleting it as "historical" made the digest depend on which build wrote it, so
    # a record the DEPLOYED release is writing right now stops verifying here. Measured: a spec
    # hashed dev's way then recovered on this branch fails `reallocation_spec_from_status`, which
    # `server/platform/runtime.py:697-710` turns into `unrecoverable` -- retiring in-flight runs on
    # upgrade. The genuinely historical half (`VersionedPersistedSpecEnvelope.rewind`) stays deleted.
    #
    # absent and empty are the same value for all of these, so hashing them alike costs no binding:
    # a set field is non-empty and stays covered, and every key is derived from the spec rather than
    # read from the snapshot, so nothing here can be forged by tampering with the stored record.
    if not public_payload["environment"].get("pip"):
        public_payload["environment"].pop("pip", None)
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
    payload = {
        "version": PREPARATION_ENVELOPE_VERSION,
        "public_spec": public_payload,
        "worker_spec": worker_payload,
        "adapter_identity": adapter_identity,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_effective_spec(public_spec: JobSpec, worker_spec: JobSpec) -> None:
    public = public_spec.to_internal_dict()
    effective = worker_spec.to_internal_dict()
    # run_id is a platform-managed top-level field stripped from the public spec, so the
    # reconstructed public spec carries only its default ("local"). exclude these from the
    # structural check; their integrity is covered by the sha256 preparation digest, and the
    # worker spec is already keyed by run_id at the persist boundary.
    for managed_top in (
        "run_id",
        # provenance of a runner-assigned pin: stripped from the public spec (which must stay
        # re-parseable by the submission schema), so the reconstructed public spec always reads
        # False while the worker half carries the real value. Comparing them would reject every
        # auto-pinned run here -- the same runs the deploy guard was just relaxed to admit.
        "model_revision",
        "model_revision_auto",
        "gpu_count_auto",
        "workload_profile_input_digest",
        "workload_profile_producer_version",
        "workload_profile",
    ):
        effective[managed_top] = public.get(managed_top)
    public_train = dict(public["train"])
    effective_train = dict(effective["train"])
    public_ref = public_train.get("init_from_adapter") or ""
    internal_ref = effective_train.get("init_from_adapter") or ""
    excluded_train_fields = [
        "init_from_adapter",
        "init_from_adapter_revision",
        # platform-managed artifact repo: stripped from the public spec (digest-protected), so the
        # reconstructed public spec carries only the default. exclude it from the structural check.
        "hf_repo",
    ]
    # rank and alpha diverge between the halves ONLY on a warm start, where the source adapter's
    # topology is authoritative and _prepared_warm_start_specs() writes it onto the worker spec.
    # Without one they are equal by construction, so comparing them costs nothing and is the only
    # thing covering them on a plain run: that path reaches neither digest branch below (no
    # workload profile, no warm-start source), so an edited public rank/alpha would otherwise be
    # accepted and recovery would train with a topology the reported spec disagrees with.
    if public_ref or internal_ref:
        excluded_train_fields += ["lora_rank", "lora_alpha"]
    for train_field in excluded_train_fields:
        effective_train[train_field] = public_train.get(train_field)
    effective["train"] = effective_train
    public_environment = dict(public["environment"])
    effective_environment = dict(effective["environment"])
    # the staged package is controller-managed: staging writes it onto the worker half only, so the
    # public half never carries one and comparing the two directly would fail every staged run.
    # this exclusion cannot be tightened into a tamper check here. `to_internal_dict` OMITS the key
    # when unset, so a stripped package is byte-identical to a run that simply has not staged yet --
    # which is the state of every run between submit and allocation, and of every finished run in
    # the hosted tests. rejecting that shape fails 96 legitimate runs
    # (tests/test_server_api.py, tests/test_client_server_integration.py), and widening the digest
    # trigger instead fails ordinary runs whose gpu.type the allocator rewrote at provisioning
    # (tests/test_server_api.py::test_deploy_ignores_stored_training_gpu).
    # stripping it is contained at the consumer rather than here: `stage_environment_package` sees
    # no package and re-stages from the SAME digest-bound `resolved_sha`, and the worker's loader
    # raises "worker job spec has no staged environment package" rather than importing anything.
    # a substituted package is the case that must fail closed, and does: it is bound to the pin by
    # `verify_staged_environment`, which re-derives both digests before any environment code loads.
    effective_environment.pop("package", None)
    public_sha = public_environment.get("resolved_sha")
    effective_sha = effective_environment.get("resolved_sha")
    if not public_sha and isinstance(effective_sha, str):
        from flash.envs.loading.loader import is_commit_sha

        if is_commit_sha(effective_sha):
            effective_environment["resolved_sha"] = ""
    effective["environment"] = effective_environment
    public_gpu = dict(public["gpu"])
    # `type` is excluded for the same reason `count` is below: _spec_with_gpu writes the selected
    # class onto the worker spec. the fallbacks ride with it -- they qualify `type`, so comparing
    # them against a public half whose head has not been rewritten would fail every ordered pin
    # that allocated to anything but its first class.
    effective_gpu = {**effective["gpu"], "type": public_gpu["type"]}
    # `to_internal_dict` omits an empty fallback list rather than emitting one, so mirror the key's
    # presence as well as its value -- writing an unconditional `()` here would add a key the public
    # half does not carry and fail every single-class run.
    if public_gpu.get("type_fallbacks"):
        effective_gpu["type_fallbacks"] = public_gpu["type_fallbacks"]
    else:
        effective_gpu.pop("type_fallbacks", None)
    # _spec_with_gpu writes the SELECTED count onto the worker spec -- the worker sizes its
    # rank count from it and the provider payload rents it -- so comparing it against the
    # authored ceiling would fail every narrowed run here, before any provider is reached.
    # narrowing only: a worker spec claiming MORE cards than the run authorized is a real
    # integrity failure and still raises. the exact selected count is digest-protected and
    # persisted for launch.
    effective_count = int(effective_gpu.get("count", 1) or 1)
    public_count = int(public_gpu.get("count", 1) or 1)
    # an auto-sized run has NO authored ceiling: its public count is the digest-stable placeholder
    # 1, so a legitimately auto-sized 2+ card shape would fail the narrowing rule below and be
    # rejected as an integrity failure at persist time. the real ceiling for that run is the
    # platform maximum. this branch is still bounded: MAX_COMBINATION_CARDS is the same cap
    # allocation itself honours, and the count that lands here was produced by a VRAM fit check and
    # the attention-head geometry cap, so a marker cannot buy a shape those would refuse.
    from flash.providers.core.sharding import MAX_COMBINATION_CARDS

    ceiling = public_count if worker_spec.authored_gpu_count is not None else MAX_COMBINATION_CARDS
    if 1 <= effective_count <= ceiling:
        effective_gpu["count"] = public_gpu.get("count")
    # disk sizing, the weight-cache volume, and retry/wall-clock lifecycle policy are platform-managed
    # (MANAGED_GPU_KEYS) and stripped from the public spec, so the reconstructed public spec carries
    # only defaults for them. exclude them from the structural comparison; their integrity is covered
    # by the sha256 preparation digest, and the committed weight-cache volume is guarded against
    # illegitimate removal at the persist boundary (see _reject_managed_volume_removal).
    for managed_gpu in MANAGED_GPU_KEYS:
        effective_gpu[managed_gpu] = public_gpu.get(managed_gpu)
    effective["gpu"] = effective_gpu
    if effective != public:
        raise ValueError("persisted effective preparation does not match the public run")
    if not public_ref:
        if internal_ref or worker_spec.train.init_from_adapter_revision:
            raise ValueError("persisted effective preparation has an unexpected source adapter")
        return
    from flash.schema import parse_adapter_storage_ref, parse_checkpoint_ref

    public_target = parse_checkpoint_ref(public_ref)
    resolved = parse_adapter_storage_ref(internal_ref)
    if public_target is None or resolved is None:
        raise ValueError("persisted effective preparation has an invalid source adapter")
    _repo, prefix = resolved
    match = re.fullmatch(
        r"(?:sft|rl|opd)/(?P<run>[A-Za-z0-9][A-Za-z0-9._-]{0,127})"
        r"(?:/checkpoints/step-(?P<step>\d+))?",
        prefix,
    )
    if match is None:
        raise ValueError("persisted effective preparation has an invalid source adapter")
    source_run, source_step = public_target
    internal_step = int(match.group("step")) if match.group("step") is not None else None
    if match.group("run") != source_run or internal_step != source_step:
        raise ValueError("persisted effective preparation source does not match the public run")
    if not worker_spec.train.init_from_adapter_revision:
        raise ValueError("persisted effective preparation has no pinned source revision")


def _resolve_model_revision(spec: JobSpec, *, required: bool = False) -> JobSpec:
    if spec.model_revision and not spec.model_revision_auto:
        raise ValueError("unmanaged model_revision is unsupported")
    from flash.core.catalog import MODELS

    catalog_pin = getattr(MODELS.get(spec.model), "managed_revision", "") or ""
    if catalog_pin and spec.model_revision and spec.model_revision != catalog_pin:
        raise ValueError(
            f"model {spec.model!r} requires immutable revision {catalog_pin}; inherited "
            f"model_revision {spec.model_revision!r} is incompatible"
        )
    if catalog_pin and not spec.model_revision:
        spec = replace(
            spec,
            model_revision=catalog_pin,
            model_revision_auto=True,
            model_revision_force_pin=True,
        )
    if not spec.model_revision and not required:
        return spec
    # an inherited warm-start pin is already an immutable sha chosen by a previous run, so there is
    # nothing left to resolve. a forced pin is excluded because it is a one-shot request to verify
    # that exact immutable commit before clearing the request marker.
    if (
        spec.model_revision_auto
        and not spec.model_revision_force_pin
        and re.fullmatch(r"[0-9a-f]{40}", spec.model_revision or "")
    ):
        return spec
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(
            spec.model,
            revision=spec.model_revision if spec.model_revision_force_pin else None,
        )
        reported = str(getattr(info, "sha", "") or "").strip()
        resolved = reported.lower()
        if re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
            raise ValueError("resolved revision is not an immutable commit")
        if spec.model_revision_force_pin and reported != spec.model_revision:
            raise ValueError("resolved revision does not match the forced immutable pin")
    except Exception as exc:
        raise ValueError(
            f"could not resolve model_revision for model {spec.model!r}; "
            "verify that the revision exists and the operator token can access it"
        ) from exc
    return replace(
        spec,
        model_revision=resolved,
        model_revision_auto=True,
        model_revision_force_pin=False,
    )


def _profile_producer_version() -> str:
    from flash import __version__

    return str(__version__)


def _require_pinned_profile_environment(spec: JobSpec) -> JobSpec:
    from flash.runner.accounting.artifacts import _pin_env_sha_with_reason

    pinned, reason = _pin_env_sha_with_reason(spec)
    if not pinned.environment.id:
        raise WorkloadProfileUnavailable("sft workload profiling requires an environment id")
    if not pinned.environment.resolved_sha:
        # the pin is best-effort, so reaching here says only that it did not happen -- and the four
        # causes (a ref that does not exist, a rate limit, an outage, a private repo the token
        # cannot read) need four different fixes. GitHub already answered with which one; name it
        # instead of describing the missing pin it produced.
        detail = f": {reason}" if reason else ""
        raise WorkloadProfileUnavailable(
            f"sft workload profiling requires a pinned environment package revision, but "
            f"{spec.environment.id!r} could not be resolved{detail}"
        )
    return pinned


def _require_sft_workload_profile(spec: JobSpec) -> JobSpec:
    """attach the control-plane sft estimate built from the packaged dataset file."""
    from flash.engine.profiling import dataset_profile
    from flash.engine.profiling.workload_profile import (
        require_matching_sft_profile,
        sft_profile_input_digest,
    )

    producer_version = _profile_producer_version()
    tokenizer_revision = spec.model_revision
    input_digest = sft_profile_input_digest(
        spec,
        tokenizer_revision=tokenizer_revision,
        producer_version=producer_version,
    )
    profiled_spec = replace(
        spec,
        workload_profile_input_digest=input_digest,
        workload_profile_producer_version=producer_version,
        workload_profile={},
    )
    try:
        measured = dataset_profile.profile_packaged_sft_dataset(
            profiled_spec,
            producer_version=producer_version,
        )
        profile = require_matching_sft_profile(
            measured.to_dict(),
            input_digest=input_digest,
            producer_version=producer_version,
            tokenizer_revision=tokenizer_revision,
        )
    except WorkloadProfileUnavailable:
        raise
    except dataset_profile.PackagedDatasetUnavailable as exc:
        raise WorkloadProfileUnavailable(str(exc)) from exc
    return replace(profiled_spec, workload_profile=profile.to_dict())
