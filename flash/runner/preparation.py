"""Everything a submission has to settle before a run exists: warm-start, validation, profile.

Continuing from an existing adapter is the bulk of it. The parent run's identity, ownership and
preparation digest all have to be verified before its weights are reused, because a run that
silently continued from the wrong adapter would train and bill normally and only look wrong much
later. `_require_sft_workload_profile` is the other half: SFT cannot be quoted without a measured
profile for the exact config, so it either finds one or launches the job that produces it.

Split out of `flash.runner` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from dataclasses import replace
from typing import TYPE_CHECKING

from flash.core.spec import JobSpec

if TYPE_CHECKING:
    # annotation-only: defined in `flash.runner` above the point where it imports this module, so a
    # runtime import would be circular. the value is read through `_runner()` where it is needed.
    from flash.runner import PreparedJob


def _runner():
    """The runner package, imported lazily because it re-exports this module.

    Five of these helpers are patched as attributes of `flash.runner` by the warm-start and
    workload-profile tests, and they call each other, so every cross-helper call and every shared
    constant is resolved through the package rather than bound here. Binding by value would make a
    patch on `flash.runner` rebind a name this module never reads.
    """
    import flash.runner as runner

    return runner


def _require_supported_adapter_continuation(spec: JobSpec) -> None:
    if spec.algorithm == "sft" and spec.train.init_from_adapter:
        raise ValueError(
            "train.init_from_adapter is supported only for GRPO and OPD continue-in-place runs; "
            "SFT adapter continuation is not supported"
        )


def _adopted_warmstart_revision(spec: JobSpec, src_spec: JobSpec) -> JobSpec:
    """Take the warm-start source's pin when the runner, not the author, chose it.

    SFT is always force-pinned by ``_resolve_model_revision(required=True)``, and the warm-start
    check demands the child's revision equal the source's. Satisfying it meant the AUTHOR writing
    the sha into rl.toml, which makes the child's pin author-supplied, which deploy refuses. So a
    warm start off SFT could pass that check or be deployable, never both. Inheriting the pin AND
    its provenance gives the child the same immutable base its parent trained against and keeps it
    deployable for the same reason the parent is.
    """
    if spec.model_revision or not src_spec.model_revision or not src_spec.model_revision_auto:
        return spec
    return replace(spec, model_revision=src_spec.model_revision, model_revision_auto=True)


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
    src_org_id = _runner()._status_org_id(src_status)
    if src_org_id:
        return src_org_id == owner_org_id
    return _runner()._source_owned_by_key(src_run_id, owner_key_id)


def _inherit_warmstart_revision(
    spec: JobSpec,
    *,
    owner_org_id: str = "",
    owner_key_id: int | None = None,
) -> JobSpec:
    """Adopt a warm-start source's runner-assigned pin BEFORE the spec is sized against it.

    Sizing reads the revision: ``resolve_model`` re-derives params/vocab/disk from the pinned
    commit's geometry, and ``min_disk_gb`` becomes ``params_b * 2 + 64``, which for half of today's
    catalog is strictly larger than the catalog default. Adopting the pin only inside
    ``_prepare_init_from_adapter`` -- which runs after ``resolve_model``, ``_with_model_disk``, and
    ``_assign_weight_cache_volume`` -- would provision the child as if unpinned while training it
    pinned, and skip the geometry validation the pin exists to enforce.

    Best-effort by design: every way the source can be unusable (unknown run, wrong org, wrong
    model, missing artifacts) is diagnosed by ``_prepare_init_from_adapter`` with its own message
    and its own error type. Raising here would report those as generic submission failures instead,
    so an unreadable source simply leaves the spec untouched and the real check speaks.

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
    if spec.model_revision or not ref or spec.algorithm == "sft":
        return spec
    from flash.schema import parse_checkpoint_ref

    parsed = parse_checkpoint_ref(ref)
    if parsed is None:
        return spec
    try:
        src_status = _runner().get_status(parsed[0])
        if not _runner()._warmstart_source_is_authorized(
            src_status, parsed[0], owner_org_id=owner_org_id, owner_key_id=owner_key_id
        ):
            return spec  # _prepare_init_from_adapter raises the same-org error
        src_spec = _runner().effective_spec_from_status(src_status)
    except Exception:
        return spec
    if src_spec.model != spec.model:
        return spec  # _prepare_init_from_adapter raises on this with the specific message
    return _runner()._adopted_warmstart_revision(spec, src_spec)


def _prepare_init_from_adapter(
    spec: JobSpec,
    *,
    owner_org_id: str = "",
    owner_key_id: int | None = None,
    token: str | None = None,
) -> tuple[JobSpec, JobSpec, dict | None]:
    """prepare public and worker specs with source-authoritative adapter metadata.

    Failures here are genuinely about the warm-start source, so they are tagged
    ``_runner().WarmStartPreparationError`` for the submit route. Everything else in ``prepare_job`` (gpu
    sizing, budget, environment resolution) must keep its own message rather than be reported as a
    bad adapter, since those run for non-warm-start runs too and have nothing to do with the
    adapter.
    """
    try:
        return _runner()._prepare_init_from_adapter_inner(
            spec, owner_org_id=owner_org_id, owner_key_id=owner_key_id, token=token
        )
    except _runner().WarmStartPreparationError:
        raise
    except Exception as exc:
        raise _runner().WarmStartPreparationError(str(exc)) from exc


def _prepare_init_from_adapter_inner(
    spec: JobSpec,
    *,
    owner_org_id: str = "",
    owner_key_id: int | None = None,
    token: str | None = None,
) -> tuple[JobSpec, JobSpec, dict | None]:
    _runner()._require_supported_adapter_continuation(spec)
    ref = spec.train.init_from_adapter
    if not ref:
        return spec, spec, None
    from flash.adapters.lora_rank import (
        adapter_artifact_identity,
        load_hf_adapter_config,
        preflight_init_adapter_lora_rank,
        resolve_hf_dataset_revision,
    )
    from flash.runner.results.checkpoints import CheckpointListingError, adapter_artifact_exists
    from flash.schema import checkpoint_storage_ref, parse_checkpoint_ref

    parsed = parse_checkpoint_ref(ref)
    if parsed is None:
        raise ValueError(
            "train.init_from_adapter must be `<run_id>` or `<run_id>/step-N` "
            f"(a checkpoint listed by `flash runs checkpoint`); got {ref!r}"
        )
    src_run_id, step = parsed
    try:
        src_status = _runner().get_status(src_run_id)
    except FileNotFoundError:
        raise ValueError(f"train.init_from_adapter references unknown run {src_run_id!r}") from None
    owner_org_id = owner_org_id.strip()
    if not _runner()._warmstart_source_is_authorized(
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
    src_spec = _runner().effective_spec_from_status(src_status)
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
        target = f"{src_run_id}/step-{step}" if step is not None else src_run_id
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
    return public_spec, worker_spec, identity


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


def _prepared_before_public_alpha(raw_public: object) -> bool:
    """True when this run's PERSISTED public spec predates ``lora_alpha`` becoming authorable.

    The discriminator is the stored bytes, not this build's serialization: a snapshot prepared
    before the change hashed a public spec with no alpha key, and its digest can only be reproduced
    the same way. status.spec is never rewritten, so this answer is stable for the run's life.

    A warm-start spec is excluded even though it also omits alpha -- to_dict() strips rank and alpha
    for those in every version, so absence there says nothing about when the run was prepared, and
    treating it as legacy would drop a binding that costs nothing to keep.
    """
    if not isinstance(raw_public, dict):
        return False
    train = raw_public.get("train")
    if not isinstance(train, dict) or "lora_alpha" in train:
        return False
    return not train.get("init_from_adapter")


def _preparation_digest(
    public_spec: JobSpec,
    worker_spec: JobSpec,
    adapter_identity: dict | None,
    *,
    legacy_keys: dict | None = None,
    legacy_public_keys: dict | None = None,
    legacy_public_alpha: bool = False,
) -> str:
    worker_payload = worker_spec.to_internal_dict()
    public_payload = public_spec.to_dict()
    # omit empty fields so existing version-1 snapshots keep their historical digest.
    for key in (
        "model_revision_auto",
        "workload_profile_kind",
        "workload_profile_input_digest",
        "workload_profile_producer_version",
        "workload_profile",
    ):
        if not worker_payload.get(key):
            worker_payload.pop(key, None)
    # Restore since-removed keys the STORED payload carried, for the same reason as the omissions
    # above: the digest has to reproduce the bytes that were hashed, not today's serialization. A
    # pre-upgrade snapshot hashed `model_policy` in (to_internal_dict was asdict, so it emitted the
    # defaulted value), and the field no longer exists -- so rehashing without it mismatches and a
    # still-valid warm-start or workload-profile run fails integrity validation on recovery. Only
    # keys the spec itself has dropped are honoured; anything the dataclass still defines comes from
    # worker_spec, so this cannot be used to forge a field.
    for key, value in (legacy_keys or {}).items():
        if key in _runner()._DROPPED_TOP_LEVEL_KEYS:
            worker_payload[key] = value
    # a dropped key that was USER-AUTHORABLE was hashed on the public side too, not just the worker
    # side. model_policy never was, so restoring only the worker payload was enough for it; worker_env
    # was, so a pre-upgrade public_spec carried it and the digest cannot reproduce without it.
    for key, value in (legacy_public_keys or {}).items():
        if key in _runner()._DROPPED_TOP_LEVEL_KEYS:
            public_payload[key] = value
    # ``lora_alpha`` became user-authorable, so to_dict() now emits it for non-warm-start runs. A
    # snapshot prepared BEFORE that change hashed a public spec without alpha, so rehashing it with
    # alpha would fail a still-valid run's integrity check on recovery -- same reason as the
    # omissions and restorations above. That omission is scoped to those legacy snapshots only: a
    # digest created from here on binds the public alpha, so tampering with the persisted public
    # value is caught. Without that binding the two halves can disagree silently, because
    # _runner()._validate_effective_spec() excludes lora_alpha from its structural comparison (the worker's
    # warm-start-inherited alpha legitimately differs from the public one), leaving the digest as
    # the only thing that could cover it.
    if legacy_public_alpha:
        public_payload["train"].pop("lora_alpha", None)
    payload = {
        "version": 1,
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
        "model_revision_auto",
        "workload_profile_kind",
        "workload_profile_input_digest",
        "workload_profile_producer_version",
        "workload_profile",
    ):
        effective[managed_top] = public.get(managed_top)
    # the pin VALUE travels with its marker: `to_dict()` drops a runner-assigned revision so the
    # public spec cannot advertise a SHA it has no way to label, which leaves the halves legitimately
    # asymmetric. exclude the value only in that case -- an AUTHORED pin is still compared, so a
    # tampered authored revision keeps failing here. the auto-pinned value is not left unguarded:
    # it is hashed into the preparation digest, and `effective_spec_from_status` now verifies that
    # digest whenever the marker is set.
    if worker_spec.model_revision_auto and not public.get("model_revision"):
        effective["model_revision"] = public.get("model_revision")
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
    public_sha = public_environment.get("resolved_sha")
    effective_sha = effective_environment.get("resolved_sha")
    if not public_sha and isinstance(effective_sha, str):
        from flash.envs.loader import is_commit_sha

        if is_commit_sha(effective_sha):
            effective_environment["resolved_sha"] = ""
    effective["environment"] = effective_environment
    public_gpu = dict(public["gpu"])
    effective_gpu = {**effective["gpu"], "type": public_gpu["type"]}
    # _spec_with_gpu writes the SELECTED count onto the worker spec -- the worker sizes its
    # rank count from it and the provider payload rents it -- so comparing it against the
    # authored ceiling would fail every narrowed run here, before any provider is reached.
    # narrowing only: a worker spec claiming MORE cards than the run authorized is a real
    # integrity failure and still raises. the exact selected count is digest-protected and
    # persisted for launch.
    effective_count = int(effective_gpu.get("count", 1) or 1)
    public_count = int(public_gpu.get("count", 1) or 1)
    if 1 <= effective_count <= public_count:
        effective_gpu["count"] = public_gpu.get("count")
    # disk sizing, the weight-cache volume, and retry/wall-clock lifecycle policy are platform-managed
    # (_runner().MANAGED_GPU_KEYS) and stripped from the public spec, so the reconstructed public spec carries
    # only defaults for them. exclude them from the structural comparison; their integrity is covered
    # by the sha256 preparation digest, and the committed weight-cache volume is guarded against
    # illegitimate removal at the persist boundary (see _reject_managed_volume_removal).
    for managed_gpu in _runner().MANAGED_GPU_KEYS:
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
    # a pin already marked runner-assigned (inherited from a warm-start source) is not authored,
    # even though it is present. reading presence alone would relabel it as the author's and hand
    # deploy a pin it refuses -- the exact failure the marker exists to prevent.
    authored = "" if spec.model_revision_auto else spec.model_revision
    if not authored and not required:
        return spec
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(
            spec.model,
            revision=authored or None,
        )
        resolved = str(getattr(info, "sha", "") or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
            raise ValueError("resolved revision is not an immutable commit")
    except Exception as exc:
        raise ValueError(
            f"could not resolve model_revision for model {spec.model!r}; "
            "verify that the revision exists and the operator token can access it"
        ) from exc
    # record WHO chose the pin, not just its value. `authored` is empty exactly when the caller
    # asked for a pin the user never wrote (SFT, required=True), and that is the only case deploy
    # may relax: serving resolves the base by name, so an auto pin asks nothing of it, while an
    # authored one is a request serving cannot honour and must still be refused.
    return replace(spec, model_revision=resolved, model_revision_auto=not authored)


def _profile_producer_version() -> str:
    from flash import __version__

    return str(__version__)


def _require_pinned_profile_environment(spec: JobSpec) -> JobSpec:
    pinned = _runner()._assign_resolved_env_sha(spec)
    if not pinned.environment.id:
        raise _runner().WorkloadProfileUnavailable(
            "sft workload profiling requires an environment id"
        )
    if not pinned.environment.resolved_sha:
        raise _runner().WorkloadProfileUnavailable(
            "sft workload profiling requires an immutable resolved environment revision"
        )
    return pinned


def _prepared_sft_profile_job(spec: JobSpec, *, input_digest: str) -> PreparedJob:
    """Prepare the cpu-only profile job that measures ``spec``'s exact sft workload.

    The profile loads a tokenizer and the pinned environment, then exits. GPU type and provider pins
    are dropped: they describe where the training run must land, and inheriting them would rent an
    H100 to tokenize on cpu.
    """
    from flash.engine.profiling.workload_profile import SFT_PROFILE_KIND, sft_profile_run_id

    profile_spec = replace(
        spec,
        run_id=sft_profile_run_id(input_digest),
        gpu=replace(
            spec.gpu,
            type="",
            provider="",
            count=1,
            disk_gb=_runner().GpuSpec.disk_gb,
            network_volume=None,
            max_wall_seconds=_runner()._WORKLOAD_PROFILE_WALL_SECONDS,
            max_retries=_runner()._WORKLOAD_PROFILE_MAX_RETRIES,
        ),
        workload_profile_kind=SFT_PROFILE_KIND,
        workload_profile_input_digest=input_digest,
        workload_profile_producer_version=_runner()._profile_producer_version(),
        workload_profile={},
    )
    profile_spec = _runner()._assign_managed_hf_repo(profile_spec)
    from flash.cost.spec import estimate_for_spec

    return _runner().PreparedJob(
        public_spec=profile_spec,
        worker_spec=profile_spec,
        estimated_cost_usd=float(estimate_for_spec(profile_spec).total_usd),
    )


def _require_sft_workload_profile(spec: JobSpec) -> JobSpec:
    """Attach the exact sft workload profile for ``spec``, or fail closed until one exists."""
    from flash.engine.profiling.workload_profile import (
        SFT_PROFILE_KIND,
        require_matching_sft_profile,
        sft_profile_input_digest,
        sft_profile_run_id,
    )

    producer_version = _runner()._profile_producer_version()
    tokenizer_revision = spec.model_revision
    input_digest = sft_profile_input_digest(
        spec,
        tokenizer_revision=tokenizer_revision,
        producer_version=producer_version,
    )
    if spec.workload_profile:
        profile = require_matching_sft_profile(
            spec.workload_profile,
            input_digest=input_digest,
            producer_version=producer_version,
            tokenizer_revision=tokenizer_revision,
        )
        return replace(
            spec,
            workload_profile_kind="",
            workload_profile_input_digest=input_digest,
            workload_profile_producer_version=producer_version,
            workload_profile=profile.to_dict(),
        )

    profile_run_id = sft_profile_run_id(input_digest)
    try:
        status = _runner().get_status(profile_run_id)
    except FileNotFoundError as exc:
        prepared = _runner()._prepared_sft_profile_job(spec, input_digest=input_digest)
        raise _runner().WorkloadProfilePending(
            profile_run_id,
            "required",
            prepared_job=prepared,
        ) from exc

    if (
        status.workload_profile_kind != SFT_PROFILE_KIND
        or status.workload_profile_input_digest != input_digest
    ):
        raise _runner().WorkloadProfileUnavailable(
            "stored workload profile identity does not match"
        )
    if status.state == "done":
        try:
            profile = require_matching_sft_profile(
                status.workload_profile,
                input_digest=input_digest,
                producer_version=producer_version,
                tokenizer_revision=tokenizer_revision,
            )
        except ValueError as exc:
            raise _runner().WorkloadProfileUnavailable(
                "stored sft workload profile failed integrity validation"
            ) from exc
        return replace(
            spec,
            workload_profile_input_digest=input_digest,
            workload_profile_producer_version=producer_version,
            workload_profile=profile.to_dict(),
        )
    if status.state in {"failed", "cancelled", "dry_run"}:
        # a spent profile is not a verdict on the workload. the id is derived from the workload
        # alone, so a preempted pod or a cancel would otherwise make this exact config unquotable
        # for everyone, forever, with nothing in the system that could ever clear it. offer a fresh
        # profile job the same way a missing one is offered; the claim decides who actually runs it.
        prepared = _runner()._prepared_sft_profile_job(spec, input_digest=input_digest)
        raise _runner().WorkloadProfilePending(
            profile_run_id,
            status.state,
            prepared_job=prepared,
            spent_at=status.created_at,
        )
    raise _runner().WorkloadProfilePending(profile_run_id, status.state)
