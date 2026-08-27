"""resolve one owned run into the exact immutable inputs a provider deployment requires.

``DeploymentBundle`` demands a complete ``ResolvedAdapter`` (immutable artifact revision, per-file
digests, base model and its revision, lora rank) plus the execution file table the serving app
materializes from. Those values live on the hub, not in a config file, so this module reads them
and refuses to synthesize any of them.

Everything here is a READ. Nothing in this module holds a provider credential: it takes the hub
token that flash already uses for artifacts, and hands its output to the provisioning layer, which
takes provider credentials request-scoped and separately.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from flash.adapters.config import (
    AdapterConfigError,
    DeclaredAdapterConfig,
    parse_declared_adapter_config,
)
from flash.schema import format_checkpoint_ref
from flash.serve.app import AdapterExecutionInput, ArtifactFile, ExecutionInputs
from flash.serve.app.materialize import MaterializationError, validate_adapter_weight_structure
from flash.serve.control import ResolvedAdapter
from flash.serve.deployment.profiles import ServingProfile
from flash.serve.provisioning import ServingImage

# a peft lora adapter is exactly these files. requiring the pair (rather than accepting whatever
# the prefix happens to contain) keeps an incomplete upload from deploying as if it were whole:
# the config alone loads no weights, and the weights alone have no rank metadata.
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
_CHECKPOINT_ADAPTER_SUBFOLDER_RE = re.compile(r"(?:^|/)checkpoints/step-(?P<step>\d+)/adapter$")


class ResolveError(ValueError):
    """the run's serving inputs are missing, incomplete, or not immutable."""


@dataclass(frozen=True, slots=True)
class ResolvedDeploymentInputs:
    """one run's complete control adapter and its execution file table."""

    adapter: ResolvedAdapter
    execution: AdapterExecutionInput


def _token() -> str | None:
    """the hub credential, or None when there is genuinely none.

    `None` and `""` are not interchangeable here. `huggingface_hub` treats a falsy-but-present
    token as a credential to send, and builds the literal header `Bearer `, which `httpx` rejects
    as an illegal header value -- so an unset `HF_TOKEN` raised `LocalProtocolError` before any
    request left the process. That failure is indistinguishable from the repo being private, and it
    made every PUBLIC serving checkpoint unreadable to a self-hoster who has no token at all.
    """

    return os.environ.get("HF_TOKEN", "").strip() or None


def _hub_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - hub is a base dependency
        raise ResolveError("huggingface_hub is required to resolve serving inputs") from exc
    return HfApi(token=_token())


def resolve_base_revision(model_id: str) -> str:
    """resolve one base model id to its immutable commit sha.

    the engine identity pins the commit, not the branch: a deployment that named `main` would
    silently change what it serves the next time the repository moves.
    """

    try:
        info = _hub_api().repo_info(repo_id=model_id, repo_type="model")
        sha = str(getattr(info, "sha", "") or "").strip().lower()
    except Exception as exc:
        raise ResolveError(f"could not resolve the commit for {model_id}") from exc
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ResolveError(f"{model_id} did not resolve to an immutable commit")
    return sha


def _digest_of(path: str) -> tuple[int, str]:
    """return one local file's exact size and sha256."""

    import hashlib

    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


@dataclass(frozen=True)
class _ArtifactDownload:
    """the exact adapter file table, plus where the two files landed on this machine."""

    files: tuple[ArtifactFile, ...]
    config_path: str
    weights_path: str


def _artifact_files(repo_id: str, repo_type: str, revision: str, subfolder: str):
    """build the exact two-file adapter table the serving manifest requires.

    the manifest accepts EXACTLY ``adapter_config.json`` then ``adapter_model.safetensors`` -- a
    real artifact prefix also holds a README and training metadata, so the table is built from the
    required pair rather than from whatever the folder contains.

    each digest comes from the file's own bytes. the hub listing carries a sha256 only for LFS
    objects, so the small non-LFS config would have no digest there, and the manifest's per-file
    verification is exactly what a missing digest would defeat.
    """

    from huggingface_hub import hf_hub_download

    prefix = subfolder.strip("/")
    files: list[ArtifactFile] = []
    local_paths: dict[str, str] = {}
    for name in (ADAPTER_CONFIG, ADAPTER_WEIGHTS):
        remote = f"{prefix}/{name}" if prefix else name
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=remote,
                repo_type=repo_type,
                revision=revision,
                token=_token(),
            )
        except Exception as exc:
            raise ResolveError(f"could not read {repo_id}@{revision}:{remote}") from exc
        size, digest = _digest_of(local)
        if size <= 0:
            raise ResolveError(f"{repo_id}@{revision}:{remote} is empty")
        files.append(ArtifactFile(path=name, size=size, sha256=digest))
        local_paths[name] = str(local)
    # every miss above raises, so reaching here means both files downloaded.
    return _ArtifactDownload(
        files=tuple(files),
        config_path=local_paths[ADAPTER_CONFIG],
        weights_path=local_paths[ADAPTER_WEIGHTS],
    )


def _declared_provenance(config_path: str) -> DeclaredAdapterConfig:
    """read the rank and base-model provenance the adapter stamps into its own config.

    the config is already on disk from the digest pass, so this costs nothing extra. reading it is
    what lets a mistyped ``--lora-rank`` or ``--model`` fail during resolution instead of inside the
    paid GPU container, which revalidates the cache entry through this same shared reader only
    after provisioning has started. hosted admission reads the same bytes through the same rules, so
    no two paths can reach opposite verdicts about one artifact.
    """

    try:
        with open(config_path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ResolveError(f"{ADAPTER_CONFIG} is not readable json") from exc
    try:
        return parse_declared_adapter_config(raw, source=ADAPTER_CONFIG)
    except AdapterConfigError as exc:
        raise ResolveError(str(exc)) from exc


def _checkpoint_step_from_subfolder(
    artifact_subfolder: str, checkpoint_step: int | None
) -> int | None:
    """attest a recognized flash artifact path against the authored checkpoint selection."""

    match = _CHECKPOINT_ADAPTER_SUBFOLDER_RE.search(artifact_subfolder)
    if match is not None:
        selected_step = int(match.group("step"))
        if checkpoint_step is None:
            raise ResolveError(
                f"--artifact-subfolder {artifact_subfolder!r} identifies checkpoint step "
                f"{selected_step}, but --checkpoint-step is unset; set --checkpoint-step "
                f"{selected_step} or select the final adapter subfolder"
            )
        if checkpoint_step != selected_step:
            raise ResolveError(
                f"--checkpoint-step {checkpoint_step} disagrees with --artifact-subfolder "
                f"{artifact_subfolder!r}, which identifies checkpoint step {selected_step}"
            )
        return selected_step

    parts = artifact_subfolder.split("/")
    if parts[-1:] == ["adapter"] and "checkpoints" not in parts:
        if checkpoint_step is not None:
            raise ResolveError(
                f"--checkpoint-step {checkpoint_step} selects a saved step, but "
                f"--artifact-subfolder {artifact_subfolder!r} identifies the final adapter; "
                "remove --checkpoint-step or select its checkpoints/step-N/adapter subfolder"
            )
        return None

    raise ResolveError(
        f"--artifact-subfolder {artifact_subfolder!r} does not identify a canonical Flash final "
        "adapter or checkpoints/step-N/adapter path"
    )


def resolve_adapter(
    *,
    run_id: str,
    artifact_repo_id: str,
    artifact_subfolder: str,
    base_model: str,
    base_model_revision: str,
    lora_rank: int,
    checkpoint_step: int | None = None,
    thinking_default: bool = False,
    structured_outputs_default_json: str | None = None,
    artifact_repo_type: str = "dataset",
) -> ResolvedDeploymentInputs:
    """resolve one owned run into its control adapter and execution file table."""

    checkpoint_step = _checkpoint_step_from_subfolder(artifact_subfolder, checkpoint_step)
    try:
        info = _hub_api().repo_info(repo_id=artifact_repo_id, repo_type=artifact_repo_type)
        artifact_revision = str(getattr(info, "sha", "") or "").strip().lower()
    except Exception as exc:
        raise ResolveError(f"could not resolve the revision for {artifact_repo_id}") from exc
    if len(artifact_revision) != 40:
        raise ResolveError(f"{artifact_repo_id} did not resolve to an immutable commit")

    download = _artifact_files(
        artifact_repo_id, artifact_repo_type, artifact_revision, artifact_subfolder
    )

    # the adapter's own config is the authority on what it was trained against. checking it here
    # turns a mistyped --lora-rank or --model into a resolution error, instead of a failure inside
    # the paid GPU container after provisioning has already begun.
    declared = _declared_provenance(download.config_path)
    if declared.lora_rank != lora_rank:
        raise ResolveError(
            f"--lora-rank {lora_rank} disagrees with {ADAPTER_CONFIG}, which declares "
            f"{declared.lora_rank}"
        )
    if declared.base_model != base_model:
        raise ResolveError(
            f"--model {base_model!r} disagrees with {ADAPTER_CONFIG}, which declares this "
            f"adapter was trained against {declared.base_model!r}"
        )
    # bind to the revision the adapter was TRAINED against, not the repo's current tip: the model
    # repo is mutable, so resolving it at deploy time can silently pair the adapter with weights it
    # never saw. the container compares these directly and would reject the mismatch.
    if declared.base_revision is not None and declared.base_revision != base_model_revision:
        base_model_revision = declared.base_revision

    try:
        validate_adapter_weight_structure(download.weights_path, declared.config, base_model)
    except MaterializationError as exc:
        prefix = artifact_subfolder.strip("/")
        remote = f"{prefix}/{ADAPTER_WEIGHTS}" if prefix else ADAPTER_WEIGHTS
        raise ResolveError(
            f"{artifact_repo_id}@{artifact_revision}:{remote} is not a deployable LoRA adapter: "
            f"{exc}"
        ) from exc

    # the aggregate digest binds the control record to the exact file table the manifest carries,
    # so a spec and a manifest cannot disagree about what was deployed.
    from flash.serve.app import aggregate_file_digest

    aggregate = aggregate_file_digest(download.files)
    checkpoint_id = format_checkpoint_ref(run_id, checkpoint_step)

    adapter = ResolvedAdapter(
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        artifact_repo_id=artifact_repo_id,
        artifact_repo_type=artifact_repo_type,
        artifact_revision=artifact_revision,
        artifact_digest=aggregate,
        artifact_subfolder=artifact_subfolder,
        base_model=base_model,
        base_model_revision=base_model_revision,
        lora_rank=lora_rank,
        thinking_default=thinking_default,
        structured_outputs_default_json=structured_outputs_default_json,
    )
    return ResolvedDeploymentInputs(
        adapter=adapter,
        execution=AdapterExecutionInput(checkpoint_id=checkpoint_id, files=download.files),
    )


def execution_inputs(
    profile: ServingProfile,
    image: ServingImage,
    resolved: tuple[ResolvedDeploymentInputs, ...],
) -> ExecutionInputs:
    """build the execution inputs bound to this image and this profile's runtime kwargs."""

    if not resolved:
        raise ResolveError("a deployment requires at least one resolved adapter")
    return ExecutionInputs(
        expected_oci_digest=image.digest,
        engine_args=dict(profile.engine_args),
        tokenizer_kwargs=dict(profile.tokenizer_kwargs),
        processor_kwargs=dict(profile.processor_kwargs),
        adapters=tuple(entry.execution for entry in resolved),
    )
