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
from dataclasses import dataclass

from flash.schema import format_adapter_revision
from flash.serve.app import AdapterExecutionInput, ArtifactFile, ExecutionInputs
from flash.serve.control import AdapterAliasIntent, ResolvedAdapter
from flash.serve.profiles import ServingProfile
from flash.serve.provisioning import ServingImage

# a peft lora adapter is exactly these files. requiring the pair (rather than accepting whatever
# the prefix happens to contain) keeps an incomplete upload from deploying as if it were whole:
# the config alone loads no weights, and the weights alone have no rank metadata.
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"


class ResolveError(ValueError):
    """the run's serving inputs are missing, incomplete, or not immutable."""


@dataclass(frozen=True, slots=True)
class ResolvedDeploymentInputs:
    """one run's complete control adapter and its execution file table."""

    adapter: ResolvedAdapter
    execution: AdapterExecutionInput


def _token() -> str:
    return os.environ.get("HF_TOKEN", "")


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
    return tuple(files)


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
    activate_alias: bool = True,
    artifact_repo_type: str = "dataset",
) -> ResolvedDeploymentInputs:
    """resolve one owned run into its control adapter and execution file table."""

    try:
        info = _hub_api().repo_info(repo_id=artifact_repo_id, repo_type=artifact_repo_type)
        artifact_revision = str(getattr(info, "sha", "") or "").strip().lower()
    except Exception as exc:
        raise ResolveError(f"could not resolve the revision for {artifact_repo_id}") from exc
    if len(artifact_revision) != 40:
        raise ResolveError(f"{artifact_repo_id} did not resolve to an immutable commit")

    files = _artifact_files(
        artifact_repo_id, artifact_repo_type, artifact_revision, artifact_subfolder
    )
    # the aggregate digest binds the control record to the exact file table the manifest carries,
    # so a spec and a manifest cannot disagree about what was deployed.
    from flash.serve.app import aggregate_file_digest

    aggregate = aggregate_file_digest(files)
    revision = format_adapter_revision(run_id, checkpoint_step, artifact_revision)
    # the control record's `checkpoint` is the suffix of its own revision, not the `run/step`
    # storage reference used elsewhere in flash. validate_resolved_adapter compares them directly.
    checkpoint = "final" if checkpoint_step is None else f"step-{checkpoint_step}"

    adapter = ResolvedAdapter(
        run_id=run_id,
        checkpoint=checkpoint,
        adapter_revision=revision,
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
        alias_intent=AdapterAliasIntent(
            activate=activate_alias,
            expected_adapter_revision=None,
        ),
    )
    return ResolvedDeploymentInputs(
        adapter=adapter,
        execution=AdapterExecutionInput(adapter_revision=revision, files=files),
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
