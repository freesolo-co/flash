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

import json
import os
import re
from dataclasses import dataclass

from flash.adapters.lora_rank import rank_from_adapter_config
from flash.schema import format_checkpoint_ref
from flash.serve.app import AdapterExecutionInput, ArtifactFile, ExecutionInputs
from flash.serve.app.materialize import MaterializationError, validate_adapter_weight_structure
from flash.serve.contract.protocol import reject_non_finite_json_constant
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
    config_path: str | None = None
    weights_path: str | None = None
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
        if name == ADAPTER_CONFIG:
            config_path = str(local)
        else:
            weights_path = str(local)
    return tuple(files), config_path, weights_path


class _DuplicateConfigKey(ValueError):
    """raised through `json.load`, so it stays a ValueError for any caller that expects one."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """mirror of the materializer's rule, so both boundaries read the same bytes the same way."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateConfigKey(f"{ADAPTER_CONFIG} contains a duplicate key")
        result[key] = value
    return result


def _declared_provenance(
    config_path: str | None,
) -> tuple[dict[str, object] | None, int | None, str | None, str | None]:
    """read the rank and base-model provenance the adapter stamps into its own config.

    the config is already on disk from the digest pass, so this costs nothing extra. reading it is
    what lets a mistyped ``--lora-rank`` or ``--model`` fail during resolution instead of inside the
    paid GPU container, where ``_validate_adapter_config`` catches the same mismatch only after
    provisioning has started.
    """

    if config_path is None:
        return None, None, None, None
    try:
        with open(config_path, "rb") as handle:
            raw = handle.read()
        # decode as strict utf-8 first, exactly as `_load_strict_config` does on the gpu side.
        # handing the bytes straight to `json.load` instead lets it auto-detect utf-16 and accept
        # a bom (rfc 4627), so a config the container refuses outright resolved cleanly here --
        # and the deployment failed only after the provider resources had been allocated.
        #
        # the duplicate-key rule is shared for the same reason: plain `json.load` takes the last
        # value, so a config declaring `r` twice would resolve against one rank and then be
        # rejected inside the container this function exists to avoid paying for.
        config = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_non_finite_json_constant,
        )
    except _DuplicateConfigKey as exc:
        raise ResolveError(str(exc)) from exc
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ResolveError(f"{ADAPTER_CONFIG} is not readable json") from exc
    if not isinstance(config, dict):
        raise ResolveError(f"{ADAPTER_CONFIG} must be a json object")

    try:
        rank = rank_from_adapter_config(config, source=ADAPTER_CONFIG)
    except Exception as exc:
        raise ResolveError(f"{ADAPTER_CONFIG} declares no usable lora rank: {exc}") from exc

    def _text(key: str) -> str | None:
        """read a config string exactly as the container will compare it.

        `_validate_adapter_config` compares these raw bytes for equality, so a value that matches
        only after stripping resolved clean here and was then rejected inside the paid container --
        the outcome this function exists to prevent. normalizing the padding away instead would be
        worse for `revision`, which the resolver *adopts* into the immutable manifest: that would
        launder a padded string into the record rather than surface it. rejecting keeps the
        resolve-time and container-time verdicts identical.
        """
        value = config.get(key)
        if not isinstance(value, str):
            return None
        if value != value.strip():
            raise ResolveError(f"{ADAPTER_CONFIG} {key} has surrounding whitespace: {value!r}")
        return value or None

    # the container compares `base_model_name_or_path` for equality, so an absent, empty, or
    # non-string one can never match and the deployment is already doomed. returning None here
    # skipped the check instead, which deferred a certain failure until after the provider had
    # allocated and started billing -- the exact outcome this function exists to prevent.
    declared_base = _text("base_model_name_or_path")
    if declared_base is None:
        raise ResolveError(f"{ADAPTER_CONFIG} declares no base_model_name_or_path")

    # the same reasoning applied to the rest of what the container checks about these exact bytes.
    # `_validate_adapter_config` rejects a non-LORA `peft_type`, an unsupported `task_type` and a
    # nonempty `modules_to_save` deterministically -- the verdict depends only on the config, not on
    # anything the gpu learns at runtime -- so leaving them out meant a config that could never
    # serve still resolved, provisioned, and billed before failing for a reason readable here for
    # free. a non-string `revision` is the same case: the container compares it for equality, so a
    # non-string can never match, and `_text` would otherwise quietly map it to None.
    peft_type = config.get("peft_type")
    if peft_type != "LORA":
        raise ResolveError(f"{ADAPTER_CONFIG} peft_type must be LORA, not {peft_type!r}")
    task_type = config.get("task_type")
    if task_type not in {None, "CAUSAL_LM"}:
        raise ResolveError(
            f"{ADAPTER_CONFIG} task_type must be absent or CAUSAL_LM, not {task_type!r}"
        )
    modules_to_save = config.get("modules_to_save")
    if modules_to_save is not None and modules_to_save != []:
        raise ResolveError(f"{ADAPTER_CONFIG} modules_to_save adapters are not supported")
    revision = config.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise ResolveError(f"{ADAPTER_CONFIG} revision must be a string when present")

    return config, rank, declared_base, _text("revision")


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

    files, config_path, weights_path = _artifact_files(
        artifact_repo_id, artifact_repo_type, artifact_revision, artifact_subfolder
    )

    # the adapter's own config is the authority on what it was trained against. checking it here
    # turns a mistyped --lora-rank or --model into a resolution error, instead of a failure inside
    # the paid GPU container after provisioning has already begun.
    config, declared_rank, declared_base, declared_base_revision = _declared_provenance(config_path)
    if declared_rank is not None and declared_rank != lora_rank:
        raise ResolveError(
            f"--lora-rank {lora_rank} disagrees with {ADAPTER_CONFIG}, which declares "
            f"{declared_rank}"
        )
    if declared_base is not None and declared_base != base_model:
        raise ResolveError(
            f"--model {base_model!r} disagrees with {ADAPTER_CONFIG}, which declares this adapter "
            f"was trained against {declared_base!r}"
        )
    # bind to the revision the adapter was TRAINED against, not the repo's current tip: the model
    # repo is mutable, so resolving it at deploy time can silently pair the adapter with weights it
    # never saw. the container compares these directly and would reject the mismatch anyway.
    if declared_base_revision is not None and declared_base_revision != base_model_revision:
        base_model_revision = declared_base_revision

    if config is None or weights_path is None:
        raise ResolveError(
            f"{artifact_repo_id}@{artifact_revision} did not resolve both required adapter files"
        )
    try:
        validate_adapter_weight_structure(weights_path, config, base_model)
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

    aggregate = aggregate_file_digest(files)
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
        execution=AdapterExecutionInput(checkpoint_id=checkpoint_id, files=files),
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
