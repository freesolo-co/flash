"""LoRA config + warm-start adapter loading for the fine-tuning worker."""

from __future__ import annotations

import json
import os
import shutil

from flash.adapters.lora_rank import resolve_adapter_ref
from flash.engine.plan.recipe import RECIPE
from flash.engine.worker.io.hf import (
    RetriableInfraError,
    _has_deployable_adapter,
    _prefetch_error_is_retriable,
    _require_hf_deadline_allowance,
    _sleep_with_hf_deadline,
)
from flash.engine.worker.model.lora import (
    _read_adapter_tensor_keys,
)
from flash.engine.worker.runtime.pkg_proxy import W as _w

_ADAPTER_DOWNLOAD_RETRIES = 4
_ADAPTER_DOWNLOAD_BACKOFF_S = 5.0
_QWEN35_EXPERT_TARGET_PARAMETERS = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)


def lora_target_parameters(model_id: str | None) -> list[str] | None:
    """return direct parameter targets required by the model's fused expert layout."""
    if model_id == "Qwen/Qwen3.6-35B-A3B":
        return list(_QWEN35_EXPERT_TARGET_PARAMETERS)
    return None


def adapter_has_fused_expert_tensors(adapter_dir: str | None, model_id: str) -> bool:
    """did this adapter train ALL of the model's fused routed experts?

    the weights, not the config, are the ground truth: peft wraps each targeted `nn.Parameter` in a
    `ParamWrapper` whose lora tensors sit under the OWNING MODULE's path. the targeted parameter's
    own name -- `gate_up_proj` / `down_proj` -- never appears in a tensor key, so a wrapper cannot
    be matched to the specific parameter it adapts.

    what it can be matched to is how MANY wrappers exist. peft nests them: wrapping a second
    parameter on an already-wrapped module puts the new wrapper outside the first, so N parameters
    targeted on one module yield N distinct lora paths (`...mlp.experts.lora_A`, then
    `...mlp.experts.base_layer.lora_A`). counting distinct paths under an owner therefore counts
    the parameters trained on it.

    requiring the full count is what makes the recovery safe. "any expert tensor" would accept a
    truncated adapter that trained only one of the two, and the caller would then write the
    complete target list into its config -- at which point peft freshly initializes the parameter
    that never trained, silently, instead of the load failing. a partial adapter must be rejected.
    """
    targets = lora_target_parameters(model_id)
    if not targets or not adapter_dir:
        return False
    try:
        keys = _read_adapter_tensor_keys(adapter_dir) or []
    except Exception:
        return False
    # how many parameters each owning module ("mlp.experts.*" -> "experts") must account for.
    required_per_owner: dict[str, int] = {}
    for target in targets:
        if "." in target:
            owner = target.split(".")[-2]
            required_per_owner[owner] = required_per_owner.get(owner, 0) + 1
    if not required_per_owner:
        return False
    for owner, needed in required_per_owner.items():
        # the wrapper's identity is its path from the owner down ("experts", "experts.base_layer"),
        # which is shared across layers, so the distinct set is the per-module wrapper count.
        wrappers = set()
        for key in keys:
            if ".lora_" not in key:
                continue
            segments = key.split(".lora_")[0].split(".")
            if owner in segments:
                wrappers.add(".".join(segments[segments.index(owner) :]))
        if len(wrappers) < needed:
            return False
    return True


def validate_lora_target_parameters(
    config: dict, model_id: str, adapter_dir: str | None = None
) -> None:
    """fail closed when a warm-start adapter omits required fused expert parameters.

    `target_parameters` is the only thing that adapts a fused routed-expert `nn.Parameter`, so an
    adapter that lacks it really is unusable for warm start. but the saved config is not a reliable
    record of what was trained: verl's exporter rebuilds `adapter_config.json` from rank, alpha and
    `target_modules` alone, so it wrote `target_parameters: null` for every adapter produced before
    `restore_fused_expert_targets` began repairing the export. rejecting on the config alone
    therefore condemned correctly-trained adapters -- including ones that deploy and serve -- with
    advice ("retrain with the current Flash version") that reproduced the same file.

    so when the config is silent, fall back to the adapter's own tensors, which record what was
    actually adapted. only a directory with no expert lora tensors is rejected, and that adapter is
    genuinely missing the expert training this model requires.

    the recovery is written back to `adapter_config.json`, not just to the caller's dict, because
    the consumer is verl: flash hands it the adapter DIRECTORY (`model.lora_adapter_path`) and it
    loads that path itself through `PeftModel.from_pretrained`. an in-memory repair would leave the
    file peft actually reads unchanged, and the load would fail on the same config this call just
    accepted -- the repair has to reach the same bytes.
    """
    required = set(lora_target_parameters(model_id) or ())
    if not required:
        return
    actual = set(config.get("target_parameters") or ())
    missing = sorted(required - actual)
    if not missing:
        return
    if actual or not adapter_has_fused_expert_tensors(adapter_dir, model_id):
        raise ValueError(
            f"warm-start adapter for {model_id} omits required expert targets {missing}; "
            "retrain the source adapter with the current Flash version"
        )
    # config dropped the field, weights prove the experts were trained: recover in the caller's
    # dict and in the file, so both this run's reader and verl's own load see the real targeting.
    from flash.engine.worker.verl.checkpoints import restore_fused_expert_targets

    restore_fused_expert_targets(config, model_id)
    config_path = os.path.join(str(adapter_dir), "adapter_config.json")
    with open(config_path, encoding="utf-8") as config_file:
        on_disk = json.load(config_file)
    restore_fused_expert_targets(on_disk, model_id)
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(on_disk, config_file, indent=2)
    print(
        f"[init-adapter] recovered fused expert targets {sorted(required)} from adapter weights; "
        "the source config predates the export fix"
    )


def make_lora(model_id: str | None = None):
    """build the model's complete serve-compatible lora target set."""
    from peft import LoraConfig

    targets = "all-linear"
    rank = _w.JOB_SPEC.train.lora_rank if _w.JOB_SPEC else RECIPE.lora.rank
    alpha = _w.JOB_SPEC.train.lora_alpha if _w.JOB_SPEC else RECIPE.lora.alpha
    model_revision = getattr(_w.JOB_SPEC, "model_revision", "") if _w.JOB_SPEC else ""
    kwargs = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": RECIPE.lora.dropout,
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
        "revision": model_revision or None,
    }
    if target_parameters := lora_target_parameters(model_id):
        kwargs["target_parameters"] = target_parameters
    # pissa removed: it mutates the base, so its adapter corrupts serve + warm-start on the unmodified base.
    kwargs["init_lora_weights"] = True
    print(
        "[lora] init_lora_weights=True (standard zero-B; PiSSA removed for serve/warm-start safety)"
    )
    # rsLoRA removed: ~5.6x effective LR inflation with catalog LRs causes SFT divergence + serve corruption.
    kwargs["use_rslora"] = False
    return LoraConfig(**kwargs)


def _warmstart_adapter_is_loadable(adir: str) -> bool:
    """return true only for a structurally complete adapter config and weight file."""
    if not _has_deployable_adapter(adir):
        return False
    try:
        with open(os.path.join(adir, "adapter_config.json"), encoding="utf-8") as config_file:
            config = json.load(config_file)
        if not isinstance(config, dict) or str(config.get("peft_type", "")).upper() != "LORA":
            return False
        return bool(_read_adapter_tensor_keys(adir))
    except Exception:
        return False


def _download_adapter(adapter_prefix: str | None) -> str | None:
    """Download an init_from_adapter LoRA to /tmp/evdl/<prefix>/adapter and return its dir.

    ``adapter_prefix`` is the internal storage ref resolved by the control plane:
    ``<owner>/<repo>:<phase>/<run_id>[/checkpoints/step-N]``.
    """
    if not adapter_prefix:
        return None
    resolved = resolve_adapter_ref(adapter_prefix)
    if not resolved:
        return None
    repo, prefix = resolved
    from huggingface_hub import snapshot_download

    adir = os.path.join("/tmp/evdl", prefix, "adapter")
    # start from a clean path so the loadable-check can only ever accept files THIS download
    # materialized -- leftover materialization from an earlier worker subprocess, attempt, or a
    # different run sharing the same prefix must not satisfy the post-exception loadable check and
    # mask a terminal 404/403/429 for the current repo/revision.
    shutil.rmtree(adir, ignore_errors=True)
    for attempt in range(_ADAPTER_DOWNLOAD_RETRIES):
        _require_hf_deadline_allowance()
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=[f"{prefix}/adapter/*"],
                local_dir="/tmp/evdl",
                token=os.environ.get("HF_TOKEN"),
                revision=(_w.JOB_SPEC.train.init_from_adapter_revision if _w.JOB_SPEC else None)
                or None,
            )
        except Exception as error:
            # a later nonessential sidecar may fail after the config and weights are already complete.
            if _warmstart_adapter_is_loadable(adir):
                return adir
            if not _prefetch_error_is_retriable(error):
                raise RuntimeError(
                    "the prepared warm-start source adapter could not be downloaded"
                ) from None
        else:
            # a returned snapshot can still be incomplete: an interrupted transfer, or hf falling
            # back to a partial local_dir when a throttled metadata call cannot confirm the file set.
            if _warmstart_adapter_is_loadable(adir):
                return adir
        # discard partial local_dir materialization so the next attempt cannot reuse stale files.
        shutil.rmtree(adir, ignore_errors=True)
        if attempt + 1 < _ADAPTER_DOWNLOAD_RETRIES:
            try:
                if not _sleep_with_hf_deadline(_ADAPTER_DOWNLOAD_BACKOFF_S * (attempt + 1)):
                    break
            except Exception:
                break
    raise RetriableInfraError(
        "the prepared warm-start source adapter could not be downloaded after transient failures"
    ) from None
