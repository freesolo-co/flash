"""LoRA config + warm-start adapter loading for the fine-tuning worker."""

from __future__ import annotations

import json
import os
import shutil

from flash.adapters.fused_experts import (
    expected_fused_expert_modules,
    has_complete_fused_expert_tensors,
    lora_target_parameters,
    restore_fused_expert_targets,
)
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


def adapter_has_fused_expert_tensors(adapter_dir: str | None, model_id: str) -> bool:
    """Return whether the downloaded adapter has complete fused-expert tensor coverage."""
    if not adapter_dir:
        return False
    try:
        keys = _read_adapter_tensor_keys(adapter_dir) or []
    except Exception:
        return False
    return has_complete_fused_expert_tensors(keys, model_id)


def _rewrite_adapter_config(config: dict, model_id: str, adapter_dir: str) -> None:
    """apply the fused-expert repair to the caller's dict AND to the file peft will read.

    verl gets the adapter DIRECTORY (`model.lora_adapter_path`) and reloads it itself through
    `PeftModel.from_pretrained`, so an in-memory-only repair leaves the bytes peft actually reads
    unchanged and the load fails on the config this call just accepted.

    written through a temporary file and replaced atomically: a truncating in-place rewrite that is
    interrupted leaves an adapter directory with a corrupt `adapter_config.json`, which is a worse
    state than either the original or the repaired one.
    """
    restore_fused_expert_targets(config, model_id)
    config_path = os.path.join(str(adapter_dir), "adapter_config.json")
    with open(config_path, encoding="utf-8") as config_file:
        on_disk = json.load(config_file)
    restore_fused_expert_targets(on_disk, model_id)
    tmp_path = f"{config_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as config_file:
        json.dump(on_disk, config_file, indent=2)
    os.replace(tmp_path, config_path)


def prepare_warmstart_adapter_config(config: dict, model_id: str, adapter_dir: str | None) -> None:
    """Validate and, when proven safe, repair the warm-start adapter config.

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
        synthetic = expected_fused_expert_modules(model_id) & {
            str(module) for module in (config.get("target_modules") or ())
        }
        if not adapter_dir:
            if synthetic:
                raise ValueError(
                    f"warm-start adapter for {model_id} retains unloadable synthetic modules "
                    f"{sorted(synthetic)} and no adapter directory was provided for repair"
                )
            return
        # declarations are not evidence: peft accepts missing adapter keys and initializes them.
        # prove the weights before stripping the synthetic names that would otherwise fail loudly.
        if not adapter_has_fused_expert_tensors(adapter_dir, model_id):
            raise ValueError(
                f"warm-start adapter for {model_id} does not contain complete fused expert LoRA "
                "weights"
            )
        if synthetic:
            _rewrite_adapter_config(config, model_id, adapter_dir)
        return
    if actual or not adapter_dir or not adapter_has_fused_expert_tensors(adapter_dir, model_id):
        raise ValueError(
            f"warm-start adapter for {model_id} omits required expert targets {missing}; "
            "retrain the source adapter with the current Flash version"
        )
    # config dropped the field, weights prove the experts were trained: recover in the caller's
    # dict and in the file, so both this run's reader and verl's own load see the real targeting.
    _rewrite_adapter_config(config, model_id, adapter_dir)
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
