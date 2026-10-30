"""Preparation, allocation and resume authorization for managed custom runs."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import replace
from pathlib import Path


def session_token(run_id, attempt):
    secret = os.environ.get("HF_TOKEN", "")
    if not secret:
        raise RuntimeError("managed custom sessions require artifact credentials")
    message = json.dumps(["flash-custom-session-v1", run_id, attempt]).encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def allocation(spec):
    from flash.providers.core.base import Allocation, Candidate, get_gpu_info
    from flash.providers.modal.pricing import hourly_rate

    gpu = get_gpu_info(spec.gpu.type)
    rate = hourly_rate(gpu.name)
    candidate = Candidate("modal", gpu.name, rate, gpu.vram_gb)
    return Allocation("modal", gpu.name, rate, gpu.vram_gb, (candidate,))


def estimate(spec):
    from flash.cost.types import CostEstimate

    rented = allocation(spec)
    seconds = spec.custom["max_wall_seconds"]
    return CostEstimate(
        model_id=spec.model,
        method="custom",
        steps=spec.train.max_steps,
        gpu=rented.gpu,
        provider=rented.provider,
        gpu_vram_gb=rented.min_vram_gb,
        required_vram_gb=rented.min_vram_gb,
        gpu_hourly_usd=rented.hourly_usd,
        setup_seconds=0,
        seconds_per_step=seconds / spec.train.max_steps,
        train_seconds=seconds,
        wall_clock_seconds=seconds,
        wall_capped=True,
        total_usd=seconds * rented.hourly_usd / 3600,
        notes=(
            "Upper bound for a dedicated custom session, including time waiting for the local script.",
        ),
    )


def prepare(spec, *, billing_context, platform_context, owner_key_id):
    from flash.providers.core.registry import available_providers
    from flash.runner.accounting.artifacts import _assign_managed_hf_repo
    from flash.runner.lifecycle import preparation, state
    from flash.runner.lifecycle.submit import PreparedJob

    if "modal" not in available_providers():
        raise ValueError("Modal is not configured on this Flash server")
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("managed custom training requires durable artifact storage")
    if spec.custom.get("resume_from"):
        spec = resolve_resume(
            spec,
            owner_key_id=owner_key_id,
            owner_org_id=preparation._context_org_id(billing_context)
            or preparation._context_org_id(platform_context),
        )
    spec = preparation._resolve_model_revision(spec, required=True)
    spec = replace(spec, run_id=spec.run_id if spec.run_id != "local" else state.new_run_id())
    spec = _assign_managed_hf_repo(spec)
    from flash.cost.currency import usd_amount

    return PreparedJob(spec, spec, usd_amount(estimate(spec).total_usd))


def resolve_resume(spec, *, owner_org_id, owner_key_id):
    from huggingface_hub import HfApi, hf_hub_download

    from flash.runner.lifecycle import preparation, status
    from flash.schema import parse_checkpoint_ref

    source_id, step = parse_checkpoint_ref(spec.custom["resume_from"])
    source_status = status.get_status(source_id)
    if not preparation._warmstart_source_is_authorized(
        source_status,
        source_id,
        owner_org_id=owner_org_id,
        owner_key_id=owner_key_id,
    ):
        raise ValueError("custom resume checkpoint is not accessible")
    source = status.effective_spec_from_status(source_status)
    if source.algorithm != "custom" or source.model != spec.model:
        raise ValueError("custom resume requires a custom checkpoint of the same base model")
    if (source.train.lora_rank, source.train.lora_alpha) != (
        spec.train.lora_rank,
        spec.train.lora_alpha,
    ):
        raise ValueError("custom resume requires the source LoRA rank and alpha")
    repo = source.train.hf_repo
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    revision = api.repo_info(repo, repo_type="dataset").sha
    path = f"custom/{source_id}/checkpoints/step-{step}"
    required = {
        f"{path}/{name}"
        for name in ("metadata.json", "state.pt", "algorithm.json", "adapter/adapter_config.json")
    }
    if not required <= set(api.list_repo_files(repo, repo_type="dataset", revision=revision)):
        raise ValueError("custom checkpoint is incomplete")
    metadata_file = hf_hub_download(
        repo,
        f"{path}/metadata.json",
        repo_type="dataset",
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
    )
    metadata = json.loads(Path(metadata_file).read_text())
    if metadata.get("format") != "flash-custom-v1" or metadata.get("step") != step:
        raise ValueError("checkpoint has no compatible custom training state")
    if metadata.get("model_revision") != source.model_revision:
        raise ValueError("checkpoint base revision does not match the source run")
    if spec.train.max_steps <= step:
        raise ValueError("train.max_steps must exceed the resumed optimizer step")
    return replace(
        spec,
        model_revision=source.model_revision,
        model_revision_auto=True,
        custom={
            **spec.custom,
            "resolved_resume": {"repo": repo, "revision": revision, "path": path},
        },
    )
