"""sft training via verl in a separate interpreter.

flash writes exact conversation ids and completion-only masks to parquet. the parent streams progress
and checkpoints without holding cuda while torchrun owns the devices.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from functools import reduce
from math import gcd

from flash.engine.plan.recipe import RECIPE
from flash.engine.plan.steps import final_save_due, validate_save_steps
from flash.engine.profiling.sft_workload import prepare_sft_workload, sft_tokens_for_updates
from flash.engine.worker.backend_common import (
    fused_ce_backend,
    gdn_probe_module,
    gdn_reset_arch_from_caps,
    latest_global_step_dir,
    parse_verl_metric,
    parse_wandb_link,
    probe_verl_capabilities,
    render_gdn_varlen_shim,
    render_wandb_link_shim,
    resolve_verl_loggers,
    resolve_verl_python,
    run_verl_training,
    stage_verl_resume,
    verl_step_number,
)
from flash.engine.worker.entry.sft import _model_arch_dims, sft_under_ran
from flash.engine.worker.io.heartbeat import liveness_heartbeat
from flash.engine.worker.model.packing import model_is_gdn_hybrid
from flash.engine.worker.runtime.pkg_proxy import W as _w
from flash.engine.worker.runtime.rng import seed_training_rngs

# todo: run the two-gpu sft smoke on the exact runpod image and command assembled below.
_SFT_LORAPLUS_RATIO = 16.0
# consecutive zero-grad-norm steps tolerated before the run is failed as untrainable (GRAD-001).
# any nonzero grad norm is proof the backward graph is intact and resets the count. 2 is enough to
# separate a one-off fully-masked batch from a severed graph, and keeps the wasted spend to a couple
# of steps rather than the whole run.
_MAX_ZERO_GRAD_STEPS = 2


def _cached_model_path(model_id: str, model_revision: str) -> str:
    from huggingface_hub import snapshot_download

    from flash.engine.worker.io.hf import _shared_weight_cache_dir
    from flash.engine.worker.perf import RetriableInfraError

    # prefetch lands weights on the shared volume when attached; resolve from the same cache first.
    for cache_dir in (_shared_weight_cache_dir(), None):
        try:
            return snapshot_download(
                repo_id=model_id,
                revision=model_revision or None,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except Exception:
            continue
    # the prefetch step reported success but the cache has no resolvable snapshot: a transient
    # volume/link glitch, not a config error — retriable so the run lands on a healthy worker.
    raise RetriableInfraError(
        f"model {model_id} not resolvable from the local HF cache after prefetch"
    )


def _warmstart_adapter_path(model_id: str, model_revision: str, expected_rank: int) -> str | None:
    spec = _w.JOB_SPEC
    source = spec.train.init_from_adapter if spec else ""
    if not source:
        return None
    adapter_dir = _w._download_adapter(source)
    if not adapter_dir:
        raise RuntimeError("the prepared SFT warm-start adapter could not be downloaded")
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as file:
        config = json.load(file)
    rank = int(config.get("r") or 0)
    if rank != expected_rank:
        raise ValueError(
            f"SFT warm-start adapter rank {rank} does not match the prepared train.lora_rank "
            f"{expected_rank}; rank changes are not supported"
        )
    _w.validate_lora_target_parameters(config, model_id)
    base = str(config.get("base_model_name_or_path") or "").strip()
    if base and base != model_id:
        raise ValueError("SFT warm-start adapter base model does not match the target model")
    revision = str(config.get("revision") or "").strip()
    if revision and model_revision and revision != model_revision:
        raise ValueError("SFT warm-start adapter revision does not match the target model revision")
    return adapter_dir


def _verl_image_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        raise ValueError("multimodal message content must be text or content blocks")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("multimodal message content blocks must be objects")
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif block.get("type") == "image":
            parts.append("<image>")
        else:
            raise ValueError(f"unsupported multimodal SFT content block {block.get('type')!r}")
    return "".join(parts)


def _restore_verl_resume(local_dir: str) -> int:
    resume = _w.hf_resume_checkpoint()
    if not resume:
        return 0
    return stage_verl_resume(resume, local_dir, job_label="SFT")


def _durable_required_save_steps(required_steps: tuple[int, ...], resume_step: int) -> set[int]:
    candidates = [step for step in required_steps if step <= resume_step]
    if not candidates:
        return set()
    if not _w.HF_REPO:
        raise RuntimeError("required SFT saves have no artifact repository")
    durable: set[int] = set()
    for step in candidates:
        marker = f"{_w.hf_prefix()}/checkpoints/step-{step}/adapter/adapter_config.json"
        try:
            exists = _w.hf_api().file_exists(
                repo_id=_w.HF_REPO,
                filename=marker,
                repo_type="dataset",
            )
        except Exception as error:
            raise _w.RetriableInfraError(
                f"could not verify required SFT save step {step} on hf"
            ) from error
        if exists:
            durable.add(step)
    return durable


_CHILD_ENV_EXACT = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_HUB_DISABLE_XET",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "FLASH_VERL_PYTHON",
    }
)
_CHILD_ENV_PREFIXES = (
    "CUDA_",
    "NCCL_",
    "TORCH_",
    "PYTORCH_",
    "VERL_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "LC_",
    # the parent picks the gated-deltanet kernel backend before any model import (see
    # perf._force_fla_triton_gdn_on_sm100), and on sm100 FLA_TILELANG=0 is a correctness floor, not
    # a preference: tilelang's backward computed dq/dk at ~0.72 relative error there and training
    # diverged to grad_norm ~1e8. the model runs in the CHILD, so a choice the child never sees is
    # no choice at all -- grpo already passes the whole environment through and keeps it.
    "FLA_",
)


def _build_verl_child_env(*, shim_dir: str, wandb_enabled: bool) -> dict[str, str]:
    child = {
        key: value
        for key, value in os.environ.items()
        if key in _CHILD_ENV_EXACT or key.startswith(_CHILD_ENV_PREFIXES)
    }
    if wandb_enabled:
        child.update({key: value for key, value in os.environ.items() if key.startswith("WANDB_")})
    child["PYTHONPATH"] = os.pathsep.join(
        item for item in (shim_dir, os.environ.get("PYTHONPATH", "")) if item
    )
    child["PYTHONUNBUFFERED"] = "1"
    child["HYDRA_FULL_ERROR"] = "1"
    child["HF_HUB_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    return child


def _probe_gpu_in_subprocess(requested_gpu: str | None, exact_type: str = "") -> dict:
    script = r"""
import json
import sys

from flash.engine.worker.perf.lifecycle import wait_for_gpu

requested, exact = json.loads(sys.argv[1])
wait_for_gpu(requested, gpu_type=exact)
import torch
print("FLASH_GPU_PROBE=" + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    "capability": list(torch.cuda.get_device_capability(0)),
}), flush=True)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, json.dumps([requested_gpu, exact_type])],
            capture_output=True,
            text=True,
            timeout=150,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _w.RetriableInfraError("gpu readiness probe failed in its subprocess") from error
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        raise _w.RetriableInfraError(f"gpu readiness probe exited with status {result.returncode}")
    for line in result.stdout.splitlines():
        if line.startswith("FLASH_GPU_PROBE="):
            return json.loads(line.split("=", 1)[1])
    raise _w.RetriableInfraError("gpu readiness probe returned no device metadata")


class _NvidiaSmiPeakSampler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                if result.returncode == 0:
                    values = [
                        float(value.strip())
                        for value in result.stdout.splitlines()
                        if value.strip()
                    ]
                    if values:
                        self.peak_mib = max(self.peak_mib, max(values))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass

    def start(self):
        self._thread.start()
        return self

    def stop_gb(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2)
        return round(self.peak_mib / 1024, 3)


def run_sft_train(spec=None) -> None:
    """run flash sft through verl's out-of-process fsdp trainer."""
    from flash.core.catalog import MODELS, resolve_vocab_size
    from flash.engine.plan.vram import sft_chunked_nll_enabled, sft_grad_accum

    spec = spec or _w.JOB_SPEC
    env = _w.require_active_env()
    # the child trainer is seeded through its shim, but the environment's dataset/completion calls
    # run HERE in the parent. without this the documented top-level seed no longer reproduces sft
    # targets for any env whose row construction uses python/numpy randomness.
    seed_training_rngs(_w.SEED)
    started_at = time.time()
    _w.heartbeat("sft_start", gpu=_w.gpu_diagnostics(include_torch=False))
    gpu_probe = _probe_gpu_in_subprocess(
        spec.gpu.type if spec else None,
        exact_type=spec.gpu.type if spec else "",
    )

    model_id = spec.model if spec else RECIPE.hf_model_id
    model_revision = getattr(spec, "model_revision", "") if spec else ""
    train_spec = spec.train if spec else None

    def train_opt(name, default):
        value = getattr(train_spec, name, None) if train_spec else None
        return value if value is not None else default

    epochs = int(train_opt("epochs", RECIPE.sft.num_epochs))
    learning_rate = float(train_opt("learning_rate", RECIPE.sft.learning_rate))
    effective_batch = int(train_opt("batch_size", RECIPE.sft.effective_batch))
    max_steps = int(train_opt("max_steps", 0) or 0)
    save_at_steps = tuple(getattr(train_spec, "save_at_steps", ()) or ())
    save_every = int(train_opt("save_every", 50))
    gpu_count = int(getattr(getattr(spec, "gpu", None), "count", 1) or 1)

    workdir = os.path.join("/tmp", "flash-sft-verl", _w.RUN_ID, f"seed-{_w.SEED}")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    data_dir = os.path.join(workdir, "data")
    image_dir = os.path.join(workdir, "images")
    local_dir = os.path.join(workdir, "checkpoints")
    export_root = os.path.join(workdir, "checkpoint-adapters")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)

    with liveness_heartbeat("sft_data_loading"):
        from flash.engine.profiling.workload_profile import require_matching_sft_profile

        # carried on the spec, not read from `flash.__version__` here: the worker runs the plane's
        # source snapshot off PYTHONPATH with no flash distribution installed, so a locally derived
        # version is the "0+unknown" fallback and would reject every profile the plane ever froze.
        producer_version = spec.workload_profile_producer_version
        prepared_workload = prepare_sft_workload(
            spec,
            env,
            tokenizer_loader=lambda candidate, revision: _w.load_tokenizer(
                candidate,
                revision=revision,
            ),
            producer_version=producer_version,
            image_dir=image_dir,
            allow_packing=True,
        )
        expected_profile = require_matching_sft_profile(
            spec.workload_profile,
            input_digest=spec.workload_profile_input_digest,
            producer_version=producer_version,
            tokenizer_revision=model_revision,
        )
        if prepared_workload.profile != expected_profile:
            raise ValueError("sft workload changed after the quote was frozen")

        rows = prepared_workload.rows
        multimodal = prepared_workload.multimodal
        profile = prepared_workload.profile
        # the context window comes from the profile, not a second reading of the train fields. the
        # rows were truncated at the profile's max_length and the quote was priced at it, so a
        # locally re-derived value could disagree with both while the parity check above still
        # passed: that check compares two values the workload module produced, so it cannot see a
        # third derivation living here.
        max_length = profile.max_length
        dropped = profile.dropped_examples
        selected_count = profile.selected_examples
        sampled_texts = prepared_workload.sampled_texts
        multiturn_targets = prepared_workload.multiturn_targets
        if dropped:
            print(
                f"[sft] dropped {dropped} rows with no real completion target "
                "(sft_max_len truncated away the whole completion, or it was content-free)"
            )
        if multiturn_targets:
            print(
                f"[sft] multi-turn SFT: {multiturn_targets}/{selected_count} rows train on a full "
                "target transcript"
            )
        elif getattr(env, "multi_turn", False):
            print(
                "[sft][warn] this is a multi-turn environment but no row ships a multi-turn "
                "target completion"
            )
        if _w.THINKING and not any("<think>" in text for text in sampled_texts[:256]):
            print(
                "WARN: thinking mode is ON but no sampled SFT target contains a <think> trace; "
                "training on non-reasoning targets teaches the model to skip thinking"
            )

        total_tokens_per_epoch = profile.real_tokens_per_epoch
        realized_max_length = profile.realized_max_length
        masked_tokens = total_tokens_per_epoch - profile.supervised_tokens_per_epoch
        print(
            f"[sft] completion-only loss: masking {masked_tokens}/{total_tokens_per_epoch} "
            f"({masked_tokens / total_tokens_per_epoch:.0%}) prompt tokens"
        )
        train_file = os.path.join(data_dir, "train.parquet")
        _write_sft_parquet(rows, train_file)

    download_seconds = _w.prefetch_model(model_id, revision=model_revision)
    setup_seconds = time.time() - started_at
    _w.heartbeat(
        "sft_model_load",
        setup_seconds=setup_seconds,
        gpu=_w.gpu_diagnostics(include_torch=False),
    )

    lora_config = _w.make_lora(model_id)
    lora_rank = int(lora_config.r)
    lora_alpha = int(lora_config.lora_alpha)
    target_modules = lora_config.target_modules
    if isinstance(target_modules, set | frozenset):
        target_modules = sorted(target_modules)
    warmstart_adapter = _warmstart_adapter_path(model_id, model_revision, lora_rank)

    vocab_size = resolve_vocab_size(model_id, model_revision)
    fused_ce = sft_chunked_nll_enabled(model_id)
    per_device_batch, _ = sft_grad_accum(
        effective_batch,
        seq_len=realized_max_length,
        vocab=vocab_size,
        fused=fused_ce,
    )
    train_batch_size = profile.examples_per_update
    micro_batch = max(1, min(per_device_batch, train_batch_size))
    steps_per_epoch = max(1, math.ceil(len(rows) / train_batch_size))
    update_horizon = profile.authoritative_steps
    validate_save_steps(save_at_steps, update_horizon)
    loop_epochs = max(epochs, math.ceil(update_horizon / steps_per_epoch))
    save_freq = reduce(gcd, save_at_steps) if save_at_steps else save_every

    card_vram_gb = float(gpu_probe.get("memory_gb") or 0.0)
    raw_capability = gpu_probe.get("capability")
    capability = tuple(raw_capability) if raw_capability else None
    hidden, layers = _model_arch_dims(model_id, revision=model_revision)
    info = MODELS.get(model_id)
    active_params_b = float(getattr(info, "active_params_b", 0.0) or 0.0) or None
    gradient_checkpointing = _w.grad_checkpointing_on(
        model_id,
        realized_max_length,
        allow_disable=True,
        card_vram_gb=card_vram_gb,
        capability=capability,
        active_params_b=active_params_b,
        hidden=hidden,
        num_layers=layers,
        fused_ce=fused_ce,
        per_device_bs=micro_batch,
        lora_rank=lora_rank,
        revision=model_revision,
    )
    reentrant_gradient_checkpointing = bool(
        gradient_checkpointing and _w.grpo_use_reentrant(model_id)
    )

    # provisioning the verl interpreter builds a venv and installs the whole training stack when the
    # run has no prebuilt worker image, which is minutes of silence with no training step to report
    # and no liveness thread otherwise running here -- long enough for the stall watchdog to fail a
    # healthy run. no progress= : there is no monotonic counter to read, only the keepalive.
    with liveness_heartbeat("sft_configuring"):
        python_bin = resolve_verl_python(
            workdir, install_wandb=bool(os.environ.get("WANDB_API_KEY"))
        )
        # packed gdn hybrids require child support for seq_idx and cu_seqlens resets; no-fla fallbacks
        # accept and discard them. probe before packing, otherwise use boundary-correct padded input.
        # resolve the modeling module in the parent to avoid repeating the hub/cache read; empty means
        # non-hybrid.
        gdn_hybrid = model_is_gdn_hybrid(model_id, model_revision)
        gdn_module = gdn_probe_module(model_id, model_revision) if gdn_hybrid else ""
        # ONE child answers every independent capability question. each used to cost its own
        # interpreter, and the torch/verl import -- not the question -- was the price.
        caps = probe_verl_capabilities(python_bin, gdn_module)
    model_path = _cached_model_path(model_id, model_revision)
    # verl logs from the verl interpreter, so gate wandb on THAT env (see resolve_verl_loggers).
    loggers = resolve_verl_loggers(caps)
    project_name = (spec.wandb.project if spec and spec.wandb else None) or "flash"
    experiment_name = _w.wandb_run_name()
    shim_dir = os.path.join(workdir, "shim")
    os.makedirs(shim_dir, exist_ok=True)
    custom_dataset_path = os.path.join(shim_dir, "flash_verl_sft_dataset.py")

    # the gdn boundary shim resets conv and recurrent state at packed example boundaries, but only
    # when the verl child has the kernels that read seq_idx and cu_seqlens; the no-fla fallbacks
    # accept both and discard them. so the shim is installed only when the child proves it can reset.
    # `gdn_hybrid`/`gdn_module` and the child's answer are resolved above, inside the configuring
    # liveness wrap, because the probe is part of the setup silence that wrap exists to cover.
    gdn_reset_arch = gdn_reset_arch_from_caps(caps, gdn_module) if gdn_hybrid else None
    # remove-padding is required by this custom dataset and verl's no_padding loss; disabling it
    # hands sft_loss a strided tensor and fails on the first step. gdn remains safe because unsupported
    # packing pins examples_per_update and train_batch_size to 1, leaving no adjacent example state to
    # contaminate. batch size 1 is the isolation lever, not the tensor-layout flag.
    use_remove_padding = True

    config = {
        "train_files": train_file,
        "train_batch_size": train_batch_size,
        "max_length": max_length,
        "micro_batch": micro_batch,
        "max_token_len_per_gpu": realized_max_length * micro_batch,
        "custom_dataset_path": custom_dataset_path,
        "model_path": model_path,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "target_parameters": _w.lora_target_parameters(model_id),
        "lora_adapter_path": warmstart_adapter,
        "ulysses_sp_size": gpu_count,
        "lr": learning_rate,
        "warmup_ratio": RECIPE.sft.warmup_frac,
        "optimizer_impl": _VERL_OPTIMIZER_IMPL,
        "optimizer_name": _VERL_OPTIMIZER_NAME,
        "optimizer_kwargs": None,
        "local_dir": local_dir,
        "save_freq": save_freq,
        "n_gpus_per_node": gpu_count,
        "seed": _w.backend_seed(_w.SEED),
        "project_name": project_name,
        "experiment_name": experiment_name,
        "loop_epochs": loop_epochs,
        "loggers": loggers,
        # liger produces a 0.0 lora grad norm under fsdp2 + peft + gradient checkpointing, versus
        # 7.02 off in a matched qwen3.5-9b test. fused linear ce remains provided by
        # use_fused_kernels with the impl_backend resolved below.
        "use_liger": False,
        "gradient_checkpointing": gradient_checkpointing and not reentrant_gradient_checkpointing,
        "total_training_steps": update_horizon if max_steps > 0 else None,
        "total_epochs": epochs if max_steps <= 0 else None,
        "use_remove_padding": use_remove_padding,
        # resolved from the out-of-process capability probe, never by opening cuda in this parent.
        "fused_ce_backend": fused_ce_backend(caps),
    }
    overrides = build_sft_overrides(config)

    shim_source = _render_sft_sitecustomize(
        seed=config["seed"],
        loraplus_ratio=_SFT_LORAPLUS_RATIO,
        save_at_steps=save_at_steps,
        total_steps=update_horizon,
        reentrant_gradient_checkpointing=reentrant_gradient_checkpointing,
    )
    # both shims, not either: this one patches DistributedSampler/StatefulDataLoader so the child's
    # row order matches the profile's byte for byte, and the gdn one patches the model's text
    # forward to reset linear-attention state at packed example boundaries. different objects,
    # no interaction -- a gdn hybrid needs both, and dropping either is a silent correctness bug.
    shim_source += render_exact_sft_dataloader_shim()
    if gdn_reset_arch is not None:
        shim_source += render_gdn_varlen_shim(gdn_reset_arch)
    if "wandb" in loggers:
        shim_source += render_wandb_link_shim()
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as file:
        file.write(shim_source)
    with open(custom_dataset_path, "w", encoding="utf-8") as file:
        file.write(_render_sft_dataset_module())

    resume_step = _restore_verl_resume(local_dir)
    watcher = _VerlCheckpointWatcher(
        local_dir=local_dir,
        export_root=export_root,
        python_bin=python_bin,
        model_id=model_id,
        model_revision=model_revision,
        required_steps=save_at_steps,
    )
    watcher.processed_steps.update(_durable_required_save_steps(save_at_steps, resume_step))
    if resume_step >= update_horizon:
        missing = sorted(watcher.required_steps - watcher.processed_steps)
        if missing:
            raise RuntimeError(f"required saves were not durably published: {missing}")

    child_env = _build_verl_child_env(
        shim_dir=shim_dir,
        wandb_enabled="wandb" in loggers,
    )
    command = [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={gpu_count}",
        "-m",
        "verl.trainer.sft_trainer",
        *overrides,
    ]

    progress = {"step": resume_step, "loss": None, "grad_norm": None, "lr": None}
    # consecutive steps seen with grad_norm == 0.0 at a nonzero lr. one step can legitimately be
    # zero (a fully-masked micro-batch), so require a short run of them before failing.
    zero_grad_steps: list[int] = []
    # every grad_norm this session observed, so a horizon too short to trip the consecutive-run
    # guard above can still be rejected at the end. a one-update run appends exactly one zero and
    # never reaches _MAX_ZERO_GRAD_STEPS, which shipped the GRAD-001 failure the guard exists to
    # stop: done, billed, and an adapter identical to the base weights.
    observed_grad_norms: list[float] = []
    loss_curve: list[float] = []
    train_tokens = (
        sft_tokens_for_updates(
            rows,
            examples_per_update=train_batch_size,
            updates=resume_step,
            field="input_ids",
        )
        if resume_step > 0
        else 0
    )
    loraplus_applied = resume_step >= update_horizon
    wandb_link: dict[str, str | None] = {}

    def on_line(line: str) -> None:
        nonlocal loraplus_applied
        watcher.raise_if_failed()
        if _LORAPLUS_READY_MARKER in line:
            loraplus_applied = True
        link = parse_wandb_link(line)
        if link is not None:
            wandb_link.update(link)
        if verl_step_number(line) is None:
            return
        if _SFT_LORAPLUS_RATIO > 1 and not loraplus_applied:
            raise RuntimeError(
                "verl reached an optimizer step before the required lora+ shim succeeded"
            )
        # these metrics are currently floats, but use the shared parser to tolerate upstream metric
        # wrapper changes and reject nan/inf before strict-json heartbeat serialization.
        loss = parse_verl_metric(line, "train/loss")
        grad_norm = parse_verl_metric(line, "train/grad_norm")
        learning_rate_value = parse_verl_metric(line, "train/lr")
        if loss is not None:
            loss_curve.append(round(loss, 4))
            progress["loss"] = loss
        if grad_norm is not None:
            progress["grad_norm"] = grad_norm
            observed_grad_norms.append(grad_norm)
            # a 0.0 grad norm means backward produced nothing for every trainable parameter. fail the
            # run instead of billing and serving an unchanged adapter (GRAD-001).
            #
            # VERL-138: do not condition on lr. transformer_impl.py:683-688 computes grad_norm from
            # p.grad before optimizer.step() and scheduler advance, so lr cannot make the gradient zero.
            if grad_norm == 0.0:
                zero_grad_steps.append(int(progress["step"] or 0))
                if len(zero_grad_steps) >= _MAX_ZERO_GRAD_STEPS:
                    raise RuntimeError(
                        "verl reported train/grad_norm=0.0 on "
                        f"{len(zero_grad_steps)} steps: no gradient is reaching the "
                        "lora parameters, so this run would train nothing. see GRAD-001"
                    )
            else:
                zero_grad_steps.clear()
        if learning_rate_value is not None:
            progress["lr"] = learning_rate_value

    def on_step(step: int) -> None:
        progress["step"] = step
        payload = {
            "step": step,
            "loss": progress["loss"],
            "grad_norm": progress["grad_norm"],
            "learning_rate": progress["lr"],
        }
        _w.heartbeat(
            "sft_step", **{key: value for key, value in payload.items() if value is not None}
        )

    def child_heartbeat() -> None:
        _w.heartbeat("sft_step", liveness=True, step=int(progress["step"] or 0))

    gpu_sampler = _NvidiaSmiPeakSampler().start()
    train_started_at = time.time()
    return_code = 0
    if resume_step < update_horizon:
        watcher.start()
        # check save completeness only after normal training completion. callback failures occur before
        # return_code assignment; checking anyway would replace the real zero-grad or lora+ diagnosis
        # with a missing-save error. opd_train uses the same guard.
        training_completed = False
        try:
            with liveness_heartbeat(
                "sft_step",
                progress=lambda: int(progress["step"] or 0),
                progress_step=True,
            ):
                return_code = run_verl_training(
                    command,
                    env=child_env,
                    on_step=on_step,
                    on_line=on_line,
                    heartbeat=child_heartbeat,
                )
                training_completed = return_code == 0
        finally:
            watcher.stop(require_complete=training_completed)
    train_wall = time.time() - train_started_at
    device_peak_gpu_gb = gpu_sampler.stop_gb()
    if return_code != 0:
        raise RuntimeError(f"verl SFT subprocess exited with status {return_code}")
    if _SFT_LORAPLUS_RATIO > 1 and not loraplus_applied:
        raise RuntimeError("required lora+ shim did not emit its success marker")

    actor_dir, final_step = latest_global_step_dir(local_dir)
    if sft_under_ran(final_step, update_horizon, max_steps):
        raise RuntimeError(
            f"sft completed {final_step}/{update_horizon} requested optimizer updates"
        )
    # short runs may finish before the consecutive zero-grad guard fires, so reject sessions where
    # every observed update had a dead gradient. tolerate isolated zeros; abstain on resume because
    # restored weights include unseen earlier updates.
    if not resume_step and observed_grad_norms and not any(observed_grad_norms):
        raise RuntimeError(
            f"verl reported train/grad_norm=0.0 on every one of {len(observed_grad_norms)} "
            "observed optimizer updates: no gradient is reaching the lora parameters, so this "
            "run would train nothing. see GRAD-001"
        )
    train_tokens = sft_tokens_for_updates(
        rows,
        examples_per_update=train_batch_size,
        updates=final_step,
        field="input_ids",
    )

    with liveness_heartbeat(
        "sft_finalizing",
        progress=lambda: final_step,
        progress_step=True,
        keepalive=True,
    ):
        adapter_dir = os.path.join(workdir, "adapter")
        _export_checkpoint_adapter(
            actor_dir,
            adapter_dir,
            model_id=model_id,
            model_revision=model_revision,
            python_bin=python_bin,
        )
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        if final_save_due(final_step, save_at_steps) and final_step not in watcher.processed_steps:
            _w.publish_deployable_checkpoint(adapter_dir, final_step)

    _w.heartbeat(
        "sft_trained",
        train_wall=train_wall,
        step=final_step,
        gpu=_w.gpu_diagnostics(include_torch=False),
    )
    _w.write_train_meta(
        phase="sft",
        adapter_dir=adapter_dir,
        model_id=model_id,
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=train_tokens,
        generated_tokens=0,
        step=final_step,
        notes={
            "epochs": epochs,
            "resumed": bool(resume_step),
            "warm_started": bool(warmstart_adapter),
            "download_seconds": download_seconds,
            "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
            "thinking": _w.THINKING,
            "multimodal": multimodal,
            "gradient_checkpointing": gradient_checkpointing,
            "gradient_checkpointing_reentrant": reentrant_gradient_checkpointing,
            "configured_max_length": max_length,
            "realized_max_length": realized_max_length,
            "runtime_max_length": realized_max_length,
            "per_device_train_batch_size": micro_batch,
            "gradient_accumulation_steps": math.ceil(train_batch_size / micro_batch),
            # verl concatenates either way; the profile's mode records whether more than one
            # example was allowed to share a concatenated batch, which is what a reader of these
            # metrics needs in order to compare a run's step count against its row count.
            "packing": profile.packing_mode,
            # the tensor layout verl actually ran. always remove-padding now: it is the only layout
            # FlashTokenizedSFTDataset fits, and the quoted step count comes from
            # profile.examples_per_update rather than from this. kept because a reader comparing
            # realized step time against the quote still needs the executed layout stated, not
            # inferred.
            "realized_packing": "verl_remove_padding",
            "gdn_boundary_resets": (gdn_reset_arch is not None) if gdn_hybrid else None,
            "loss_curve": loss_curve[:400],
            "peak_gpu_gb": device_peak_gpu_gb,
            "device_peak_gpu_gb": device_peak_gpu_gb,
            "loraplus_optim": _VERL_OPTIMIZER_NAME,
            "loraplus_applied": loraplus_applied,
            "verl_backend": "fsdp2",
            "ulysses_sequence_parallel_size": gpu_count,
            "wandb_project": project_name if "wandb" in loggers else None,
            "wandb_run_name": experiment_name if "wandb" in loggers else None,
            # the sdk's link_wandb reads notes["wandb_url"]; trl gets it from the parent's live
            # wandb.run, verl from the child marker (see backend_common.render_wandb_link_shim).
            "wandb_url": wandb_link.get("wandb_url"),
            "wandb_id": wandb_link.get("wandb_id"),
        },
    )


# re-exported so the child-render helpers stay reachable as `sft_train.<name>`: the other
# trainers import `_hydra_val` from here, and the sft tests import the renderers and the
# LoRA+ marker from here.
# re-exported because the sft tests patch `sft_train._export_checkpoint_adapter` and
# `sft_train._VerlCheckpointWatcher`, and `run_sft_train` below constructs the watcher.
from flash.engine.worker.train.sft_checkpoints import (  # noqa: E402,F401
    _copy_processing_sidecars,
    _export_checkpoint_adapter,
    _VerlCheckpointWatcher,
)
from flash.engine.worker.train.sft_config import (  # noqa: E402,F401
    _LORAPLUS_READY_MARKER,
    _REQUIRED_OVERRIDE_KEYS,
    _VERL_OPTIMIZER_IMPL,
    _VERL_OPTIMIZER_NAME,
    _hydra_val,
    _optimizer_override_config,
    _render_sft_dataset_module,
    _render_sft_sitecustomize,
    _sft_parquet_features,
    _write_sft_parquet,
    build_sft_overrides,
    render_exact_sft_dataloader_shim,
    render_loraplus_shim,
)
