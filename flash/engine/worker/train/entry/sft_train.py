"""sft training via verl in a separate interpreter.

flash writes exact conversation ids and completion-only masks to parquet. the parent streams progress
and checkpoints without holding cuda while torchrun owns the devices.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from functools import partial

import flash.engine.worker.io.heartbeat as _worker_heartbeat
import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.model.adapter as _worker_adapter
import flash.engine.worker.perf as _worker_perf
import flash.engine.worker.runtime.state as _worker_state
import flash.engine.worker.train.core.lifecycle.finalize as _worker_finalize
from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
from flash.adapters.lora_rank import alpha_from_adapter_config, rank_from_adapter_config
from flash.engine.worker.runtime.kernel_warmup import KERNEL_CACHE_ENV_SUBDIRS
from flash.engine.worker.verl.checkpoints import restore_verl_resume
from flash.engine.worker.verl.parallelism import ULYSSES_SEQUENCE_PARALLEL_SIZE

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


def _warmstart_adapter_path(
    model_id: str,
    model_revision: str,
    expected_rank: int,
    expected_alpha: int,
    targeting=None,
) -> str | None:
    """Stage and verify this run's warm-start source adapter; None when it is not a warm start.

    Shared by the SFT and OPD runners (see ``opd_train``), and the source adapter may come from a
    run of any algorithm, so nothing here may describe either end as SFT: warm start continues one
    LoRA in place across every algorithm pair.
    """
    spec = _worker_state.JOB_SPEC
    source = spec.train.init_from_adapter if spec else ""
    if not source:
        return None
    adapter_dir = _worker_adapter._download_adapter(source)
    if not adapter_dir:
        raise RuntimeError("the prepared warm-start adapter could not be downloaded")
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    rank = rank_from_adapter_config(config, source=config_path)
    if rank != expected_rank:
        raise ValueError(
            f"warm-start adapter rank {rank} does not match the prepared train.lora_rank "
            f"{expected_rank}; rank changes are not supported"
        )
    # alpha is checked alongside rank because the pair sets the adapter's scaling: continuing a
    # LoRA at the source rank but a different alpha silently rescales every trained delta. grpo
    # already rereads both from the source (see rl/inputs.py::_resolve_warmstart_config); without
    # this, sft and opd failed closed on rank and open on alpha.
    alpha = alpha_from_adapter_config(config, source=config_path)
    if alpha != expected_alpha:
        raise ValueError(
            f"warm-start adapter alpha {alpha} does not match the prepared train.lora_alpha "
            f"{expected_alpha}; alpha changes are not supported"
        )
    if targeting is None:
        _worker_adapter.validate_warmstart_adapter(config, model_id, adapter_dir)
    else:
        _worker_adapter.validate_warmstart_adapter(config, model_id, adapter_dir, targeting)
    base = str(config.get("base_model_name_or_path") or "").strip()
    if base and base != model_id:
        raise ValueError("warm-start adapter base model does not match the target model")
    revision = str(config.get("revision") or "").strip()
    if revision and model_revision and revision != model_revision:
        raise ValueError("warm-start adapter revision does not match the target model revision")
    return adapter_dir


_restore_verl_resume = partial(restore_verl_resume, job_label="SFT")


def _durable_required_save_steps(required_steps: tuple[int, ...], resume_step: int) -> set[int]:
    candidates = [step for step in required_steps if step <= resume_step]
    if not candidates:
        return set()
    if not _worker_state.HF_REPO:
        raise RuntimeError("required SFT saves have no artifact repository")
    durable: set[int] = set()
    for step in candidates:
        marker = f"{_worker_hf.hf_prefix()}/checkpoints/step-{step}/adapter/adapter_config.json"
        try:
            exists = _worker_hf.hf_api().file_exists(
                repo_id=_worker_state.HF_REPO,
                filename=marker,
                repo_type="dataset",
            )
        except Exception as error:
            raise _worker_perf.RetriableInfraError(
                f"could not verify required SFT save step {step} on hf"
            ) from error
        if exists:
            durable.add(step)
    return durable


def _seed_resume_lifecycle(watcher, required_steps: tuple[int, ...], resume_step: int) -> None:
    """record what a previous attempt already made durable, before the watcher's first sweep.

    the resume-state half of this is the shared rule and lives on the ledger. what stays here is the
    part only sft and opd can answer: the prior attempt's deployable publish was best-effort for a
    periodic save, so a required step is credited only when ``_durable_required_save_steps`` finds
    its adapter on hf. a required step whose adapter is missing stays undiscovered, so this watcher
    stages and publishes it without re-uploading the full state hf already has.
    """
    watcher.lifecycle.seed_resumed_step(resume_step, frozenset(required_steps))
    for step in _durable_required_save_steps(required_steps, resume_step):
        watcher.lifecycle.mark_deployable_published(step)
        watcher.lifecycle.mark_discovered(step)


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
    # the baked per-sm kernel cache is addressed ONLY through these vars, and the child is the
    # interpreter that actually trains: a cache the child never sees is no cache at all, so without
    # them it falls back to $HOME/.cache and re-JITs the FA2/GDN triton kernels the bake already paid
    # ~124s to build. taken from kernel_warmup rather than restated so the two cannot drift -- and
    # they must be named EXACTly, because TORCHINDUCTOR_CACHE_DIR does NOT match the "TORCH_" prefix
    # below (its sixth character is "I"), which is how it was dropped despite looking like it passes.
    # safe to share: /opt/verl-venv resolves the SAME torch as the parent (both 2.10.0+cu128 on the
    # published cu128-sm90 image), and both are image-local, so this shares nothing across tenants --
    # see providers/_lifecycle/worker.py, which keeps JIT caches OFF the multi-tenant volume.
    | KERNEL_CACHE_ENV_SUBDIRS.keys()
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
    applied_secret_names = frozenset(
        name.strip() for name in os.environ.get(SECRET_ENV_KEYS_ENV, "").split(",") if name.strip()
    )
    child = {
        key: value
        for key, value in os.environ.items()
        if key in _CHILD_ENV_EXACT or key.startswith(_CHILD_ENV_PREFIXES)
    }
    if wandb_enabled:
        child.update({key: value for key, value in os.environ.items() if key.startswith("WANDB_")})
    # retained runtime namespaces are filtered by applied-secret provenance. authenticated child-side
    # wandb logging is the sole secret exception, and only when the existing logger gate enables it.
    for name in applied_secret_names:
        child.pop(name, None)
    if wandb_enabled and "WANDB_API_KEY" in os.environ:
        child["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    parent_pythonpath = (
        os.environ.get("PYTHONPATH", "") if "PYTHONPATH" not in applied_secret_names else ""
    )
    child["PYTHONPATH"] = os.pathsep.join(item for item in (shim_dir, parent_pythonpath) if item)
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
        raise _worker_perf.RetriableInfraError(
            "gpu readiness probe failed in its subprocess"
        ) from error
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        raise _worker_perf.RetriableInfraError(
            f"gpu readiness probe exited with status {result.returncode}"
        )
    for line in result.stdout.splitlines():
        if line.startswith("FLASH_GPU_PROBE="):
            return json.loads(line.split("=", 1)[1])
    raise _worker_perf.RetriableInfraError("gpu readiness probe returned no device metadata")


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


def _resolve_sft_vocab_size(model_id: str, model_revision: str) -> int:
    from flash.core.catalog import resolve_vocab_size

    return resolve_vocab_size(model_id, model_revision)


def _resolve_sft_grad_accum(effective_batch: int, **kwargs):
    from flash.engine.plan.vram import sft_grad_accum

    return sft_grad_accum(effective_batch, **kwargs)


def _resolve_sft_gradient_checkpointing(model_id: str, max_length: int, **kwargs) -> bool:
    return _worker_perf.grad_checkpointing_on(model_id, max_length, **kwargs)


def _resolve_sft_reentrant_gradient_checkpointing(model_id: str) -> bool:
    return _worker_perf.grpo_use_reentrant(model_id)


def _resolve_sft_fused_ce_backend(caps):
    return fused_ce_backend(caps)


def _sft_profile_max_length(profile) -> int:
    max_length = profile.max_length
    return max_length  # noqa: RET504


def _sft_liger_config() -> dict[str, bool]:
    return {"use_liger": False}


# re-exported so the child-render helpers stay reachable as `sft_train.<name>`: the other
# trainers import `_hydra_val` from here, and the sft tests import the renderers and the
# LoRA+ marker from here.
# the runner resolves patched seams through this parent module at call time, so these imports must
# remain attributes of `sft_train` rather than direct imports in the split module.
from flash.engine.plan.recipe import RECIPE  # noqa: E402,F401
from flash.engine.plan.steps import final_save_due, validate_save_steps  # noqa: E402,F401
from flash.engine.profiling.sft_workload import (  # noqa: E402,F401
    prepare_sft_workload,
    sft_tokens_for_updates,
)
from flash.engine.worker.entry.sft import _model_arch_dims, sft_under_ran  # noqa: E402,F401
from flash.engine.worker.io.heartbeat import liveness_heartbeat  # noqa: E402
from flash.engine.worker.model.packing import model_is_gdn_hybrid  # noqa: E402
from flash.engine.worker.runtime.rng import seed_training_rngs  # noqa: E402,F401
from flash.engine.worker.train.entry.backend_common import (  # noqa: E402,F401
    SHIM_FRAGMENT_FAILED_EXIT_CODE,
    fused_ce_backend,
    gdn_probe_module,
    gdn_reset_arch_from_caps,
    latest_global_step_dir,
    parse_verl_metric,
    parse_wandb_link,
    probe_verl_capabilities,
    render_sitecustomize_bootstrap,
    require_gdn_boundary_resets,
    resolve_verl_loggers,
    resolve_verl_python,
    run_verl_training,
    shim_marker_file,
    strict_gdn_probe_module,
    verify_applied_shim_markers,
    verl_step_number,
)
from flash.engine.worker.train.sft.setup.checkpoints import (  # noqa: E402,F401
    _copy_processing_sidecars,
    _export_checkpoint_adapter,
    _VerlCheckpointWatcher,
)
from flash.engine.worker.train.sft.setup.config import (  # noqa: E402,F401
    _LORAPLUS_READY_MARKER,
    _REQUIRED_OVERRIDE_KEYS,
    _VERL_OPTIMIZER_IMPL,
    _VERL_OPTIMIZER_NAME,
    _hydra_val,
    _optimizer_override_config,
    _render_sft_dataset_module,
    _sft_parquet_features,
    _write_sft_parquet,
    build_sft_overrides,
)

# isort: split
from flash.engine.worker.train.entry.sft_train_runner import (  # noqa: E402
    _consume_sft_marker_line,
    _finish_sft_child,
    _invoke_sft_child,
    _prepare_sft_child,
    _prepare_sft_data,
    _prepare_sft_model,
    _prepare_sft_progress,
    _record_sft_step_metrics,
    _resolve_sft_options,
    _SftCapabilities,
    _SftOutputs,
    _SftProgressCallbacks,
    _verify_sft_run,
)


def _write_sft_result(options, data, model, child, progress, verified, outputs) -> None:
    adapter_dir = outputs.adapter_dir
    train_wall = outputs.train_wall
    device_peak_gpu_gb = outputs.device_peak_gpu_gb
    _worker_heartbeat.heartbeat(
        "sft_trained",
        train_wall=train_wall,
        step=verified.final_step,
        gpu=_worker_perf.gpu_diagnostics(include_torch=False),
    )
    _worker_finalize.write_train_meta(
        phase="sft",
        adapter_dir=adapter_dir,
        model_id=options.model_id,
        train_wall=train_wall,
        setup_seconds=model.setup_seconds,
        train_tokens=verified.train_tokens,
        generated_tokens=0,
        step=verified.final_step,
        notes={
            "epochs": options.epochs,
            "resumed": bool(child.resume_step),
            "warm_started": bool(model.warmstart_adapter),
            "download_seconds": model.download_seconds,
            "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
            "thinking": _worker_state.THINKING,
            "multimodal": data.multimodal,
            "gradient_checkpointing": model.gradient_checkpointing,
            "gradient_checkpointing_reentrant": model.reentrant_gradient_checkpointing,
            "configured_max_length": data.max_length,
            "realized_max_length": data.realized_max_length,
            "runtime_max_length": data.realized_max_length,
            # the EXECUTED micro-batch, not the requested one: data parallelism caps it to a rank's
            # share of the batch, so a reader reconstructing the token budget off the request would
            # believe each rank held rows it never received.
            "per_device_train_batch_size": child.micro_batch,
            # over one RANK'S share of the batch, not the global batch. `micro_batch` is already
            # capped to `train_batch_size // world_size` (see `_resolve_sft_width_and_micro_batch`),
            # so dividing the global batch by it was the sequence-parallel formula, where every rank
            # sees the whole batch. under data parallelism it over-counts by the world size, and a
            # reader reconstructing tokens as micro-batch x grad-accum x DP size lands world_size
            # times too high.
            "gradient_accumulation_steps": math.ceil(
                (model.train_batch_size / max(1, child.world_size)) / child.micro_batch
            ),
            # verl concatenates either way; the profile's mode records whether more than one
            # example was allowed to share a concatenated batch, which is what a reader of these
            # metrics needs in order to compare a run's step count against its row count.
            "packing": data.profile.packing_mode,
            # the tensor layout verl actually ran. always remove-padding now: it is the only layout
            # FlashTokenizedSFTDataset fits, and the quoted step count comes from
            # profile.examples_per_update rather than from this. kept because a reader comparing
            # realized step time against the quote still needs the executed layout stated, not
            # inferred.
            "realized_packing": "verl_remove_padding",
            "gdn_boundary_resets": (
                (child.gdn_reset_arch is not None) if child.gdn_hybrid else None
            ),
            "loss_curve": progress.loss_curve[:400],
            "peak_gpu_gb": device_peak_gpu_gb,
            "device_peak_gpu_gb": device_peak_gpu_gb,
            "loraplus_optim": _VERL_OPTIMIZER_NAME,
            "loraplus_applied": progress.loraplus_applied,
            "verl_backend": "fsdp2",
            # sft shards by DATA -- see ULYSSES_SEQUENCE_PARALLEL_SIZE for why. fsdp splits the batch
            # across the ranks actually LAUNCHED, which is the allocated card count only when the
            # batch divides by it, so report the executed width rather than the allocation ceiling.
            "ulysses_sequence_parallel_size": ULYSSES_SEQUENCE_PARALLEL_SIZE,
            "data_parallel_size": child.world_size,
            "wandb_project": child.project_name if "wandb" in child.loggers else None,
            "wandb_run_name": child.experiment_name if "wandb" in child.loggers else None,
            # the sdk's link_wandb reads notes["wandb_url"]; trl gets it from the parent's live
            # wandb.run, verl from the child marker (see backend_common.render_wandb_link_shim).
            "wandb_url": progress.wandb_link.get("wandb_url"),
            "wandb_id": progress.wandb_link.get("wandb_id"),
        },
    )


def run_sft_train(spec=None) -> None:
    """run flash sft through verl's out-of-process fsdp trainer."""
    options = _resolve_sft_options(spec)
    with liveness_heartbeat("sft_data_loading"):
        data = _prepare_sft_data(options)
    model = _prepare_sft_model(options, data)

    # provisioning the verl interpreter builds a venv and installs the whole training stack when the
    # run has no prebuilt worker image, which is minutes of silence with no training step to report
    # and no liveness thread otherwise running here -- long enough for the stall watchdog to fail a
    # healthy run. no progress= : there is no monotonic counter to read, only the keepalive.
    with liveness_heartbeat("sft_configuring"):
        python_bin = resolve_verl_python(
            options.paths.workdir, install_wandb=bool(os.environ.get("WANDB_API_KEY"))
        )
        # packed gdn hybrids require child support for seq_idx and cu_seqlens resets; no-fla fallbacks
        # accept and discard them. probe before packing, otherwise use boundary-correct padded input.
        #
        # the PROFILE is authoritative for "is this a gdn hybrid": it was frozen by a RAISING probe
        # (see `_packing_mode`), while `model_is_gdn_hybrid` swallows a failed hub/cache read and
        # answers False. trusting the swallow alone would let a transient failure skip the
        # boundary-reset requirement below on an already-packed profile -- the one combination that
        # trains across example boundaries unprotected.
        gdn_hybrid = data.profile.architecture_mode == "gdn-hybrid" or model_is_gdn_hybrid(
            options.model_id, options.model_revision
        )
        # and the MODULE has to fail closed for the same reason one layer down: `gdn_model_type`
        # swallows a failed config read and defaults to "qwen3_5". on a packed `qwen3_5_moe` model
        # (Qwen/Qwen3.6-35B-A3B is in the catalog) that default names the DENSE module, so the child
        # would clear it, the shim would patch it, and the real MoE layers would stay unpatched --
        # state crossing packed boundaries while the log says resets are active. resolve it strictly
        # for a packed run and let the failure surface instead.
        # keyed on the REALIZED batch, matching the reset gate below: a packed profile whose
        # examples_per_update is 1 has no neighbours, so it needs neither the strict resolve nor
        # the hard gate.
        if (
            gdn_hybrid
            and data.profile.packing_mode == "packed"
            and data.profile.examples_per_update > 1
        ):
            gdn_module = strict_gdn_probe_module(options.model_id, options.model_revision)
        else:
            gdn_module = (
                gdn_probe_module(options.model_id, options.model_revision) if gdn_hybrid else ""
            )
        # ONE child answers every independent capability question. each used to cost its own
        # interpreter, and the torch/verl import -- not the question -- was the price.
        caps = probe_verl_capabilities(python_bin, gdn_module)
        capabilities = _SftCapabilities(
            python_bin=python_bin,
            caps=caps,
            gdn_hybrid=gdn_hybrid,
            gdn_module=gdn_module,
        )

    # the control-plane gate cannot answer whether resets actually run: it is device-independent by
    # construction, so it can prove the kernels are installed
    # but not that the conv kernel runs on THIS card. the child probe is the only place that
    # question can be answered, and continuing without resets would train across packed example
    # boundaries while looking patched. an exact-unpacked run keeps the soft form:
    # examples_per_update is 1, so there are no packed neighbours to contaminate.
    # `packed` with examples_per_update == 1 is still boundary-safe: min(batch_size, len(rows)) can
    # land on 1, and one example per update has no neighbour to contaminate. requiring resets there
    # would reject a run the unpacked path executes happily, so key the demand on the REALIZED
    # batch rather than the label.
    packed_neighbours = (
        data.profile.packing_mode == "packed" and data.profile.examples_per_update > 1
    )
    if gdn_hybrid and packed_neighbours:
        gdn_reset_arch = require_gdn_boundary_resets(caps, gdn_module)
    else:
        gdn_reset_arch = gdn_reset_arch_from_caps(caps, gdn_module) if gdn_hybrid else None
    # remove-padding is required by this custom dataset and verl's no_padding loss; disabling it
    # hands sft_loss a strided tensor and fails on the first step. an unsupported gdn stack stays
    # safe because packing pins examples_per_update and train_batch_size to 1, leaving no adjacent
    # example state to contaminate; a supported one is safe because the shim above resets at every
    # boundary. batch size 1 is the isolation lever, not the tensor-layout flag.
    use_remove_padding = True
    child = _prepare_sft_child(
        options, data, model, capabilities, use_remove_padding, gdn_reset_arch
    )
    child_progress = _prepare_sft_progress(data, model, child)
    progress = child_progress.values

    def on_line(line: str) -> None:
        child.watcher.raise_if_failed()
        if not _consume_sft_marker_line(child_progress, line):
            return
        # these metrics are currently floats, but use the shared parser to tolerate upstream metric
        # wrapper changes and reject nan/inf before strict-json heartbeat serialization.
        loss = parse_verl_metric(line, "train/loss")
        grad_norm = parse_verl_metric(line, "train/grad_norm")
        learning_rate_value = parse_verl_metric(line, "train/lr")
        _record_sft_step_metrics(child_progress, loss, grad_norm, learning_rate_value)

    callbacks = _SftProgressCallbacks(child_progress)
    gpu_sampler = _NvidiaSmiPeakSampler().start()
    train_started_at = time.time()
    return_code = 0
    if child.resume_step < model.update_horizon:
        child.watcher.start()
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
                return_code = _invoke_sft_child(child, callbacks, on_line)
                training_completed = return_code == 0
        finally:
            child.watcher.stop(require_complete=training_completed)
    train_wall, device_peak_gpu_gb = _finish_sft_child(
        gpu_sampler, train_started_at, return_code, child_progress
    )
    verified = _verify_sft_run(options, data, model, child_progress, child.resume_step)
    actor_dir = verified.actor_dir
    final_step = verified.final_step
    with liveness_heartbeat(
        "sft_finalizing",
        progress=lambda: final_step,
        progress_step=True,
        keepalive=True,
    ):
        adapter_dir = os.path.join(options.paths.workdir, "adapter")
        _export_checkpoint_adapter(
            actor_dir,
            adapter_dir,
            model_id=options.model_id,
            model_revision=options.model_revision,
            exclude_modules=model.exclude_modules,
            python_bin=child.python_bin,
            preprocessor=data.processor,
        )
        _worker_hf.hf_upload_folder(adapter_dir, "adapter", required=True)
        # only a durably published adapter may suppress the final publish. the seeded resume step no
        # longer needs excluding by hand: it is credited as deployable_published only when its
        # adapter was actually found on hf, so a resume that carried resumable state without a
        # servable adapter falls through to this publish instead of being skipped.
        if (
            final_save_due(final_step, options.save_at_steps)
            and final_step not in child.watcher.lifecycle.deployable_published_steps
        ):
            _worker_hf.publish_deployable_checkpoint(adapter_dir, final_step)
        outputs = _SftOutputs(adapter_dir, train_wall, device_peak_gpu_gb)
    _write_sft_result(options, data, model, child, child_progress, verified, outputs)
