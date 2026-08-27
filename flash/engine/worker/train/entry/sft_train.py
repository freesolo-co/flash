"""sft training via verl in a separate interpreter.

flash writes exact conversation ids and completion-only masks to parquet. the parent streams progress
and checkpoints without holding cuda while torchrun owns the devices.
"""

from __future__ import annotations

import math
import os
import time

import flash.engine.worker.io.heartbeat as _worker_heartbeat
import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.perf as _worker_perf
import flash.engine.worker.runtime.state as _worker_state
import flash.engine.worker.train.core.lifecycle.finalize as _worker_finalize
import flash.engine.worker.train.sft.orchestration as _sft
import flash.engine.worker.train.sft.setup.checkpoints as _sft_checkpoints
import flash.engine.worker.verl.install as _verl_install
from flash.engine.plan.steps import final_save_due
from flash.engine.worker.model.packing import model_is_gdn_hybrid
from flash.engine.worker.train.entry.sft_train_runner import (
    _consume_sft_marker_line,
    _finish_sft_child,
    _invoke_sft_child,
    _prepare_sft_child,
    _prepare_sft_data,
    _prepare_sft_model,
    _prepare_sft_progress,
    _record_sft_step_metrics,
    _resolve_sft_options,
    _SftProgressCallbacks,
    _verify_sft_run,
)
from flash.engine.worker.train.sft.orchestration import (
    _CHILD_ENV_EXACT as _CHILD_ENV_EXACT,
)
from flash.engine.worker.train.sft.orchestration import (
    _CHILD_ENV_PREFIXES as _CHILD_ENV_PREFIXES,
)
from flash.engine.worker.train.sft.orchestration import (
    _MAX_ZERO_GRAD_STEPS as _MAX_ZERO_GRAD_STEPS,
)
from flash.engine.worker.train.sft.orchestration import (
    _SFT_LORAPLUS_RATIO as _SFT_LORAPLUS_RATIO,
)
from flash.engine.worker.train.sft.orchestration import (
    RECIPE as RECIPE,
)
from flash.engine.worker.train.sft.orchestration import (
    SHIM_FRAGMENT_FAILED_EXIT_CODE as SHIM_FRAGMENT_FAILED_EXIT_CODE,
)
from flash.engine.worker.train.sft.orchestration import (
    _build_verl_child_env as _build_verl_child_env,
)
from flash.engine.worker.train.sft.orchestration import (
    _cached_model_path as _cached_model_path,
)
from flash.engine.worker.train.sft.orchestration import (
    _durable_required_save_steps as _durable_required_save_steps,
)
from flash.engine.worker.train.sft.orchestration import (
    _model_arch_dims as _model_arch_dims,
)
from flash.engine.worker.train.sft.orchestration import (
    _NvidiaSmiPeakSampler as _NvidiaSmiPeakSampler,
)
from flash.engine.worker.train.sft.orchestration import (
    _probe_gpu_in_subprocess as _probe_gpu_in_subprocess,
)
from flash.engine.worker.train.sft.orchestration import (
    _resolve_sft_fused_ce_backend as _resolve_sft_fused_ce_backend,
)
from flash.engine.worker.train.sft.orchestration import (
    _resolve_sft_grad_accum as _resolve_sft_grad_accum,
)
from flash.engine.worker.train.sft.orchestration import (
    _resolve_sft_gradient_checkpointing as _resolve_sft_gradient_checkpointing,
)
from flash.engine.worker.train.sft.orchestration import (
    _resolve_sft_reentrant_gradient_checkpointing as _resolve_sft_reentrant_gradient_checkpointing,
)
from flash.engine.worker.train.sft.orchestration import (
    _resolve_sft_vocab_size as _resolve_sft_vocab_size,
)
from flash.engine.worker.train.sft.orchestration import (
    _restore_verl_resume as _restore_verl_resume,
)
from flash.engine.worker.train.sft.orchestration import (
    _seed_resume_lifecycle as _seed_resume_lifecycle,
)
from flash.engine.worker.train.sft.orchestration import (
    _sft_liger_config as _sft_liger_config,
)
from flash.engine.worker.train.sft.orchestration import (
    _sft_profile_max_length as _sft_profile_max_length,
)
from flash.engine.worker.train.sft.orchestration import (
    _SftCapabilities as _SftCapabilities,
)
from flash.engine.worker.train.sft.orchestration import (
    _SftOutputs as _SftOutputs,
)
from flash.engine.worker.train.sft.orchestration import (
    _verl_image_message_content as _verl_image_message_content,
)
from flash.engine.worker.train.sft.orchestration import (
    _warmstart_adapter_path as _warmstart_adapter_path,
)
from flash.engine.worker.train.sft.orchestration import (
    prepare_sft_workload as prepare_sft_workload,
)
from flash.engine.worker.train.sft.orchestration import (
    seed_training_rngs as seed_training_rngs,
)
from flash.engine.worker.train.sft.orchestration import (
    sft_tokens_for_updates as sft_tokens_for_updates,
)
from flash.engine.worker.train.sft.orchestration import (
    sft_under_ran as sft_under_ran,
)
from flash.engine.worker.train.sft.setup.checkpoints import (
    _export_checkpoint_adapter as _export_checkpoint_adapter,
)
from flash.engine.worker.train.sft.setup.checkpoints import (
    _VerlCheckpointWatcher as _VerlCheckpointWatcher,
)
from flash.engine.worker.train.sft.setup.config import (
    _LORAPLUS_READY_MARKER as _LORAPLUS_READY_MARKER,
)
from flash.engine.worker.train.sft.setup.config import (
    _VERL_OPTIMIZER_IMPL as _VERL_OPTIMIZER_IMPL,
)
from flash.engine.worker.train.sft.setup.config import (
    _VERL_OPTIMIZER_NAME,
)
from flash.engine.worker.train.sft.setup.config import (
    _hydra_val as _hydra_val,
)
from flash.engine.worker.train.sft.setup.config import (
    _render_sft_dataset_module as _render_sft_dataset_module,
)
from flash.engine.worker.train.sft.setup.config import (
    _write_sft_parquet as _write_sft_parquet,
)
from flash.engine.worker.train.sft.setup.config import (
    build_sft_overrides as build_sft_overrides,
)
from flash.engine.worker.verl.capabilities import (
    gdn_probe_module,
    gdn_reset_arch_from_caps,
    probe_verl_capabilities,
    require_gdn_boundary_resets,
    strict_gdn_probe_module,
)
from flash.engine.worker.verl.child_io import parse_verl_metric
from flash.engine.worker.verl.parallelism import ULYSSES_SEQUENCE_PARALLEL_SIZE


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
    with _sft.liveness_heartbeat("sft_data_loading"):
        data = _prepare_sft_data(options)
    model = _prepare_sft_model(options, data)

    # provisioning the verl interpreter builds a venv and installs the whole training stack when the
    # run has no prebuilt worker image, which is minutes of silence with no training step to report
    # and no liveness thread otherwise running here -- long enough for the stall watchdog to fail a
    # healthy run. no progress= : there is no monotonic counter to read, only the keepalive.
    with _sft.liveness_heartbeat("sft_configuring"):
        python_bin = _verl_install.resolve_verl_python(
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
        capabilities = _sft._SftCapabilities(
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
    gpu_sampler = _sft._NvidiaSmiPeakSampler().start()
    train_started_at = time.time()
    return_code = 0
    if child.resume_step < model.update_horizon:
        child.watcher.start()
        # check save completeness only after normal training completion. callback failures occur before
        # return_code assignment; checking anyway would replace the real zero-grad or lora+ diagnosis
        # with a missing-save error. opd_train uses the same guard.
        training_completed = False
        try:
            with _sft.liveness_heartbeat(
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
    with _sft.liveness_heartbeat(
        "sft_finalizing",
        progress=lambda: final_step,
        progress_step=True,
        keepalive=True,
    ):
        adapter_dir = os.path.join(options.paths.workdir, "adapter")
        _sft_checkpoints._export_checkpoint_adapter(
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
        outputs = _sft._SftOutputs(adapter_dir, train_wall, device_peak_gpu_gb)
    _write_sft_result(options, data, model, child, child_progress, verified, outputs)
