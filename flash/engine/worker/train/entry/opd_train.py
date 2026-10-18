"""Flash OPD orchestration through verl 0.8.0 in an isolated child interpreter."""

from __future__ import annotations

import os
import time

import flash.engine.worker.io.heartbeat as _worker_heartbeat
import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.perf as _worker_perf
import flash.engine.worker.runtime.state as _worker_state
import flash.engine.worker.train.core.lifecycle.finalize as _worker_finalize
from flash.engine.plan.steps import final_save_due
from flash.engine.worker.io.heartbeat import liveness_heartbeat
from flash.engine.worker.train.entry.backend_common import (
    fused_ce_backend,
    gdn_probe_module,
    probe_verl_capabilities,
    require_gdn_boundary_resets,
    resolve_blackwell_attention_backends,
    resolve_rollout_enforce_eager,
    resolve_verl_python,
    rollout_fp8_kv,
    rollout_sleep_unsupported,
    verl_device_capability,
)
from flash.engine.worker.train.entry.opd_train_runner import (
    _build_base_config,
    _build_train_note_sections,
    _export_and_upload_adapter,
    _materialize_child_files,
    _prepare_request,
    _prepare_workload,
    _report_training_complete,
    _resolve_opd_gpu_mem_util,
    _run_child,
    _validate_aligned_sequences,
    _validate_teacher_transport,
    _with_structured_validation,
)
from flash.engine.worker.train.entry.sft_train import _probe_gpu_in_subprocess
from flash.engine.worker.train.opd.orchestration.failures import _failure_accounting_metadata
from flash.engine.worker.train.opd.orchestration.gkd import (
    generation_eos_from_cached_config,
)
from flash.engine.worker.train.opd.orchestration.prompt_preparation import (
    prepare_prompts,
    render_prompt_rows,
)
from flash.teacher.limits import OPD_TEACHER_SCORING_CONCURRENCY


def _validate_multimodal_opd(request, spec, model_id: str) -> None:
    """Re-check an image-bearing OPD job now that the env class is loaded.

    Runs before the GPU probe and weight download so capability failures cost no paid setup.
    """
    from flash.content.multimodal import validate_multimodal_training

    validate_multimodal_training(
        model_id,
        "opd",
        getattr(spec.train, "teacher_model", None),
    )


def _load_opd_model(model_id: str, model_revision: str, prompt_state) -> tuple[float, list]:
    """Pull the base weights and read back the generation EOS ids, reporting progress throughout.

    Weights come AFTER the prompt-budget filter: a dataset whose every prompt is over budget is a
    deterministic input error, and downloading tens of GB before raising it burns paid worker
    minutes for a verdict the tokenizer already had. The tokenizer/processor/config loads before
    this fetch kilobytes, not weights, so they are cheap to run first.

    `opd_model_load` is the stage the provider classifies as setup and TRAINING.md tells users to
    expect here, but nothing emitted it -- so the documented stage never appeared and this span
    reported as whatever ping came before it. Emit the transition, then hold it open across the
    config read, which is minutes of silence on a cold cache mount.
    """
    download_seconds = _worker_hf.prefetch_model(model_id, revision=model_revision)
    _worker_heartbeat.heartbeat(
        "opd_model_load",
        download_seconds=download_seconds,
        gpu=_worker_perf.gpu_diagnostics(include_torch=False),
    )
    with liveness_heartbeat("opd_model_load"):
        # reads the snapshot with local_files_only, so it has to follow the prefetch.
        eos_token_ids = generation_eos_from_cached_config(
            model_id, model_revision, prompt_state.tokenizer
        )
    return download_seconds, eos_token_ids


def _cuda_supports_fp8_kv() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9))
    except Exception:  # no cuda or probe failure, so use conservative bf16 kv
        return False


def run_opd_train(spec=None) -> None:
    """Run flash OPD through verl's native rollout and weight-sync path."""
    from flash.engine.worker.train.opd.orchestration import validation

    request = _prepare_request(spec := spec or _worker_state.JOB_SPEC)
    knobs, model_id, model_revision = request.knobs, request.model_id, request.model_revision
    structured_validation = validation.validate_opd_structured_outputs(
        knobs.structured_outputs,
        model_id=model_id,
        model_revision=model_revision,
    )
    request = _with_structured_validation(request, structured_validation)
    prompt_rows, multimodal = render_prompt_rows(request)
    if multimodal:
        _validate_multimodal_opd(request, spec, model_id)
    started_at = time.time()
    capability, control_panel_url = _validate_teacher_transport()
    _worker_heartbeat.heartbeat("opd_start", gpu=_worker_perf.gpu_diagnostics(include_torch=False))
    _probe_gpu_in_subprocess(
        spec.gpu.type if spec else None,
        exact_type=spec.gpu.type if spec else "",
    )
    prompt_state = prepare_prompts(request, prompt_rows, multimodal, capability, control_panel_url)
    multi_turn, max_model_len = request.multi_turn, prompt_state.max_model_len
    if not prompt_state.prompts:
        raise RuntimeError("every OPD prompt exceeds the configured prompt budget")
    download_seconds, eos_token_ids = _load_opd_model(model_id, model_revision, prompt_state)
    workload = _prepare_workload(request, prompt_state, multimodal)
    update_horizon, prompts_per_step = workload.update_horizon, workload.prompts_per_step
    # same silent boundary the sft path guards: with no prebuilt worker image this builds a venv and installs the training stack, minutes long with nothing to report and no liveness thread running.
    with liveness_heartbeat("opd_configuring"):
        python_bin = resolve_verl_python(
            workload.workdir, install_wandb=bool(os.environ.get("WANDB_API_KEY"))
        )
        # the architecture question is asked on its OWN, before the batched probe, because it reads the checkpoint config and has nothing to do with the child's capabilities.
        # model_is_gdn_hybrid already returns False on its own probe failure, so it needs no guard. the modeling module is resolved HERE, in the parent, because it needs a hub/cache read the child must not repeat; "" skips the gdn question for a non-hybrid.
        from flash.engine.worker.model.packing import model_is_gdn_hybrid

        gdn_hybrid = model_is_gdn_hybrid(model_id, revision=model_revision)
        gdn_module = gdn_probe_module(model_id, model_revision) if gdn_hybrid else ""
        # ONE child answers every independent capability question. each used to cost its own interpreter, and the torch/verl import -- not the question -- was the price.
        caps = probe_verl_capabilities(python_bin, gdn_module)
    # enable fp8 kv on cc >= 8.9. a gdn hybrid qualifies only when the catalog pins its rollout engine resident: it is sleep/wake that crashes on the hybrid cache, not gdn itself. see rollout_fp8_kv. keep this aligned with vram.py, and keep the device probe separate from gdn classification.
    fp8_kv = rollout_fp8_kv(_cuda_supports_fp8_kv(), gdn_hybrid, model_id)
    # gdn packing requires child support for seq_idx and cu_seqlens; fallbacks discard both and silently bleed state across examples. record whether the child can reset gdn state because successful runs upload no console and failure here is silent contamination; none means non-gdn. see require_gdn_boundary_resets.
    gdn_reset_arch = require_gdn_boundary_resets(caps, gdn_module)
    # run sm86 eagerly because vllm 0.19.1 graph capture degenerates there. sm89 capture is empirically acceptable; enforce_eager overrides async cudagraph settings last at config/vllm.py:1024.
    # reuse one capability probe for both rollout decisions.
    verl_cc = verl_device_capability(caps)
    enforce_eager = resolve_rollout_enforce_eager(verl_cc)
    # pin both rollout attention backends on blackwell: vllm 0.19.1's ViT CUTE default fails with missing cutlass.cute.core.ThrMma, including text-only rollouts on VL models.
    attention_backend, mm_encoder_attn_backend = resolve_blackwell_attention_backends(caps, verl_cc)
    runtime = _materialize_child_files(
        request, prompt_state, workload, python_bin, caps, gdn_reset_arch, eos_token_ids
    )
    try:
        config = _build_base_config(request, prompt_state, workload, runtime, eos_token_ids)
        config.update(
            {
                "fp8_kv": fp8_kv,
                "enforce_eager": enforce_eager,
                "attention_backend": attention_backend,
                "mm_encoder_attn_backend": mm_encoder_attn_backend,
                "sleep_unsupported": rollout_sleep_unsupported(model_id),
                # caps the agent-worker fan-out: each is a ray actor with its own processor copy,
                # and on an image run that fan-out exhausted the grpo worker container's threads.
                "multimodal": bool(multimodal),
                "gpu_mem_util": _resolve_opd_gpu_mem_util(
                    request, prompt_state, workload, runtime, model_id, fp8_kv
                ),
                "loggers": runtime.loggers,
                # resolved from the out-of-process capability probe, never by opening cuda in this parent -- see fused_ce_backend.
                "fused_ce_backend": fused_ce_backend(caps),
            }
        )
        result = _run_child(request, prompt_state, workload, runtime, config, eos_token_ids)
        final_accounting, final_step = result.final_accounting, result.final_step
        if len(final_accounting["loss_curve"]) != final_step:
            # record_step only checks that each metric line FOLLOWS the last one, so a missing trailing metric (on_line skips any step-tagged line whose loss it cannot parse) leaves a curve shorter than the checkpoint verl actually wrote.
            # nothing later arrives to catch it. opt_steps is published from this curve, so a short curve would understate the updates applied. fail loud instead of reporting a number the curve cannot support.
            raise RuntimeError(
                f"verl OPD recorded {len(final_accounting['loss_curve'])} distillation-loss metrics "
                f"for {final_step} optimizer updates; refusing to publish an accounting that does "
                "not cover every update"
            )
        _validate_aligned_sequences(final_accounting)
        with liveness_heartbeat(
            "opd_finalizing", progress=lambda: final_step, progress_step=True, keepalive=True
        ):
            adapter_dir = _export_and_upload_adapter(request, workload, runtime, result)
            # preserve the final checkpoint only when save_at_steps is empty, matching grpo. watcher
            # and final-save paths are disjoint, so the watcher's lifecycle must not suppress it.
            if final_save_due(final_step, knobs.save_at_steps):
                _worker_hf.publish_deployable_checkpoint(
                    adapter_dir, final_step, _provenance_ready=True
                )
        setup_seconds = _report_training_complete(result, started_at)
        initial_notes, accounting_notes, training_notes, backend_notes = _build_train_note_sections(
            request, prompt_state, workload, runtime, result, download_seconds
        )
        _worker_finalize.write_train_meta(
            phase="opd",
            step=final_step,
            adapter_dir=adapter_dir,
            model_id=model_id,
            train_wall=result.train_wall,
            setup_seconds=setup_seconds,
            train_tokens=0,
            generated_tokens=int(final_accounting["generated_tokens"]),
            notes={
                "steps": update_horizon,
                # optimizer updates that actually produced a distillation loss. record_step enforces loss_curve length == the metric step, and the guard above rejects a curve shorter than final_step, so this is measured, not assumed.
                "opt_steps": len(final_accounting["loss_curve"]),
                **initial_notes,
                # the real alignment-health signal. mean_coverage reads ~1.0 even when the alignment has collapsed every student token onto one group, so it cannot flag that failure mode on its own; this ratio can.
                "mean_align_granularity": (
                    float(final_accounting["align_group_sum"])
                    / int(final_accounting["align_group_n"])
                    if final_accounting["align_group_n"]
                    else 0.0
                ),
                **accounting_notes,
                **_failure_accounting_metadata(final_accounting),
                **training_notes,
                # the engine length actually handed to vllm (prompt + completion), already clamped to the model's own limit. the prompt filter is carved out of this same number.
                "vllm_max_model_len": max_model_len,
                # only single-turn text uses the fixed serial batcher; multimodal and multi-turn use bridge threads. cap the reported batch by samples the step can produce.
                "opd_teacher_batch_size": (
                    min(
                        OPD_TEACHER_SCORING_CONCURRENCY, max(1, prompts_per_step * knobs.group_size)
                    )
                    if not multimodal and not multi_turn
                    else None
                ),
                "opd_teacher_workers": 1 if not multimodal and not multi_turn else None,
                **backend_notes[0],
                "gdn_boundary_resets": gdn_hybrid or None,
                **backend_notes[1],
                "wandb_url": result.wandb_url,  # the sdk's link_wandb reads notes["wandb_url"]; trl gets it from the child marker emitted by backend_common.render_wandb_link_shim.
                "wandb_id": result.wandb_id,
            },
        )
    finally:
        runtime.bridge.close()
