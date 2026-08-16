"""Flash OPD orchestration through verl 0.8.0 in an isolated child interpreter."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from flash.engine.plan.recipe import RECIPE  # noqa: F401
from flash.engine.plan.steps import (  # noqa: F401
    final_save_due,
    on_policy_steps,
    resolve_update_horizon,
    validate_save_steps,
)
from flash.engine.profiling.sft_workload import (  # noqa: F401
    _materialize_verl_images,
)
from flash.engine.worker.backend_common import (  # noqa: F401
    ChildOutputTail,
    ChildTailStaleness,
    VerlChildSilenceWatchdog,
    clamp_engine_len,
    fused_ce_backend,
    gdn_probe_module,
    latest_global_step_dir,
    model_max_position_embeddings,
    parse_verl_metric,
    parse_wandb_link,
    probe_verl_capabilities,
    render_sitecustomize_bootstrap,
    require_gdn_boundary_resets,
    resolve_blackwell_attention_backends,
    resolve_rollout_enforce_eager,
    resolve_verl_loggers,
    resolve_verl_python,
    rollout_sleep_unsupported,
    run_verl_training,
    shim_marker_file,
    stall_tail_fields,
    verify_applied_shim_markers,
    verl_device_capability,
    verl_step_number,
)
from flash.engine.worker.entry.opd import (  # noqa: F401
    _resolve_opd_knobs,
    _thinking_prefill_text,
)
from flash.engine.worker.io.heartbeat import liveness_heartbeat
from flash.engine.worker.runtime.pkg_proxy import W as _w
from flash.engine.worker.runtime.rng import seed_training_rngs  # noqa: F401
from flash.engine.worker.sft_train import (  # noqa: F401
    _cached_model_path,
    _export_checkpoint_adapter,
    _NvidiaSmiPeakSampler,
    _probe_gpu_in_subprocess,
    _seed_resume_lifecycle,
    _verl_image_message_content,
    _warmstart_adapter_path,
)
from flash.engine.worker.train.core.child.glue import (  # noqa: F401
    validate_glue_template,
    validate_transcript_messages,
)
from flash.engine.worker.train.opd.gkd import (
    generation_eos_from_cached_config,
)
from flash.teacher.limits import OPD_TEACHER_SCORING_CONCURRENCY

_PERMANENT_TEACHER_EXIT = 86
_TRANSIENT_TEACHER_EXIT = 87
_TEXT_TEACHER_FLUSH_WAIT_S = 0.1
_TEXT_TEACHER_SHUTDOWN_WAIT_S = 5.0

# opd supervises the teacher's distribution, not a task reward, so every rollout scores zero. the
# score is unreachable either way: use_task_rewards=false makes verl zero the whole policy loss
# (distillation/losses.py:211), so nothing a scorer returns can enter the gradient. this exists
# only to keep the reward loop out of its builtin data_source registry -- see the call site.
_OPD_ZERO_REWARD_SOURCE = '''"""flash opd reward shim (generated). opd carries no task reward."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return 0.0
'''


class _RecordedMutationCallbackFailure(RuntimeError):
    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


@dataclass(frozen=True)
class _BridgePrompt:
    student_messages: list[dict]
    teacher_messages: list[dict]
    prompt_ids: tuple[int, ...]
    image_descriptors: tuple[str, ...]
    package_root: str | None
    example: dict | None = None


class _OpdProgressState:
    def __init__(self, resume_state: dict | None = None) -> None:
        state = resume_state or {}
        self._condition = threading.Condition()
        self.loss_curve = [float(value) for value in state.get("loss_curve", [])]
        self.coverage_curve = [float(value) for value in state.get("coverage_curve", [])]
        self.base_train_wall_seconds = float(state.get("train_wall_seconds", 0.0))
        self._prev_aligned = int(state.get("aligned_sequences", 0))
        self._prev_cov_sum = float(state.get("coverage_sum", 0.0))
        self._prev_truncated = int(state.get("truncated_rollouts", 0))
        self._prev_samples_seen = int(state.get("samples_seen", 0))
        self._prev_no_signal_skipped_steps = int(state.get("no_signal_skipped_steps", 0))
        self._train_started_at: float | None = None
        self._step_states: dict[int, dict] = {}
        if resume_state is not None:
            self._step_states[int(state["opt_steps"])] = dict(state)

    def start_training(self) -> None:
        self._train_started_at = time.time()

    def _train_wall_seconds(self) -> float:
        elapsed = 0.0
        if self._train_started_at is not None:
            elapsed = max(0.0, time.time() - self._train_started_at)
        return self.base_train_wall_seconds + elapsed

    def record_step(
        self, step: int, loss: float, bridge: _TeacherAlignmentBridge
    ) -> tuple[float, int]:
        with self._condition:
            expected_step = len(self.loss_curve) + 1
            if step != expected_step:
                raise RuntimeError(
                    f"verl OPD metric step {step} does not follow accumulated step {expected_step - 1}"
                )
            snapshot = bridge.accounting_snapshot()
            self.loss_curve.append(float(loss))
            aligned = int(snapshot["aligned_sequences"])
            cov_sum = float(snapshot["coverage_sum"])
            # per-step coverage: delta over the previous snapshot, so the curve shows each step's
            # own alignment quality instead of a cumulative average that flattens regressions.
            d_aligned = aligned - self._prev_aligned
            d_cov = cov_sum - self._prev_cov_sum
            self._prev_aligned, self._prev_cov_sum = aligned, cov_sum
            coverage = (
                (d_cov / d_aligned) if d_aligned > 0 else (cov_sum / aligned if aligned else 0.0)
            )
            self.coverage_curve.append(coverage)
            truncated = int(snapshot["truncated_rollouts"])
            samples_seen = int(snapshot["samples_seen"])
            # truncation can change abruptly with prompt mix, so report this step's rollout share
            # rather than the cumulative average. a zero-delta snapshot reports 0.0 because no
            # rollouts belong to this step, and reusing history would misattribute old truncations.
            d_truncated = truncated - self._prev_truncated
            d_samples_seen = samples_seen - self._prev_samples_seen
            self._prev_truncated, self._prev_samples_seen = truncated, samples_seen
            self._prev_no_signal_skipped_steps = int(snapshot.get("no_signal_skipped_steps", 0))
            truncation_rate = (
                min(1.0, max(0.0, d_truncated / d_samples_seen)) if d_samples_seen > 0 else 0.0
            )
            # bound by this step's own sample delta for the same reason the rate clamps to 1.0: an
            # in-flight snapshot can read truncated_rollouts and samples_seen from different steps,
            # and an unbounded count would report more discards than the step drew.
            #
            # the clamp bounds the magnitude; it does NOT establish ownership. these are deltas of
            # counters the bridge advances asynchronously, so when stdout parsing lags a wave, a
            # truncation raised by the next step can still be attributed here (and the next step
            # then reads zero). the same asynchrony has always applied to truncation_rate. read
            # both as a rolling indicator of truncation pressure, not an exact per-step ledger.
            discarded_rollouts = max(0, min(d_truncated, d_samples_seen))
            # the per-step values are returned for the heartbeat and deliberately kept OUT of the
            # snapshot: this dict is spread verbatim into the persisted opd_state.json resume
            # contract, whose schema is fail-closed and holds cumulative counters. per-step display
            # values have no meaning on resume and nothing reads them back.
            snapshot.update(
                {
                    "train_wall_seconds": self._train_wall_seconds(),
                    "loss_curve": list(self.loss_curve),
                    "coverage_curve": list(self.coverage_curve),
                }
            )
            self._step_states[step] = snapshot
            self._condition.notify_all()
            return truncation_rate, discarded_rollouts

    def truncation_window(
        self,
        bridge: _TeacherAlignmentBridge,
        max_completion: int,
    ) -> _TruncationWindow:
        snapshot = bridge.accounting_snapshot()
        with self._condition:
            # the fatal no-signal diagnosis must describe the attempts after the last completed step.
            # cumulative rollout history can otherwise blame an old truncation spike for a current
            # empty-alignment failure.
            return _TruncationWindow(
                truncated_rollouts=max(
                    0, int(snapshot["truncated_rollouts"]) - self._prev_truncated
                ),
                samples_seen=max(0, int(snapshot["samples_seen"]) - self._prev_samples_seen),
                no_signal_skipped_steps=max(
                    0,
                    int(snapshot.get("no_signal_skipped_steps", 0))
                    - self._prev_no_signal_skipped_steps,
                ),
                max_completion=max_completion,
            )

    def checkpoint_state(self, step: int, *, timeout_s: float = 300.0) -> dict:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while step not in self._step_states:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"timed out waiting for honest OPD accounting through checkpoint step {step}"
                    )
                self._condition.wait(remaining)
            return dict(self._step_states[step])

    def final_state(self, bridge: _TeacherAlignmentBridge) -> dict:
        snapshot = bridge.accounting_snapshot()
        snapshot.update(
            {
                "train_wall_seconds": self._train_wall_seconds(),
                "loss_curve": list(self.loss_curve),
                "coverage_curve": list(self.coverage_curve),
            }
        )
        return snapshot


def _validate_multimodal_opd(request, spec, model_id: str) -> None:
    """Re-check an image-bearing OPD job now that the env class is loaded.

    The submit-time preflight in multimodal.py reads `multi_turn` out of the dataset params, so a
    multi-turn env that declares itself by CLASS reaches the worker unflagged; `request.multi_turn`
    here comes from that class and is authoritative. Runs before the GPU probe and weight download
    so a rejected job costs no paid setup.
    """
    from flash.content.multimodal import validate_multimodal_training

    validate_multimodal_training(
        model_id,
        "opd",
        getattr(spec.train, "teacher_model", None),
    )
    if request.multi_turn:
        raise ValueError("multi-turn image-bearing opd is not supported")


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
    download_seconds = _w.prefetch_model(model_id, revision=model_revision)
    _w.heartbeat(
        "opd_model_load",
        download_seconds=download_seconds,
        gpu=_w.gpu_diagnostics(include_torch=False),
    )
    with liveness_heartbeat("opd_model_load"):
        # reads the snapshot with local_files_only, so it has to follow the prefetch.
        eos_token_ids = generation_eos_from_cached_config(
            model_id, model_revision, prompt_state.tokenizer
        )
    return download_seconds, eos_token_ids


def run_opd_train(spec=None) -> None:
    """Run flash OPD through verl's native rollout and weight-sync path."""
    from flash.engine.worker.train.opd.validation import validate_opd_structured_outputs

    request = _prepare_request(spec := spec or _w.JOB_SPEC)
    knobs, model_id, model_revision = request.knobs, request.model_id, request.model_revision
    structured_validation = validate_opd_structured_outputs(
        knobs.structured_outputs,
        model_id=model_id,
        model_revision=model_revision,
    )
    request = _with_structured_validation(request, structured_validation)
    prompt_rows, multimodal = _render_prompt_rows(request)
    if multimodal:
        _validate_multimodal_opd(request, spec, model_id)
    started_at = time.time()
    capability, control_panel_url = _validate_teacher_transport()
    _w.heartbeat("opd_start", gpu=_w.gpu_diagnostics(include_torch=False))
    _probe_gpu_in_subprocess(
        spec.gpu.type if spec else None,
        exact_type=spec.gpu.type if spec else "",
    )
    prompt_state = _prepare_prompts(request, prompt_rows, multimodal, capability, control_panel_url)
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
    # enable fp8 kv on cc >= 8.9 only for non-gdn models; gdn sleep/wake crashes on hybrid caches. keep this aligned with vram.py, and keep the device probe separate from gdn classification.
    try:
        import torch as _torch_cc

        _cc_ok = bool(
            _torch_cc.cuda.is_available() and _torch_cc.cuda.get_device_capability() >= (8, 9)
        )
    except Exception:  # no cuda / probe failure -> conservative bf16 kv
        _cc_ok = False
    fp8_kv = _cc_ok and not gdn_hybrid
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
                _w.publish_deployable_checkpoint(adapter_dir, final_step, _provenance_ready=True)
        setup_seconds = _report_training_complete(result, started_at)
        initial_notes, accounting_notes, training_notes, backend_notes = _build_train_note_sections(
            request, prompt_state, workload, runtime, result, download_seconds
        )
        _w.write_train_meta(
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


# opd run preparation and child execution, implemented in the sibling runner module.
from flash.engine.worker.opd_train_runner import (  # noqa: E402
    _build_base_config,
    _build_train_note_sections,
    _export_and_upload_adapter,
    _materialize_child_files,
    _prepare_prompts,
    _prepare_request,
    _prepare_workload,
    _render_prompt_rows,
    _report_training_complete,
    _run_child,
    _validate_aligned_sequences,
    _validate_teacher_transport,
    _with_structured_validation,
)

# the teacher-alignment bridge, implemented in `.train.opd.bridge`. imported at the BOTTOM because
# that module reaches back here for the names the opd tests patch on this module, so a top-level
# import would be circular. re-exported rather than referenced through the submodule so
# `_TeacherAlignmentBridge` stays importable from `opd_train`, which is where the tests construct it.
# text-teacher batching and response validation, implemented in `.train.opd.batching`. imported at
# the BOTTOM because that module reads this one's flush/shutdown budgets, so a top-level import
# would be circular. re-exported because the opd tests import these names from `opd_train`.
from flash.engine.worker.train.opd.batching import (  # noqa: E402,F401
    _TEXT_TEACHER_REQUEST_BACKLOG,
    _align_granularity,
    _permanent_teacher_error,
    _teacher_batch_error,
    _TeacherBridgeHTTPServer,
    _TextTeacherBatcher,
    _validate_text_teacher_batch,
)
from flash.engine.worker.train.opd.bridge import _TeacherAlignmentBridge  # noqa: E402

# failure accounting and resume staging, implemented in `.train.opd.failures`. imported at the
# BOTTOM because that module reads this one's teacher exit codes, so a top-level import would be
# circular. re-exported because the opd tests import these names from `opd_train`.
from flash.engine.worker.train.opd.failures import (  # noqa: E402,F401
    _canonical_skip_reasons,
    _failure_accounting_metadata,
    _find_checkpoint_file,
    _OpdVerlCheckpointWatcher,
    _raise_verl_failure,
    _read_classified_failure_fallback,
    _read_failure_fallback_records,
    _reconcile_no_signal_notification_failure,
    _reconcile_score_delivery_failure,
    _restore_verl_resume,
    _stage_retry_contract,
    _TruncationWindow,
)

# hydra overrides, child env, and parquet writing, implemented in `.train.opd.overrides`.
# re-exported because the opd tests import these names from `opd_train`.
from flash.engine.worker.train.opd.overrides import (  # noqa: E402,F401
    _OPD_PARQUET_WRITE_BATCH_ROWS,
    _build_opd_child_env,
    _build_opd_plugin_config,
    _opd_multimodal_parquet_features,
    _write_opd_parquet,
    build_opd_overrides,
)

# prompt fingerprinting and token-mask helpers, implemented in `.train.opd.prompts`.
# re-exported because `run_opd_train` above and the opd tests both reach them here.
from flash.engine.worker.train.opd.prompts import (  # noqa: E402,F401
    _normalize_prompt_ids,
    _processor_expanded_prompt_ids,
    _prompt_pool_fingerprint,
    _trim_response_and_forced,
    _validate_forced_mask,
    encode_shifted_group_metadata,
)
