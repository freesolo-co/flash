from __future__ import annotations

import math
import os
import shutil
import time
from functools import partial
from typing import Any

import flash.engine.worker.io.heartbeat as _worker_heartbeat
import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.io.wandb_log as _worker_wandb
import flash.engine.worker.model.adapter as _worker_adapter
import flash.engine.worker.perf as _worker_perf
import flash.engine.worker.runtime.rng as _worker_rng
import flash.engine.worker.train.entry.backend_common as _backend
import flash.engine.worker.train.entry.sft_train as _sft
import flash.engine.worker.train.opd.bridging.bridge as _opd_bridge
import flash.engine.worker.train.opd.bridging.prompts as _opd_prompts
import flash.engine.worker.train.opd.child.bridge as _opd_child
import flash.engine.worker.train.opd.orchestration.failures as _opd_failures
import flash.engine.worker.train.opd.orchestration.overrides as _opd_overrides
import flash.engine.worker.train.opd.orchestration.progress as _opd_progress
import flash.engine.worker.train.opd.orchestration.protocol as _opd_protocol
from flash.adapters.targets import resolve_lora_targeting
from flash.core.catalog import get_model
from flash.engine.plan import steps as _steps
from flash.engine.plan.recipe import RECIPE
from flash.engine.plan.steps import rl_data_parallel_cards
from flash.engine.profiling.sft_workload import _materialize_verl_images
from flash.engine.worker.entry import opd as _opd_entry
from flash.engine.worker.runtime.state import WorkerContext
from flash.engine.worker.train.entry.prompt_rows import canonical_prompt_messages
from flash.engine.worker.train.opd.orchestration.reporting import (
    _build_train_note_sections as _build_train_note_sections,
)
from flash.engine.worker.train.opd.orchestration.state import (
    _ChildCallbacks,
    _ChildResult,
    _OpdRequest,
    _PromptState,
    _RuntimeState,
    _WorkloadState,
)
from flash.engine.worker.verl.parallelism import (
    ULYSSES_SEQUENCE_PARALLEL_SIZE,
    resolve_reshard_after_forward,
)


def _prepare_request(spec: Any, *, context: WorkerContext) -> _OpdRequest:
    env = context.require_active_env()
    if getattr(env, "is_tool_env", False):
        raise RuntimeError("native tool-calling OPD environments are not supported")
    multi_turn = bool(getattr(env, "multi_turn", False))
    if multi_turn:
        required_methods = (
            "new_rollout_state",
            "record_model_turn",
            "env_reply",
            "rollout_done",
        )
        missing = [name for name in required_methods if not callable(getattr(env, name, None))]
        if missing:
            raise RuntimeError(
                f"multi-turn OPD environment is missing required rollout methods: {missing}"
            )
        max_turns = int(getattr(env, "max_turns", 0) or 0)
        if max_turns <= 0:
            raise RuntimeError("multi-turn OPD environment requires a positive bounded turn limit")
    else:
        max_turns = 0
    knobs = _opd_entry._resolve_opd_knobs(context=context)
    if multi_turn and knobs.structured_outputs:
        raise RuntimeError(
            "multi-turn structured-output OPD is not supported until a per-turn constraint contract exists"
        )
    model_id = spec.model if spec else RECIPE.hf_model_id
    model_revision = getattr(spec, "model_revision", "") if spec else ""
    return _OpdRequest(spec, env, multi_turn, max_turns, knobs, model_id, model_revision)


def _with_structured_validation(request: _OpdRequest, validation: Any) -> _OpdRequest:
    return _OpdRequest(
        request.spec,
        request.env,
        request.multi_turn,
        request.max_turns,
        request.knobs,
        request.model_id,
        request.model_revision,
        validation.constraint,
        validation.model_vocab_size,
    )


def _validate_teacher_transport() -> tuple[str, str]:
    # validate the control-panel broker transport before the gpu probe and model prefetch so a malformed
    # attempt fails before any additional paid setup. raw managed-teacher provider credentials never
    # enter the worker.
    from flash.core.spec import PUBLIC_URL_ENV, TEACHER_CAPABILITY_ENV

    control_panel_url = os.environ.get(PUBLIC_URL_ENV, "").strip()
    capability = os.environ.get(TEACHER_CAPABILITY_ENV, "").strip()
    if not control_panel_url or not capability:
        raise RuntimeError(
            "managed teacher control-panel transport is missing from the OPD parent worker"
        )
    return capability, control_panel_url


def _reset_workdir(workdir: str) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
    if os.path.lexists(workdir):
        raise RuntimeError(f"could not clear stale OPD attempt workdir {workdir!r}")
    os.makedirs(workdir)


def _prepare_workload(
    request: _OpdRequest, prompt_state: _PromptState, multimodal: bool, *, context: WorkerContext
) -> _WorkloadState:
    prompts = prompt_state.prompts
    knobs = request.knobs
    prompts_per_step = min(knobs.prompts_per_step, len(prompts))
    derived_steps = _steps.on_policy_steps(
        epochs=knobs.epochs,
        prompt_count=len(prompts),
        prompts_per_step=prompts_per_step,
    )
    update_horizon = _steps.resolve_update_horizon(derived_steps, knobs.max_steps)
    _steps.validate_save_steps(knobs.save_at_steps, update_horizon)
    prompt_pool_fingerprint = _opd_prompts._prompt_pool_fingerprint(prompts)
    workdir = os.path.join(
        "/tmp", "flash-opd-verl", context.config.run_id, f"seed-{context.config.seed}"
    )
    _reset_workdir(workdir)
    data_dir = os.path.join(workdir, "data")
    image_dir = os.path.join(workdir, "images")
    runtime_dir = os.path.join(workdir, "runtime")
    local_dir = os.path.join(workdir, "checkpoints")
    export_root = os.path.join(workdir, "checkpoint-adapters")
    mutation_failure_path = os.path.join(workdir, "mutation-failure")
    score_delivery_failure_path = os.path.join(workdir, "score-delivery-failure")
    rollout_failure_path = os.path.join(workdir, "rollout-failure")
    abandonment_failure_path = os.path.join(workdir, "abandonment-failure")
    resample_failure_path = os.path.join(workdir, "resample-failure")
    cycle_commit_failure_path = os.path.join(workdir, "cycle-commit-failure")
    for path in (data_dir, runtime_dir, local_dir, export_root):
        os.makedirs(path, exist_ok=True)
    materialized_images: dict[int, list[dict[str, str]]] = {}
    if multimodal:
        for index, prompt in enumerate(prompts):
            uris = _materialize_verl_images(
                list(prompt.image_descriptors),
                prompt.package_root,
                image_dir,
                index,
            )
            materialized_images[index] = [{"image": uri} for uri in uris]
    rows = []
    for ordinal in range(update_horizon * prompts_per_step):
        index = ordinal % len(prompts)
        prompt = prompts[index]
        row = {
            "prompt": canonical_prompt_messages(
                prompt.student_messages,
                multimodal=multimodal,
            ),
            "data_source": "flash_opd",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"index": index},
        }
        if multimodal:
            row["images"] = materialized_images[index]
        rows.append(row)
    train_file = os.path.join(data_dir, "train.parquet")
    val_file = os.path.join(data_dir, "val.parquet")
    _opd_overrides._write_opd_parquet(rows, train_file)
    _opd_overrides._write_opd_parquet([rows[0]], val_file)
    lora_config = _worker_adapter.make_lora(request.model_id, context=context)
    targeting = resolve_lora_targeting(request.model_id, algorithm="opd", multimodal=multimodal)
    target_modules = targeting.target_modules
    if isinstance(target_modules, set | frozenset):
        target_modules = sorted(target_modules)
    target_parameters = tuple(targeting.target_parameters) if targeting.target_parameters else None
    lora_rank = int(lora_config.r)
    return _WorkloadState(
        prompts_per_step,
        update_horizon,
        prompt_pool_fingerprint,
        workdir,
        runtime_dir,
        local_dir,
        export_root,
        mutation_failure_path,
        score_delivery_failure_path,
        rollout_failure_path,
        abandonment_failure_path,
        resample_failure_path,
        cycle_commit_failure_path,
        train_file,
        val_file,
        lora_rank,
        int(lora_config.lora_alpha),
        target_modules,
        target_parameters,
        targeting.exclude_modules,
        _sft._warmstart_adapter_path(
            request.model_id,
            request.model_revision,
            lora_rank,
            int(lora_config.lora_alpha),
            targeting,
            context=context,
        ),
        multimodal=multimodal,
    )


def _materialize_child_files(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    python_bin: str,
    caps: dict[str, Any],
    gdn_reset_arch: str | None,
    eos_token_ids: tuple[int, ...],
    *,
    context: WorkerContext,
) -> _RuntimeState:
    knobs = request.knobs
    model_path = _sft._cached_model_path(request.model_id, request.model_revision)
    # the ranks verl will RUN, not the cards rented: with ulysses pinned off every rank is a dp rank,
    # and verl chunks the step's sequences across them with an exact-divisibility assert. bound here
    # at the single source so the launch width, the resume world_size and the run metadata cannot
    # disagree about how wide the attempt actually was.
    gpu_count = rl_data_parallel_cards(
        int(getattr(request.spec.gpu, "count", 1) or 1),
        workload.prompts_per_step * knobs.group_size,
    )
    default_save_freq = max(1, min(knobs.save_every, workload.update_horizon))
    save_freq = math.gcd(*knobs.save_at_steps) if knobs.save_at_steps else default_save_freq
    # verl logs from the verl interpreter, so gate wandb on THAT env (see resolve_verl_loggers).
    loggers = _backend.resolve_verl_loggers(caps)
    wandb = request.spec.wandb if request.spec else None
    project_name = wandb.project if wandb and wandb.project else "flash"
    experiment_name = _worker_wandb.wandb_run_name(context=context)
    entry_path, reward_path = _child_module_paths()
    resume_step, resume_state = _opd_failures._restore_verl_resume(
        workload.local_dir,
        prompt_pool_fingerprint=workload.prompt_pool_fingerprint,
        update_horizon=workload.update_horizon,
        # the same count this attempt hands verl as n_gpus_per_node, which is the data-parallel
        # width: ulysses is pinned to 1, so every rank is a dp rank.
        world_size=gpu_count,
        context=context,
    )
    bridge = _opd_bridge._TeacherAlignmentBridge(
        prompts=prompt_state.prompts,
        processor=prompt_state.processor,
        tokenizer=prompt_state.tokenizer,
        teacher=prompt_state.teacher,
        thinking_prefill=prompt_state.thinking_prefill,
        eos_token_ids=eos_token_ids,
        stop_sequences=tuple(str(value) for value in knobs.stop_sequences),
        structured=request.structured_outputs is not None,
        active_env=request.env if request.multi_turn else None,
        multi_turn=request.multi_turn,
        max_turns=request.max_turns,
        thinking=bool(context.config.thinking),
        mutation_callback=partial(_worker_hf.publish_opd_optimizer_start_marker, context=context),
        initial_state=resume_state,
    )
    bridge.start()
    return _RuntimeState(
        python_bin,
        model_path,
        gpu_count,
        save_freq,
        loggers,
        project_name,
        experiment_name,
        gdn_reset_arch,
        entry_path,
        reward_path,
        resume_step,
        resume_state,
        bridge,
    )


def _child_module_paths() -> tuple[str, str]:
    child_dir = os.path.join(os.path.dirname(__file__), "../opd/child")
    return "flash.engine.worker.train.opd.child.entry", os.path.join(child_dir, "reward.py")


def _resolve_opd_gpu_mem_util(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    model_id: str,
    fp8_kv: bool,
) -> float:
    """Size vLLM's colocated executor budget from this run's geometry, as the GRPO path does.

    Left unset, verl substitutes its own default of 0.5 and the engine claims half the CARD on
    every wake regardless of what the trainer already holds. That is not a spare-capacity request:
    `wake_up` re-acquires the physical pages it released to sleep, so an overcommit is a hard
    `CUDA Error: out of memory` in cumem_allocator rather than a smaller pool. Observed on a 27B
    image OPD run on one H200 -- the trainer reserved 83.35 GB, the default handed vLLM 70.5 GB of
    a 141 GB card, and the weight-sync wake after step 2 died 12.85 GB short.

    Shares GRPO's resolver rather than restating it: both run the same colocated sleep/wake path,
    so a second equation here could only drift from the one preflight admits against.
    """
    from flash.engine.worker.train.entry.backend_common import rollout_sleep_unsupported
    from flash.engine.worker.train.rl.launch.verl_config import resolve_gpu_mem_util

    return resolve_gpu_mem_util(
        {
            "model_id": model_id,
            "model_revision": str(getattr(request, "model_revision", "") or ""),
            "engine_len": int(prompt_state.max_model_len),
            "lora_rank": int(workload.lora_rank),
            # opd submits prompts_per_step * group_size generations to the rollout engine together.
            # grpo's concurrency is group_size because its resolver receives one prompt group here.
            "group_size": int(workload.prompts_per_step) * int(request.knobs.group_size),
        },
        gpu_type=_opd_protocol.spec_gpu_type(getattr(request, "spec", None)),
        n_gpus=int(runtime.gpu_count),
        fp8_kv=bool(fp8_kv),
        sleep_unsupported=rollout_sleep_unsupported(model_id),
        preserve_legacy_floor=True,
    )


def _build_base_config(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    eos_token_ids: tuple[int, ...],
    *,
    context: WorkerContext,
) -> dict[str, Any]:
    knobs = request.knobs
    bridge = runtime.bridge
    return {
        "train_files": [workload.train_file],
        "val_files": [workload.val_file],
        "train_batch_size": workload.prompts_per_step,
        "max_prompt_length": prompt_state.prompt_budget,
        "max_response_length": knobs.max_completion,
        "model_path": runtime.model_path,
        "lora_rank": workload.lora_rank,
        "lora_alpha": workload.lora_alpha,
        "target_modules": workload.target_modules,
        "exclude_modules": None,
        "target_parameters": workload.target_parameters,
        "lora_adapter_path": workload.warmstart_adapter,
        "learning_rate": knobs.learning_rate,
        "local_dir": workload.local_dir,
        "save_freq": runtime.save_freq,
        "n_gpus_per_node": runtime.gpu_count,
        # opd shards by DATA -- see ULYSSES_SEQUENCE_PARALLEL_SIZE for why.
        "ulysses_sequence_parallel_size": ULYSSES_SEQUENCE_PARALLEL_SIZE,
        # zero-2 vs zero-3, decided by the allocator's own fit model so the worker cannot spend
        # memory the shape was not admitted with. the spec carries the SELECTED class and count
        # (`_spec_with_gpu`), so this asks about the hardware the run actually landed on.
        # read it off `request.spec` like every other spec lookup here: the caller may pass a spec
        # that is not the process-global job spec (`opd_train.py`: `spec or _worker_state.JOB_SPEC`), and
        # sizing the gate off different hardware than the run uses is the exact allocator/worker
        # divergence this gate exists to prevent.
        "reshard_after_forward": resolve_reshard_after_forward(
            model_id=request.model_id,
            algorithm="opd",
            gpu_type=_opd_protocol.spec_gpu_type(getattr(request, "spec", None)),
            n_gpus=int(runtime.gpu_count),
            train=getattr(getattr(request, "spec", None), "train", None),
            thinking=bool(context.config.thinking),
            model_revision=str(getattr(request, "model_revision", "") or ""),
        ),
        "seed": _worker_rng.backend_seed(context.config.seed),
        "project_name": runtime.project_name,
        "experiment_name": runtime.experiment_name,
        "total_training_steps": workload.update_horizon,
        "group_size": knobs.group_size,
        "bridge_url": bridge.url,
        "bridge_token": bridge.token,
        "reward_path": runtime.reward_path,
        "kl_penalty_coef": knobs.kl_coef,
        "temperature": knobs.temperature,
        "top_p": knobs.top_p,
        # the job's own engine length (prompt + completion), already clamped to the model's
        # limit. prompt_budget above is carved out of this same value, so the engine, the prompt
        # filter, and the token budget cannot disagree. a hardcoded engine would size vllm's kv
        # cache for a context the job never uses, and -- above it -- admit prompts the engine
        # cannot hold.
        "max_sequence_length": prompt_state.max_model_len,
        "multi_turn": request.multi_turn,
        "thinking": bool(context.config.thinking),
        "structured_outputs": request.structured_outputs,
    }


def _build_child_callbacks(
    watcher: Any,
    progress_state: Any,
    bridge: Any,
    resume_step: int,
    readiness_path: str,
    expected_components: tuple[str, ...],
    *,
    context: WorkerContext,
) -> _ChildCallbacks:
    progress = {
        "step": resume_step,
        "loss": None,
        "truncation_rate": None,
        "discarded_rollouts": None,
        "truncation_step": None,
    }
    wandb_link: dict[str, str | None] = {}
    runtime_verified = False

    def on_line(line: str) -> None:
        nonlocal runtime_verified
        context.phase_timings.observe_line(line)
        watcher.raise_if_failed()
        link = _backend.parse_wandb_link(line)
        if link is not None:
            wandb_link.update(link)
        step_number = _backend.verl_step_number(line)
        if step_number is None:
            return
        # require successful trainer initialization before accepting progress.
        if not runtime_verified:
            _backend.verify_runtime_ready(readiness_path, expected_components)
            runtime_verified = True
        # use parse_verl_metric because numpy 2 pprint emits np.float64(...); float() would drop
        # every step and leave a trained run with an empty loss curve.
        loss = _backend.parse_verl_metric(line, "actor/distillation/loss")
        if loss is None:
            loss = _backend.parse_verl_metric(line, "distillation/loss")
        if loss is None:
            # verl emits step-tagged lines that are not metric summaries (timers, val lines);
            # skip those rather than killing the run. the end-of-run guard still fails loud
            # when NO step ever produced a distillation loss.
            return
        progress["loss"] = loss
        (
            progress["truncation_rate"],
            progress["discarded_rollouts"],
        ) = progress_state.record_step(step_number, loss, bridge)
        progress["truncation_step"] = step_number

    def on_step(step: int) -> None:
        progress["step"] = step
        payload = {"step": step}
        if progress["loss"] is not None:
            payload["loss"] = progress["loss"]
        if progress["truncation_step"] == step and progress["truncation_rate"] is not None:
            payload["truncation_rate"] = progress["truncation_rate"]
            payload["discarded_rollouts"] = progress["discarded_rollouts"]
        _worker_heartbeat.heartbeat("opd_step", **payload, context=context)

    def child_heartbeat() -> None:
        _worker_heartbeat.heartbeat(
            "opd_step", liveness=True, step=int(progress["step"] or 0), context=context
        )

    child_tail = _backend.ChildOutputTail()
    # one instance for the whole run: it measures silence across ticks, so it cannot live inside
    # the per-tick callback.
    tail_staleness = _backend.ChildTailStaleness()
    silence_watchdog = _backend.VerlChildSilenceWatchdog(
        child_tail,
        baseline_step=resume_step,
        parent_work=bridge.parent_work,
    )

    def liveness_fields() -> dict[str, object]:
        return _backend.stall_tail_fields(
            int(progress["step"] or 0), child_tail, staleness=tail_staleness
        )

    return _ChildCallbacks(
        on_line,
        on_step,
        child_heartbeat,
        liveness_fields,
        progress,
        wandb_link,
        child_tail,
        silence_watchdog,
    )


def _run_child(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    config: dict[str, Any],
    eos_token_ids: tuple[int, ...],
    *,
    context: WorkerContext,
) -> _ChildResult:
    overrides = _opd_overrides.build_opd_overrides(config)
    progress_state = _opd_progress._OpdProgressState(runtime.resume_state)
    watcher = _build_checkpoint_watcher(request, workload, runtime, progress_state, context=context)
    readiness_path = _backend.readiness_file(workload.runtime_dir)
    expected_components = ("runtime-ready",)
    callbacks = _build_child_callbacks(
        watcher,
        progress_state,
        runtime.bridge,
        runtime.resume_step,
        readiness_path,
        expected_components,
        context=context,
    )
    child_env = _build_child_env(
        request, prompt_state, workload, runtime, eos_token_ids, context=context
    )
    command = [
        runtime.python_bin,
        "-m",
        runtime.entry_path,
        "--flash-config",
        os.path.join(workload.runtime_dir, "launch.json"),
        *overrides,
    ]
    gpu_sampler = _sft._NvidiaSmiPeakSampler().start()
    train_started_at = time.time()
    return_code = 0
    training_completed = runtime.resume_step >= workload.update_horizon
    watcher_started = False
    try:
        watcher.start()
        watcher_started = True
        if runtime.resume_step < workload.update_horizon:
            progress_state.start_training()
            with _worker_heartbeat.liveness_heartbeat(
                "opd_step",
                progress=lambda: int(callbacks.progress["step"] or 0),
                progress_step=True,
                fields=callbacks.liveness_fields,
                sample_off_thread=True,
                context=context,
            ):
                return_code = _backend.run_verl_training(
                    command,
                    env=child_env,
                    on_step=callbacks.on_step,
                    on_line=callbacks.on_line,
                    heartbeat=callbacks.child_heartbeat,
                    tail=callbacks.child_tail,
                    silence_watchdog=callbacks.silence_watchdog,
                )
                training_completed = return_code == 0
    finally:
        # the watcher stamps checkpoints, and stamping BLOCKS on this run's accounting. a child
        # that died before printing its last step will never record it, so tell the gate the run
        # is over first -- otherwise `watcher.stop` waits out the accounting timeout and reports a
        # bookkeeping stall in place of the exit that actually ended the run.
        try:
            if not training_completed:
                progress_state.fail(
                    f"verl child exited with code {return_code}"
                    if return_code
                    else "verl child ended without completing training"
                )
            if watcher_started:
                watcher.stop(require_complete=training_completed)
        finally:
            # the sampler polls nvidia-smi on a thread of its own. stop it even when either the
            # child callback or watcher cleanup raises, because this worker outlives the run.
            peak_gpu_gb = gpu_sampler.stop_gb()
    truncation_window = None
    if return_code != 0:
        truncation_window = progress_state.truncation_window(
            runtime.bridge,
            request.knobs.max_completion,
        )
    _reconcile_child_failures(
        workload,
        runtime.bridge,
        return_code,
        truncation_window=truncation_window,
    )
    final_accounting = progress_state.final_state(runtime.bridge)
    actor_dir, final_step = _backend.latest_global_step_dir(workload.local_dir)
    result = _ChildResult(
        final_accounting,
        actor_dir,
        final_step,
        float(final_accounting["train_wall_seconds"]),
        peak_gpu_gb,
        train_started_at,
        callbacks.wandb_link.get("wandb_url"),
        callbacks.wandb_link.get("wandb_id"),
    )
    _validate_checkpoint_progress(result, workload.update_horizon)
    return result


def _build_checkpoint_watcher(
    request: _OpdRequest,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    progress_state: Any,
    *,
    context: WorkerContext,
) -> Any:
    watcher = _opd_failures._OpdVerlCheckpointWatcher(
        local_dir=workload.local_dir,
        export_root=workload.export_root,
        python_bin=runtime.python_bin,
        model_id=request.model_id,
        model_revision=request.model_revision,
        required_steps=request.knobs.save_at_steps,
        final_step=workload.update_horizon,
        exclude_modules=workload.exclude_modules,
        preprocessor=runtime.bridge.processor,
        seed=int(context.config.seed),
        prompt_pool_fingerprint=workload.prompt_pool_fingerprint,
        prompts_per_step=workload.prompts_per_step,
        group_size=request.knobs.group_size,
        accounting_state=progress_state.checkpoint_state,
        context=context,
    )
    _sft._seed_resume_lifecycle(
        watcher, request.knobs.save_at_steps, runtime.resume_step, context=context
    )
    return watcher


def _build_child_env(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    eos_token_ids: tuple[int, ...],
    *,
    context: WorkerContext,
) -> dict[str, str]:
    bridge = runtime.bridge
    return _opd_overrides._build_opd_child_env(
        runtime_dir=workload.runtime_dir,
        wandb_enabled="wandb" in runtime.loggers,
        bridge_url=bridge.url,
        bridge_token=bridge.token,
        seed=int(context.config.seed),
        stop_sequences=request.knobs.stop_sequences,
        eos_token_ids=eos_token_ids,
        structured_outputs=request.structured_outputs,
        model_vocab_size=request.model_vocab_size,
        thinking=bool(context.config.thinking),
        multi_turn=request.multi_turn,
        max_turns=request.max_turns,
        max_model_len=prompt_state.max_model_len,
        mutation_failure_path=workload.mutation_failure_path,
        score_delivery_failure_path=workload.score_delivery_failure_path,
        rollout_failure_path=workload.rollout_failure_path,
        abandonment_failure_path=workload.abandonment_failure_path,
        resample_failure_path=workload.resample_failure_path,
        cycle_commit_failure_path=workload.cycle_commit_failure_path,
        launch_config=_opd_overrides._build_opd_launch_config(
            runtime_dir=workload.runtime_dir,
            save_at_steps=request.knobs.save_at_steps,
            total_steps=workload.update_horizon,
            lora_language_prefix=(
                get_model(request.model_id).lora_language_prefix if workload.exclude_modules else ""
            ),
            structured_outputs=request.structured_outputs,
            gdn_model_type=runtime.gdn_reset_arch,
            reentrant_checkpointing=_worker_perf.grpo_use_reentrant(request.model_id),
            multimodal=workload.multimodal,
            loggers=runtime.loggers,
        ),
        context=context,
    )


def _reconcile_child_failures(
    workload: _WorkloadState,
    bridge: Any,
    return_code: int,
    *,
    truncation_window: _opd_failures._TruncationWindow | None,
) -> None:
    score_delivery_failure = _opd_failures._reconcile_score_delivery_failure(
        bridge,
        _opd_failures._read_classified_failure_fallback(workload.score_delivery_failure_path),
    )
    no_signal_failure = _opd_failures._reconcile_no_signal_notification_failure(
        bridge,
        (
            _opd_failures._read_classified_failure_fallback(workload.resample_failure_path),
            _opd_failures._read_classified_failure_fallback(workload.abandonment_failure_path),
        ),
    )
    fallback_mutation_failure = _opd_failures._read_classified_failure_fallback(
        workload.mutation_failure_path
    )
    if fallback_mutation_failure is not None:
        bridge._record_mutation_failure(*fallback_mutation_failure)
    cycle_commit_failure = _opd_failures._read_classified_failure_fallback(
        workload.cycle_commit_failure_path
    )
    rollout_failure = _opd_child._read_rollout_failure_fallback(workload.rollout_failure_path)
    _opd_failures._raise_verl_failure(
        return_code,
        bridge.teacher_failure,
        bridge.mutation_failure,
        cycle_commit_failure,
        no_signal_failure,
        score_delivery_failure,
        rollout_failure=rollout_failure,
        truncation_window=truncation_window,
    )


def _validate_checkpoint_progress(result: _ChildResult, update_horizon: int) -> None:
    if result.final_step < update_horizon:
        raise RuntimeError(
            f"opd completed {result.final_step}/{update_horizon} requested optimizer updates"
        )
    if not result.final_accounting["loss_curve"]:
        raise RuntimeError(
            "verl OPD produced no distillation-loss metrics for the whole run — the "
            "distillation path never engaged; refusing to publish"
        )


def _validate_aligned_sequences(final_accounting: dict[str, Any]) -> None:
    if int(final_accounting.get("aligned_sequences", 0) or 0) <= 0:
        # zeroed-mask pass-through batches still emit a (zero) loss metric, so the loss-curve
        # check alone cannot distinguish real distillation from a run where the teacher never
        # aligned once. require at least one aligned sequence before publishing.
        raise RuntimeError(
            "verl OPD saw zero aligned teacher sequences for the whole run — every batch was "
            "no-signal; refusing to publish an unchanged adapter"
        )


def _report_training_complete(
    result: _ChildResult, started_at: float, *, context: WorkerContext
) -> float:
    _worker_heartbeat.heartbeat(
        "opd_trained",
        step=result.final_step,
        train_wall=result.train_wall,
        gpu=_worker_perf.gpu_diagnostics(),
        context=context,
    )
    return result.train_started_at - started_at


def _export_and_upload_adapter(
    request: _OpdRequest,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    result: _ChildResult,
    *,
    context: WorkerContext,
) -> str:
    adapter_dir = os.path.join(workload.workdir, "adapter")
    _sft._export_checkpoint_adapter(
        result.actor_dir,
        adapter_dir,
        model_id=request.model_id,
        model_revision=request.model_revision,
        exclude_modules=workload.exclude_modules,
        python_bin=runtime.python_bin,
        preprocessor=runtime.bridge.processor,
    )
    _worker_hf.hf_upload_folder(adapter_dir, "adapter", required=True, context=context)
    return adapter_dir
