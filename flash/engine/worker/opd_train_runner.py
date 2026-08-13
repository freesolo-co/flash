"""OPD training preparation, child execution, and finalization helpers.

Split out of ``flash.engine.worker.opd_train`` to keep that module under the file-size limit.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash.engine.plan.steps import rl_data_parallel_cards
from flash.engine.worker import opd_train as _opd_train
from flash.engine.worker.verl.parallelism import ULYSSES_SEQUENCE_PARALLEL_SIZE


@dataclass(frozen=True)
class _OpdRequest:
    spec: Any
    env: Any
    multi_turn: bool
    max_turns: int
    knobs: Any
    model_id: str
    model_revision: str
    structured_outputs: Any = None
    model_vocab_size: int | None = None


@dataclass(frozen=True)
class _PromptState:
    teacher: Any
    tokenizer: Any
    thinking_prefill: str
    max_model_len: int
    prompt_budget: int
    prompts: list[Any]
    dropped_long: int


@dataclass(frozen=True)
class _WorkloadState:
    prompts_per_step: int
    update_horizon: int
    prompt_pool_fingerprint: str
    workdir: str
    shim_dir: str
    local_dir: str
    export_root: str
    mutation_failure_path: str
    score_delivery_failure_path: str
    abandonment_failure_path: str
    resample_failure_path: str
    cycle_commit_failure_path: str
    teacher_worker_failure_path: str
    train_file: str
    val_file: str
    lora_rank: int
    lora_alpha: int
    target_modules: Any
    warmstart_adapter: str | None


@dataclass(frozen=True)
class _RuntimeState:
    python_bin: str
    model_path: str
    gpu_count: int
    save_freq: int
    loggers: list[str]
    project_name: str
    experiment_name: str
    gdn_reset_arch: str | None
    entry_path: str
    reward_path: str
    resume_step: int
    resume_state: dict[str, Any] | None
    bridge: Any


@dataclass(frozen=True)
class _ChildCallbacks:
    on_line: Any
    on_step: Any
    child_heartbeat: Any
    liveness_fields: Any
    progress: dict[str, Any]
    wandb_link: dict[str, str | None]
    child_tail: Any


@dataclass(frozen=True)
class _ChildResult:
    final_accounting: dict[str, Any]
    actor_dir: str
    final_step: int
    train_wall: float
    peak_gpu_gb: float
    train_started_at: float
    wandb_url: str | None
    wandb_id: str | None


def _prepare_request(spec: Any) -> _OpdRequest:
    env = _opd_train._w.require_active_env()
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
    knobs = _opd_train._resolve_opd_knobs()
    if multi_turn and knobs.structured_outputs:
        raise RuntimeError(
            "multi-turn structured-output OPD is not supported until a per-turn constraint contract exists"
        )
    model_id = spec.model if spec else _opd_train.RECIPE.hf_model_id
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


def _render_prompt_rows(request: _OpdRequest) -> tuple[list[tuple[Any, Any]], bool]:
    from flash.content.multimodal import record_has_images

    # the child trainer is seeded through its own config, but the environment's dataset /
    # prompt_messages calls run HERE in the parent. an unseeded parent can build a different prompt
    # pool across attempts, whose fingerprint then rejects a valid resume checkpoint. seed just
    # before the first env call that can consume randomness, so the cheap fail-closed guards above
    # still raise without paying for the torch import.
    _opd_train.seed_training_rngs(_opd_train._w.SEED)
    train = list(request.env.dataset())
    if not train:
        raise RuntimeError("opd environment dataset is empty")
    max_examples = int(getattr(request.spec.train, "max_examples", 0) or 0) if request.spec else 0
    if max_examples > 0:
        train = train[:max_examples]
    scanned = [0]
    with _opd_train.liveness_heartbeat("opd_prompt_scan", progress=lambda: scanned[0]):
        # rendering prompts for a large dataset can outlast the heartbeat window; keep the worker
        # alive while scanning (the scan is O(dataset) tokenizer/template work).
        prompt_rows = []
        for example in train:
            prompt_rows.append((example, request.env.prompt_messages(example)))
            scanned[0] += 1
    multimodal = any(record_has_images(example, messages) for example, messages in prompt_rows)
    # shuffle cached rendered rows, not examples: prompt_messages may be stateful and a second
    # render could change multimodal classification. the same seeded permutation preserves resume order.
    random.Random(_opd_train._w.SEED).shuffle(prompt_rows)
    return prompt_rows, multimodal


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


def _prepare_prompts(
    request: _OpdRequest,
    prompt_rows: list[tuple[Any, Any]],
    multimodal: bool,
    capability: str,
    control_panel_url: str,
) -> _PromptState:
    from flash.content.multimodal import (
        image_teacher_prompt_messages,
        normalize_prompt_images,
        record_has_images,
    )
    from flash.engine.worker.teacher.client import TeacherClient

    teacher = TeacherClient(capability, control_panel_url, request.knobs.teacher_model)
    processor = None
    if multimodal:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            request.model_id,
            trust_remote_code=True,
            **_opd_train._w.model_revision_kwargs(request.model_revision),
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = _opd_train._w.load_tokenizer(request.model_id, revision=request.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    thinking_prefill = _opd_train._thinking_prefill_text(tokenizer)
    requested_len = request.knobs.max_length or (
        _opd_train.RECIPE.opd.max_prompt_len + request.knobs.max_completion
    )
    # clamp to the architecture BEFORE deriving the prompt budget, so every downstream length agrees.
    # clamping only the engine would admit prompts sized against the unclamped budget and then fail
    # them at rollout instead of training on the shorter context.
    max_model_len = _opd_train.clamp_engine_len(
        requested_len,
        _opd_train.model_max_position_embeddings(request.model_id, request.model_revision),
    )
    if max_model_len < requested_len:
        print(
            f"[opd-verl] max_context_tokens {requested_len} exceeds the {request.model_id} context limit; "
            f"training at {max_model_len}",
            flush=True,
        )
    prompt_budget = max_model_len - request.knobs.max_completion
    if prompt_budget < 1:
        raise RuntimeError("opd max_context_tokens leaves no room for a prompt")
    if request.multi_turn:
        _opd_train.validate_glue_template(tokenizer, thinking=bool(_opd_train._w.THINKING))
    prompts: list[Any] = []
    dropped_long = 0
    package_root_value = getattr(request.env, "package_root", None)
    package_root = str(Path(package_root_value).resolve()) if package_root_value else None
    prepped = [0]
    with _opd_train.liveness_heartbeat("opd_image_prep", progress=lambda: prepped[0]):
        for example, messages in prompt_rows:
            prepped[0] += 1
            if request.multi_turn:
                messages = _opd_train.validate_transcript_messages(
                    messages, source="environment initial prompt"
                )
            if record_has_images(example, messages):
                assert processor is not None
                normalized = normalize_prompt_images(example, messages, package_root)
                student_messages = normalized.messages
                image_descriptors = tuple(normalized.descriptors)
                teacher_messages = image_teacher_prompt_messages(
                    student_messages, len(image_descriptors)
                )
                prompt_ids = _opd_train._processor_expanded_prompt_ids(
                    processor,
                    student_messages,
                    image_descriptors,
                    package_root,
                    enable_thinking=bool(_opd_train._w.THINKING),
                )
            else:
                student_messages = messages
                teacher_messages = messages
                image_descriptors = ()
                if processor is not None:
                    # mixed job: the verl child tokenizes EVERY row through the multimodal dataset
                    # path (the processor), so text-only rows must freeze via the same path or the
                    # bridge's exact prompt-id check trips on tokenizer-vs-processor differences.
                    prompt_ids = _opd_train._processor_expanded_prompt_ids(
                        processor,
                        student_messages,
                        (),
                        package_root,
                        enable_thinking=bool(_opd_train._w.THINKING),
                    )
                else:
                    prompt_ids = _opd_train._normalize_prompt_ids(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=_opd_train._w.THINKING,
                        )
                    )
            if len(prompt_ids) > prompt_budget:
                dropped_long += 1
                continue
            prompts.append(
                _opd_train._BridgePrompt(
                    student_messages=student_messages,
                    teacher_messages=teacher_messages,
                    prompt_ids=prompt_ids,
                    image_descriptors=image_descriptors,
                    package_root=package_root,
                    example=example if request.multi_turn else None,
                )
            )
    return _PromptState(
        teacher,
        tokenizer,
        thinking_prefill,
        max_model_len,
        prompt_budget,
        prompts,
        dropped_long,
    )


def _prepare_workload(
    request: _OpdRequest,
    prompt_state: _PromptState,
    multimodal: bool,
) -> _WorkloadState:
    prompts = prompt_state.prompts
    knobs = request.knobs
    prompts_per_step = min(knobs.prompts_per_step, len(prompts))
    derived_steps = _opd_train.on_policy_steps(
        epochs=knobs.epochs,
        prompt_count=len(prompts),
        prompts_per_step=prompts_per_step,
    )
    update_horizon = _opd_train.resolve_update_horizon(derived_steps, knobs.max_steps)
    _opd_train.validate_save_steps(knobs.save_at_steps, update_horizon)
    prompt_pool_fingerprint = _opd_train._prompt_pool_fingerprint(prompts)
    workdir = os.path.join(
        "/tmp", "flash-opd-verl", _opd_train._w.RUN_ID, f"seed-{_opd_train._w.SEED}"
    )
    shutil.rmtree(workdir, ignore_errors=True)
    data_dir = os.path.join(workdir, "data")
    image_dir = os.path.join(workdir, "images")
    shim_dir = os.path.join(workdir, "shim")
    local_dir = os.path.join(workdir, "checkpoints")
    export_root = os.path.join(workdir, "checkpoint-adapters")
    mutation_failure_path = os.path.join(workdir, "mutation-failure")
    score_delivery_failure_path = os.path.join(workdir, "score-delivery-failure")
    abandonment_failure_path = os.path.join(workdir, "abandonment-failure")
    resample_failure_path = os.path.join(workdir, "resample-failure")
    cycle_commit_failure_path = os.path.join(workdir, "cycle-commit-failure")
    teacher_worker_failure_path = os.path.join(workdir, "teacher-worker-failure")
    for path in (data_dir, shim_dir, local_dir, export_root):
        os.makedirs(path, exist_ok=True)
    materialized_images: dict[int, list[dict[str, str]]] = {}
    if multimodal:
        for index, prompt in enumerate(prompts):
            uris = _opd_train._materialize_verl_images(
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
            "prompt": (
                [
                    {
                        "role": str(message.get("role") or ""),
                        "content": _opd_train._verl_image_message_content(message.get("content")),
                    }
                    for message in prompt.student_messages
                ]
                if multimodal
                else prompt.student_messages
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
    _opd_train._write_opd_parquet(rows, train_file)
    _opd_train._write_opd_parquet([rows[0]], val_file)
    lora_config = _opd_train._w.make_lora(request.model_id)
    target_modules = lora_config.target_modules
    if isinstance(target_modules, set | frozenset):
        target_modules = sorted(target_modules)
    lora_rank = int(lora_config.r)
    return _WorkloadState(
        prompts_per_step,
        update_horizon,
        prompt_pool_fingerprint,
        workdir,
        shim_dir,
        local_dir,
        export_root,
        mutation_failure_path,
        score_delivery_failure_path,
        abandonment_failure_path,
        resample_failure_path,
        cycle_commit_failure_path,
        teacher_worker_failure_path,
        train_file,
        val_file,
        lora_rank,
        int(lora_config.lora_alpha),
        target_modules,
        _opd_train._warmstart_adapter_path(request.model_id, request.model_revision, lora_rank),
    )


def _materialize_child_files(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    python_bin: str,
    caps: dict[str, Any],
    gdn_reset_arch: str | None,
    eos_token_ids: tuple[int, ...],
) -> _RuntimeState:
    knobs = request.knobs
    model_path = _opd_train._cached_model_path(request.model_id, request.model_revision)
    # the ranks verl will RUN, not the cards rented: with ulysses pinned off every rank is a dp rank,
    # and verl chunks the step's sequences across them with an exact-divisibility assert. bound here
    # at the single source so the launch width, the resume world_size and the run metadata cannot
    # disagree about how wide the attempt actually was.
    gpu_count = rl_data_parallel_cards(
        int(getattr(request.spec.gpu, "count", 1) or 1),
        workload.prompts_per_step * knobs.group_size,
    )
    save_freq = math.gcd(*knobs.save_at_steps) if knobs.save_at_steps else knobs.save_every
    # verl logs from the verl interpreter, so gate wandb on THAT env (see resolve_verl_loggers).
    loggers = _opd_train.resolve_verl_loggers(caps)
    project_name = (
        request.spec.wandb.project if request.spec and request.spec.wandb else None
    ) or "flash"
    experiment_name = _opd_train._w.wandb_run_name()
    entry_path, reward_path = _write_child_shims(
        request,
        workload,
        gdn_reset_arch,
        loggers,
    )
    resume_step, resume_state = _opd_train._restore_verl_resume(
        workload.local_dir,
        prompt_pool_fingerprint=workload.prompt_pool_fingerprint,
        update_horizon=workload.update_horizon,
        # the same count this attempt hands verl as n_gpus_per_node, which is the DATA-parallel
        # width: ulysses is pinned to 1, so every rank is a dp rank.
        world_size=gpu_count,
    )
    bridge = _opd_train._TeacherAlignmentBridge(
        prompts=prompt_state.prompts,
        tokenizer=prompt_state.tokenizer,
        teacher=prompt_state.teacher,
        thinking_prefill=prompt_state.thinking_prefill,
        eos_token_ids=eos_token_ids,
        stop_sequences=tuple(str(value) for value in knobs.stop_sequences),
        structured=request.structured_outputs is not None,
        active_env=request.env if request.multi_turn else None,
        multi_turn=request.multi_turn,
        max_turns=request.max_turns,
        thinking=bool(_opd_train._w.THINKING),
        mutation_callback=_opd_train._w.publish_opd_optimizer_start_marker,
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


def _write_child_shims(
    request: _OpdRequest,
    workload: _WorkloadState,
    gdn_reset_arch: str | None,
    loggers: list[str],
) -> tuple[str, str]:
    shim_dir = workload.shim_dir
    parent_dir = os.path.dirname(_opd_train.__file__)
    copies = (
        ("train/opd/child/plugin.py", "flash_opd_plugin.py"),
        # the plugin imports this by its flat name at child-import time, so it has to land next to it.
        ("train/opd/child/bridge.py", "flash_opd_bridge.py"),
        ("train/opd/child/structured.py", "flash_opd_structured.py"),
        ("train/opd/child/multiturn.py", "flash_opd_multiturn.py"),
        ("train/core/child/glue.py", "flash_multiturn_glue.py"),
    )
    for source, target in copies:
        shutil.copy2(os.path.join(parent_dir, *source.split("/")), os.path.join(shim_dir, target))
    entry_path = os.path.join(shim_dir, "flash_opd_entry.py")
    with open(entry_path, "w", encoding="utf-8") as file:
        file.write("import verl\nfrom flash_opd_plugin import main\nmain()\n")
    # use a zero custom reward: verl still runs scoring when use_task_rewards=false, and its default
    # registry has no flash_opd entry (reward_loop.py:146-155).
    reward_path = os.path.join(shim_dir, "flash_opd_reward.py")
    with open(reward_path, "w", encoding="utf-8") as file:
        file.write(_opd_train._OPD_ZERO_REWARD_SOURCE)
    opd_shim_source = _opd_train._render_opd_sitecustomize(
        save_at_steps=request.knobs.save_at_steps,
        total_steps=workload.update_horizon,
    )
    if gdn_reset_arch is not None:
        opd_shim_source += _opd_train.render_gdn_varlen_shim(gdn_reset_arch)
    if "wandb" in loggers:
        opd_shim_source += _opd_train.render_wandb_link_shim()
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as file:
        file.write(opd_shim_source)
    return entry_path, reward_path


def _build_base_config(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    eos_token_ids: tuple[int, ...],
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
        "target_parameters": _opd_train._w.lora_target_parameters(request.model_id),
        "lora_adapter_path": workload.warmstart_adapter,
        "learning_rate": knobs.learning_rate,
        "local_dir": workload.local_dir,
        "save_freq": runtime.save_freq,
        "n_gpus_per_node": runtime.gpu_count,
        # opd shards by DATA -- see ULYSSES_SEQUENCE_PARALLEL_SIZE for why.
        "ulysses_sequence_parallel_size": ULYSSES_SEQUENCE_PARALLEL_SIZE,
        "seed": _opd_train._w.backend_seed(_opd_train._w.SEED),
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
        "thinking": bool(_opd_train._w.THINKING),
        "structured_outputs": request.structured_outputs,
    }


def _build_child_callbacks(
    watcher: Any,
    progress_state: Any,
    bridge: Any,
    resume_step: int,
) -> _ChildCallbacks:
    progress = {
        "step": resume_step,
        "loss": None,
        "truncation_rate": None,
        "truncation_step": None,
    }
    wandb_link: dict[str, str | None] = {}

    def on_line(line: str) -> None:
        watcher.raise_if_failed()
        link = _opd_train.parse_wandb_link(line)
        if link is not None:
            wandb_link.update(link)
        step_number = _opd_train.verl_step_number(line)
        if step_number is None:
            return
        # use parse_verl_metric because numpy 2 pprint emits np.float64(...); float() would drop
        # every step and leave a trained run with an empty loss curve.
        loss = _opd_train.parse_verl_metric(line, "actor/distillation/loss")
        if loss is None:
            loss = _opd_train.parse_verl_metric(line, "distillation/loss")
        if loss is None:
            # verl emits step-tagged lines that are not metric summaries (timers, val lines);
            # skip those rather than killing the run. the end-of-run guard still fails loud
            # when NO step ever produced a distillation loss.
            return
        progress["loss"] = loss
        progress["truncation_rate"] = progress_state.record_step(step_number, loss, bridge)
        progress["truncation_step"] = step_number

    def on_step(step: int) -> None:
        progress["step"] = step
        payload = {"step": step}
        if progress["loss"] is not None:
            payload["loss"] = progress["loss"]
        if progress["truncation_step"] == step and progress["truncation_rate"] is not None:
            payload["truncation_rate"] = progress["truncation_rate"]
        _opd_train._w.heartbeat("opd_step", **payload)

    def child_heartbeat() -> None:
        _opd_train._w.heartbeat("opd_step", liveness=True, step=int(progress["step"] or 0))

    child_tail = _opd_train.ChildOutputTail()
    # one instance for the whole run: it measures silence ACROSS ticks, so it cannot live inside
    # the per-tick callback.
    tail_staleness = _opd_train.ChildTailStaleness()

    def liveness_fields() -> dict[str, object]:
        return _opd_train.stall_tail_fields(
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
    )


def _run_child(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    config: dict[str, Any],
    eos_token_ids: tuple[int, ...],
) -> _ChildResult:
    overrides = _opd_train.build_opd_overrides(config)
    progress_state = _opd_train._OpdProgressState(runtime.resume_state)
    watcher = _build_checkpoint_watcher(request, workload, runtime, progress_state)
    callbacks = _build_child_callbacks(watcher, progress_state, runtime.bridge, runtime.resume_step)
    child_env = _build_child_env(request, prompt_state, workload, runtime, eos_token_ids)
    command = [runtime.python_bin, runtime.entry_path, *overrides]
    gpu_sampler = _opd_train._NvidiaSmiPeakSampler().start()
    train_started_at = time.time()
    return_code = 0
    training_completed = runtime.resume_step >= workload.update_horizon
    watcher.start()
    try:
        if runtime.resume_step < workload.update_horizon:
            progress_state.start_training()
            with _opd_train.liveness_heartbeat(
                "opd_step",
                progress=lambda: int(callbacks.progress["step"] or 0),
                progress_step=True,
                fields=callbacks.liveness_fields,
            ):
                return_code = _opd_train.run_verl_training(
                    command,
                    env=child_env,
                    on_step=callbacks.on_step,
                    on_line=callbacks.on_line,
                    heartbeat=callbacks.child_heartbeat,
                    tail=callbacks.child_tail,
                )
                training_completed = return_code == 0
    finally:
        watcher.stop(require_complete=training_completed)
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
    actor_dir, final_step = _opd_train.latest_global_step_dir(workload.local_dir)
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
) -> Any:
    watcher = _opd_train._OpdVerlCheckpointWatcher(
        local_dir=workload.local_dir,
        export_root=workload.export_root,
        python_bin=runtime.python_bin,
        model_id=request.model_id,
        model_revision=request.model_revision,
        required_steps=request.knobs.save_at_steps,
        seed=int(_opd_train._w.SEED),
        prompt_pool_fingerprint=workload.prompt_pool_fingerprint,
        prompts_per_step=workload.prompts_per_step,
        group_size=request.knobs.group_size,
        accounting_state=progress_state.checkpoint_state,
    )
    watcher.processed_steps.update(
        _opd_train._processed_resume_steps(request.knobs.save_at_steps, runtime.resume_step)
    )
    return watcher


def _build_child_env(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    eos_token_ids: tuple[int, ...],
) -> dict[str, str]:
    bridge = runtime.bridge
    return _opd_train._build_opd_child_env(
        shim_dir=workload.shim_dir,
        wandb_enabled="wandb" in runtime.loggers,
        bridge_url=bridge.url,
        bridge_token=bridge.token,
        seed=int(_opd_train._w.SEED),
        stop_sequences=request.knobs.stop_sequences,
        eos_token_ids=eos_token_ids,
        structured_outputs=request.structured_outputs,
        model_vocab_size=request.model_vocab_size,
        thinking=bool(_opd_train._w.THINKING),
        multi_turn=request.multi_turn,
        max_turns=request.max_turns,
        max_model_len=prompt_state.max_model_len,
        mutation_failure_path=workload.mutation_failure_path,
        score_delivery_failure_path=workload.score_delivery_failure_path,
        abandonment_failure_path=workload.abandonment_failure_path,
        resample_failure_path=workload.resample_failure_path,
        cycle_commit_failure_path=workload.cycle_commit_failure_path,
        teacher_worker_failure_path=workload.teacher_worker_failure_path,
    )


def _reconcile_child_failures(
    workload: _WorkloadState,
    bridge: Any,
    return_code: int,
    *,
    truncation_window: _opd_train._TruncationWindow | None,
) -> None:
    score_delivery_failure = _opd_train._reconcile_score_delivery_failure(
        bridge,
        _opd_train._read_classified_failure_fallback(workload.score_delivery_failure_path),
    )
    no_signal_failure = _opd_train._reconcile_no_signal_notification_failure(
        bridge,
        (
            _opd_train._read_classified_failure_fallback(workload.resample_failure_path),
            _opd_train._read_classified_failure_fallback(workload.abandonment_failure_path),
        ),
    )
    fallback_mutation_failure = _opd_train._read_classified_failure_fallback(
        workload.mutation_failure_path
    )
    if fallback_mutation_failure is not None:
        bridge._record_mutation_failure(*fallback_mutation_failure)
    cycle_commit_failure = _opd_train._read_classified_failure_fallback(
        workload.cycle_commit_failure_path
    )
    # a teacher-path exit kills the ray agent-loop actor, not the child driver, so its exit code
    # never reaches the return-code branches in _raise_verl_failure. this record is the only
    # surviving evidence of why that worker died -- without it the run is reported as a bare
    # "worker died" or, before the sample deadline existed, as an indefinite hang.
    teacher_worker_failure = _opd_train._read_classified_failure_fallback(
        workload.teacher_worker_failure_path
    )
    _opd_train._raise_verl_failure(
        return_code,
        bridge.teacher_failure,
        bridge.mutation_failure,
        cycle_commit_failure,
        no_signal_failure,
        score_delivery_failure,
        truncation_window=truncation_window,
        teacher_worker_failure=teacher_worker_failure,
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


def _report_training_complete(result: _ChildResult, started_at: float) -> float:
    _opd_train._w.heartbeat(
        "opd_trained",
        step=result.final_step,
        train_wall=result.train_wall,
        gpu=_opd_train._w.gpu_diagnostics(include_torch=False),
    )
    return result.train_started_at - started_at


def _export_and_upload_adapter(
    request: _OpdRequest,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    result: _ChildResult,
) -> str:
    adapter_dir = os.path.join(workload.workdir, "adapter")
    _opd_train._export_checkpoint_adapter(
        result.actor_dir,
        adapter_dir,
        model_id=request.model_id,
        model_revision=request.model_revision,
        python_bin=runtime.python_bin,
    )
    _opd_train._w.hf_upload_folder(adapter_dir, "adapter", required=True)
    return adapter_dir


def _build_train_note_sections(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    result: _ChildResult,
    download_seconds: float,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], dict[str, Any]],
]:
    final_accounting = result.final_accounting
    knobs = request.knobs
    initial = {
        "epochs": knobs.epochs,
        "retained_prompts": len(prompt_state.prompts),
        "dropped_long_prompts": prompt_state.dropped_long,
        "method": "gkd",
        "init_from_adapter": request.spec.train.init_from_adapter or None,
        "teacher_model": knobs.teacher_model,
        "download_seconds": download_seconds,
        "thinking": _opd_train._w.THINKING,
        "loss_curve": final_accounting["loss_curve"],
        "mean_coverage": (
            float(final_accounting["coverage_sum"]) / int(final_accounting["aligned_sequences"])
            if final_accounting["aligned_sequences"]
            else 0.0
        ),
    }
    accounting = {
        "truncated_rollouts": int(final_accounting["truncated_rollouts"]),
        "forced_tokens": int(final_accounting["forced_tokens"]),
        "dropped_forced_groups": int(final_accounting["dropped_forced_groups"]),
        "teacher_input_tokens": int(final_accounting["teacher_input_tokens"]),
        "teacher_output_tokens": int(final_accounting["teacher_output_tokens"]),
        "aligned_sequences": int(final_accounting["aligned_sequences"]),
        "empty_alignments": int(final_accounting["empty_alignments"]),
        "teacher_ok": int(final_accounting["teacher_ok"]),
    }
    training = {
        "temperature": knobs.temperature,
        "group_size": knobs.group_size,
        "prompts_per_step": workload.prompts_per_step,
        "max_completion_len": knobs.max_completion,
        "multi_turn": request.multi_turn,
        "max_turns": request.max_turns if request.multi_turn else None,
        "episodes": int(final_accounting["episodes_seen"]) if request.multi_turn else None,
        "mean_turns_per_episode": (
            int(final_accounting["mt_turn_records"]) / int(final_accounting["episodes_seen"])
            if request.multi_turn and final_accounting["episodes_seen"]
            else None
        ),
    }
    backend = (
        {
            "rollout_backend": "verl_vllm",
            "verl_version": "0.8.0",
            "verl_backend": "fsdp",
            # report the EXECUTED width, not the allocation: the card count here would claim a
            # sequence-parallel run that did not happen. token-balanced batching makes every
            # allocated rank a dp rank, so unlike sft the executed dp width IS the card count.
            "ulysses_sequence_parallel_size": ULYSSES_SEQUENCE_PARALLEL_SIZE,
            "data_parallel_size": runtime.gpu_count,
        },
        {
            "peak_gpu_gb": result.peak_gpu_gb,
            "warm_started": bool(workload.warmstart_adapter),
            "resumed": bool(runtime.resume_step),
            "wandb_project": runtime.project_name if "wandb" in runtime.loggers else None,
            "wandb_run_name": runtime.experiment_name if "wandb" in runtime.loggers else None,
        },
    )
    return initial, accounting, training, backend
