"""Flash OPD orchestration through verl 0.8.0 in an isolated child interpreter."""

from __future__ import annotations

import math
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from flash.engine.plan.recipe import RECIPE
from flash.engine.plan.steps import (
    final_save_due,
    on_policy_steps,
    resolve_update_horizon,
    validate_save_steps,
)
from flash.engine.profiling.sft_workload import (
    _materialize_verl_images,
)
from flash.engine.worker.backend_common import (
    ChildOutputTail,
    ChildTailStaleness,
    clamp_engine_len,
    fused_ce_backend,
    gdn_probe_module,
    latest_global_step_dir,
    model_max_position_embeddings,
    parse_verl_metric,
    parse_wandb_link,
    probe_verl_capabilities,
    render_gdn_varlen_shim,
    render_wandb_link_shim,
    require_gdn_boundary_resets,
    resolve_blackwell_attention_backends,
    resolve_rollout_enforce_eager,
    resolve_verl_loggers,
    resolve_verl_python,
    rollout_sleep_unsupported,
    run_verl_training,
    stall_tail_fields,
    verl_device_capability,
    verl_step_number,
)
from flash.engine.worker.entry.opd import (
    _resolve_opd_knobs,
    _thinking_prefill_text,
)
from flash.engine.worker.io.heartbeat import liveness_heartbeat
from flash.engine.worker.runtime.pkg_proxy import W as _w
from flash.engine.worker.runtime.rng import seed_training_rngs
from flash.engine.worker.sft_train import (
    _cached_model_path,
    _export_checkpoint_adapter,
    _NvidiaSmiPeakSampler,
    _probe_gpu_in_subprocess,
    _verl_image_message_content,
    _warmstart_adapter_path,
)
from flash.engine.worker.train.core.child.glue import (
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
        self._prev_aligned = int(state.get("aligned_sequences", state.get("granularity_n", 0)))
        self._prev_cov_sum = float(state.get("coverage_sum", state.get("granularity_sum", 0.0)))
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

    def record_step(self, step: int, loss: float, bridge: _TeacherAlignmentBridge) -> None:
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
            snapshot.update(
                {
                    "train_wall_seconds": self._train_wall_seconds(),
                    "loss_curve": list(self.loss_curve),
                    "coverage_curve": list(self.coverage_curve),
                }
            )
            self._step_states[step] = snapshot
            self._condition.notify_all()

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


def run_opd_train(spec=None) -> None:
    """Run flash OPD through verl's native rollout and weight-sync path."""
    from flash.content.multimodal import (
        image_teacher_prompt_messages,
        normalize_prompt_images,
        record_has_images,
        validate_multimodal_training,
    )
    from flash.engine.worker.teacher.client import TeacherClient

    spec = spec or _w.JOB_SPEC
    env = _w.require_active_env()
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
    knobs = _resolve_opd_knobs()
    if multi_turn and knobs.structured_outputs:
        raise RuntimeError(
            "multi-turn structured-output OPD is not supported until a per-turn constraint contract exists"
        )
    model_id = spec.model if spec else RECIPE.hf_model_id
    model_revision = getattr(spec, "model_revision", "") if spec else ""
    from flash.engine.worker.train.opd.validation import validate_opd_structured_outputs

    structured_validation = validate_opd_structured_outputs(
        knobs.structured_outputs,
        model_id=model_id,
        model_revision=model_revision,
    )
    structured_outputs = structured_validation.constraint
    model_vocab_size = structured_validation.model_vocab_size
    # the child trainer is seeded through its own config, but the environment's dataset /
    # prompt_messages calls run HERE in the parent. an unseeded parent can build a different prompt
    # pool across attempts, whose fingerprint then rejects a valid resume checkpoint. seed just
    # before the first env call that can consume randomness, so the cheap fail-closed guards above
    # still raise without paying for the torch import.
    seed_training_rngs(_w.SEED)
    train = list(env.dataset())
    if not train:
        raise RuntimeError("opd environment dataset is empty")
    max_examples = int(getattr(spec.train, "max_examples", 0) or 0) if spec else 0
    if max_examples > 0:
        train = train[:max_examples]
    _scanned = [0]
    with liveness_heartbeat("opd_prompt_scan", progress=lambda: _scanned[0]):
        # rendering prompts for a large dataset can outlast the heartbeat window; keep the worker
        # alive while scanning (the scan is O(dataset) tokenizer/template work).
        prompt_rows = []
        for example in train:
            prompt_rows.append((example, env.prompt_messages(example)))
            _scanned[0] += 1
    multimodal = any(record_has_images(example, messages) for example, messages in prompt_rows)
    if multimodal:
        validate_multimodal_training(
            model_id,
            "opd",
            getattr(spec.train, "teacher_model", None),
        )
        if multi_turn:
            raise ValueError("multi-turn image-bearing opd is not supported")
    # shuffle cached rendered rows, not examples: prompt_messages may be stateful and a second
    # render
    # could change multimodal classification. the same seeded permutation preserves resume order.
    random.Random(_w.SEED).shuffle(prompt_rows)

    started_at = time.time()
    # validate the control-panel broker transport before the gpu probe and model prefetch so a malformed
    # attempt fails
    # before any additional paid setup. raw managed-teacher provider credentials never enter the worker.
    from flash.core.spec import CONTROL_PANEL_URL_ENV, TEACHER_CAPABILITY_ENV

    control_panel_url = os.environ.get(CONTROL_PANEL_URL_ENV, "").strip()
    capability = os.environ.get(TEACHER_CAPABILITY_ENV, "").strip()
    if not control_panel_url or not capability:
        raise RuntimeError(
            "managed teacher control-panel transport is missing from the OPD parent worker"
        )
    _w.heartbeat("opd_start", gpu=_w.gpu_diagnostics(include_torch=False))
    _probe_gpu_in_subprocess(
        spec.gpu.type if spec else None,
        exact_type=spec.gpu.type if spec else "",
    )
    teacher = TeacherClient(capability, control_panel_url, knobs.teacher_model)
    processor = None
    if multimodal:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            **_w.model_revision_kwargs(model_revision),
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = _w.load_tokenizer(model_id, revision=model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    thinking_prefill = _thinking_prefill_text(tokenizer)
    requested_len = knobs.max_length or (RECIPE.opd.max_prompt_len + knobs.max_completion)
    # clamp to the architecture BEFORE deriving the prompt budget, so every downstream length agrees.
    # clamping only the engine would admit prompts sized against the unclamped budget and then fail
    # them at rollout instead of training on the shorter context.
    max_model_len = clamp_engine_len(
        requested_len, model_max_position_embeddings(model_id, model_revision)
    )
    if max_model_len < requested_len:
        print(
            f"[opd-verl] max_context_tokens {requested_len} exceeds the {model_id} context limit; "
            f"training at {max_model_len}",
            flush=True,
        )
    prompt_budget = max_model_len - knobs.max_completion
    if prompt_budget < 1:
        raise RuntimeError("opd max_context_tokens leaves no room for a prompt")
    if multi_turn:
        validate_glue_template(tokenizer, thinking=bool(_w.THINKING))

    prompts: list[_BridgePrompt] = []
    dropped_long = 0
    package_root_value = getattr(env, "package_root", None)
    package_root = str(Path(package_root_value).resolve()) if package_root_value else None
    _prepped = [0]
    with liveness_heartbeat("opd_image_prep", progress=lambda: _prepped[0]):
        for example, messages in prompt_rows:
            _prepped[0] += 1
            if multi_turn:
                messages = validate_transcript_messages(
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
                prompt_ids = _processor_expanded_prompt_ids(
                    processor,
                    student_messages,
                    image_descriptors,
                    package_root,
                    enable_thinking=bool(_w.THINKING),
                )
            else:
                student_messages = messages
                teacher_messages = messages
                image_descriptors = ()
                if processor is not None:
                    # mixed job: the verl child tokenizes EVERY row through the multimodal dataset
                    # path (the processor), so text-only rows must freeze via the same path or the
                    # bridge's exact prompt-id check trips on tokenizer-vs-processor differences.
                    prompt_ids = _processor_expanded_prompt_ids(
                        processor,
                        student_messages,
                        (),
                        package_root,
                        enable_thinking=bool(_w.THINKING),
                    )
                else:
                    prompt_ids = _normalize_prompt_ids(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=_w.THINKING,
                        )
                    )
            if len(prompt_ids) > prompt_budget:
                dropped_long += 1
                continue
            prompts.append(
                _BridgePrompt(
                    student_messages=student_messages,
                    teacher_messages=teacher_messages,
                    prompt_ids=prompt_ids,
                    image_descriptors=image_descriptors,
                    package_root=package_root,
                    example=example if multi_turn else None,
                )
            )
    if not prompts:
        raise RuntimeError("every OPD prompt exceeds the configured prompt budget")
    # weights come AFTER the budget filter: a dataset whose every prompt is over budget is a
    # deterministic input error, and downloading tens of GB before raising it burns paid worker
    # minutes for a verdict the tokenizer already had. the tokenizer/processor/config loads above
    # fetch kilobytes, not weights, so they are cheap to run first.
    download_seconds = _w.prefetch_model(model_id, revision=model_revision)
    # reads the snapshot with local_files_only, so it has to follow the prefetch.
    eos_token_ids = generation_eos_from_cached_config(model_id, model_revision, tokenizer)
    prompts_per_step = min(knobs.prompts_per_step, len(prompts))
    derived_steps = on_policy_steps(
        epochs=knobs.epochs,
        prompt_count=len(prompts),
        prompts_per_step=prompts_per_step,
    )
    update_horizon = resolve_update_horizon(derived_steps, knobs.max_steps)
    validate_save_steps(knobs.save_at_steps, update_horizon)
    prompt_pool_fingerprint = _prompt_pool_fingerprint(prompts)

    workdir = os.path.join("/tmp", "flash-opd-verl", _w.RUN_ID, f"seed-{_w.SEED}")
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
    for path in (data_dir, shim_dir, local_dir, export_root):
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
            "prompt": (
                [
                    {
                        "role": str(message.get("role") or ""),
                        "content": _verl_image_message_content(message.get("content")),
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
    _write_opd_parquet(rows, train_file)
    _write_opd_parquet([rows[0]], val_file)

    lora_config = _w.make_lora(model_id)
    lora_rank = int(lora_config.r)
    lora_alpha = int(lora_config.lora_alpha)
    target_modules = lora_config.target_modules
    if isinstance(target_modules, set | frozenset):
        target_modules = sorted(target_modules)
    warmstart_adapter = _warmstart_adapter_path(model_id, model_revision, lora_rank)
    # same silent boundary the sft path guards: with no prebuilt worker image this builds a venv and
    # installs the training stack, minutes long with nothing to report and no liveness thread running.
    with liveness_heartbeat("opd_configuring"):
        python_bin = resolve_verl_python(
            workdir, install_wandb=bool(os.environ.get("WANDB_API_KEY"))
        )
        # the architecture question is asked on its OWN, before the batched probe, because it reads
        # the checkpoint config and has nothing to do with the child's capabilities.
        # model_is_gdn_hybrid already returns False on its own probe failure, so it needs no guard.
        # the modeling module is resolved HERE, in the parent, because it needs a hub/cache read the
        # child must not repeat; "" skips the gdn question for a non-hybrid.
        from flash.engine.worker.model.packing import model_is_gdn_hybrid

        gdn_hybrid = model_is_gdn_hybrid(model_id, revision=model_revision)
        gdn_module = gdn_probe_module(model_id, model_revision) if gdn_hybrid else ""
        # ONE child answers every independent capability question. each used to cost its own
        # interpreter, and the torch/verl import -- not the question -- was the price.
        caps = probe_verl_capabilities(python_bin, gdn_module)
    model_path = _cached_model_path(model_id, model_revision)
    gpu_count = int(getattr(spec.gpu, "count", 1) or 1)
    save_freq = math.gcd(*knobs.save_at_steps) if knobs.save_at_steps else knobs.save_every
    # verl logs from the verl interpreter, so gate wandb on THAT env (see resolve_verl_loggers).
    loggers = resolve_verl_loggers(caps)
    project_name = (spec.wandb.project if spec and spec.wandb else None) or "flash"
    experiment_name = _w.wandb_run_name()
    # enable fp8 kv on cc >= 8.9 only for non-gdn models; gdn sleep/wake crashes on hybrid caches.
    # keep this aligned with vram.py, and keep the device probe separate from gdn classification.
    try:
        import torch as _torch_cc

        _cc_ok = bool(
            _torch_cc.cuda.is_available() and _torch_cc.cuda.get_device_capability() >= (8, 9)
        )
    except Exception:  # no cuda / probe failure -> conservative bf16 kv
        _cc_ok = False
    fp8_kv = _cc_ok and not gdn_hybrid

    # gdn packing requires child support for seq_idx and cu_seqlens; fallbacks discard both and
    # silently bleed state across examples. see require_gdn_boundary_resets.
    gdn_reset_arch = require_gdn_boundary_resets(caps, gdn_module)

    # run sm86 eagerly because vllm 0.19.1 graph capture degenerates there.
    # sm89 capture is empirically acceptable; enforce_eager overrides async cudagraph settings last
    # at config/vllm.py:1024. reuse one capability probe for both rollout decisions.
    verl_cc = verl_device_capability(caps)
    enforce_eager = resolve_rollout_enforce_eager(verl_cc)
    # pin both rollout attention backends on blackwell: vllm 0.19.1's ViT CUTE default fails with
    # missing cutlass.cute.core.ThrMma, including text-only rollouts on VL models.
    attention_backend, mm_encoder_attn_backend = resolve_blackwell_attention_backends(caps, verl_cc)

    plugin_path = os.path.join(shim_dir, "flash_opd_plugin.py")
    shutil.copy2(
        os.path.join(os.path.dirname(__file__), "train", "opd", "child", "plugin.py"), plugin_path
    )
    # the plugin imports this by its flat name at child-import time, so it has to land next to it.
    bridge_helper_path = os.path.join(shim_dir, "flash_opd_bridge.py")
    shutil.copy2(
        os.path.join(os.path.dirname(__file__), "train", "opd", "child", "bridge.py"),
        bridge_helper_path,
    )
    structured_helper_path = os.path.join(shim_dir, "flash_opd_structured.py")
    shutil.copy2(
        os.path.join(os.path.dirname(__file__), "train", "opd", "child", "structured.py"),
        structured_helper_path,
    )
    multiturn_helper_path = os.path.join(shim_dir, "flash_opd_multiturn.py")
    shutil.copy2(
        os.path.join(os.path.dirname(__file__), "train", "opd", "child", "multiturn.py"),
        multiturn_helper_path,
    )
    glue_helper_path = os.path.join(shim_dir, "flash_multiturn_glue.py")
    shutil.copy2(
        os.path.join(os.path.dirname(__file__), "train", "core", "child", "glue.py"),
        glue_helper_path,
    )
    entry_path = os.path.join(shim_dir, "flash_opd_entry.py")
    with open(entry_path, "w", encoding="utf-8") as file:
        file.write("import verl\nfrom flash_opd_plugin import main\nmain()\n")
    # use a zero custom reward: verl still runs scoring when use_task_rewards=false, and its default
    # registry has no flash_opd entry (reward_loop.py:146-155).
    reward_path = os.path.join(shim_dir, "flash_opd_reward.py")
    with open(reward_path, "w", encoding="utf-8") as file:
        file.write(_OPD_ZERO_REWARD_SOURCE)
    opd_shim_source = _render_opd_sitecustomize(
        save_at_steps=knobs.save_at_steps,
        total_steps=update_horizon,
    )
    if gdn_reset_arch is not None:
        opd_shim_source += render_gdn_varlen_shim(gdn_reset_arch)
    if "wandb" in loggers:
        opd_shim_source += render_wandb_link_shim()
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as file:
        file.write(opd_shim_source)

    resume_step, resume_state = _restore_verl_resume(
        local_dir,
        prompt_pool_fingerprint=prompt_pool_fingerprint,
        update_horizon=update_horizon,
    )
    bridge = _TeacherAlignmentBridge(
        prompts=prompts,
        tokenizer=tokenizer,
        teacher=teacher,
        thinking_prefill=thinking_prefill,
        eos_token_ids=eos_token_ids,
        stop_sequences=tuple(str(value) for value in knobs.stop_sequences),
        structured=structured_outputs is not None,
        active_env=env if multi_turn else None,
        multi_turn=multi_turn,
        max_turns=max_turns,
        thinking=bool(_w.THINKING),
        mutation_callback=_w.publish_opd_optimizer_start_marker,
        initial_state=resume_state,
    )
    bridge.start()
    try:
        config = {
            "train_files": [train_file],
            "val_files": [val_file],
            "train_batch_size": prompts_per_step,
            "max_prompt_length": prompt_budget,
            "max_response_length": knobs.max_completion,
            "model_path": model_path,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
            "target_parameters": _w.lora_target_parameters(model_id),
            "lora_adapter_path": warmstart_adapter,
            "learning_rate": knobs.learning_rate,
            "local_dir": local_dir,
            "save_freq": save_freq,
            "n_gpus_per_node": gpu_count,
            "ulysses_sequence_parallel_size": gpu_count,
            "seed": _w.backend_seed(_w.SEED),
            "project_name": project_name,
            "experiment_name": experiment_name,
            "total_training_steps": update_horizon,
            "group_size": knobs.group_size,
            "bridge_url": bridge.url,
            "bridge_token": bridge.token,
            "reward_path": reward_path,
            "kl_penalty_coef": knobs.kl_coef,
            "temperature": knobs.temperature,
            "top_p": knobs.top_p,
            # the job's own engine length (prompt + completion), already clamped to the model's
            # limit. prompt_budget above is carved out of this same value, so the engine, the prompt
            # filter, and the token budget cannot disagree. a hardcoded engine would size vllm's kv
            # cache for a context the job never uses, and -- above it -- admit prompts the engine
            # cannot hold.
            "max_sequence_length": max_model_len,
            "multi_turn": multi_turn,
            "thinking": bool(_w.THINKING),
            "structured_outputs": structured_outputs,
            "fp8_kv": fp8_kv,
            "enforce_eager": enforce_eager,
            "attention_backend": attention_backend,
            "mm_encoder_attn_backend": mm_encoder_attn_backend,
            "sleep_unsupported": rollout_sleep_unsupported(model_id),
            "loggers": loggers,
            # resolved from the out-of-process capability probe, never by opening cuda in this
            # parent -- see fused_ce_backend.
            "fused_ce_backend": fused_ce_backend(caps),
        }
        overrides = build_opd_overrides(config)
        progress_state = _OpdProgressState(resume_state)
        watcher = _OpdVerlCheckpointWatcher(
            local_dir=local_dir,
            export_root=export_root,
            python_bin=python_bin,
            model_id=model_id,
            model_revision=model_revision,
            required_steps=knobs.save_at_steps,
            seed=int(_w.SEED),
            prompt_pool_fingerprint=prompt_pool_fingerprint,
            prompts_per_step=prompts_per_step,
            group_size=knobs.group_size,
            accounting_state=progress_state.checkpoint_state,
        )
        watcher.processed_steps.update(_processed_resume_steps(knobs.save_at_steps, resume_step))
        child_env = _build_opd_child_env(
            shim_dir=shim_dir,
            wandb_enabled="wandb" in loggers,
            bridge_url=bridge.url,
            bridge_token=bridge.token,
            seed=int(_w.SEED),
            stop_sequences=knobs.stop_sequences,
            eos_token_ids=eos_token_ids,
            structured_outputs=structured_outputs,
            model_vocab_size=model_vocab_size,
            thinking=bool(_w.THINKING),
            multi_turn=multi_turn,
            max_turns=max_turns,
            max_model_len=max_model_len,
            mutation_failure_path=mutation_failure_path,
            score_delivery_failure_path=score_delivery_failure_path,
            abandonment_failure_path=abandonment_failure_path,
            resample_failure_path=resample_failure_path,
            cycle_commit_failure_path=cycle_commit_failure_path,
        )
        command = [python_bin, entry_path, *overrides]
        progress = {"step": resume_step, "loss": None}
        wandb_link: dict[str, str | None] = {}

        def on_line(line: str) -> None:
            watcher.raise_if_failed()
            link = parse_wandb_link(line)
            if link is not None:
                wandb_link.update(link)
            step_number = verl_step_number(line)
            if step_number is None:
                return
            # use parse_verl_metric because numpy 2 pprint emits np.float64(...); float() would drop
            # every step and leave a trained run with an empty loss curve.
            loss = parse_verl_metric(line, "actor/distillation/loss")
            if loss is None:
                loss = parse_verl_metric(line, "distillation/loss")
            if loss is None:
                # verl emits step-tagged lines that are not metric summaries (timers, val lines);
                # skip those rather than killing the run. the end-of-run guard still fails loud
                # when NO step ever produced a distillation loss.
                return
            progress["loss"] = loss
            progress_state.record_step(step_number, loss, bridge)

        def on_step(step: int) -> None:
            progress["step"] = step
            payload = {"step": step}
            if progress["loss"] is not None:
                payload["loss"] = progress["loss"]
            _w.heartbeat("opd_step", **payload)

        def child_heartbeat() -> None:
            _w.heartbeat("opd_step", liveness=True, step=int(progress["step"] or 0))

        child_tail = ChildOutputTail()
        # one instance for the whole run: it measures silence ACROSS ticks, so it cannot live inside
        # the per-tick callback.
        tail_staleness = ChildTailStaleness()

        def liveness_fields() -> dict[str, object]:
            return stall_tail_fields(
                int(progress["step"] or 0), child_tail, staleness=tail_staleness
            )

        gpu_sampler = _NvidiaSmiPeakSampler().start()
        train_started_at = time.time()
        return_code = 0
        training_completed = resume_step >= update_horizon
        watcher.start()
        try:
            if resume_step < update_horizon:
                progress_state.start_training()
                with liveness_heartbeat(
                    "opd_step",
                    progress=lambda: int(progress["step"] or 0),
                    progress_step=True,
                    fields=liveness_fields,
                ):
                    return_code = run_verl_training(
                        command,
                        env=child_env,
                        on_step=on_step,
                        on_line=on_line,
                        heartbeat=child_heartbeat,
                        tail=child_tail,
                    )
                    training_completed = return_code == 0
        finally:
            watcher.stop(require_complete=training_completed)
        peak_gpu_gb = gpu_sampler.stop_gb()
        score_delivery_failure = _reconcile_score_delivery_failure(
            bridge,
            _read_classified_failure_fallback(score_delivery_failure_path),
        )
        no_signal_failure = _reconcile_no_signal_notification_failure(
            bridge,
            (
                _read_classified_failure_fallback(resample_failure_path),
                _read_classified_failure_fallback(abandonment_failure_path),
            ),
        )
        fallback_mutation_failure = _read_classified_failure_fallback(mutation_failure_path)
        if fallback_mutation_failure is not None:
            bridge._record_mutation_failure(*fallback_mutation_failure)
        cycle_commit_failure = _read_classified_failure_fallback(cycle_commit_failure_path)
        _raise_verl_failure(
            return_code,
            bridge.teacher_failure,
            bridge.mutation_failure,
            cycle_commit_failure,
            no_signal_failure,
            score_delivery_failure,
        )
        final_accounting = progress_state.final_state(bridge)
        train_wall = float(final_accounting["train_wall_seconds"])

        actor_dir, final_step = latest_global_step_dir(local_dir)
        if final_step < update_horizon:
            raise RuntimeError(
                f"opd completed {final_step}/{update_horizon} requested optimizer updates"
            )
        if not final_accounting["loss_curve"]:
            raise RuntimeError(
                "verl OPD produced no distillation-loss metrics for the whole run — the "
                "distillation path never engaged; refusing to publish"
            )
        if len(final_accounting["loss_curve"]) != final_step:
            # record_step only checks that each metric line FOLLOWS the last one, so a missing
            # trailing metric (on_line skips any step-tagged line whose loss it cannot parse) leaves
            # a curve shorter than the checkpoint verl actually wrote, and nothing later arrives to
            # catch it. opt_steps is published from this curve, so a short curve would understate the
            # updates applied. fail loud instead of reporting a number the curve cannot support.
            raise RuntimeError(
                f"verl OPD recorded {len(final_accounting['loss_curve'])} distillation-loss metrics "
                f"for {final_step} optimizer updates; refusing to publish an accounting that does "
                "not cover every update"
            )
        if int(final_accounting.get("aligned_sequences", 0) or 0) <= 0:
            # zeroed-mask pass-through batches still emit a (zero) loss metric, so the loss-curve
            # check alone cannot distinguish real distillation from a run where the teacher never
            # aligned once. require at least one aligned sequence before publishing.
            raise RuntimeError(
                "verl OPD saw zero aligned teacher sequences for the whole run — every batch was "
                "no-signal; refusing to publish an unchanged adapter"
            )
        adapter_dir = os.path.join(workdir, "adapter")
        with liveness_heartbeat(
            "opd_finalizing", progress=lambda: final_step, progress_step=True, keepalive=True
        ):
            _export_checkpoint_adapter(
                actor_dir,
                adapter_dir,
                model_id=model_id,
                model_revision=model_revision,
                python_bin=python_bin,
            )
            _w.hf_upload_folder(adapter_dir, "adapter", required=True)
            # preserve the final checkpoint only when save_at_steps is empty, matching grpo.
            # watcher and final-save paths are disjoint, so processed_steps must not suppress it.
            if final_save_due(final_step, knobs.save_at_steps):
                _w.publish_deployable_checkpoint(adapter_dir, final_step, _provenance_ready=True)

        setup_seconds = train_started_at - started_at
        _w.heartbeat(
            "opd_trained",
            step=final_step,
            train_wall=train_wall,
            gpu=_w.gpu_diagnostics(include_torch=False),
        )
        _w.write_train_meta(
            phase="opd",
            step=final_step,
            adapter_dir=adapter_dir,
            model_id=model_id,
            train_wall=train_wall,
            setup_seconds=setup_seconds,
            train_tokens=0,
            generated_tokens=int(final_accounting["generated_tokens"]),
            notes={
                "steps": update_horizon,
                # optimizer updates that actually produced a distillation loss. record_step enforces
                # loss_curve length == the metric step, and the guard above rejects a curve shorter
                # than final_step, so this is measured, not assumed.
                "opt_steps": len(final_accounting["loss_curve"]),
                "epochs": knobs.epochs,
                "retained_prompts": len(prompts),
                "dropped_long_prompts": dropped_long,
                "method": "gkd",
                "init_from_adapter": spec.train.init_from_adapter or None,
                "teacher_model": knobs.teacher_model,
                "download_seconds": download_seconds,
                "thinking": _w.THINKING,
                "loss_curve": final_accounting["loss_curve"],
                "mean_coverage": (
                    float(final_accounting["coverage_sum"])
                    / int(final_accounting["aligned_sequences"])
                    if final_accounting["aligned_sequences"]
                    else 0.0
                ),
                # the real alignment-health signal. mean_coverage reads ~1.0 even when the alignment
                # has collapsed every student token onto one group, so it cannot flag that failure
                # mode on its own; this ratio can.
                "mean_align_granularity": (
                    float(final_accounting["align_group_sum"])
                    / int(final_accounting["align_group_n"])
                    if final_accounting["align_group_n"]
                    else 0.0
                ),
                "truncated_rollouts": int(final_accounting["truncated_rollouts"]),
                "forced_tokens": int(final_accounting["forced_tokens"]),
                "dropped_forced_groups": int(final_accounting["dropped_forced_groups"]),
                "teacher_input_tokens": int(final_accounting["teacher_input_tokens"]),
                "teacher_output_tokens": int(final_accounting["teacher_output_tokens"]),
                "aligned_sequences": int(final_accounting["aligned_sequences"]),
                "empty_alignments": int(final_accounting["empty_alignments"]),
                "teacher_ok": int(final_accounting["teacher_ok"]),
                **_failure_accounting_metadata(final_accounting),
                "temperature": knobs.temperature,
                "group_size": knobs.group_size,
                "prompts_per_step": prompts_per_step,
                "max_completion_len": knobs.max_completion,
                "multi_turn": multi_turn,
                "max_turns": max_turns if multi_turn else None,
                "episodes": int(final_accounting["episodes_seen"]) if multi_turn else None,
                "mean_turns_per_episode": (
                    int(final_accounting["mt_turn_records"])
                    / int(final_accounting["episodes_seen"])
                    if multi_turn and final_accounting["episodes_seen"]
                    else None
                ),
                # the engine length actually handed to vllm (prompt + completion), already clamped to
                # the model's own limit. the prompt filter is carved out of this same number.
                "vllm_max_model_len": max_model_len,
                # only single-turn text uses the fixed serial batcher; multimodal and multi-turn use
                # bridge threads. cap the reported batch by samples the step can produce.
                "opd_teacher_batch_size": (
                    min(
                        OPD_TEACHER_SCORING_CONCURRENCY, max(1, prompts_per_step * knobs.group_size)
                    )
                    if not multimodal and not multi_turn
                    else None
                ),
                "opd_teacher_workers": 1 if not multimodal and not multi_turn else None,
                "rollout_backend": "verl_vllm",
                "verl_version": "0.8.0",
                "verl_backend": "fsdp",
                "ulysses_sequence_parallel_size": gpu_count,
                # record whether the child can reset gdn state at packed boundaries; successful runs
                # upload no console, and failure here is silent contamination. None means non-gdn.
                "gdn_boundary_resets": gdn_hybrid or None,
                "peak_gpu_gb": peak_gpu_gb,
                "warm_started": bool(warmstart_adapter),
                "resumed": bool(resume_step),
                "wandb_project": project_name if "wandb" in loggers else None,
                "wandb_run_name": experiment_name if "wandb" in loggers else None,
                # the sdk's link_wandb reads notes["wandb_url"]; trl gets it from the parent's live
                # wandb.run, verl from the child marker (see backend_common.render_wandb_link_shim).
                "wandb_url": wandb_link.get("wandb_url"),
                "wandb_id": wandb_link.get("wandb_id"),
            },
        )
    finally:
        bridge.close()


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
    _processed_resume_steps,
    _raise_verl_failure,
    _read_classified_failure_fallback,
    _read_failure_fallback_records,
    _reconcile_no_signal_notification_failure,
    _reconcile_score_delivery_failure,
    _restore_verl_resume,
    _stage_retry_contract,
)

# hydra overrides, child env, and parquet writing, implemented in `.train.opd.overrides`.
# re-exported because the opd tests import these names from `opd_train`.
from flash.engine.worker.train.opd.overrides import (  # noqa: E402,F401
    _OPD_PARQUET_WRITE_BATCH_ROWS,
    _build_opd_child_env,
    _opd_multimodal_parquet_features,
    _render_opd_sitecustomize,
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
