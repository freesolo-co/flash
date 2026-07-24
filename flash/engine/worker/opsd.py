"""On-policy self-distillation training path for algorithm="opsd".

The student samples one completion from the base model plus its trainable LoRA adapter. The teacher is
the same base model with that adapter disabled, conditioned on the environment's gold completion, and
teacher-forces the student's exact rollout token ids. The objective is the paper's clipped
full-vocabulary forward KL with gradients through the student only.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from flash.engine.recipe import RECIPE
from flash.engine.steps import on_policy_steps, resolve_update_horizon
from flash.engine.vram import opd_completion_len
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.opd import (
    _generate_many_vllm,
    _render_prompt_ids,
    _save_adapter,
    _student_model,
)
from flash.engine.worker.opd import (
    _publish_opd_deployable as _publish_opsd_deployable,
)
from flash.engine.worker.opd_gkd import _generation_eos_ids
from flash.engine.worker.opd_vllm import (
    OpdVllmRolloutEngine,
)
from flash.engine.worker.opd_vllm import (
    opd_lora_rank as _opd_lora_rank,
)
from flash.engine.worker.opd_vllm import (
    opd_vllm_kwargs as _opd_vllm_kwargs,
)
from flash.engine.worker.perf import (
    free_gpu,
    gpu_diagnostics,
    grad_checkpointing_on,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.engine.worker.rng import backend_seed, rollout_request_seed, seed_training_rngs

# paper recipe (siyan-zhao/OPSD, scripts/run_opsd_4b.sh + opsd_trainer.py). forward kl (beta 0)
# with a per-vocab clamp(max=token_clip) that keeps negative summands, softmax temperature 1.1,
# on-policy rollouts at temperature 1.1 / top_p 0.95, adamw lr 5e-6, and gradient clipping at 0.1.
_OPSD_TEMPERATURE = 1.1
_OPSD_CLIP_TAU = 0.05
_OPSD_LEARNING_RATE = 5e-6
_OPSD_ROLLOUT_TEMPERATURE = 1.1
_OPSD_ROLLOUT_TOP_P = 0.95
_OPSD_MAX_GRAD_NORM = 0.1


@dataclass(frozen=True)
class OpsdKnobs:
    epochs: int
    learning_rate: float
    temperature: float
    top_p: float
    max_completion: int
    prompts_per_step: int
    max_steps: int
    max_length: int
    stop_sequences: tuple[str, ...]


@dataclass(frozen=True)
class _PromptRecord:
    example: object
    prompt_ids: list[int]
    teacher_prompt_ids: list[int]


def _resolve_opsd_knobs() -> OpsdKnobs:
    train = _w.JOB_SPEC.train if _w.JOB_SPEC else None
    defaults = RECIPE.opd

    def opt(name, default):
        value = getattr(train, name, None) if train else None
        return default if value is None else value

    return OpsdKnobs(
        epochs=int(opt("epochs", defaults.num_epochs)),
        learning_rate=float(opt("learning_rate", _OPSD_LEARNING_RATE)),
        temperature=float(opt("temperature", _OPSD_ROLLOUT_TEMPERATURE)),
        top_p=_OPSD_ROLLOUT_TOP_P,
        max_completion=opd_completion_len(opt("max_completion_tokens", 0), _w.THINKING),
        prompts_per_step=int(opt("batch_size", defaults.prompts_per_step)),
        max_steps=int(opt("max_steps", 0) or 0),
        max_length=int(opt("max_context_tokens", 0) or 0),
        stop_sequences=tuple(opt("stop_sequences", ()) or ()),
    )


def _completion_text(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    return "\n".join(part for part in parts if part).strip()


def _privileged_teacher_messages(prompt_messages: object, gold_messages: object) -> list[dict]:
    gold = _completion_text(gold_messages)
    if not gold:
        raise RuntimeError(
            "opsd requires a nonempty gold completion from env.sft_completion(example) for every "
            "selected row"
        )
    context = (
        "=== Reference Solution Begin ===\n"
        f"{gold}\n"
        "=== Reference Solution End ===\n"
        "Do not copy the reference solution verbatim. Derive the answer independently, reason step by "
        "step, and then produce the final answer."
    )
    return [*list(prompt_messages), {"role": "user", "content": context}]


def _opsd_kl_loss(
    student_logits,
    teacher_logits,
    completion_mask,
    *,
    temperature: float = _OPSD_TEMPERATURE,
    clip_tau: float = _OPSD_CLIP_TAU,
):
    """Compute the paper's clipped full-vocabulary forward KL over non-pad completion positions.

    Forward KL is ``KL(p_teacher || p_student) = sum_v p_T(v) (log p_T(v) - log p_S(v))``, gradients
    flowing through the student only. This is the objective the paper adopts (its divergence
    ablation ranks forward KL best and reverse KL below the baseline).

    Each per-vocabulary summand is clipped from above at ``clip_tau`` only; negative summands are
    kept, so the returned loss can be negative by design. This is verbatim to the released
    ``opsd_trainer.py`` (``jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none",
    log_target=True)`` then ``jsd.clamp(max=token_clip)`` then ``jsd.sum() / mask.sum()``), and it
    keeps high-probability style tokens from dominating the gradient signal.
    """
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if clip_tau <= 0:
        raise ValueError("clip_tau must be > 0")
    mask = completion_mask.to(device=student_logits.device, dtype=torch.bool)
    if mask.shape != student_logits.shape[:-1] or teacher_logits.shape != student_logits.shape:
        raise ValueError("opsd logits and completion mask shapes do not align")
    if not bool(mask.any()):
        raise ValueError("opsd loss requires at least one completion position")

    student = student_logits.float()
    teacher = teacher_logits.detach().to(device=student.device, dtype=torch.float32)
    safe_mask = mask.unsqueeze(-1)
    student = torch.where(safe_mask, student, torch.zeros_like(student))
    teacher = torch.where(safe_mask, teacher, torch.zeros_like(teacher))
    logp_student = torch.log_softmax(student / temperature, dim=-1)
    logp_teacher = torch.log_softmax(teacher / temperature, dim=-1)
    contributions = logp_teacher.exp() * (logp_teacher - logp_student)
    # upper-only clip: cap each per-vocab summand at clip_tau but keep negative summands, so the
    # summed loss can legitimately go negative. this stops high-probability style tokens from
    # dominating the objective and matches the released opsd clamp exactly.
    per_position = contributions.clamp_max(clip_tau).sum(dim=-1)
    # global equal-token average: sum clipped contributions over every completion position in the
    # batch, divide by the total completion-token count (not a per-example mean).
    return per_position.masked_select(mask).sum() / mask.sum()


def _forward_completion_logits(model, prefix_ids, completion_ids, device):
    import torch

    sequence = [*prefix_ids, *completion_ids]
    if not prefix_ids or not completion_ids:
        raise ValueError("opsd forward requires nonempty prompt and completion token ids")
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = attention_mask.cumsum(dim=-1) - 1
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    logits = output.logits
    start = len(prefix_ids) - 1
    end = start + len(completion_ids)
    return logits[:, start:end, :]


def _opsd_sample_loss(model, record: _PromptRecord, completion_ids, device):
    import torch

    model.eval()
    with model.disable_adapter(), torch.no_grad():
        teacher_logits = _forward_completion_logits(
            model, record.teacher_prompt_ids, completion_ids, device
        )
    model.train()
    student_logits = _forward_completion_logits(model, record.prompt_ids, completion_ids, device)
    completion_mask = torch.ones(
        student_logits.shape[:-1], dtype=torch.bool, device=student_logits.device
    )
    return _opsd_kl_loss(student_logits, teacher_logits, completion_mask)


def run_opsd():
    import torch

    from flash.engine.worker.hf import load_tokenizer, model_revision_kwargs
    from flash.multimodal import record_has_images, validate_multimodal_training

    seed_training_rngs(_w.SEED)
    env = _w.require_active_env()
    if getattr(env, "is_tool_env", False) or getattr(env, "multi_turn", False):
        raise RuntimeError("opsd phase 1 supports only single-turn, text-only environments")
    train = env.dataset()
    if not train:
        raise RuntimeError("opsd: the environment dataset is empty")

    knobs = _resolve_opsd_knobs()
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    model_revision = getattr(_w.JOB_SPEC, "model_revision", "") if _w.JOB_SPEC else ""
    max_examples = int(getattr(getattr(_w.JOB_SPEC, "train", None), "max_examples", 0) or 0)
    if max_examples > 0:
        train = train[:max_examples]
    random.Random(_w.SEED).shuffle(train)

    prompt_rows = []
    for example in train:
        prompt_messages = env.prompt_messages(example)
        if record_has_images(example, prompt_messages):
            validate_multimodal_training(model_id, "opsd")
        teacher_messages = _privileged_teacher_messages(
            prompt_messages, env.sft_completion(example)
        )
        prompt_rows.append((example, prompt_messages, teacher_messages))

    _w.heartbeat("opsd_start", gpu=gpu_diagnostics())
    wait_for_gpu(
        _w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None,
        gpu_type=_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else "",
    )
    setup_perf_backends()
    tok = load_tokenizer(model_id, revision=model_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if knobs.max_length:
        prompt_budget = knobs.max_length - knobs.max_completion
        if prompt_budget < 1:
            raise RuntimeError(
                "opsd: max_context_tokens must exceed the effective max_completion_tokens"
            )
    else:
        prompt_budget = RECIPE.opd.max_prompt_len

    examples: list[_PromptRecord] = []
    for example, prompt_messages, teacher_messages in prompt_rows:
        prompt_ids = _render_prompt_ids(tok, prompt_messages, thinking=_w.THINKING)
        teacher_prompt_ids = _render_prompt_ids(tok, teacher_messages, thinking=_w.THINKING)
        if len(prompt_ids) <= prompt_budget and len(teacher_prompt_ids) <= prompt_budget:
            examples.append(
                _PromptRecord(
                    example=example,
                    prompt_ids=list(prompt_ids),
                    teacher_prompt_ids=list(teacher_prompt_ids),
                )
            )
    if not examples:
        raise RuntimeError(
            f"opsd: every student or privileged teacher prompt exceeds the {prompt_budget}-token budget"
        )

    prompts_per_step = min(knobs.prompts_per_step, len(examples))
    derived_steps = on_policy_steps(
        epochs=knobs.epochs,
        prompt_count=len(examples),
        prompts_per_step=prompts_per_step,
    )
    steps = resolve_update_horizon(derived_steps, knobs.max_steps)

    download_seconds = (
        _w.prefetch_model(model_id, revision=model_revision)
        if model_revision
        else _w.prefetch_model(model_id)
    )
    seed_training_rngs(_w.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = optimal_attn_impl()
    model_init_kwargs = {"dtype": torch.bfloat16, **model_revision_kwargs(model_revision)}
    if attn:
        model_init_kwargs["attn_implementation"] = attn
    model, rollout_model_source = _student_model(model_id, model_init_kwargs, device)
    seq_cap = knobs.max_length or (RECIPE.opd.max_prompt_len + knobs.max_completion)
    if grad_checkpointing_on(model_id, seq_cap, revision=model_revision):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.config.use_cache = False
    free_gpu()

    eos_ids = _generation_eos_ids(model, tok)
    lora_rank = _opd_lora_rank(
        model, getattr(_w.JOB_SPEC.train, "lora_rank", 32) if _w.JOB_SPEC else 32
    )
    vllm_kwargs = _opd_vllm_kwargs(
        model_id,
        knobs,
        seq_cap,
        prompts_per_step=prompts_per_step,
        lora_rank=lora_rank,
        model_revision=model_revision,
    )
    rollout = OpdVllmRolloutEngine(
        model_source=rollout_model_source,
        model_revision=model_revision,
        max_model_len=seq_cap,
        temperature=knobs.temperature,
        top_p=knobs.top_p,
        stop_sequences=knobs.stop_sequences,
        eos_token_ids=tuple(sorted(eos_ids)),
        lora_rank=lora_rank,
        seed=backend_seed(_w.SEED),
        **vllm_kwargs,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=knobs.learning_rate,
    )
    rollout.sync_from_model(model)

    out_dir = f"/tmp/opsd_seed{_w.SEED}"
    adapter_dir = f"{out_dir}/adapter"
    os.makedirs(out_dir, exist_ok=True)
    loss_curve: list[float] = []
    generated_tokens = 0
    train_started = time.time()
    try:
        for step in range(steps):
            batch = [
                examples[(step * prompts_per_step + offset) % len(examples)]
                for offset in range(prompts_per_step)
            ]
            prompt_ids_batch = [record.prompt_ids for record in batch]
            request_seeds = [
                rollout_request_seed(_w.SEED, step * prompts_per_step + offset)
                for offset in range(len(batch))
            ]
            generations = _generate_many_vllm(
                rollout,
                tok,
                prompt_ids_batch,
                knobs,
                eos_ids,
                max_tokens=knobs.max_completion,
                request_seeds=request_seeds,
            )
            usable = [
                (record, generation)
                for record, generation in zip(batch, generations, strict=True)
                if not (generation.truncated or generation.skip or not generation.completion_ids)
            ]
            if not usable:
                # every student rollout truncated before naturally terminating, so this step has
                # no teacher-forceable target (opsd distills only over naturally-completed
                # rollouts). on long-trace thinking envs an occasional all-truncate step is
                # expected, so skip it (no optimizer update) instead of aborting the whole run.
                _w.heartbeat("opsd_step_skipped", step=step + 1, reason="all_rollouts_truncated")
                continue
            optimizer.zero_grad(set_to_none=True)
            # per-sample backward with gradient accumulation: mathematically identical to
            # torch.stack(losses).mean().backward() (grad of a mean == mean of the grads), but it
            # holds only one sample's dense-vocab autograd graph at a time instead of every usable
            # rollout's at once. thinking-mode opsd emits long completions, so each sample's
            # [completion, vocab] teacher/student/contribution tensors are large; accumulating them
            # across a full step overflows even the largest single gpu. dividing each sample loss by
            # the usable count keeps the effective objective the batch mean.
            usable_count = len(usable)
            step_loss = 0.0
            for record, generation in usable:
                generated_tokens += generation.gen_tokens
                sample_loss = (
                    _opsd_sample_loss(model, record, list(generation.completion_ids), device)
                    / usable_count
                )
                sample_loss.backward()
                step_loss += float(sample_loss.detach())
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, _OPSD_MAX_GRAD_NORM)
            optimizer.step()
            rollout.sync_from_model(model)
            loss_curve.append(step_loss)
            _w.heartbeat("opsd_step", step=step + 1, loss=loss_curve[-1])

        if not loss_curve:
            # every step skipped: all student rollouts truncated on every step, so no optimizer
            # update ran and the adapter is untrained. this only happens on a misconfigured env
            # (e.g. max_completion_tokens far below the trace length), so fail loud instead of
            # silently publishing an untrained adapter.
            raise RuntimeError(
                f"opsd trained no step across {steps} steps: every step's student rollouts "
                "truncated before terminating (max_completion_tokens likely too low for this env)"
            )

        train_wall = time.time() - train_started
        _save_adapter(model, tok, adapter_dir)
        _publish_opsd_deployable(
            adapter_dir,
            steps,
            as_default=True,
            publish_checkpoint=False,
        )
        _w.heartbeat("opsd_trained", step=steps, train_wall=train_wall, gpu=gpu_diagnostics())
        _w.write_train_meta(
            phase="opsd",
            step=steps,
            adapter_dir=adapter_dir,
            model_id=model_id,
            train_wall=train_wall,
            setup_seconds=0.0,
            train_tokens=0,
            generated_tokens=generated_tokens,
            notes={
                "steps": steps,
                "epochs": knobs.epochs,
                "retained_prompts": len(examples),
                "method": "opsd",
                "teacher": "frozen_base_adapter_disabled",
                "objective": "forward_kl",
                "temperature": _OPSD_TEMPERATURE,
                "clip_tau": _OPSD_CLIP_TAU,
                "loss_curve": loss_curve,
                "download_seconds": download_seconds,
                "thinking": _w.THINKING,
                "prompts_per_step": prompts_per_step,
                "max_completion_len": knobs.max_completion,
                "rollout_backend": "vllm",
            },
        )
    finally:
        rollout.close()
        free_gpu(model)
