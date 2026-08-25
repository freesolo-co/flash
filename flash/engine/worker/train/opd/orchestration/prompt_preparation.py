"""parent-side OPD prompt rendering and tokenizer preparation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.runtime.state as _worker_state
import flash.engine.worker.train.entry.backend_common as _backend
import flash.engine.worker.train.opd.bridging.prompts as _opd_prompts
from flash.content.thinking import messages_for_chat_template
from flash.engine.support.huggingface import model_revision_kwargs
from flash.engine.worker.entry.opd import _thinking_prefill_text
from flash.engine.worker.io.heartbeat import liveness_heartbeat
from flash.engine.worker.model.decoding import prompt_opens_thinking
from flash.engine.worker.runtime.rng import seed_training_rngs
from flash.engine.worker.train.core.child.glue import (
    parent_image_digests,
    validate_glue_template,
    validate_transcript_messages,
)
from flash.engine.worker.train.opd.orchestration.state import (
    _BridgePrompt,
    _OpdRequest,
    _PromptState,
)


def render_prompt_rows(request: _OpdRequest) -> tuple[list[tuple[Any, Any]], bool]:
    from flash.content.multimodal import record_has_images

    seed_training_rngs(_worker_state.SEED)
    train = list(request.env.dataset())
    if not train:
        raise RuntimeError("opd environment dataset is empty")
    max_examples = int(getattr(request.spec.train, "max_examples", 0) or 0) if request.spec else 0
    if max_examples > 0:
        train = train[:max_examples]
    scanned = [0]
    with liveness_heartbeat("opd_prompt_scan", progress=lambda: scanned[0]):
        prompt_rows = []
        for example in train:
            prompt_rows.append((example, request.env.prompt_messages(example)))
            scanned[0] += 1
    multimodal = bool(getattr(request.env, "image_observations", False)) or any(
        record_has_images(example, messages) for example, messages in prompt_rows
    )
    random.Random(_worker_state.SEED).shuffle(prompt_rows)
    return prompt_rows, multimodal


def _prepare_prompt_messages(
    example: dict,
    messages: list[dict],
    *,
    multi_turn: bool,
    package_root: str | None,
) -> tuple[list[dict], tuple[str, ...]]:
    from flash.content.multimodal import normalize_prompt_images, record_has_images

    if record_has_images(example, messages):
        normalized = normalize_prompt_images(example, messages, package_root)
        if multi_turn:
            validate_transcript_messages(
                normalized.messages,
                source="environment initial prompt",
                allow_content_blocks=True,
            )
        return normalized.messages, tuple(normalized.descriptors)
    if multi_turn:
        messages = validate_transcript_messages(messages, source="environment initial prompt")
    return messages, ()


def prepare_prompts(
    request: _OpdRequest,
    prompt_rows: list[tuple[Any, Any]],
    multimodal: bool,
    capability: str,
    control_panel_url: str,
) -> _PromptState:
    from flash.content.multimodal import image_teacher_prompt_messages
    from flash.engine.worker.teacher.client import TeacherClient

    teacher = TeacherClient(capability, control_panel_url, request.knobs.teacher_model)
    processor = None
    if multimodal:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            request.model_id,
            trust_remote_code=True,
            **model_revision_kwargs(request.model_revision),
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = _worker_hf.load_tokenizer(request.model_id, revision=request.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    thinking_prefill = _thinking_prefill_text(tokenizer)
    from flash.engine.plan.vram import opd_rollout_seq_len

    requested_len = opd_rollout_seq_len(
        request.knobs.max_length,
        request.knobs.max_completion,
        bool(_worker_state.THINKING),
    )
    max_model_len = _backend.clamp_engine_len(
        requested_len,
        _backend.model_max_position_embeddings(request.model_id, request.model_revision),
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
        validate_glue_template(tokenizer, thinking=bool(_worker_state.THINKING))
    prompts: list[_BridgePrompt] = []
    dropped_long = 0
    package_root_value = getattr(request.env, "package_root", None)
    package_root = str(Path(package_root_value).resolve()) if package_root_value else None
    prepped = [0]
    thinking_semantics_set = False
    with liveness_heartbeat("opd_image_prep", progress=lambda: prepped[0]):
        for example, messages in prompt_rows:
            prepped[0] += 1
            student_messages, image_descriptors = _prepare_prompt_messages(
                example,
                messages,
                multi_turn=request.multi_turn,
                package_root=package_root,
            )
            student_messages = messages_for_chat_template(student_messages)
            if image_descriptors:
                assert processor is not None
                teacher_messages = image_teacher_prompt_messages(
                    student_messages, len(image_descriptors)
                )
                prompt_ids, rendered_prompt = _opd_prompts._processor_expanded_prompt(
                    processor,
                    student_messages,
                    image_descriptors,
                    package_root,
                    enable_thinking=bool(_worker_state.THINKING),
                )
            else:
                teacher_messages = student_messages
                if processor is not None:
                    prompt_ids, rendered_prompt = _opd_prompts._processor_expanded_prompt(
                        processor,
                        student_messages,
                        (),
                        package_root,
                        enable_thinking=bool(_worker_state.THINKING),
                    )
                else:
                    rendered_prompt = tokenizer.apply_chat_template(
                        student_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=_worker_state.THINKING,
                        preserve_thinking=False,
                    )
                    prompt_ids = _opd_prompts._normalize_prompt_ids(
                        tokenizer.apply_chat_template(
                            student_messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=_worker_state.THINKING,
                            preserve_thinking=False,
                        )
                    )
            if len(prompt_ids) > prompt_budget:
                dropped_long += 1
                continue
            if not thinking_semantics_set:
                thinking = bool(_worker_state.THINKING)
                if hasattr(request.env, "thinking"):
                    request.env.thinking = thinking
                if hasattr(request.env, "prompt_opens_thinking"):
                    request.env.prompt_opens_thinking = thinking and prompt_opens_thinking(
                        rendered_prompt
                    )
                thinking_semantics_set = True
            prompts.append(
                _BridgePrompt(
                    student_messages=student_messages,
                    teacher_messages=teacher_messages,
                    prompt_ids=prompt_ids,
                    image_descriptors=image_descriptors,
                    package_root=package_root,
                    example=example if request.multi_turn else None,
                    image_digests=tuple(
                        parent_image_digests(processor, image_descriptors, package_root)
                    ),
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
        processor,
    )


__all__ = ["prepare_prompts", "render_prompt_rows"]
