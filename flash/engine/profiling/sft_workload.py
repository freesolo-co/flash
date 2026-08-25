"""shared sft tokenization and workload construction for estimates and training."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

from flash.content.multimodal import NormalizedImages
from flash.content.thinking import messages_for_chat_template
from flash.engine.plan.recipe import RECIPE
from flash.engine.plan.steps import resolve_update_horizon, sft_update_steps
from flash.engine.profiling.sft_image_rows import (
    estimate_sft_image_row,
    process_sft_image_row,
)
from flash.engine.profiling.workload_profile import (
    SftWorkloadProfile,
    horizon_row_count,
    marked_reasoning_end,
    reasoned_assistant_turns,
    reasoning_marker_prefix,
    reasoning_markers,
    reasoning_warning_rows,
    rendered_reasoning_loss_warning,
    sft_sample_policy,
    strip_reasoning_markers,
    unpacked_batch_warning,
    with_marked_reasoning,
)
from flash.engine.worker.entry.sft import (
    _pretokenize_completion_only,
    _reject_image_completion,
    has_real_target,
    select_sft_examples,
)
from flash.engine.worker.model.packing import (
    gdn_packing_contract_available,
    probe_is_gdn_hybrid,
    probe_is_pure_attention,
)


@dataclass
class PreparedSftWorkload:
    rows: list[dict[str, Any]]
    profile: SftWorkloadProfile
    multimodal: bool
    tokenizer: Any
    processor: Any | None
    sampled_texts: list[str]
    multiturn_targets: int
    coerced_singleturn_targets: int
    role_aware_multiturn_targets: int
    fallback_multiturn_targets: int
    authored_reasoning_turns: int
    rendered_reasoning_spans: int
    truncated_reasoning_spans: int


_ImageRowResult = tuple[list[int], list[int], bytes, int, bool]


class _NormalizeSftImageRow(Protocol):
    def __call__(
        self,
        record: dict,
        messages: list[dict],
        package_root: str | Path | None,
    ) -> NormalizedImages: ...


class _TokenizeSftImageRow(Protocol):
    def __call__(
        self,
        prompt_messages: list[dict],
        completion_messages: list[dict],
        descriptors: list[str],
        *,
        package_root: str | Path | None,
    ) -> _ImageRowResult: ...


@dataclass(frozen=True)
class _SftImagePipeline:
    """the bound normalization and tokenization path for every image row in one run."""

    normalize: _NormalizeSftImageRow
    tokenize: _TokenizeSftImageRow


@dataclass(frozen=True)
class _SftTokenization:
    """the tokenizer and optional structurally complete image pipeline for one run."""

    tokenizer: Any
    processor: Any | None
    image: _SftImagePipeline | None


def _resolve_sft_tokenization(
    spec,
    *,
    multimodal: bool,
    require_processor: bool,
    tokenizer_loader: Callable[[str, str], Any],
    processor_loader: Callable[[str, str], Any] | None,
    max_length: int,
) -> _SftTokenization:
    """resolve the tokenizer and bind the image-row tokenizer this run will use."""
    from flash.content.multimodal import normalize_prompt_images, validate_multimodal_training
    from flash.engine.profiling.image_tokens import ImageProfileValidationState, load_image_geometry

    if not multimodal:
        tokenizer = tokenizer_loader(spec.model, spec.model_revision)
        resolved = _SftTokenization(tokenizer, None, None)
    else:
        validate_multimodal_training(
            spec.model,
            "sft",
            getattr(spec.train, "teacher_model", None),
        )
        if require_processor:
            processor = (processor_loader or _default_processor_loader)(
                spec.model,
                spec.model_revision,
            )
            tokenizer = processor.tokenizer
            image = _SftImagePipeline(
                normalize=normalize_prompt_images,
                tokenize=partial(
                    process_sft_image_row,
                    processor,
                    max_length=max_length,
                    thinking=bool(spec.thinking),
                ),
            )
            resolved = _SftTokenization(tokenizer, processor, image)
        else:
            geometry = load_image_geometry(spec.model, spec.model_revision)
            tokenizer = tokenizer_loader(spec.model, spec.model_revision)
            image = _SftImagePipeline(
                # the estimator validates each descriptor through one cached, budgeted pass, so
                # normalization defers the eager decode instead of decoding every image twice.
                normalize=partial(normalize_prompt_images, defer_validation=True),
                tokenize=partial(
                    estimate_sft_image_row,
                    tokenizer,
                    geometry=geometry,
                    validation_state=ImageProfileValidationState(),
                    max_length=max_length,
                    thinking=bool(spec.thinking),
                ),
            )
            resolved = _SftTokenization(tokenizer, None, image)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return resolved


def _materialize_verl_images(
    descriptors: list[str],
    package_root,
    image_dir: str | None,
    row_index: int,
) -> list[str]:
    """decode image descriptors to files verl can load; estimate-only callers write none."""
    if image_dir is None:
        return []
    from flash.content.multimodal import decode_image_descriptors

    os.makedirs(image_dir, exist_ok=True)
    images = cast("list[Any]", decode_image_descriptors(descriptors, package_root))
    rows: list[str] = []
    try:
        for image_index, image in enumerate(images):
            path = Path(image_dir, f"row-{row_index}-image-{image_index}.png").resolve()
            image.save(path, format="PNG")
            rows.append(path.as_uri())
        return rows
    finally:
        for image in images:
            image.close()


def _default_processor_loader(model_id: str, revision: str):
    from transformers import AutoProcessor

    from flash.engine.support.huggingface import model_revision_kwargs

    return AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        **model_revision_kwargs(revision),
    )


def _packing_mode(
    model_id: str,
    revision: str,
    *,
    multimodal: bool,
    allow_packing: bool,
    packing_support: Callable[[str, str], tuple[str, bool]] | None,
) -> tuple[str, str]:
    if multimodal:
        return "exact-unpacked", "multimodal"
    if packing_support is not None:
        architecture_mode, supported = packing_support(model_id, revision)
    else:
        # use raising probes because these labels are frozen into the profile and compared in
        # `sft_train.py`. a swallowed timeout can mint `unsupported` and later fail parity against
        # `gdn-hybrid` despite identical training behavior.
        try:
            if probe_is_pure_attention(model_id, revision=revision):
                architecture_mode, supported = "pure-attention", True
            elif probe_is_gdn_hybrid(model_id, revision=revision):
                # a gdn hybrid packs only when the stack can reset the linear-attention recurrence
                # (fla's cu_seq_lens_q) and the causal conv (seq_idx) at example boundaries. without
                # both, transformers' fallbacks accept the kwargs and DISCARD them, so state bleeds
                # across examples inside a packed block while looking patched.
                #
                # the contract probe is device-independent on purpose: the control-plane estimate
                # and gpu worker must derive the same packing capability from the pinned stack, not
                # from whether cuda is locally visible. see gdn_packing_contract_available.
                supported = gdn_packing_contract_available(model_id, revision=revision)
                architecture_mode = "gdn-hybrid"
            else:
                architecture_mode, supported = "unsupported", False
        except Exception as e:
            raise RuntimeError(
                f"architecture probe for {model_id!r} could not resolve the model config, so the "
                "packing mode cannot be frozen into a workload profile"
            ) from e
    return ("packed" if allow_packing and supported else "exact-unpacked"), architecture_mode


def sft_tokens_for_updates(
    rows: list[dict[str, Any]],
    *,
    examples_per_update: int,
    updates: int,
    field: str,
) -> int:
    """Return the exact fixed-order token total consumed by a verl update horizon."""
    if not rows:
        raise ValueError("sft token accounting requires at least one row")
    batch_size = int(examples_per_update)
    if batch_size < 1:
        raise ValueError("examples_per_update must be >= 1")
    update_count = int(updates)
    if update_count < 0:
        raise ValueError("updates must be >= 0")
    # a horizon of zero updates consumed nothing; never round it up to one batch.
    if update_count == 0:
        return 0
    per_update = [
        sum(
            sum(row[field]) if field == "loss_mask" else len(row[field])
            for row in rows[start : start + batch_size]
        )
        for start in range(0, len(rows), batch_size)
    ]
    full_passes, remainder = divmod(update_count, len(per_update))
    return full_passes * sum(per_update) + sum(per_update[:remainder])


def sft_max_length(spec) -> int:
    """The context window the rows are truncated at: the authored cap, else the cap for the mode.

    This single profile `max_length` reaches both trainer and quote; neither may re-derive it.
    """
    authored = spec.train.max_context_tokens
    if authored is not None:
        return int(authored)
    return int(RECIPE.sft.max_seq_len_thinking if spec.thinking else RECIPE.sft.max_seq_len)


@dataclass(frozen=True)
class _TokenizedSftRows:
    row_by_index: dict[int, dict[str, Any]]
    untruncated_by_index: dict[int, int]
    sampled_texts: list[str]
    multiturn_targets: int
    coerced_singleturn_targets: int
    # multi-turn rows only: present means the row has a multi-turn target, and the value says whether
    # its mask was role-aware or fell back to the contiguous span. one map rather than a set plus a
    # parallel dict, so a row can never be counted as multi-turn while its mask status is missing.
    multiturn_mask_applied: dict[int, bool]
    reasoning_by_index: dict[int, _RowReasoning]
    dropped: int


@dataclass(frozen=True)
class _RetainedSftRows:
    rows: list[dict[str, Any]]
    untruncated_lengths: list[int]
    authored_reasoning_turns: int
    rendered_reasoning_spans: int
    truncated_reasoning_spans: int
    role_aware_multiturn_targets: int
    fallback_multiturn_targets: int
    dropped: int
    # kept PER ROW, in the retained order, so the warning can be bounded to the rows the optimizer
    # actually consumes. the totals above describe the whole retained dataset, which is what the
    # profile records; a run stopped early by max_steps trains on a PREFIX of these rows.
    row_reasoning: list[_RowReasoning]


@dataclass(frozen=True)
class _SftTokenMeasurements:
    real_tokens: int
    supervised_tokens: int
    padded_compute_tokens: int
    realized_max_length: int
    untruncated_max_length: int
    truncated_rows: int


@dataclass(frozen=True)
class _SftStepHorizon:
    examples_per_update: int
    packed_blocks: int
    derived_steps: int
    authoritative_steps: int
    authoritative_real_tokens: int
    authoritative_supervised_tokens: int


def _sft_completion_with_provenance(env, example: dict) -> tuple[list[dict], bool]:
    completion_with_provenance = getattr(env, "sft_completion_with_provenance", None)
    if callable(completion_with_provenance):
        return completion_with_provenance(example)
    return env.sft_completion(example), False


def _encoded_length(tokenizer, text: str) -> int:
    """Token count of ``text``, for either the batched or unbatched ``input_ids`` shape.

    A tokenizer handed a single string may answer with one id list or with a batch holding one
    row. Reading ``len()`` off the batch would count ROWS -- always 1 -- and silently declare every
    span to be within the cap.
    """
    ids = tokenizer(text)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


@dataclass(frozen=True)
class _RowReasoning:
    """One row's reasoning accounting, with the two causes of loss kept apart.

    They have opposite remedies: reasoning the TEMPLATE dropped is fixed by splitting a K-turn
    transcript into K single-turn rows, while reasoning the CAP cut is fixed by raising
    ``max_context_tokens``. Folding both into one survivor count makes the warning prescribe
    restructuring for a dataset whose only problem is that the row is too long.
    """

    authored_turns: int
    rendered_spans: int
    truncated_spans: int


def _row_reasoning(
    prompt_messages: list[dict],
    completion_messages: list[dict],
    *,
    render: Callable[[list[dict]], str],
    tokenizer,
    max_length: int,
) -> _RowReasoning:
    """One row's authored reasoning, and how much of it reaches the loss.

    The row is rendered once with each reasoning-authoring turn's reasoning marked. A turn survives
    exactly when its marker reaches the render: a marker rides only inside text the template chose
    to keep as THIS turn's reasoning, so a ``<think>`` an answer merely quotes carries none and a
    span in a system or user message -- never supervised, since the split is assistant-only -- cannot
    be credited either. Counting rendered tags instead gets both wrong, and scores the total-loss
    case as one survivor because the template injects an empty block on trailing assistant turns.

    A survivor also has to fit inside ``max_length``: the cap slices the token row, so a block past
    it never reaches the loss. Truncation is judged per turn rather than per row, because the cap
    usually cuts the answer tail while leaving an earlier reasoning block whole.

    Markers are stripped before measuring, so the length charged against the cap is the real render's
    rather than one inflated by the measurement's own bytes.
    """
    authored = reasoned_assistant_turns(completion_messages)
    if not authored:
        # nothing authored means nothing to lose, and the marked render would equal the full one.
        # skipped rather than rendered so a dataset with no reasoning pays nothing for this check.
        return _RowReasoning(0, 0, 0)
    prefix = reasoning_marker_prefix(render([*prompt_messages, *completion_messages]))
    marked = render([*prompt_messages, *with_marked_reasoning(completion_messages, prefix)])
    rendered = 0
    truncated = 0
    for marker in reasoning_markers(completion_messages, prefix):
        end = marked_reasoning_end(marked, marker)
        if end is None:
            continue
        rendered += 1
        through = strip_reasoning_markers(marked[:end], prefix)
        if _encoded_length(tokenizer, through) > max_length:
            truncated += 1
    return _RowReasoning(authored, rendered, truncated)


def _text_sft_row_spec(
    spec,
    prompt_messages: list[dict],
    completion_messages: list[dict],
    *,
    tokenizer,
    render_transcript: Callable[[list[dict]], str],
    max_length: int,
    row_index: int,
    multiturn: bool,
) -> tuple[str, _RowReasoning, dict[str, Any]]:
    normalized_prompt = messages_for_chat_template(prompt_messages)
    normalized_completion = messages_for_chat_template(completion_messages)
    normalized_source = [*normalized_prompt, *normalized_completion]
    text = render_transcript(normalized_source)
    prompt_text = tokenizer.apply_chat_template(
        normalized_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=spec.thinking,
        preserve_thinking=False,
    )
    reasoning = _row_reasoning(
        prompt_messages,
        completion_messages,
        render=render_transcript,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return (
        text,
        reasoning,
        {
            "text": text,
            "prompt_text": prompt_text,
            "target_messages": normalized_completion,
            "source_messages": normalized_source,
            "template_kwargs": {"enable_thinking": spec.thinking, "preserve_thinking": False},
            # measurement carried alongside the row rather than in it: the parquet row is built from
            # an explicit key list below, so neither key reaches verl.
            "row_index": row_index,
            "multiturn": multiturn,
        },
    )


def _tokenize_prompt_rows(
    spec,
    prompt_rows: list[tuple[Any, list[dict], list[dict], bool]],
    *,
    package_root,
    tokenizer,
    image: _SftImagePipeline | None,
    max_length: int,
    image_dir: str | None,
    record_has_images: Callable,
    text_only_prompt_messages: Callable,
) -> _TokenizedSftRows:
    """Compute rendered text and token rows before target filtering."""
    row_by_index: dict[int, dict[str, Any]] = {}
    # kept OUT of the row dicts: a row carries exactly the columns the parquet schema declares, and
    # an extra key would be dropped on the way to verl. this is measurement, not training input.
    untruncated_by_index: dict[int, int] = {}
    text_specs: list[dict[str, Any]] = []
    sampled_texts: list[str] = []
    multiturn_targets = 0
    coerced_singleturn_targets = 0
    multiturn_mask_applied: dict[int, bool] = {}
    # kept per row rather than summed here, for the same reason as `untruncated_by_index`: rows that
    # lose their whole completion to the cap are dropped below, and folding their reasoning into a
    # running total would report loss from rows the run never trains on against a retained-row
    # denominator. summed after filtering instead.
    reasoning_by_index: dict[int, _RowReasoning] = {}

    def render_transcript(messages: list[dict]) -> str:
        return tokenizer.apply_chat_template(
            messages_for_chat_template(messages),
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=spec.thinking,
            preserve_thinking=False,
        )

    for row_index, (
        example,
        prompt_messages,
        completion_messages,
        coerced_scalar_output,
    ) in enumerate(prompt_rows):
        has_images = record_has_images(example, prompt_messages)
        _reject_image_completion(completion_messages, image_bearing=has_images)
        # read before the image branch rewrites `completion_messages` to its text-only form.
        multiturn = len(completion_messages) > 1
        if multiturn:
            multiturn_targets += 1
        elif (
            coerced_scalar_output
            and len(completion_messages) == 1
            and completion_messages[0].get("role") == "assistant"
        ):
            coerced_singleturn_targets += 1
        if has_images:
            if image is None:
                raise RuntimeError("multimodal sft row has no image pipeline")
            normalized = image.normalize(example, prompt_messages, package_root)
            completion_messages = text_only_prompt_messages(completion_messages)
            (
                input_ids,
                loss_mask,
                multimodal_inputs,
                untruncated_length,
                assistant_mask_applied,
            ) = image.tokenize(
                normalized.messages,
                completion_messages,
                normalized.descriptors,
                package_root=package_root,
            )
            untruncated_by_index[row_index] = untruncated_length
            if multiturn:
                multiturn_mask_applied[row_index] = assistant_mask_applied
            row_by_index[row_index] = {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "images": _materialize_verl_images(
                    normalized.descriptors,
                    package_root,
                    image_dir,
                    row_index,
                ),
                "multimodal_inputs": multimodal_inputs,
            }
            text = render_transcript([*normalized.messages, *completion_messages])
            sampled_texts.append(text)
            # every image is in the prompt, so visual expansion shifts every completion position by
            # the same amount; charge that inflation before measuring reasoning against the cap.
            visual_inflation = max(0, untruncated_length - _encoded_length(tokenizer, text))
            reasoning_by_index[row_index] = _row_reasoning(
                normalized.messages,
                completion_messages,
                render=render_transcript,
                tokenizer=tokenizer,
                max_length=max_length - visual_inflation,
            )
        else:
            text, reasoning, text_spec = _text_sft_row_spec(
                spec,
                prompt_messages,
                completion_messages,
                tokenizer=tokenizer,
                render_transcript=render_transcript,
                max_length=max_length,
                row_index=row_index,
                multiturn=multiturn,
            )
            sampled_texts.append(text)
            reasoning_by_index[row_index] = reasoning
            text_specs.append(text_spec)

    dropped = 0
    if text_specs:
        kept_specs, tokenized_rows, text_dropped = _pretokenize_completion_only(
            text_specs,
            tokenizer,
            max_length,
        )
        dropped += text_dropped
        for spec_row, tokenized in zip(kept_specs, tokenized_rows, strict=True):
            row_index = spec_row["row_index"]
            input_ids = tokenized["input_ids"]
            untruncated_by_index[row_index] = tokenized["untruncated_length"]
            if spec_row["multiturn"]:
                multiturn_mask_applied[row_index] = tokenized["assistant_mask_applied"]
            row_by_index[row_index] = {
                "input_ids": input_ids,
                "loss_mask": tokenized["completion_mask"],
                "images": [],
                "multimodal_inputs": b"",
            }
    return _TokenizedSftRows(
        row_by_index=row_by_index,
        untruncated_by_index=untruncated_by_index,
        sampled_texts=sampled_texts,
        multiturn_targets=multiturn_targets,
        coerced_singleturn_targets=coerced_singleturn_targets,
        multiturn_mask_applied=multiturn_mask_applied,
        reasoning_by_index=reasoning_by_index,
        dropped=dropped,
    )


def _filter_retained_rows(tokenized: _TokenizedSftRows, tokenizer) -> _RetainedSftRows:
    """Compute the ordered rows that retain a real supervised target."""
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    rows = []
    retained_untruncated: list[int] = []
    authored_reasoning = 0
    rendered_reasoning = 0
    truncated_reasoning = 0
    role_aware_multiturn_targets = 0
    fallback_multiturn_targets = 0
    per_row: list[_RowReasoning] = []
    dropped = tokenized.dropped
    for row_index in sorted(tokenized.row_by_index):
        row = tokenized.row_by_index[row_index]
        if has_real_target(row["input_ids"], row["loss_mask"], special_ids):
            rows.append(row)
            # appended in lockstep with the row it measures, so the truncation counts below describe
            # the rows that are actually trained on rather than the ones that were dropped.
            retained_untruncated.append(tokenized.untruncated_by_index[row_index])
            if row_index in tokenized.multiturn_mask_applied:
                if tokenized.multiturn_mask_applied[row_index]:
                    role_aware_multiturn_targets += 1
                else:
                    fallback_multiturn_targets += 1
            # summed here for the same reason, so the reasoning-loss warning describes the rows the
            # run trains on. a dropped row contributes neither its authored reasoning nor its
            # survivors, which would otherwise be reported against a retained-row denominator.
            row_reasoning = tokenized.reasoning_by_index.get(row_index)
            if row_reasoning is not None:
                authored_reasoning += row_reasoning.authored_turns
                rendered_reasoning += row_reasoning.rendered_spans
                truncated_reasoning += row_reasoning.truncated_spans
            # appended for EVERY retained row, including one that authored nothing, so this list
            # stays index-aligned with ``rows``. a prefix of it is then the same prefix of rows.
            per_row.append(row_reasoning if row_reasoning is not None else _RowReasoning(0, 0, 0))
        else:
            dropped += 1
    if not rows:
        raise ValueError(
            "every SFT example has an empty completion after sft_max_len truncation "
            "(nothing to train on); increase sft_max_len or shorten the prompts"
        )
    return _RetainedSftRows(
        rows=rows,
        untruncated_lengths=retained_untruncated,
        authored_reasoning_turns=authored_reasoning,
        rendered_reasoning_spans=rendered_reasoning,
        truncated_reasoning_spans=truncated_reasoning,
        role_aware_multiturn_targets=role_aware_multiturn_targets,
        fallback_multiturn_targets=fallback_multiturn_targets,
        dropped=dropped,
        row_reasoning=per_row,
    )


def _measure_sft_tokens(
    rows: list[dict[str, Any]],
    untruncated_lengths: list[int],
    max_length: int,
) -> _SftTokenMeasurements:
    """Compute per-epoch token totals and truncation measurements."""
    real_tokens = sum(len(row["input_ids"]) for row in rows)
    supervised_tokens = sum(sum(int(item) for item in row["loss_mask"]) for row in rows)
    padded_compute_tokens = real_tokens
    realized_max_length = max(len(row["input_ids"]) for row in rows)
    # measured against the UNTRUNCATED encode, so it reports what the rows actually need rather
    # than what the cap allowed. realized_max_length cannot do this: it is taken after the slice,
    # so it saturates at max_length exactly when the cap binds and the censoring becomes invisible.
    # mirrors the rollout profile's truncated_rollouts/truncation_rate reasoning.
    untruncated_max_length = max(untruncated_lengths)
    truncated_rows = sum(1 for length in untruncated_lengths if length > max_length)
    if truncated_rows:
        print(
            f"warning: [train] max_context_tokens {max_length} truncated {truncated_rows} of "
            f"{len(rows)} sft rows; the longest row needs {untruncated_max_length} tokens. "
            "training on the truncated rows as configured; set max_context_tokens to at least "
            f"{untruncated_max_length} to keep every row whole.",
            file=sys.stderr,
        )
    return _SftTokenMeasurements(
        real_tokens=real_tokens,
        supervised_tokens=supervised_tokens,
        padded_compute_tokens=padded_compute_tokens,
        realized_max_length=realized_max_length,
        untruncated_max_length=untruncated_max_length,
        truncated_rows=truncated_rows,
    )


def _resolve_sft_step_horizon(
    rows: list[dict[str, Any]],
    *,
    effective_batch: int,
    packing_mode: str,
    epochs: int,
    max_steps: int,
) -> _SftStepHorizon:
    """Compute optimizer updates and exact tokens consumed by the resolved horizon."""
    # batch size follows the packing mode: `exact-unpacked` keeps one example per update, which is
    # what makes an unpacked gdn run boundary-safe (no packed neighbours to contaminate). a `packed`
    # gdn run has earned that mode through the boundary-reset contract above. this CPU-side choice
    # sets quoted and executed steps. the child probe controls verl layout, not batch size.
    examples_per_update = min(effective_batch, len(rows)) if packing_mode == "packed" else 1
    # packed_blocks is already the optimizer batches verl runs per epoch, so the horizon is one
    # update per block per epoch. do not divide by examples_per_update again.
    packed_blocks = math.ceil(len(rows) / examples_per_update)
    derived_steps = sft_update_steps(
        epochs=epochs,
        example_count=len(rows),
        examples_per_update=1,
        packed_block_count=packed_blocks,
    )
    authoritative_steps = resolve_update_horizon(derived_steps, max_steps)
    authoritative_real_tokens = sft_tokens_for_updates(
        rows,
        examples_per_update=examples_per_update,
        updates=authoritative_steps,
        field="input_ids",
    )
    authoritative_supervised_tokens = sft_tokens_for_updates(
        rows,
        examples_per_update=examples_per_update,
        updates=authoritative_steps,
        field="loss_mask",
    )
    return _SftStepHorizon(
        examples_per_update=examples_per_update,
        packed_blocks=packed_blocks,
        derived_steps=derived_steps,
        authoritative_steps=authoritative_steps,
        authoritative_real_tokens=authoritative_real_tokens,
        authoritative_supervised_tokens=authoritative_supervised_tokens,
    )


def _build_sft_profile(
    spec,
    *,
    producer_version: str,
    source_examples: int,
    selected_examples: int,
    retained: _RetainedSftRows,
    epochs: int,
    max_length: int,
    max_examples: int,
    packing_mode: str,
    architecture_mode: str,
    measurements: _SftTokenMeasurements,
    horizon: _SftStepHorizon,
) -> SftWorkloadProfile:
    """Compute the immutable profile from the retained workload measurements."""
    return SftWorkloadProfile(
        input_digest=spec.workload_profile_input_digest,
        producer_version=producer_version,
        tokenizer_revision=spec.model_revision,
        environment_id=spec.environment.id,
        environment_revision=spec.environment.resolved_sha,
        source_examples=source_examples,
        selected_examples=selected_examples,
        retained_examples=len(retained.rows),
        dropped_examples=retained.dropped,
        epochs=epochs,
        max_length=max_length,
        packing_mode=packing_mode,
        architecture_mode=architecture_mode,
        packed_blocks=horizon.packed_blocks,
        real_tokens_per_epoch=measurements.real_tokens,
        supervised_tokens_per_epoch=measurements.supervised_tokens,
        padded_compute_tokens_per_epoch=measurements.padded_compute_tokens,
        authoritative_real_tokens=horizon.authoritative_real_tokens,
        authoritative_supervised_tokens=horizon.authoritative_supervised_tokens,
        authoritative_compute_tokens=horizon.authoritative_real_tokens,
        realized_max_length=measurements.realized_max_length,
        untruncated_max_length=measurements.untruncated_max_length,
        truncated_examples=measurements.truncated_rows,
        examples_per_update=horizon.examples_per_update,
        derived_steps=horizon.derived_steps,
        authoritative_steps=horizon.authoritative_steps,
        packing_efficiency=measurements.real_tokens / measurements.padded_compute_tokens,
        sample_policy=sft_sample_policy(max_examples),
        # bounded to the rows the horizon reaches, not the whole retained dataset. these three
        # fields exist only to carry the reasoning-loss warning, and the warning is rendered TWICE:
        # here on the worker's stderr, and again by the CLI off the serialized profile, because
        # control-plane profiling runs server-side where its stderr never reaches the submitter.
        # bounding them at the source is what keeps those two renderings from disagreeing.
        #
        # `retained_examples` deliberately stays whole-dataset: it sizes the GPU allocation and
        # carries the profile invariant retained + dropped == selected.
        **_horizon_reasoning_fields(retained, horizon),
    )


class _HorizonReasoningFields(TypedDict):
    authored_reasoning_turns: int
    rendered_reasoning_spans: int
    truncated_reasoning_spans: int
    reasoning_rows: int


def _horizon_reasoning_fields(
    retained: _RetainedSftRows, horizon: _SftStepHorizon
) -> _HorizonReasoningFields:
    """The three reasoning counts, plus the row count they were totalled over.

    ``retained.row_reasoning`` is index-aligned with the retained rows, so the horizon's row count
    is also its prefix length. An empty prefix -- a horizon of zero updates -- totals zero and
    stays silent, which is correct: a run that performs no update cannot lose reasoning to one.

    The prefix length travels with the counts as ``reasoning_rows``. Both a bounded and an
    unbounded profile carry the same horizon inputs, so a reader that re-derived this could not
    tell which it held.
    """
    rows = horizon_row_count(
        len(retained.row_reasoning),
        examples_per_update=horizon.examples_per_update,
        updates=horizon.authoritative_steps,
    )
    consumed = retained.row_reasoning[:rows]
    return {
        "authored_reasoning_turns": sum(row.authored_turns for row in consumed),
        "rendered_reasoning_spans": sum(row.rendered_spans for row in consumed),
        "truncated_reasoning_spans": sum(row.truncated_spans for row in consumed),
        "reasoning_rows": rows,
    }


def _print_workload_warnings(
    profile: SftWorkloadProfile,
    retained: _RetainedSftRows,
    *,
    batch_size: object,
) -> None:
    """Emit the pre-allocation warnings a finished run's metrics would report too late.

    Both run for the control-plane estimate and again on the training worker, so they land in both
    logs -- before a GPU is paid for, while the config or dataset can still be changed.
    """
    # one example per update instead of the authored batch is an optimization-semantics change
    # whose only other trace is `notes["packing"]` in the finished run's metrics, which is not
    # visible until after the run is paid for. pass the authored batch_size rather than
    # `effective_batch`: the helper resolves None to the same recipe default but keeps the value's
    # source, so an omitted knob is not reported to the user as one they configured.
    warning = unpacked_batch_warning(
        packing_mode=profile.packing_mode,
        architecture_mode=profile.architecture_mode,
        examples_per_update=profile.examples_per_update,
        configured_batch_size=batch_size,
    )
    if warning:
        print(f"warning: [train] {warning}", file=sys.stderr)
    # read straight off the profile, which already carries the horizon-bounded counts. this line is
    # rendered twice -- here, and again by the CLI off the serialized profile -- and recomputing it
    # from `retained` would let the worker's stderr and the submitter's warning disagree about the
    # same run. one source, one answer.
    reasoning_warning = rendered_reasoning_loss_warning(
        authored_turns=profile.authored_reasoning_turns,
        rendered_spans=profile.rendered_reasoning_spans,
        truncated_spans=profile.truncated_reasoning_spans,
        rows=reasoning_warning_rows(profile),
    )
    if reasoning_warning:
        print(f"warning: [train] {reasoning_warning}", file=sys.stderr)


def prepare_sft_workload(
    spec,
    env,
    *,
    tokenizer_loader: Callable[[str, str], Any],
    producer_version: str,
    processor_loader: Callable[[str, str], Any] | None = None,
    image_dir: str | None = None,
    require_processor: bool = True,
    allow_packing: bool = True,
    packing_support: Callable[[str, str], tuple[str, bool]] | None = None,
    source_examples: int | None = None,
    examples_preselected: bool = False,
) -> PreparedSftWorkload:
    """render, tokenize, filter, and pack the exact rows consumed by sft."""
    from flash.content.multimodal import record_has_images, text_only_prompt_messages

    train_spec = spec.train
    max_length = sft_max_length(spec)
    epochs = int(train_spec.epochs if train_spec.epochs is not None else RECIPE.sft.num_epochs)
    effective_batch = int(
        train_spec.batch_size if train_spec.batch_size is not None else RECIPE.sft.effective_batch
    )
    max_examples = int(train_spec.max_examples or 0)
    max_steps = int(train_spec.max_steps or 0)

    source = list(env.dataset())
    selected = (
        source if examples_preselected else select_sft_examples(source, max_examples, spec.seed)
    )
    source_count = len(source) if source_examples is None else int(source_examples)
    if source_count < len(selected):
        raise ValueError("source_examples cannot be smaller than the selected sft sample")
    prompt_rows = []
    for example in selected:
        prompt_messages = env.prompt_messages(example)
        completion_messages, coerced_scalar_output = _sft_completion_with_provenance(env, example)
        prompt_rows.append((example, prompt_messages, completion_messages, coerced_scalar_output))
    package_root = getattr(env, "package_root", None)
    multimodal = any(
        record_has_images(example, prompt_messages)
        for example, prompt_messages, _completion, _used_fallback in prompt_rows
    )
    tokenization = _resolve_sft_tokenization(
        spec,
        multimodal=multimodal,
        require_processor=require_processor,
        tokenizer_loader=tokenizer_loader,
        processor_loader=processor_loader,
        max_length=max_length,
    )
    tokenizer = tokenization.tokenizer

    tokenized = _tokenize_prompt_rows(
        spec,
        prompt_rows,
        package_root=package_root,
        tokenizer=tokenizer,
        image=tokenization.image,
        max_length=max_length,
        image_dir=image_dir,
        record_has_images=record_has_images,
        text_only_prompt_messages=text_only_prompt_messages,
    )
    retained = _filter_retained_rows(tokenized, tokenizer)
    packing_mode, architecture_mode = _packing_mode(
        spec.model,
        spec.model_revision,
        multimodal=multimodal,
        allow_packing=allow_packing,
        packing_support=packing_support,
    )
    measurements = _measure_sft_tokens(
        retained.rows,
        retained.untruncated_lengths,
        max_length,
    )
    horizon = _resolve_sft_step_horizon(
        retained.rows,
        effective_batch=effective_batch,
        packing_mode=packing_mode,
        epochs=epochs,
        max_steps=max_steps,
    )
    profile = _build_sft_profile(
        spec,
        producer_version=producer_version,
        source_examples=source_count,
        selected_examples=len(selected),
        retained=retained,
        epochs=epochs,
        max_length=max_length,
        max_examples=max_examples,
        packing_mode=packing_mode,
        architecture_mode=architecture_mode,
        measurements=measurements,
        horizon=horizon,
    )
    _print_workload_warnings(profile, retained, batch_size=train_spec.batch_size)
    return PreparedSftWorkload(
        rows=retained.rows,
        profile=profile,
        multimodal=multimodal,
        tokenizer=tokenizer,
        processor=tokenization.processor,
        sampled_texts=tokenized.sampled_texts,
        multiturn_targets=tokenized.multiturn_targets,
        coerced_singleturn_targets=tokenized.coerced_singleturn_targets,
        role_aware_multiturn_targets=retained.role_aware_multiturn_targets,
        fallback_multiturn_targets=retained.fallback_multiturn_targets,
        authored_reasoning_turns=retained.authored_reasoning_turns,
        rendered_reasoning_spans=retained.rendered_reasoning_spans,
        truncated_reasoning_spans=retained.truncated_reasoning_spans,
    )
