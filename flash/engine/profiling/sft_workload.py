"""Shared exact SFT preprocessing for profile and training workers."""

from __future__ import annotations

import io
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash.engine.plan.recipe import RECIPE
from flash.engine.plan.steps import resolve_update_horizon, sft_update_steps
from flash.engine.profiling.workload_profile import (
    SftWorkloadProfile,
    sft_sample_policy,
    unpacked_batch_warning,
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


def _serialize_multimodal_inputs(values: dict) -> bytes:
    if not values:
        return b""
    import numpy as np

    arrays = {}
    for key, value in values.items():
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays[key] = np.asarray(value)
    if not arrays:
        return b""
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    return payload.getvalue()


def _multimodal_messages_with_images(messages: list[dict], images: list[object]) -> list[dict]:
    image_iter = iter(images)
    prepared = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                block = dict(block)
                if block.get("type") == "image":
                    block["image"] = next(image_iter)
                blocks.append(block)
            copied["content"] = blocks
        prepared.append(copied)
    try:
        next(image_iter)
    except StopIteration:
        return prepared
    raise ValueError("unused decoded image while preparing multimodal sft tokens")


def _processor_tokenized_row(
    processor,
    prompt_messages: list[dict],
    completion_messages: list[dict],
    images: list[object],
    *,
    max_length: int,
    thinking: bool,
) -> tuple[list[int], list[int], bytes, int]:
    from flash.engine.worker.model.packing import completion_mask_from_ids

    prepared_prompt = _multimodal_messages_with_images(prompt_messages, images)
    full_messages = [*prepared_prompt, *completion_messages]
    common = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": thinking,
    }
    full = dict(
        processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            **common,
        )
    )
    prompt = dict(
        processor.apply_chat_template(
            prepared_prompt,
            add_generation_prompt=True,
            **common,
        )
    )

    def ids(value) -> list[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        return [int(item) for item in value]

    untruncated_ids = ids(full.pop("input_ids"))
    # true length BEFORE the cap: realized_max_length is measured after this slice, so it can never
    # report more than max_length and cannot say whether the cap actually bound.
    untruncated_length = len(untruncated_ids)
    input_ids = untruncated_ids[:max_length]
    prompt_ids = ids(prompt["input_ids"])[:max_length]
    loss_mask = completion_mask_from_ids(prompt_ids, input_ids)
    full.pop("attention_mask", None)
    return input_ids, loss_mask, _serialize_multimodal_inputs(full), untruncated_length


def _materialize_verl_images(
    descriptors: list[str],
    package_root,
    image_dir: str | None,
    row_index: int,
) -> list[str]:
    """Decode image descriptors to files verl can load; a profile run passes no dir and writes none."""
    if image_dir is None:
        return []
    from flash.content.multimodal import decode_image_descriptors

    os.makedirs(image_dir, exist_ok=True)
    images = decode_image_descriptors(descriptors, package_root)
    rows: list[str] = []
    for image_index, image in enumerate(images):
        path = Path(image_dir, f"row-{row_index}-image-{image_index}.png").resolve()
        image.save(path, format="PNG")
        rows.append(path.as_uri())
    return rows


def _default_processor_loader(model_id: str, revision: str):
    from transformers import AutoProcessor

    from flash.engine.worker.io.hf import model_revision_kwargs

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
                # the contract probe is device-independent on purpose: this same function runs in
                # the cpu-only profile job that freezes the quote AND on the gpu worker, and
                # sft_train compares the two profiles byte-for-byte. see
                # gdn_packing_contract_available.
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
    dropped: int


@dataclass(frozen=True)
class _RetainedSftRows:
    rows: list[dict[str, Any]]
    untruncated_lengths: list[int]
    dropped: int


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


def _tokenize_prompt_rows(
    spec,
    prompt_rows: list[tuple[Any, list[dict], list[dict]]],
    *,
    package_root,
    tokenizer,
    processor,
    max_length: int,
    image_dir: str | None,
    decode_image_descriptors: Callable,
    normalize_prompt_images: Callable,
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
    for row_index, (example, prompt_messages, completion_messages) in enumerate(prompt_rows):
        _reject_image_completion(completion_messages)
        if len(completion_messages) > 1:
            multiturn_targets += 1
        if record_has_images(example, prompt_messages):
            if processor is None:
                raise RuntimeError("multimodal sft row has no processor")
            normalized = normalize_prompt_images(example, prompt_messages, package_root)
            completion_messages = text_only_prompt_messages(completion_messages)
            decoded_images = decode_image_descriptors(normalized.descriptors, package_root)
            input_ids, loss_mask, multimodal_inputs, untruncated_length = _processor_tokenized_row(
                processor,
                normalized.messages,
                completion_messages,
                decoded_images,
                max_length=max_length,
                thinking=bool(spec.thinking),
            )
            untruncated_by_index[row_index] = untruncated_length
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
            sampled_texts.append(
                tokenizer.apply_chat_template(
                    [*normalized.messages, *completion_messages],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=spec.thinking,
                )
            )
        else:
            text = tokenizer.apply_chat_template(
                [*prompt_messages, *completion_messages],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=spec.thinking,
            )
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=spec.thinking,
            )
            sampled_texts.append(text)
            text_specs.append({"text": text, "prompt_text": prompt_text, "row_index": row_index})

    dropped = 0
    if text_specs:
        kept_specs, tokenized_rows, text_dropped = _pretokenize_completion_only(
            text_specs,
            tokenizer,
            max_length,
        )
        dropped += text_dropped
        for spec_row, tokenized in zip(kept_specs, tokenized_rows, strict=True):
            input_ids = tokenized["input_ids"]
            untruncated_by_index[spec_row["row_index"]] = tokenized["untruncated_length"]
            row_by_index[spec_row["row_index"]] = {
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
        dropped=dropped,
    )


def _filter_retained_rows(tokenized: _TokenizedSftRows, tokenizer) -> _RetainedSftRows:
    """Compute the ordered rows that retain a real supervised target."""
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    rows = []
    retained_untruncated: list[int] = []
    dropped = tokenized.dropped
    for row_index in sorted(tokenized.row_by_index):
        row = tokenized.row_by_index[row_index]
        if has_real_target(row["input_ids"], row["loss_mask"], special_ids):
            rows.append(row)
            # appended in lockstep with the row it measures, so the truncation counts below describe
            # the rows that are actually trained on rather than the ones that were dropped.
            retained_untruncated.append(tokenized.untruncated_by_index[row_index])
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
        dropped=dropped,
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
    )


def prepare_sft_workload(
    spec,
    env,
    *,
    tokenizer_loader: Callable[[str, str], Any],
    producer_version: str,
    processor_loader: Callable[[str, str], Any] | None = None,
    image_dir: str | None = None,
    allow_packing: bool = True,
    packing_support: Callable[[str, str], tuple[str, bool]] | None = None,
) -> PreparedSftWorkload:
    """Render, tokenize, filter, and pack the exact rows consumed by SFT."""
    from flash.content.multimodal import (
        decode_image_descriptors,
        normalize_prompt_images,
        record_has_images,
        text_only_prompt_messages,
        validate_multimodal_training,
    )

    train_spec = spec.train
    max_length = sft_max_length(spec)
    epochs = int(train_spec.epochs if train_spec.epochs is not None else RECIPE.sft.num_epochs)
    effective_batch = int(
        train_spec.batch_size if train_spec.batch_size is not None else RECIPE.sft.effective_batch
    )
    max_examples = int(train_spec.max_examples or 0)
    max_steps = int(train_spec.max_steps or 0)

    source = list(env.dataset())
    selected = select_sft_examples(source, max_examples, spec.seed)
    prompt_rows = [
        (example, env.prompt_messages(example), env.sft_completion(example)) for example in selected
    ]
    package_root = getattr(env, "package_root", None)
    multimodal = any(
        record_has_images(example, prompt_messages)
        for example, prompt_messages, _completion in prompt_rows
    )
    processor = None
    if multimodal:
        validate_multimodal_training(spec.model, "sft")
        processor = (processor_loader or _default_processor_loader)(
            spec.model,
            spec.model_revision,
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = tokenizer_loader(spec.model, spec.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = _tokenize_prompt_rows(
        spec,
        prompt_rows,
        package_root=package_root,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
        image_dir=image_dir,
        decode_image_descriptors=decode_image_descriptors,
        normalize_prompt_images=normalize_prompt_images,
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
        source_examples=len(source),
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
    # the batch override is a large optimization-semantics change (one example per update instead
    # of the authored batch) and its only other trace is `notes["packing"]` in the finished run's
    # metrics, which nobody reads before paying for the run. say it out loud on the same stderr
    # channel the truncation warning above uses; this runs in the profile job and again on the
    # training worker, so both logs carry it.
    warning = unpacked_batch_warning(
        packing_mode=profile.packing_mode,
        architecture_mode=profile.architecture_mode,
        examples_per_update=profile.examples_per_update,
        configured_batch_size=effective_batch,
    )
    if warning:
        print(f"warning: [train] {warning}", file=sys.stderr)
    return PreparedSftWorkload(
        rows=retained.rows,
        profile=profile,
        multimodal=multimodal,
        tokenizer=tokenizer,
        processor=processor,
        sampled_texts=tokenized.sampled_texts,
        multiturn_targets=tokenized.multiturn_targets,
    )
