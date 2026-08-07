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

from flash.engine.recipe import RECIPE
from flash.engine.steps import resolve_update_horizon, sft_update_steps
from flash.engine.worker.packing import probe_is_gdn_hybrid, probe_is_pure_attention
from flash.engine.worker.sft import (
    _pretokenize_completion_only,
    _reject_image_completion,
    _select_indexed_sft_examples,
    has_real_target,
)
from flash.workload_profile import SftWorkloadProfile, sft_sample_policy


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
    from flash.engine.worker.packing import completion_mask_from_ids

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
    from flash.multimodal import decode_image_descriptors

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

    from flash.engine.worker.hf import model_revision_kwargs

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
        # the RAISING probes, not the swallowing ones. both labels below are frozen into the
        # profile and compared byte-for-byte by the training worker (sft_train.py), so a probe that
        # answered False because the hub timed out would freeze "unsupported", and a later
        # re-derivation that reached the config would produce "gdn-hybrid" and fail the run with
        # "sft workload changed after the quote was frozen" -- while every token, step and packing
        # decision was in fact identical. a probe that could not answer must fail the PROFILE, which
        # is retryable, rather than mint a label that poisons every training run built on it.
        try:
            if probe_is_pure_attention(model_id, revision=revision):
                architecture_mode, supported = "pure-attention", True
            elif probe_is_gdn_hybrid(model_id, revision=revision):
                architecture_mode, supported = "gdn-hybrid", False
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

    The one producer of this number. It reaches the trainer and the quote as the profile's
    ``max_length`` rather than by either of them re-deriving it, because a second derivation would
    silently disagree while the worker's parity check -- which compares two values this module
    produced -- still passed.
    """
    authored = spec.train.max_context_tokens
    if authored is not None:
        return int(authored)
    return int(RECIPE.sft.max_seq_len_thinking if spec.thinking else RECIPE.sft.max_seq_len)


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
    from flash.multimodal import (
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
    indexed_train = _select_indexed_sft_examples(source, max_examples, spec.seed)
    selected = [example for _, example in indexed_train]
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

    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    unpacked_rows = []
    retained_untruncated: list[int] = []
    for row_index in sorted(row_by_index):
        row = row_by_index[row_index]
        if has_real_target(row["input_ids"], row["loss_mask"], special_ids):
            unpacked_rows.append(row)
            # appended in lockstep with the row it measures, so the truncation counts below describe
            # the rows that are actually trained on rather than the ones that were dropped.
            retained_untruncated.append(untruncated_by_index[row_index])
        else:
            dropped += 1
    if not unpacked_rows:
        raise ValueError(
            "every SFT example has an empty completion after sft_max_len truncation "
            "(nothing to train on); increase sft_max_len or shorten the prompts"
        )

    packing_mode, architecture_mode = _packing_mode(
        spec.model,
        spec.model_revision,
        multimodal=multimodal,
        allow_packing=allow_packing,
        packing_support=packing_support,
    )
    rows = unpacked_rows
    real_tokens = sum(len(row["input_ids"]) for row in rows)
    supervised_tokens = sum(sum(int(item) for item in row["loss_mask"]) for row in rows)
    padded_compute_tokens = real_tokens
    realized_max_length = max(len(row["input_ids"]) for row in rows)
    # measured against the UNTRUNCATED encode, so it reports what the rows actually need rather
    # than what the cap allowed. realized_max_length cannot do this: it is taken after the slice,
    # so it saturates at max_length exactly when the cap binds and the censoring becomes invisible.
    # mirrors the rollout profile's truncated_rollouts/truncation_rate reasoning.
    untruncated_max_length = max(retained_untruncated)
    truncated_rows = sum(1 for length in retained_untruncated if length > max_length)
    if truncated_rows:
        print(
            f"warning: [train] max_context_tokens {max_length} truncated {truncated_rows} of "
            f"{len(rows)} sft rows; the longest row needs {untruncated_max_length} tokens. "
            "training on the truncated rows as configured; set max_context_tokens to at least "
            f"{untruncated_max_length} to keep every row whole.",
            file=sys.stderr,
        )
    # one example per update is this layer's isolation lever for an architecture it cannot pack.
    # a micro-batch reaches the model as one (1, total_nnz) row with `attention_mask=None`: softmax
    # layers recover their boundaries from the per-example `position_ids` restarts, but
    # GatedDeltaNet layers read theirs out of `seq_idx` (causal conv) and `cu_seq_lens_q`
    # (recurrence), so on a gdn hybrid every example after the first would train on state carried
    # over from its predecessor -- silently, with no error and no metric. keeping one example per
    # batch leaves nothing to carry.
    #
    # this decision is made WITHOUT a gpu, so it can only go on the architecture, and it is the one
    # that binds: `examples_per_update` feeds the quoted step count below, and the worker replays it
    # verbatim (sft_train reads `profile.examples_per_update` and `profile.authoritative_steps`), so
    # the horizon the user was quoted is the horizon that runs. the worker's child probe decides
    # something ELSE -- verl's tensor layout, `use_remove_padding` -- which is why that flag must not
    # be gated on this mode: it selects a code path, not a batch size.
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

    profile = SftWorkloadProfile(
        input_digest=spec.workload_profile_input_digest,
        producer_version=producer_version,
        tokenizer_revision=spec.model_revision,
        environment_id=spec.environment.id,
        environment_revision=spec.environment.resolved_sha,
        source_examples=len(source),
        selected_examples=len(selected),
        retained_examples=len(unpacked_rows),
        dropped_examples=dropped,
        epochs=epochs,
        max_length=max_length,
        packing_mode=packing_mode,
        architecture_mode=architecture_mode,
        packed_blocks=packed_blocks,
        real_tokens_per_epoch=real_tokens,
        supervised_tokens_per_epoch=supervised_tokens,
        padded_compute_tokens_per_epoch=padded_compute_tokens,
        authoritative_real_tokens=authoritative_real_tokens,
        authoritative_supervised_tokens=authoritative_supervised_tokens,
        authoritative_compute_tokens=authoritative_real_tokens,
        realized_max_length=realized_max_length,
        untruncated_max_length=untruncated_max_length,
        truncated_examples=truncated_rows,
        examples_per_update=examples_per_update,
        derived_steps=derived_steps,
        authoritative_steps=authoritative_steps,
        packing_efficiency=real_tokens / padded_compute_tokens,
        sample_policy=sft_sample_policy(max_examples),
    )
    return PreparedSftWorkload(
        rows=rows,
        profile=profile,
        multimodal=multimodal,
        tokenizer=tokenizer,
        processor=processor,
        sampled_texts=sampled_texts,
        multiturn_targets=multiturn_targets,
    )
