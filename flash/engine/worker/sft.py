"""pure sft data and accounting helpers plus the thin verl delegator."""

from __future__ import annotations

import random

from flash.engine.worker._pkg import W as _w
from flash.engine.worker.packing import completion_mask_from_ids, tokenize_for_packing


def _pretokenize_completion_only(texts, tokenizer, max_length):
    """tokenize rows with the historical completion-only mask and drop empty targets."""
    full_ids = tokenize_for_packing([row["text"] for row in texts], tokenizer, max_length)
    prompt_ids = tokenizer(
        [row["prompt_text"] for row in texts],
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    tokenized = [
        {"input_ids": ids, "completion_mask": completion_mask_from_ids(prompt, ids)}
        for ids, prompt in zip(full_ids, prompt_ids, strict=True)
    ]
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])

    def has_real_target(row) -> bool:
        return any(
            mask and token_id not in special_ids
            for token_id, mask in zip(
                row["input_ids"],
                row["completion_mask"],
                strict=True,
            )
        )

    kept = [
        (text, row)
        for text, row in zip(texts, tokenized, strict=True)
        if has_real_target(row)
    ]
    return [text for text, _ in kept], [row for _, row in kept], len(tokenized) - len(kept)


def _model_arch_dims(model_id: str, revision: str = "") -> tuple[int, int]:
    """return hidden size and layer count for gradient-checkpoint sizing."""
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    catalog_hidden = int(getattr(info, "hidden_size", 0) or 0) if info else 0
    catalog_layers = int(getattr(info, "num_layers", 0) or 0) if info else 0
    if catalog_hidden and catalog_layers and not revision:
        return catalog_hidden, catalog_layers
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            **_w.model_revision_kwargs(revision),
        )
        text_config = getattr(config, "text_config", None) or config
        layers = int(
            getattr(text_config, "num_hidden_layers", 0)
            or getattr(config, "num_hidden_layers", 0)
            or 0
        )
        hidden = int(
            getattr(text_config, "hidden_size", 0)
            or getattr(config, "hidden_size", 0)
            or 0
        )
        if revision and (
            (catalog_hidden and hidden and hidden != catalog_hidden)
            or (catalog_layers and layers and layers != catalog_layers)
        ):
            raise ValueError("revision architecture does not match catalog geometry")
        return hidden or catalog_hidden, layers or catalog_layers
    except Exception as error:
        if revision:
            raise RuntimeError("could not validate revision-specific model architecture") from error
        print(
            f"[sft] arch-dims probe failed ({error}); "
            "gradient checkpointing stays conservative"
        )
        return catalog_hidden, catalog_layers


def _select_indexed_sft_examples(train, max_examples, seed):
    if max_examples > 0:
        train = train[:max_examples]
    indexed_train = list(enumerate(train))
    random.Random(seed).shuffle(indexed_train)
    return indexed_train


def select_sft_examples(train, max_examples, seed):
    """select a deterministic prefix, then shuffle it with the run seed."""
    return [example for _, example in _select_indexed_sft_examples(train, max_examples, seed)]


def sft_completed_train_tokens(
    tokens_per_epoch: int,
    epochs: int,
    derived_steps: int,
    completed_steps: int,
) -> int:
    """estimate processed tokens from the final completed update count."""
    epoch_tokens = max(0, int(tokens_per_epoch)) * max(0, int(epochs))
    completed = max(0, int(completed_steps))
    derived = max(1, int(derived_steps))
    if completed == derived:
        return epoch_tokens
    if completed == 0 or epoch_tokens == 0:
        return 0
    return max(1, round(epoch_tokens * completed / derived))


def sft_under_ran(final_step: int, update_horizon: int, max_steps: int) -> bool:
    """return whether a max-step run completed fewer updates than requested."""
    return int(max_steps) > 0 and int(final_step) < int(update_horizon)


def _reject_image_completion(completion) -> None:
    from flash.multimodal import record_has_images

    if record_has_images({}, completion):
        raise ValueError("image-bearing SFT completions are not supported")


def run_sft():
    """delegate sft execution to the out-of-process verl worker."""
    from flash.engine.worker.sft_verl import run_sft_verl

    return run_sft_verl(_w.JOB_SPEC)
