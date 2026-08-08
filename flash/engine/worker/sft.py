"""SFT entry point and the dataset helpers the verl worker shares.

``run_sft`` delegates to ``sft_train.run_sft_train``, which owns tokenization, packing, and training.
What stays here is what that worker imports rather than reimplements: example selection, the
completion-only pretokenizer, model-arch dims, the token accounting used to detect an under-run, and
the image-completion rejection.
"""

from __future__ import annotations

import random

from flash.engine.worker._pkg import W as _w
from flash.engine.worker.packing import (
    completion_mask_from_ids,
    tokenize_for_packing,
    untruncated_lengths_for_packing,
)


def has_real_target(input_ids, loss_mask, special_ids: set[int]) -> bool:
    """Does this row supervise at least one token that is not a special token?

    A row whose entire supervised span is padding/EOS teaches nothing, and both SFT paths drop such
    rows. They spell the mask field differently (``completion_mask`` when pretokenizing, ``loss_mask``
    once packed), so the check takes the two sequences rather than a row dict: the emptiness rule
    cannot then depend on which name the caller happens to use.
    """
    return any(
        mask and token_id not in special_ids
        for token_id, mask in zip(input_ids, loss_mask, strict=True)
    )


def _pretokenize_completion_only(texts, tokenizer, max_length):
    """Pre-tokenize SFT rows into ``{input_ids, completion_mask}``, dropping rows with no real completion target.

    Returns ``(kept_texts, pretok, n_dropped)``. Each pretok row also carries
    ``untruncated_length``: its token count BEFORE the cap. The post-slice length can never exceed
    max_length, so it cannot say whether the cap actually bound; this can.
    """
    full_ids = tokenize_for_packing([t["text"] for t in texts], tokenizer, max_length)
    # the same encode without the cap, so a truncated row reports its real size rather than the cap.
    untruncated = untruncated_lengths_for_packing([t["text"] for t in texts], tokenizer)
    prompt_ids = tokenizer(
        [t["prompt_text"] for t in texts], truncation=True, max_length=max_length
    )["input_ids"]
    pretok = [
        {
            "input_ids": ids,
            "completion_mask": completion_mask_from_ids(pids, ids),
            "untruncated_length": length,
        }
        for ids, pids, length in zip(full_ids, prompt_ids, untruncated, strict=True)
    ]
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    kept = [
        (t, r)
        for t, r in zip(texts, pretok, strict=True)
        if has_real_target(r["input_ids"], r["completion_mask"], special_ids)
    ]
    return [t for t, _ in kept], [r for _, r in kept], len(pretok) - len(kept)


def _model_arch_dims(model_id: str, revision: str = "") -> tuple[int, int]:
    """``(hidden_size, num_hidden_layers)`` used to size the GC-off activation estimate.

    Prefer the CURATED catalog geometry (deterministic, no network/parse risk) for known models — a
    live B200 SFT showed the runtime ``AutoConfig`` probe returning (0, 0) on the 35B-A3B's
    multimodal-nested config, which silently kept GC on. For open-model-policy ids (no catalog dims)
    fall back to the HF config, handling the ``text_config`` nesting (config.json is already cached by
    the tokenizer load). Best-effort: ``(0, 0)`` if neither is available -> the GC-off gate
    conservatively keeps gradient checkpointing on."""
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    c_hidden = int(getattr(info, "hidden_size", 0) or 0) if info else 0
    c_layers = int(getattr(info, "num_layers", 0) or 0) if info else 0
    if c_hidden and c_layers and not revision:
        return c_hidden, c_layers
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            **_w.model_revision_kwargs(revision),
        )
        tc = getattr(cfg, "text_config", None) or cfg
        layers = int(
            getattr(tc, "num_hidden_layers", 0) or getattr(cfg, "num_hidden_layers", 0) or 0
        )
        hidden = int(getattr(tc, "hidden_size", 0) or getattr(cfg, "hidden_size", 0) or 0)
        # Only a NONZERO probe dim that disagrees with the catalog is a real revision mismatch. A (0, 0)
        # probe means "couldn't parse" (the 35B-A3B multimodal-nested config does exactly this), not
        # "differs" -- treat it like the unpinned path and fall back to catalog dims below, otherwise a
        # revision pin would spuriously fail such models after the GPU is already rented.
        if revision and (
            (c_hidden and hidden and hidden != c_hidden)
            or (c_layers and layers and layers != c_layers)
        ):
            raise ValueError("revision architecture does not match catalog geometry")
        return hidden or c_hidden, layers or c_layers
    except Exception as e:
        if revision:
            raise RuntimeError("could not validate revision-specific model architecture") from e
        print(f"[sft] arch-dims probe failed ({e}); GC decision stays conservative (keep GC on)")
        return c_hidden, c_layers


def _select_indexed_sft_examples(train, max_examples, seed):
    if max_examples > 0:
        train = train[:max_examples]
    indexed_train = list(enumerate(train))
    rng = random.Random(seed)
    rng.shuffle(indexed_train)
    return indexed_train


def select_sft_examples(train, max_examples, seed):
    """Pick the SFT sample: the first ``max_examples`` rows of the dataset (file order), shuffled.

    The slice happens BEFORE the shuffle so ``max_examples`` is a deterministic prefix fence,
    not a random subsample. A train.jsonl that carries fully-labeled SFT rows first and
    prompt-only (empty-output) GRPO rows after can cap SFT to the labeled head — an empty
    completion can never be shuffled into the SFT sample and teach the model to emit nothing.
    """
    return [example for _, example in _select_indexed_sft_examples(train, max_examples, seed)]


def sft_completed_train_tokens(
    tokens_per_epoch: int,
    epochs: int,
    derived_steps: int,
    completed_steps: int,
) -> int:
    """Estimate tokens processed from completed updates while preserving epoch accounting at parity."""
    epoch_tokens = max(0, int(tokens_per_epoch)) * max(0, int(epochs))
    completed = max(0, int(completed_steps))
    derived = max(1, int(derived_steps))
    if completed == derived:
        return epoch_tokens
    if completed == 0 or epoch_tokens == 0:
        return 0
    return max(1, round(epoch_tokens * completed / derived))


def sft_under_ran(final_step: int, update_horizon: int, max_steps: int) -> bool:
    """True when a max_steps-authoritative run completed fewer updates than requested.

    with max_steps authoritative, the trainer's max_steps override lands a fresh run exactly on
    the horizon, and a resume from a checkpoint past a lowered horizon does zero new steps yet
    holds a fully-trained adapter (final_step >= horizon). fail loudly only on a genuine
    under-run, mirroring grpo (steps_run < steps) and opd (opt_steps < steps).
    """
    return int(max_steps) > 0 and int(final_step) < int(update_horizon)


def _reject_image_completion(completion) -> None:
    from flash.multimodal import record_has_images

    if record_has_images({}, completion):
        raise ValueError("image-bearing SFT completions are not supported")


def run_sft():
    """Run SFT. verl is the only backend; this module keeps the dataset helpers it shares."""
    from flash.engine.worker.sft_train import run_sft_train

    run_sft_train()
