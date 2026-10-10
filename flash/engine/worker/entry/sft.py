"""SFT entry point and the dataset helpers the verl worker shares.

``run_sft`` delegates to ``sft_train.run_sft_train``, which owns tokenization, packing, and training.
What stays here is what that worker imports rather than reimplements: example selection, the
completion-only pretokenizer, model-arch dims, the token accounting used to detect an under-run, and
the image-completion rejection.
"""

from __future__ import annotations

import random

import flash.engine.worker.io.hf as _worker_hf
from flash.engine.worker.model.chatml_mask import assistant_only_mask
from flash.engine.worker.model.packing import (
    completion_mask_from_ids,
    packing_appends_eos,
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
    max_length, so it cannot say whether the cap actually bound; this can. It also carries
    ``assistant_mask_applied``: whether the render was readable enough to mask non-assistant turns.
    """
    full_ids = tokenize_for_packing([t["text"] for t in texts], tokenizer, max_length)
    # the same encode without the cap, so a truncated row reports its real size rather than the cap.
    untruncated = untruncated_lengths_for_packing([t["text"] for t in texts], tokenizer)
    prompt_ids = tokenizer(
        [t["prompt_text"] for t in texts], truncation=True, max_length=max_length
    )["input_ids"]
    pretok = []
    for text, ids, pids, length in zip(texts, full_ids, prompt_ids, untruncated, strict=True):
        mask, role_aware = assistant_only_mask(
            completion_mask_from_ids(pids, ids),
            ids,
            tokenizer,
            text["target_messages"],
            # the appended EOS counts only when it SURVIVED the cap: an equal-length row proves the
            # token that terminates the example is still there to supervise.
            appended_eos=packing_appends_eos(text["text"], tokenizer) and len(ids) == length,
            source_messages=text["source_messages"],
            template_kwargs=text["template_kwargs"],
        )
        pretok.append(
            {
                "input_ids": ids,
                "completion_mask": mask,
                "assistant_mask_applied": role_aware,
                "untruncated_length": length,
            }
        )
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    kept = [
        (t, r)
        for t, r in zip(texts, pretok, strict=True)
        if has_real_target(r["input_ids"], r["completion_mask"], special_ids)
    ]
    return [t for t, _ in kept], [r for _, r in kept], len(pretok) - len(kept)


def _model_arch_dims(model_id: str, revision: str = "") -> tuple[int, int]:
    """``(hidden_size, num_hidden_layers)`` used to size the GC-off activation estimate.

    Prefer the CURATED catalog geometry (deterministic, no network/parse risk) — a live B200 SFT
    showed the runtime ``AutoConfig`` probe returning (0, 0) on the 35B-A3B's multimodal-nested
    config, which silently kept GC on. A PINNED revision has no curated dims (the commit's geometry
    is not validated), so it falls back to the HF config, handling the ``text_config`` nesting
    (config.json is already cached by the tokenizer load). Best-effort: ``(0, 0)`` if neither is
    available -> the GC-off gate conservatively keeps gradient checkpointing on."""
    from flash.core.catalog import MODELS

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
            **_worker_hf.model_revision_kwargs(revision),
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


def select_sft_examples(train, max_examples, seed):
    """Pick the SFT sample: the first ``max_examples`` rows of the dataset (file order), shuffled.

    The slice happens BEFORE the shuffle so ``max_examples`` is a deterministic prefix fence,
    not a random subsample. A train.jsonl that carries fully-labeled SFT rows first and
    prompt-only (empty-output) GRPO rows after can cap SFT to the labeled head: an empty
    completion can never be shuffled into the SFT sample and teach the model to emit nothing.
    """
    if max_examples > 0:
        train = train[:max_examples]
    selected = list(train)
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected


def sft_under_ran(final_step: int, update_horizon: int) -> bool:
    """True when the run completed fewer updates than the horizon it was quoted and billed for.

    the horizon caps every run (trainer.total_training_steps in _prepare_sft_child), so a fresh run
    lands exactly on it. a resume from a checkpoint past a lowered horizon does zero new steps yet
    holds a fully-trained adapter (final_step >= horizon). fail loudly only on a genuine under-run,
    mirroring grpo (steps_run < steps) and opd (opt_steps < steps).
    """
    return int(final_step) < int(update_horizon)


def _reject_image_completion(
    completion,
    *,
    image_bearing: bool,
    source_messages=None,
    template_source=None,
    template_kwargs: dict | None = None,
) -> None:
    """reject image targets always, and reserved pad tokens only on image-bearing rows.

    the estimator tokenizes an image-bearing prompt and completion as one stream, so a literal
    image-pad token in any field emitted by the chat template is indistinguishable from a real image
    block. the rendered field probe covers content and nested tool-call fields without rejecting
    unrendered metadata. on a text-only row the same registered token is ordinary authored text and
    retains the normal prompt mask or completion supervision.
    """
    from flash.content.multimodal import (
        IMAGE_PAD_TOKEN,
        completion_has_images,
        message_content_text,
    )

    if completion_has_images(completion):
        raise ValueError("image-bearing SFT completions are not supported")
    if not image_bearing:
        return
    for message in completion or []:
        if IMAGE_PAD_TOKEN in message_content_text(
            message.get("content") if isinstance(message, dict) else None
        ):
            raise ValueError(
                f"sft completion text contains the reserved image marker {IMAGE_PAD_TOKEN!r}; "
                "remove it from the target"
            )
    if template_source is not None:
        from flash.engine.worker.model.chatml_mask import reject_rendered_message_token

        messages = list(source_messages if source_messages is not None else completion or [])
        reject_rendered_message_token(
            template_source,
            messages,
            IMAGE_PAD_TOKEN,
            template_kwargs=template_kwargs or {},
        )


def run_sft():
    """Run SFT. verl is the only backend; this module keeps the dataset helpers it shares."""
    from flash.engine.worker.train.entry.sft_train import run_sft_train

    run_sft_train()
