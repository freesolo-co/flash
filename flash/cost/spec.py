"""Map a parsed training ``JobSpec`` to a cost ``RunConfig`` / step count / estimate.

Used by ``flash train --cost`` for a pre-flight quote. The control plane bills completed runs
from their final recorded ``cost_usd`` instead of charging this estimate at submit time."""

from __future__ import annotations

import math
import random

from flash.cost.analytical import estimate_cost
from flash.cost.types import CostEstimate, RunConfig


def _load_env(env_id: str, params: dict | None = None):
    if not env_id:
        return None
    try:
        from flash.envs import load_environment

        return load_environment(env_id, params or {})
    except Exception:
        return None


def _env_rows(env_id: str, params: dict | None = None) -> list | None:
    env = _load_env(env_id, params)
    if env is None:
        return None
    try:
        rows = env.dataset()
    except Exception:
        return None
    if rows is None:
        return None
    try:
        len(rows)
        return rows
    except Exception:
        return list(rows)


def count_env_examples(env_id: str, params: dict | None = None) -> int | None:
    """Training rows in ``env_id``'s dataset (the worker's train split), or ``None`` if it can't
    be loaded. Best-effort -- prices an uncapped SFT run on the real dataset size, not a guess.

    Loading may need network access for managed Freesolo environments. If the environment
    cannot be loaded in this interpreter, this returns ``None``."""
    rows = _env_rows(env_id, params)
    return len(rows) if rows is not None else None


def _sft_epochs(spec) -> int:
    from flash.engine.recipe import RECIPE

    t = spec.train
    return int(t.epochs) if t.epochs is not None else RECIPE.sft.num_epochs


def _sft_seq_len(spec) -> int:
    from flash.engine.recipe import RECIPE

    t = spec.train
    return (
        int(t.max_length)
        if t.max_length is not None
        else (RECIPE.sft.max_seq_len_thinking if spec.thinking else RECIPE.sft.max_seq_len)
    )


def _sft_example_count(spec) -> int:
    t = spec.train
    pinned_examples = int(t.max_examples) if t.max_examples else 0
    if pinned_examples > 0:
        return pinned_examples
    from flash.envs.training_params import training_env_params

    examples = count_env_examples(spec.environment.id, training_env_params(spec))
    if examples is None:
        raise ValueError(
            "cannot estimate uncapped SFT cost because the environment dataset could not be "
            "counted; set [train].max_examples or run `flash train --cost` where the "
            "environment is importable"
        )
    return examples


def _sft_realized_batch(spec) -> int:
    from flash.catalog import vocab_size_for
    from flash.engine.recipe import RECIPE
    from flash.engine.vram import resolve_params_b, sft_logits_fused, sft_realized_batch

    t = spec.train
    requested_batch = int(t.batch_size) if t.batch_size is not None else RECIPE.sft.effective_batch
    sft_seq = _sft_seq_len(spec)
    # Resolve params_b via the shared helper (catalog stat else HF safetensors for an open model) —
    # the SAME resolution the worker's run_sft uses. The fused-CE decision (and thus the big-vocab
    # micro-batch cap) hinges on the >=3B threshold, so an uncataloged >=3B model must not be priced
    # as <3B (which would flip fused off, change the realized batch via the cap, and misprice the
    # step count). Best-effort: no network -> None -> the prior <3B (cap-on) behavior.
    sft_fused = sft_logits_fused(resolve_params_b(spec.model), sft_seq)
    return sft_realized_batch(
        requested_batch, seq_len=sft_seq, vocab=vocab_size_for(spec.model), fused=sft_fused
    )


def _sft_steps_from_examples(spec, examples: int, *, apply_cap: bool) -> int:
    t = spec.train
    cap = int(t.max_steps) if t.max_steps else 0  # SFT-only optimizer-step cap (0 = uncapped)
    n = max(1, math.ceil(examples / _sft_realized_batch(spec)) * _sft_epochs(spec))
    return min(n, cap) if apply_cap and cap > 0 else n


def _indexable_dataset_rows(env) -> object | None:
    try:
        rows = env.dataset()
    except Exception:
        return None
    if rows is None:
        return None
    try:
        n = len(rows)
        if n:
            rows[0]
        return rows
    except Exception:
        try:
            return list(rows)
        except Exception:
            return None


def _sft_selected_row_indexes(rows, max_examples: int) -> list[int]:
    from flash.spec import FIXED_SEED

    order = list(range(len(rows)))
    random.Random(FIXED_SEED).shuffle(order)
    return order[:max_examples] if max_examples > 0 else order


def count_sft_train_tokens(spec) -> int | None:
    """Actual SFT training tokens across epochs, or ``None`` if local token counting is unavailable.

    This mirrors the worker's SFT text construction: render prompt + SFT completion with the model
    chat template, tokenize with the configured max length, then multiply by epochs. If
    ``max_steps`` caps the SFT run before all epochs/examples are consumed, the token count is
    reduced by the same step fraction.
    """
    if spec.algorithm != "sft":
        return None
    from flash.envs.training_params import training_env_params

    env = _load_env(spec.environment.id, training_env_params(spec))
    if env is None:
        return None
    rows = _indexable_dataset_rows(env)
    if rows is None:
        return None
    max_examples = int(spec.train.max_examples) if spec.train.max_examples else 0
    row_indexes = _sft_selected_row_indexes(rows, max_examples)
    if not row_indexes:
        return None

    try:
        from transformers import AutoTokenizer

        from flash.engine.worker.sft import _pretokenize_completion_only

        tok = AutoTokenizer.from_pretrained(spec.model, trust_remote_code=True)
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
        texts: list[dict[str, str]] = []
        for idx in row_indexes:
            ex = rows[idx]
            prompt_messages = list(env.prompt_messages(ex) or [])
            completion = list(env.sft_completion(ex) or [])
            msgs = [*prompt_messages, *completion]
            texts.append(
                {
                    "text": tok.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=bool(spec.thinking),
                    ),
                    "prompt_text": tok.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=bool(spec.thinking),
                    ),
                }
            )
        _, token_rows, _ = _pretokenize_completion_only(texts, tok, _sft_seq_len(spec))
        tokens = sum(len(row["input_ids"]) for row in token_rows)
    except Exception:
        return None
    if not token_rows:
        return None

    total = max(1, tokens * _sft_epochs(spec))
    trainable_examples = len(token_rows)
    uncapped_steps = _sft_steps_from_examples(spec, trainable_examples, apply_cap=False)
    capped_steps = _sft_steps_from_examples(spec, trainable_examples, apply_cap=True)
    if capped_steps < uncapped_steps:
        total = max(1, math.ceil(total * capped_steps / uncapped_steps))
    return total


def spec_steps(spec) -> int:
    """Per-seed optimizer steps implied by a train spec (mirrors the worker). GRPO: ``train.steps``
    (else recipe default). SFT: ``epochs x ceil(num_examples / realized_batch)`` capped by
    ``max_steps``, where ``num_examples`` is ``max_examples`` if pinned else the real env size."""
    from flash.engine.recipe import RECIPE

    t = spec.train
    if spec.algorithm == "grpo":
        if t.steps is not None:
            return max(1, int(t.steps))
        return RECIPE.rl.num_steps
    # max_examples is a CAP; 0 (like None) means "no cap" (worker trains the full dataset), so
    # don't let max_examples=0 price a single step.
    return _sft_steps_from_examples(spec, _sft_example_count(spec), apply_cap=True)


def runconfig_from_spec(spec) -> RunConfig:
    """Map a parsed ``JobSpec`` to a cost ``RunConfig``. A run trains exactly one adapter, so the
    estimate covers a single job. The estimate doesn't pin a GPU -- it does its own cheapest-fit
    (provider="auto")."""
    t, g = spec.train, spec.gpu
    is_grpo = spec.algorithm == "grpo"
    train_tokens = None if is_grpo else count_sft_train_tokens(spec)
    return RunConfig(
        model_id=spec.model,
        method=spec.algorithm,
        steps=spec_steps(spec),
        train_tokens=train_tokens,
        seq_len=t.max_length,
        completion_len=t.max_tokens if is_grpo else None,
        batch_size=t.batch_size,
        group_size=t.group_size if is_grpo else None,
        lora_rank=t.lora_rank,
        thinking=spec.thinking,
        provider="auto",
        max_wall_seconds=g.max_wall_seconds,
        environment=spec.environment.id or None,
    )


def estimate_for_spec(spec) -> CostEstimate:
    """The pre-flight ``CostEstimate`` for a parsed training ``JobSpec``."""
    return estimate_cost(runconfig_from_spec(spec))
