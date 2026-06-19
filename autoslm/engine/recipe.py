"""Frozen, shared AutoSLM fine-tuning recipe.

Single source of truth for the default fine-tuning hyperparameters: base model,
tokenizer, data, LoRA config, optimization, token budget, and decoding.
Per-run TOML configs (parsed into a ``JobSpec``) override the relevant fields.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Model identity
# ----------------------------------------------------------------------------
# Recipe fallback base model. Model selection precedence on the worker is
# JobSpec.model > env BENCH_HF_MODEL > this recipe default; worker.py resolves
# JOB_SPEC.model first and only falls back to RECIPE.hf_model_id. The RunPod launcher
# sets BENCH_HF_MODEL from the spec; Vast carries the model via the full JobSpec
# (JOB_SPEC.model), which the worker resolves before this fallback. This literal is the
# last-resort default when neither is present.
# Keep it in sync with catalog.DEFAULT_MODEL (a proven dense text-only instruction model
# that loads on the current worker stack: transformers 5.x / TRL 1.x / vLLM 0.19.x; the
# natively-multimodal Qwen3.5/3.6 checkpoints are also catalog'd, trained/served text-only).
HF_MODEL_ID = os.environ.get("BENCH_HF_MODEL", "Qwen/Qwen3.5-4B")  # catalog DEFAULT_MODEL


# ----------------------------------------------------------------------------
# LoRA (rank is the main user-controllable knob)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    # The worker adapts all linear projections, set via the LORA_TARGETS env var
    # (default "all-linear" — see engine.worker); `rank`/`alpha` are the main
    # user-controllable knobs here.


# ----------------------------------------------------------------------------
# SFT (Phase 1)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SFTConfig:
    max_seq_len: int = 1024
    # Thinking-mode sequence cap: <think> traces in targets need headroom. A deliberate
    # consumer-GPU compromise (SFT cost/VRAM scales with sequence length).
    max_seq_len_thinking: int = 2048
    learning_rate: float = 1e-4
    warmup_frac: float = 0.03
    # Effective batch = per_device_batch * grad_accum (Arm A) / batch of datums (Arm B)
    effective_batch: int = 32
    num_epochs: int = 2


# ----------------------------------------------------------------------------
# RL / GRPO (Phase 2)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RLConfig:
    learning_rate: float = 1e-5
    # Default engine prompt budget. 512 was too small for real envs with non-trivial system
    # prompts (e.g. a schema/instructions block + the user query), which made every prompt
    # overflow before training started. 2048 fits typical instruction prompts; the run's
    # [train].max_length sets the engine length explicitly when it needs more/less.
    max_prompt_len: int = 2048
    max_completion_len: int = 320
    # Thinking-mode completion budget: <think> blocks consume most of it (phase 0
    # showed 320 is hopeless — every completion hit the cap). 1536 is a consumer-GPU
    # compromise (KV cache + rollout cost scale linearly with completion length, ~5x
    # tokens/step vs non-thinking); the run's [train].max_tokens overrides it explicitly.
    max_completion_len_thinking: int = 1536
    prompts_per_step: int = 64
    group_size: int = 8  # G completions per prompt
    num_steps: int = 150  # overridable per-run via the TOML `train.steps`
    sampling_temperature: float = 1.0  # on-policy sampling for rollouts
    sampling_top_p: float = 1.0


@dataclass(frozen=True)
class Recipe:
    """The complete shared recipe."""

    hf_model_id: str = HF_MODEL_ID
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)


RECIPE = Recipe()
