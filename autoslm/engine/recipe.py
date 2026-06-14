"""Frozen, shared AutoSLM fine-tuning recipe.

Single source of truth for the default fine-tuning hyperparameters: base model,
tokenizer, data, LoRA config, optimization, token budget, eval, and decoding.
Per-run TOML configs (parsed into a ``JobSpec``) override the relevant fields.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

# ----------------------------------------------------------------------------
# Model identity
# ----------------------------------------------------------------------------
# Default base model; override via env BENCH_HF_MODEL or the TOML `model` field.
# A proven, dense, text-only instruction model that loads on the current worker stack
# (transformers 5.x / TRL 1.x / vLLM 0.19.x). The natively-multimodal Qwen3.5/3.6
# checkpoints are also in the catalog, trained/served text-only.
HF_MODEL_ID = os.environ.get("BENCH_HF_MODEL", "Qwen/Qwen3-4B-Instruct-2507")


# ----------------------------------------------------------------------------
# LoRA (rank is the main user-controllable knob)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    # All linear projections (PEFT). The worker uses "all-linear" so this list is the
    # documented default; `rank`/`alpha` are the main user-controllable knobs.
    target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


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
    max_prompt_len: int = 512
    max_completion_len: int = 320
    # Thinking-mode completion budget: <think> blocks consume most of it (phase 0
    # showed 320 is hopeless — every completion hit the cap). 1536 is a consumer-GPU
    # compromise (KV cache + rollout cost scale linearly with completion length, ~5x
    # tokens/step vs non-thinking); RL_MAX_COMPLETION remains the escape hatch.
    max_completion_len_thinking: int = 1536
    prompts_per_step: int = 64
    group_size: int = 8  # G completions per prompt
    num_steps: int = 150  # overridable per-run via the TOML `train.steps`
    sampling_temperature: float = 1.0  # on-policy sampling for rollouts
    sampling_top_p: float = 1.0


# ----------------------------------------------------------------------------
# Evaluation (identical eval set + decoding on both arms; shared grader)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    num_examples: int = 300
    subset_seed: int = 12345  # fixed seed -> identical indices on both arms
    # Deterministic greedy decoding for eval
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    # Thinking-mode eval budget: the reasoning block must close AND leave room for the
    # answer (unclosed <think> grades 0). EVAL_MAX_NEW_TOKENS remains the escape hatch.
    max_new_tokens_thinking: int = 2048


@dataclass(frozen=True)
class Recipe:
    """The complete shared recipe."""

    hf_model_id: str = HF_MODEL_ID
    seeds: tuple = (0, 1)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


RECIPE = Recipe()
