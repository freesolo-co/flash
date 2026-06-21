"""Config linter: advisory warnings (not hard errors) for a parsed ``JobSpec``.

``schema`` rejects malformed configs with ``ConfigError`` (HTTP 400). This module is the
*soft* layer on top: a config can pass that validation yet still be a likely mistake — a tiny
LoRA rank on a big model, a two-completion GRPO group, a near-zero rollout temperature, a
thinking run with a 320-token completion budget. Those parse fine and waste a paid run.

``lint_spec`` compares an already-parsed spec against the shared recipe defaults
(``flash.engine.recipe.RECIPE`` — the team's priors) plus a small set of high-precision
heuristics, and returns human-readable :class:`Advice`. It is pure and offline: the caller
passes in the ``ModelInfo`` that ``spec_from_dict`` already resolved, so the linter never
re-touches the network (the open-model HF probe) on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import ModelInfo
from .engine.recipe import RECIPE
from .engine.vram import params_b_from_str
from .spec import JobSpec

# A model is "large" (LoRA rank capacity starts to matter) at roughly this many billion
# params; below it a small rank is a fine, cheap choice.
_LARGE_MODEL_B = 7.0
# A thinking run's <think> trace needs real headroom; below this the budget truncates
# reasoning before the answer (the thinking recipe defaults are 1536 GRPO / 2048 SFT).
_THINKING_GRPO_MIN_TOKENS = 1024


@dataclass(frozen=True)
class Advice:
    """One piece of advisory feedback about a config knob."""

    field: str  # the config key the advice is about, e.g. "train.lora_rank"
    message: str  # actionable, human-readable
    severity: str = "warning"  # "warning" | "info"

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "message": self.message, "severity": self.severity}


def lint_spec(spec: JobSpec, info: ModelInfo) -> list[Advice]:
    """Advisory warnings for a parsed (already-valid) spec. Pure/offline.

    ``info`` is the ``ModelInfo`` the caller already resolved for ``spec`` (so the linter does
    not re-resolve and re-probe the network). Optional knobs are inspected only when set (an
    unset knob is never flagged); the always-present lora_rank/lora_alpha are gated on model
    size — so a config left entirely at the recipe defaults produces no advice.
    """
    out: list[Advice] = []
    t = spec.train
    grpo = spec.algorithm == "grpo"
    params_b = params_b_from_str(info.params)

    # 1. Thinking mode with a completion/sequence budget too short for <think> traces.
    if spec.thinking and grpo and t.max_tokens is not None and t.max_tokens < _THINKING_GRPO_MIN_TOKENS:
        out.append(
            Advice(
                "train.max_tokens",
                f"max_tokens={t.max_tokens} is short for a thinking run (thinking=true): "
                f"<think> traces usually need >={_THINKING_GRPO_MIN_TOKENS} tokens (the thinking "
                f"recipe default is {RECIPE.rl.max_completion_len_thinking}). A short budget "
                f"truncates reasoning before the answer.",
            )
        )
    if (
        spec.thinking
        and not grpo
        and t.max_length is not None
        and t.max_length < RECIPE.sft.max_seq_len_thinking
    ):
        out.append(
            Advice(
                "train.max_length",
                f"max_length={t.max_length} is short for a thinking SFT run (thinking=true): "
                f"reasoning targets need headroom (the thinking recipe default is "
                f"{RECIPE.sft.max_seq_len_thinking}).",
            )
        )

    # 2. LoRA rank too low to give a large model enough adapter capacity.
    if params_b is not None and params_b >= _LARGE_MODEL_B and t.lora_rank <= 8:
        out.append(
            Advice(
                "train.lora_rank",
                f"lora_rank={t.lora_rank} is low for a {params_b:g}B model "
                f"({info.display_name}): large models usually need more adapter capacity "
                f"(rank 16-64; recipe default {RECIPE.lora.rank}). Low rank can underfit.",
            )
        )

    # 3. GRPO group too small for a meaningful within-group advantage signal.
    if grpo and t.group_size is not None and t.group_size < 4:
        out.append(
            Advice(
                "train.group_size",
                f"group_size={t.group_size} is small for GRPO: advantages are computed within "
                f"each prompt's group, so <4 completions gives a weak/noisy signal "
                f"(recipe default {RECIPE.rl.group_size}).",
            )
        )

    # 4. Near-zero GRPO rollout temperature collapses the group to identical completions.
    if grpo and t.temperature is not None and t.temperature < 0.5:
        out.append(
            Advice(
                "train.temperature",
                f"temperature={t.temperature:g} is low for GRPO rollouts: near-zero temperature "
                f"makes a prompt's completions nearly identical, collapsing the within-group "
                f"advantage signal (recipe default {RECIPE.rl.sampling_temperature:g}).",
            )
        )

    # 5. Learning rate too hot. The universal "very high" check and the GRPO-specific "high"
    #    check are mutually exclusive (elif) so a single high LR never double-fires.
    if t.learning_rate is not None:
        if t.learning_rate >= 1e-3:
            out.append(
                Advice(
                    "train.learning_rate",
                    f"learning_rate={t.learning_rate:g} is very high for LoRA fine-tuning and "
                    f"risks divergence (SFT default {RECIPE.sft.learning_rate:g}, GRPO default "
                    f"{RECIPE.rl.learning_rate:g}).",
                )
            )
        elif grpo and t.learning_rate >= 1e-4:
            out.append(
                Advice(
                    "train.learning_rate",
                    f"learning_rate={t.learning_rate:g} is high for GRPO (recipe default "
                    f"{RECIPE.rl.learning_rate:g}): RL is LR-sensitive and high values can "
                    f"collapse the policy.",
                )
            )

    # 6. Structural: checkpoint cadence past the run length (GRPO is step-driven). The worker
    #    writes no mid-run checkpoint when save_every exceeds the total step count.
    if (
        grpo
        and t.steps is not None
        and t.save_every is not None
        and t.save_every > t.steps
    ):
        out.append(
            Advice(
                "train.save_every",
                f"save_every={t.save_every} exceeds steps={t.steps}: no mid-run checkpoint "
                f"will be written.",
            )
        )

    return out
