"""The estimator's input: a ``RunConfig`` plus recipe-default knob resolution.

A ``RunConfig`` is the small, dependency-free description of a training run the
estimator prices. Any knob left ``None`` is filled from the frozen ``flash`` recipe
(``engine.recipe.RECIPE``) according to the method, exactly as a real run would
default it -- so an estimate for ``RunConfig(model_id, "grpo", steps=150)`` uses the
same group size / completion budget the worker would.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flash.catalog import normalize_algorithm
from flash.engine.recipe import RECIPE


@dataclass(frozen=True)
class RunConfig:
    """One training run to price. ``None`` knobs resolve to recipe defaults."""

    model_id: str
    method: str  # "sft" | "grpo"
    steps: int

    # Number of cold-start setups the bill covers. A single run cold-starts once
    # (=1); a multi-seed run reprovisions and bills setup once PER SEED (runner.py
    # runs each seed as its own job), so the reconstruction sets this to the seed
    # count to charge the repeated boot/deps/download/vLLM-init the measured cost paid.
    setup_repeats: int = 1

    seq_len: int | None = None  # SFT max_seq_len / GRPO max_prompt_len
    completion_len: int | None = None  # GRPO only (max_tokens)
    batch_size: int | None = None  # SFT effective batch / GRPO prompts_per_step
    group_size: int | None = None  # GRPO completions per prompt (G)
    lora_rank: int | None = None
    thinking: bool = False
    # GRPO only: seconds to score one completion through the verifiers reward function.
    # None -> inferred from the environment (rewards.py); a slow reward (code-exec,
    # LLM judge) can dominate the step.
    reward_seconds_per_completion: float | None = None

    gpu: str | None = None  # pin a class; else cheapest fitting
    # Whether the run permits unvalidated GPU classes (spec ``gpu.allow_unvalidated``).
    # The allocator widens its candidate pool when this is set, so the cheapest fitting
    # class -- and thus the priced $/hr -- can be an unvalidated one. Mirror it here so a
    # preflight estimate matches the allocator's pick for such specs.
    allow_unvalidated: bool = False
    # Per-run total wall-clock cap in seconds (spec ``gpu.max_wall_seconds``; default 24h).
    # The runner applies this PER SEED (each seed is its own job), so the estimator clamps
    # each seed's setup+train to it. ``None`` -> the estimator's default cap.
    max_wall_seconds: int | None = None
    provider: str = "auto"
    # Verifiers environment slug (owner/name). Descriptive only -- the dataset/reward
    # source doesn't drive GPU-time cost -- but carried so a run is fully specified.
    environment: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        # Normalize/validate the method early so downstream code can trust it.
        object.__setattr__(self, "method", normalize_algorithm(self.method))
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.setup_repeats < 1:
            raise ValueError(f"setup_repeats must be >= 1, got {self.setup_repeats}")
        # Explicit run knobs (None -> recipe default) must be positive; a <= 0 value
        # produces a bogus quote (zero/negative tokens or completions per step). lora_rank
        # is included to mirror the real parser's ``train.lora_rank < 1`` rejection.
        for _name in ("seq_len", "batch_size", "group_size", "completion_len", "lora_rank"):
            _val = getattr(self, _name)
            if _val is not None and _val < 1:
                raise ValueError(f"{_name} must be >= 1, got {_val}")

    @property
    def is_grpo(self) -> bool:
        return self.method == "grpo"

    def display(self) -> str:
        return self.label or f"{self.model_id} {self.method} x{self.steps}"

    def normalized(self) -> RunConfig:
        """A copy with every ``None`` knob filled from the recipe for this method."""
        lora = self.lora_rank if self.lora_rank is not None else RECIPE.lora.rank
        if self.is_grpo:
            seq = self.seq_len if self.seq_len is not None else RECIPE.rl.max_prompt_len
            comp = self.completion_len
            if comp is None:
                comp = (
                    RECIPE.rl.max_completion_len_thinking
                    if self.thinking
                    else RECIPE.rl.max_completion_len
                )
            batch = self.batch_size if self.batch_size is not None else RECIPE.rl.prompts_per_step
            group = self.group_size if self.group_size is not None else RECIPE.rl.group_size
        else:
            seq = self.seq_len
            if seq is None:
                seq = RECIPE.sft.max_seq_len_thinking if self.thinking else RECIPE.sft.max_seq_len
            comp = None
            batch = self.batch_size if self.batch_size is not None else RECIPE.sft.effective_batch
            group = None
        return replace(
            self,
            seq_len=seq,
            completion_len=comp,
            batch_size=batch,
            group_size=group,
            lora_rank=lora,
        )

    def train_knobs(self) -> dict[str, int]:
        """The knob dict ``engine.vram.model_required_vram_gb`` consumes (for sizing).

        Mirrors what the REAL allocator passes: the parsed ``TrainSpec`` leaves an omitted
        ``batch_size`` as ``None``, and ``model_required_vram_gb`` then defaults it to 1 for
        SFT VRAM sizing (the activations term scales with batch). So an omitted batch_size is
        left OUT here -- the recipe effective batch (32) is the right model of compute
        throughput in ``seconds_per_step``, but feeding it into VRAM sizing over-provisions
        relative to the allocator. Only an EXPLICIT batch_size is forwarded for sizing.
        """
        n = self.normalized()
        knobs: dict[str, int] = {"lora_rank": n.lora_rank}
        if self.batch_size is not None:
            knobs["batch_size"] = self.batch_size
        if n.seq_len is not None:
            knobs["max_length"] = n.seq_len
        if n.completion_len is not None:
            knobs["max_tokens"] = n.completion_len
        if n.group_size is not None:
            knobs["group_size"] = n.group_size
        return knobs
