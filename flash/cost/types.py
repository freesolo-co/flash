"""The estimator's I/O types: ``RunConfig`` (input) and ``CostEstimate`` (result)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from flash.catalog import normalize_algorithm
from flash.engine.recipe import RECIPE
from flash.providers import PROVIDER_NAMES


@dataclass(frozen=True)
class RunConfig:
    """One training run to price. ``None`` knobs resolve to recipe defaults."""

    model_id: str
    method: str  # "sft" | "grpo"
    steps: int

    # Multi-seed runs reprovision (and re-pay boot) per seed.
    setup_repeats: int = 1

    seq_len: int | None = None
    completion_len: int | None = None  # GRPO only (max_tokens)
    batch_size: int | None = None  # SFT effective batch / GRPO prompts_per_step
    group_size: int | None = None  # GRPO completions per prompt (G)
    lora_rank: int | None = None
    thinking: bool = False
    # GRPO only: seconds to score one completion. None -> the single average grader latency.
    reward_seconds_per_completion: float | None = None

    max_wall_seconds: int | None = None  # per-seed wall cap (spec gpu.max_wall_seconds); None = 24h
    provider: str = "auto"
    environment: str | None = None  # Freesolo environment id; descriptive only

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", normalize_algorithm(self.method))
        prov = (self.provider or "auto").strip().lower() or "auto"
        if prov not in ("auto", *PROVIDER_NAMES):
            raise ValueError(f"unknown provider {self.provider!r} (auto, {', '.join(PROVIDER_NAMES)})")
        object.__setattr__(self, "provider", prov)
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.setup_repeats < 1:
            raise ValueError(f"setup_repeats must be >= 1, got {self.setup_repeats}")
        # Steps split evenly across seeds; fractional split is impossible in a real run.
        if self.steps % self.setup_repeats != 0:
            raise ValueError(
                f"steps ({self.steps}) must be a multiple of setup_repeats ({self.setup_repeats})"
            )
        # max_wall_seconds is intentionally absent: estimate_cost floors it to 60s like the runner.
        for _name in ("seq_len", "batch_size", "group_size", "completion_len", "lora_rank"):
            _val = getattr(self, _name)
            if _val is not None and _val < 1:
                raise ValueError(f"{_name} must be >= 1, got {_val}")

    @property
    def is_grpo(self) -> bool:
        return self.method == "grpo"

    def normalized(self) -> RunConfig:
        """A copy with every ``None`` knob filled from the recipe for this method."""
        lora = self.lora_rank if self.lora_rank is not None else RECIPE.lora.rank
        if self.is_grpo:
            comp = self.completion_len
            if comp is None:
                comp = (
                    RECIPE.rl.max_completion_len_thinking
                    if self.thinking
                    else RECIPE.rl.max_completion_len
                )
            # Mirror allocator: max(1024, prompt+completion), not bare prompt (would under-size).
            seq = (
                self.seq_len
                if self.seq_len is not None
                else max(1024, RECIPE.rl.max_prompt_len + int(comp))
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
        return replace(self, seq_len=seq, completion_len=comp, batch_size=batch, group_size=group, lora_rank=lora)

    def train_knobs(self) -> dict[str, int]:
        """Knob dict for ``model_required_vram_gb``. Omitted SFT batch_size is NOT forwarded — it
        would use recipe effective batch (32) instead of the worker micro-batch (4), over-provisioning."""
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


@dataclass(frozen=True)
class CostEstimate:
    """A pre-flight estimate. ``total_usd`` = ``wall_clock_hours * gpu_hourly_usd``, no multiplier."""

    model_id: str
    method: str
    steps: int
    gpu: str
    provider: str
    gpu_vram_gb: int
    required_vram_gb: int
    gpu_hourly_usd: float
    setup_seconds: float  # cold start: boot + deps + model load (+ vLLM init for GRPO)
    seconds_per_step: float
    train_seconds: float  # steps * seconds_per_step (post wall-clock cap)
    wall_clock_seconds: float
    wall_capped: bool
    total_usd: float
    notes: tuple[str, ...] = ()

    @property
    def wall_clock_hours(self) -> float:
        return self.wall_clock_seconds / 3600.0

    def breakdown(self) -> str:
        """Multi-line itemized breakdown for CLI output."""
        lines = [
            f"Run        : {self.model_id}  [{self.method.upper()}, {self.steps} steps]",
            f"GPU        : {self.gpu} on {self.provider} "
            f"({self.gpu_vram_gb} GB; run needs >= {self.required_vram_gb} GB) "
            f"@ ${self.gpu_hourly_usd:.2f}/hr",
            f"Setup      : {self.setup_seconds / 60:.1f} min (cold start: boot + deps + model load"
            + (" + vLLM init" if self.method == "grpo" else "")
            + ")",
            f"Per step   : {self.seconds_per_step:.2f} s",
            f"Train      : {self.train_seconds / 60:.1f} min"
            + ("  [CAPPED at the wall-clock limit]" if self.wall_capped else ""),
            f"Wall clock : {self.wall_clock_hours:.2f} h",
            f"TOTAL      : ${self.total_usd:.2f}",
        ]
        if self.notes:
            lines.append("Notes      :")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)
