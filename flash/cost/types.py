"""The estimator's I/O types: ``RunConfig`` (input) and ``CostEstimate`` (result)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from flash.catalog import normalize_algorithm
from flash.engine.recipe import RECIPE
from flash.providers import PROVIDER_NAMES


@dataclass(frozen=True)
class RunConfig:
    """One training run to price. ``None`` knobs resolve to recipe defaults."""

    model_id: str
    method: str  # "sft" | "grpo"
    steps: int

    # Cold-start setups the bill covers. A multi-seed run reprovisions per seed (each seed is
    # its own job in runner.py), so set this to the seed count to charge the repeated boot.
    setup_repeats: int = 1

    seq_len: int | None = None  # SFT max_seq_len / GRPO max_prompt_len
    completion_len: int | None = None  # GRPO only (max_tokens)
    batch_size: int | None = None  # SFT effective batch / GRPO prompts_per_step
    group_size: int | None = None  # GRPO completions per prompt (G)
    lora_rank: int | None = None
    thinking: bool = False
    # GRPO only: seconds to score one completion. None -> inferred from the environment.
    reward_seconds_per_completion: float | None = None

    gpu: str | None = None  # pin a class; else cheapest fitting
    # Tri-state, mirroring the spec/allocator: None means "unspecified" and is resolved at
    # selection time via ``providers.base.unvalidated_allowed`` (the managed default). Don't
    # coerce a missing value to False, or the estimate would disagree with the allocator's pick.
    allow_unvalidated: bool | None = None
    # Per-run wall-clock cap in seconds (spec ``gpu.max_wall_seconds``; default 24h), applied
    # per seed. None -> the estimator's default cap.
    max_wall_seconds: int | None = None
    provider: str = "auto"
    environment: str | None = None  # verifiers env slug (owner/name); descriptive only
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", normalize_algorithm(self.method))
        # Same for the provider: GPU selection (``pick_gpu``) and the allocator expect an exact
        # substrate ("runpod"/"vast") or the "auto" sentinel. Normalize case/whitespace, treat
        # empty as "auto", and reject an unknown substrate here (mirrors ``allocator.allocate``'s
        # guard) -- otherwise a variant like "RunPod"/" vast " filters out every candidate and
        # surfaces as a confusing "no GPU class fits" instead of a clear provider error.
        prov = (self.provider or "auto").strip().lower() or "auto"
        if prov not in ("auto", *PROVIDER_NAMES):
            raise ValueError(
                f"unknown provider {self.provider!r} (auto, {', '.join(PROVIDER_NAMES)})"
            )
        object.__setattr__(self, "provider", prov)
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.setup_repeats < 1:
            raise ValueError(f"setup_repeats must be >= 1, got {self.setup_repeats}")
        # ``estimate_cost`` divides steps evenly across seeds, so a non-divisible split would
        # price fractional steps per seed (impossible in a real run) and skew per-seed capping.
        if self.steps % self.setup_repeats != 0:
            raise ValueError(
                f"steps ({self.steps}) must be a multiple of setup_repeats "
                f"({self.setup_repeats}): each seed runs an equal share of the steps"
            )
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
        """The knob dict ``engine.vram.model_required_vram_gb`` consumes for VRAM sizing.

        Only an EXPLICIT batch_size is forwarded: the allocator leaves an omitted batch at
        None (sized as 1), so feeding the recipe effective batch here would over-provision.
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


@dataclass(frozen=True)
class CostEstimate:
    """A pre-flight training-cost estimate. ``total_usd`` = ``wall_clock_hours * gpu_hourly_usd``."""

    # the run this estimate is for (echoed for traceability)
    model_id: str
    method: str  # "sft" | "grpo"
    steps: int

    # chosen hardware
    gpu: str
    provider: str
    gpu_vram_gb: int
    required_vram_gb: int
    gpu_hourly_usd: float

    # timing breakdown (seconds)
    setup_seconds: float  # cold start: boot + deps + model download (+ vLLM init for GRPO)
    seconds_per_step: float
    train_seconds: float  # steps * seconds_per_step (post wall-clock cap)
    wall_clock_seconds: float  # setup + train
    wall_capped: bool

    total_usd: float  # wall-clock hours x market $/hr, no output adjustment
    notes: tuple[str, ...] = ()

    @property
    def wall_clock_hours(self) -> float:
        return self.wall_clock_seconds / 3600.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["wall_clock_hours"] = self.wall_clock_hours
        return d

    def describe(self) -> str:
        """One-line human summary; the figure is exactly ``$/hr x h``."""
        cap = " (wall-capped)" if self.wall_capped else ""
        return (
            f"{self.model_id} {self.method.upper()} x{self.steps} steps -> "
            f"${self.total_usd:.2f} (${self.gpu_hourly_usd:.2f}/hr x {self.wall_clock_hours:.2f}h{cap}) "
            f"on {self.gpu}@{self.provider}"
        )

    def breakdown(self) -> str:
        """Multi-line itemized breakdown for CLI output."""
        lines = [
            f"Run        : {self.model_id}  [{self.method.upper()}, {self.steps} steps]",
            f"GPU        : {self.gpu} on {self.provider} "
            f"({self.gpu_vram_gb} GB; run needs >= {self.required_vram_gb} GB) "
            f"@ ${self.gpu_hourly_usd:.2f}/hr",
            f"Setup      : {self.setup_seconds / 60:.1f} min (cold start: boot + deps + download"
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
