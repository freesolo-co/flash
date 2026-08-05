"""The estimator's I/O types: ``RunConfig`` (input) and ``CostEstimate`` (result)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from flash.catalog import normalize_algorithm, samples_on_policy
from flash.engine.recipe import RECIPE
from flash.providers import PROVIDER_NAMES
from flash.providers.base import GPU_INFO, canonical_gpu, providers_for
from flash.spec import parse_positive_int_tuple


@dataclass(frozen=True)
class RunConfig:
    """One training run to price. ``None`` knobs resolve to recipe defaults."""

    model_id: str
    method: str  # "sft" | "grpo" | "opd"
    steps: int

    # Engine context length (forwarded as [train].max_context_tokens, NOT prompt length). When unset the
    # GRPO default mirrors the worker's max(1024, max_prompt_len + completion); see normalized().
    seq_len: int | None = None
    completion_len: int | None = None  # GRPO/OPD only (max_completion_tokens)
    batch_size: int | None = None  # SFT effective batch / GRPO prompts_per_step
    group_size: int | None = None  # GRPO completions per prompt (G)
    lora_rank: int | None = None
    thinking: bool = False
    # GRPO only: seconds to score one completion. None -> the single average grader latency.
    reward_seconds_per_completion: float | None = None
    # opd only: the parasail teacher model id (already resolved from [train].teacher_model at parse).
    # prices the teacher-api estimate; an empty value resolves to the default glm 5.2 teacher (an
    # omitted [train].teacher_model).
    teacher_model: str = ""

    max_wall_seconds: int | None = None  # wall cap (spec gpu.max_wall_seconds); None = 24h
    provider: str = "auto"
    environment: str | None = None  # Freesolo environment id; descriptive only
    # sft only: actual training tokens across all epochs. when present, sft dollars are priced
    # from this token count instead of the padded batch_size * seq_len slot estimate.
    train_tokens: int | None = None
    save_at_steps: tuple[int, ...] = ()
    gpu_type: str = ""
    model_revision: str = ""
    # spec gpu.disk_gb, carried so a pinned quote allocates at the run's real disk floor (parity
    # with the launch allocate call), keeping the persisted quote aligned with the pinned hardware.
    disk_gb: float = 0.0
    # Spec gpu.count: cards the job occupies. total cost scales linearly with it (n cards for the
    # billed training wall); 1 = the historical single-gpu quote.
    gpu_count: int = 1
    opd_multi_turn: bool = False
    opd_max_turns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", normalize_algorithm(self.method))
        # Normalize like the allocator (case/whitespace, empty -> "auto") and reject an unknown
        # substrate up front (else it filters out every candidate -> confusing "no GPU fits").
        prov = (self.provider or "auto").strip().lower() or "auto"
        if prov not in ("auto", *PROVIDER_NAMES):
            raise ValueError(
                f"unknown provider {self.provider!r} (auto, {', '.join(PROVIDER_NAMES)})"
            )
        object.__setattr__(self, "provider", prov)
        exact = ""
        if self.gpu_type:
            exact = canonical_gpu(self.gpu_type)
            info = GPU_INFO.get(exact)
            if info is None or not info.validated:
                raise ValueError(f"gpu_type {exact!r} must name an active validated GPU class")
            if prov != "auto" and prov not in providers_for(exact):
                raise ValueError(f"provider {prov!r} cannot provision gpu_type {exact!r}")
        object.__setattr__(self, "gpu_type", exact)
        if not isinstance(self.model_revision, str):
            raise TypeError("model_revision must be a string")
        object.__setattr__(self, "model_revision", self.model_revision.strip())
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if isinstance(self.gpu_count, bool) or not isinstance(self.gpu_count, int):
            raise TypeError("gpu_count must be an integer")
        if self.gpu_count < 1:
            raise ValueError(f"gpu_count must be >= 1, got {self.gpu_count}")
        # upper bound mirrors GpuSpec's 1..8 so a direct RunConfig(gpu_count=...) cannot price a
        # count the spec layer would reject.
        if self.gpu_count > 8:
            raise ValueError(f"gpu_count must be <= 8, got {self.gpu_count}")
        # Reject 0/negative positive-only knobs (bogus quote). max_wall_seconds is NOT here: the
        # runner floors it to max(60, ...) and estimate_cost mirrors that, so a non-positive cap
        # is accepted (floored to 60s), not rejected.
        for _name in ("seq_len", "batch_size", "group_size", "completion_len", "lora_rank"):
            _val = getattr(self, _name)
            if _val is not None and _val < 1:
                raise ValueError(f"{_name} must be >= 1, got {_val}")
        if self.train_tokens is not None and self.train_tokens < 1:
            raise ValueError(f"train_tokens must be >= 1, got {self.train_tokens}")
        if not isinstance(self.opd_multi_turn, bool):
            raise TypeError("opd_multi_turn must be a boolean")
        if self.opd_max_turns is not None and (
            isinstance(self.opd_max_turns, bool) or not isinstance(self.opd_max_turns, int)
        ):
            raise TypeError("opd_max_turns must be an integer")
        save_at_steps = parse_positive_int_tuple(self.save_at_steps, name="save_at_steps")
        if save_at_steps and save_at_steps[-1] > self.steps:
            raise ValueError("save_at_steps cannot exceed steps")
        object.__setattr__(self, "save_at_steps", save_at_steps)

    @property
    def is_grpo(self) -> bool:
        return self.method == "grpo"

    @property
    def is_opd(self) -> bool:
        return self.method == "opd"

    @property
    def has_rollout(self) -> bool:
        """True when a step samples on-policy student completions."""
        return samples_on_policy(self.method)

    def normalized(self) -> RunConfig:
        """A copy with every ``None`` knob filled from the recipe for this method."""
        lora = self.lora_rank if self.lora_rank is not None else RECIPE.lora.rank
        if self.has_rollout:
            # rollout algorithms use either the opd-style single-rollout recipe or grpo's group recipe.
            rc = RECIPE.opd if self.is_opd else RECIPE.rl
            comp = self.completion_len
            if comp is None:
                comp = rc.max_completion_len_thinking if self.thinking else rc.max_completion_len
            # Explicit pin wins; else mirror the allocator's rollout sizing of an unset context:
            # max(1024, max_prompt_len + completion), not bare max_prompt_len (which under-sizes).
            seq = (
                self.seq_len
                if self.seq_len is not None
                else max(1024, rc.max_prompt_len + int(comp))
            )
            batch = self.batch_size if self.batch_size is not None else rc.prompts_per_step
            group = self.group_size if self.group_size is not None else rc.group_size
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
        """The knob dict ``model_required_vram_gb`` consumes. Only an EXPLICIT batch_size is
        forwarded -- an omitted SFT batch sizes as the worker's micro-batch (4), not the recipe's
        effective batch (32), which would over-provision."""
        n = self.normalized()
        knobs: dict[str, int] = {"lora_rank": n.lora_rank}
        if self.batch_size is not None:
            knobs["batch_size"] = self.batch_size
        if n.seq_len is not None:
            knobs["max_context_tokens"] = n.seq_len
        if n.completion_len is not None:
            knobs["max_completion_tokens"] = n.completion_len
        if n.group_size is not None:
            knobs["group_size"] = n.group_size
        return knobs


@dataclass(frozen=True)
class CostEstimate:
    """A pre-flight estimate.

    ``total_usd`` = training-only GPU hours * ``gpu_hourly_usd``. Setup/cold-start time is reported
    as elapsed wall time but is not billed to the user estimate. ``teacher_api_usd`` (opd only) uses
    the platform-managed teacher key and remains itemized separately from the customer GPU charge,
    so it is not included in ``total_usd``.
    """

    model_id: str
    method: str
    steps: int
    gpu: str
    provider: str
    gpu_vram_gb: int
    required_vram_gb: int
    gpu_hourly_usd: float
    setup_seconds: float  # cold start: boot + deps + model load (+ vllm init for grpo/opd)
    seconds_per_step: float
    train_seconds: float  # steps * seconds_per_step (post wall-clock cap)
    wall_clock_seconds: float
    wall_capped: bool
    total_usd: float
    # cards the job occupies; total_usd already reflects n-card billing. 1 = single-gpu quote.
    gpu_count: int = 1
    # opd only: external parasail teacher token spend (0.0 for sft/grpo). billed by parasail
    # to the platform-managed teacher key (users don't supply one), tracked separately from the
    # platform-billed gpu charge, so it is not part of total_usd and is shown as its own itemized
    # diagnostic line only.
    teacher_api_usd: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def wall_clock_hours(self) -> float:
        return self.wall_clock_seconds / 3600.0

    @property
    def billable_hours(self) -> float:
        return self.train_seconds / 3600.0

    def breakdown(self) -> str:
        """Multi-line itemized breakdown for CLI output."""
        lines = [
            f"Run        : {self.model_id}  [{self.method.upper()}, {self.steps} steps]",
            f"GPU        : {f'{self.gpu_count}x ' if self.gpu_count > 1 else ''}{self.gpu} "
            f"({self.gpu_vram_gb} GB; run needs >= {self.required_vram_gb} GB) "
            f"@ ${self.gpu_hourly_usd:.2f}/hr{' per card' if self.gpu_count > 1 else ''}",
            f"Setup      : {self.setup_seconds / 60:.1f} min (cold start: boot + deps + model load"
            + (" + vLLM init" if self.method == "grpo" else "")
            + "; not billed)",
            f"Per step   : {self.seconds_per_step:.2f} s",
            f"Train      : {self.train_seconds / 60:.1f} min"
            + ("  [CAPPED at the wall-clock limit]" if self.wall_capped else ""),
            f"Wall clock : {self.wall_clock_hours:.2f} h",
            f"Billable   : {self.billable_hours:.2f} h (training only)",
        ]
        if self.teacher_api_usd > 0:
            lines.append(
                f"Teacher API: ${self.teacher_api_usd:.2f} (Parasail teacher token spend on the "
                "platform-managed teacher key — tracked separately, NOT included in TOTAL)"
            )
        lines.append(f"TOTAL      : ${self.total_usd:.2f}")
        if self.notes:
            lines.append("Notes      :")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)
