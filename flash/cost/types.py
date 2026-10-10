"""The estimator's I/O types: ``RunConfig`` (input) and ``CostEstimate`` (result)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from flash.core.catalog import normalize_algorithm, optimizer_batch_key, samples_on_policy
from flash.core.spec import parse_positive_int_tuple
from flash.cost.currency import format_usd
from flash.engine.plan.recipe import RECIPE
from flash.providers.core.base import GPU_INFO, canonical_gpu, providers_for
from flash.providers.core.registry import PROVIDER_NAMES, validated_provider_preferences


def vram_clause(per_card_gb: int, offered_gb: int, gpu_count: int) -> str:
    """The VRAM half of a quote's GPU line, in terms comparable to the run's requirement.

    A single-card shape offers exactly its card, so it keeps the plain spelling. A multi-card
    shape states the pooled figure the fit gate actually used and keeps the per-card number
    beside it, because that is the one the user checks against a provider's listing.
    """
    if gpu_count > 1 and offered_gb != per_card_gb:
        return f"{offered_gb} GB usable across {gpu_count}x {per_card_gb} GB"
    return f"{per_card_gb} GB"


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
    # opd only: the canonical friendly teacher alias from [train].teacher_model.
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
    # spec gpu.count ceiling. none means the user left it unset, so the offline quote auto-sizes the
    # same structural shape allocation may rent; an integer is an authored hard ceiling.
    gpu_count: int | None = None
    supervised_train_tokens: int | None = None
    sft_packing_mode: str = ""
    sft_packed_blocks: int | None = None
    opd_multi_turn: bool = False
    opd_max_turns: int | None = None
    # rows the trainer iterates (profile `retained_examples`), which is what verl's sampler shards.
    # NOT derivable from sft_packed_blocks: that is ceil(rows / examples_per_update), so a packed
    # profile with 10 rows and a batch of 8 reports 2 blocks and reconstructs as 16.
    # APPENDED, not slotted beside the other sft fields: this class keeps a positional constructor
    # (see `test_runconfig_preserves_old_positional_constructor`), so inserting here would shift
    # every later field and silently rebind an old caller's opd flag to this one.
    sft_retained_examples: int | None = None
    # appended for the same positional-constructor reason as sft_retained_examples above.
    providers: tuple[str, ...] = ()
    # the remaining classes an ordered `[gpu] type` list accepts, after gpu_type takes the head.
    # allocation cost-ranks the whole acceptable set, so quoting the head alone prices a shape the
    # run may never be given: an authored ["B200", "H100"] is quoted on B200 while allocate() would
    # rent the cheaper H100, and the affordability precheck runs on that inflated estimate.
    # appended for the same positional-constructor reason as the two fields above.
    gpu_type_fallbacks: tuple[str, ...] = ()

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
        providers = validated_provider_preferences(
            self.providers, allow_empty=isinstance(self.providers, tuple)
        )
        if prov != "auto" and providers:
            raise ValueError("provider and providers cannot both be set")
        object.__setattr__(self, "providers", providers)
        exact = ""
        if self.gpu_type:
            exact = canonical_gpu(self.gpu_type)
            info = GPU_INFO.get(exact)
            if info is None or not info.validated:
                raise ValueError(f"gpu_type {exact!r} must name an active validated GPU class")
            if prov != "auto" and prov not in providers_for(exact):
                raise ValueError(f"provider {prov!r} cannot provision gpu_type {exact!r}")
        object.__setattr__(self, "gpu_type", exact)
        # canonicalized and validated exactly like the head above, so a quote cannot rank a class the
        # spec layer would have rejected. deduped against the head and each other because the
        # ranking loop would otherwise price the same class twice.
        fallbacks: list[str] = []
        for entry in self.gpu_type_fallbacks or ():
            if not isinstance(entry, str):
                raise TypeError("gpu_type_fallbacks entries must be strings")
            name = canonical_gpu(entry)
            info = GPU_INFO.get(name)
            if info is None or not info.validated:
                raise ValueError(
                    f"gpu_type_fallbacks entry {name!r} must name an active validated GPU class"
                )
            if name != exact and name not in fallbacks:
                fallbacks.append(name)
        if fallbacks and not exact:
            raise ValueError("gpu_type_fallbacks requires gpu_type")
        object.__setattr__(self, "gpu_type_fallbacks", tuple(fallbacks))
        if not isinstance(self.model_revision, str):
            raise TypeError("model_revision must be a string")
        object.__setattr__(self, "model_revision", self.model_revision.strip())
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.gpu_count is not None:
            if isinstance(self.gpu_count, bool) or not isinstance(self.gpu_count, int):
                raise TypeError("gpu_count must be an integer or none")
            if self.gpu_count < 1:
                raise ValueError(f"gpu_count must be >= 1, got {self.gpu_count}")
            # upper bound mirrors gpuspec's 1..8 so a direct runconfig cannot price a count the spec
            # layer would reject.
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
        if self.supervised_train_tokens is not None:
            if self.supervised_train_tokens < 1:
                raise ValueError("supervised_train_tokens must be >= 1")
            if self.train_tokens is None or self.supervised_train_tokens > self.train_tokens:
                raise ValueError("supervised_train_tokens cannot exceed train_tokens")
        if self.sft_packing_mode not in {"", "packed", "exact-unpacked"}:
            raise ValueError("unsupported sft_packing_mode")
        if self.sft_packed_blocks is not None and self.sft_packed_blocks < 1:
            raise ValueError("sft_packed_blocks must be >= 1")
        if self.sft_retained_examples is not None:
            # a bad value here does not raise downstream, it UNDER-CREDITS: `sft_data_parallel_cards`
            # reads a non-positive row count as "unknown, do not constrain", so 0 or a negative
            # silently credits every rented card and quotes a width the run will not launch on --
            # the exact failure the retained-example count was added to prevent. reject it here,
            # where the type boundary is, rather than teaching the width rule to distrust its input.
            if isinstance(self.sft_retained_examples, bool) or not isinstance(
                self.sft_retained_examples, int
            ):
                raise TypeError("sft_retained_examples must be an integer or none")
            if self.sft_retained_examples < 1:
                raise ValueError(
                    f"sft_retained_examples must be >= 1, got {self.sft_retained_examples}"
                )
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
            if self.is_grpo:
                batch = self.batch_size if self.batch_size is not None else rc.prompts_per_step
                group = self.group_size if self.group_size is not None else rc.group_size
            else:
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
            # `RunConfig.batch_size` is the optimizer batch whatever the algorithm; the TRAIN TABLE
            # spells it per algorithm and sizing reads it by that name, so emit the name sizing
            # will look for or an authored rl batch silently sizes as the recipe default.
            knobs[optimizer_batch_key(self.method)] = self.batch_size
        if n.seq_len is not None:
            knobs["max_context_tokens"] = n.seq_len
        if n.completion_len is not None:
            knobs["max_completion_tokens"] = n.completion_len
        if n.group_size is not None:
            knobs["group_size"] = n.group_size
        return knobs


@dataclass(frozen=True)
class CostEstimate:
    """report a pre-flight estimate.

    ``total_usd`` bills training gpu hours only. setup is elapsed but unbilled; opd teacher api cost is
    itemized separately because it uses the platform-managed key.
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
    # ranks that actually JOIN the run, which a small batch can bound below the billed gpu_count
    # (``analytical.executed_gpu_count``). The fit gate values a shape at this width, not the
    # rented one, so it is what ``offered_vram_gb`` has to quote. APPENDED with a default for the
    # same positional-constructor reason as RunConfig's trailing fields; 0 means "not stamped",
    # which falls back to gpu_count rather than claiming a width nobody computed.
    executed_gpu_count: int = 0

    @property
    def wall_clock_hours(self) -> float:
        return self.wall_clock_seconds / 3600.0

    @property
    def billable_hours(self) -> float:
        return self.train_seconds / 3600.0

    @property
    def joined_gpu_count(self) -> int:
        """Ranks that actually join the run: the stamped executed width, else the billed count."""
        return max(1, self.executed_gpu_count or self.gpu_count)

    @property
    def offered_vram_gb(self) -> int:
        """VRAM this SHAPE offers the run, valued the way the allocator's fit gate valued it.

        ``required_vram_gb`` is a whole-run figure, so pairing it with the per-card
        ``gpu_vram_gb`` compares two different quantities: a quote that passed the fit gate on
        pooled capacity then renders as "180 GB; needs >= 199 GB" and reads as a rejection of
        the shape it just recommended. ``combined_vram_gb`` is the allocator's own fit model,
        so quoting through it makes the two numbers comparable on every count.

        Valued at the EXECUTED width, not the billed one. A card that never joins the fsdp group
        contributes no memory, so crediting a rented-but-idle card would claim capacity the run
        does not have -- the same mistake ``_executed_width`` exists to stop on the allocator
        side. An sft job whose one-row batch launches a single rank on two rented cards is
        offered one card's memory, and the quote has to say so.
        """
        from flash.providers.core.sharding import combined_vram_gb

        return int(combined_vram_gb(self.gpu_vram_gb, self.joined_gpu_count))

    def breakdown(self) -> str:
        """Multi-line itemized breakdown for CLI output."""
        lines = [
            f"Run        : {self.model_id}  [{self.method.upper()}, {self.steps} steps]",
            (
                f"GPU        : {f'{self.gpu_count}x ' if self.gpu_count > 1 else ''}{self.gpu} "
                f"({vram_clause(self.gpu_vram_gb, self.offered_vram_gb, self.joined_gpu_count)}; "
                f"run needs >= {self.required_vram_gb} GB) "
                f"@ ${self.gpu_hourly_usd:.2f}/hr{' per card' if self.gpu_count > 1 else ''}"
            ),
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
        lines.append(f"TOTAL      : {format_usd(self.total_usd)}")
        if self.notes:
            lines.append("Notes      :")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)
