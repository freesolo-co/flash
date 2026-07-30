"""Which concurrent runs should share one set of gpus.

A grpo step is two halves that do not overlap: the card generates and updates, then it sits idle
while the env grades the completions one at a time. ``step_seconds_split`` already reports those
halves separately. At the default 8x4 shape a 0.85s sandbox grader leaves an H100 idle for ~82% of
every step, and a 3.1s llm judge for ~94%.

That idle is schedulable. Runs placed on the same cards interleave: while run A waits on its
grader, run B has the card to itself. The latency that decides this is a MEASURED one --
``flash.engine.reward_profile`` times the env's real grader before training starts, and the cost
model's single global average (1.0 s/completion, against real graders spanning ~0.01s to ~3s)
cannot tell a regex env from an llm judge. Those two belong on opposite sides of a placement.

Two things bound a group, and both are hard:

- **Card time.** The gpu serves one tenant at a time, so a group's duty cycles sum. Under 1.0 the
  demands fit inside each other's idle and nobody is slowed; over it, everyone stretches.
- **VRAM.** Tenants are concurrent processes with resident weights, optimizer state and kv cache.
  Unlike card time, this does not timeshare -- exceeding it is an OOM, not a slowdown. A group is
  admissible only if every member is resident simultaneously.

This module is pure arithmetic over already-computed step splits and vram figures: no i/o, no env
code, no provider calls. It decides placement; it does not perform it.
"""

from __future__ import annotations

from dataclasses import dataclass

# sharing is never free: concurrent processes on one card contend for sm time, memory bandwidth and
# the pcie link even when their busy phases nominally interleave. the multi-card scaling work
# measured the same effect across cards; an interleaved pair retains ~0.9 of its ideal packing.
_PAIR_EFFICIENCY = 0.9

# each tenant beyond the first costs again: more context switching, more allocator pressure, more
# cache thrash. efficiency is _PAIR_EFFICIENCY ** (tenants - 1), so a triple retains ~0.81 and a
# quad ~0.73. this is what stops groups growing without bound when duty cycles alone would allow it.

# a placement must beat running its members back to back by a real margin. within a percent of even,
# the difference is inside the noise of the step model itself, and paying scheduling complexity for
# it is not a trade worth making.
_MIN_THROUGHPUT_GAIN = 1.01

# a group is capped well below what card time alone would permit. beyond this the efficiency model
# is extrapolating past anything the multi-card work measured, and an over-packed card degrades in
# ways (allocator fragmentation, kv eviction) this arithmetic does not represent.
_MAX_TENANTS = 4


@dataclass(frozen=True)
class RunShape:
    """One run's step, split into the part a card does and the part it waits through.

    ``gpu_seconds`` is the gpu-bound half of a step and ``reward_seconds`` the off-gpu half --
    exactly the two values ``step_seconds_split`` returns. ``vram_gb`` is what the run needs
    resident (``model_required_vram_gb``); it does not timeshare. ``label`` identifies the run for
    the caller; nothing here interprets it.
    """

    label: str
    gpu_seconds: float
    reward_seconds: float
    vram_gb: float = 0.0

    def __post_init__(self) -> None:
        if self.gpu_seconds < 0 or self.reward_seconds < 0:
            raise ValueError(f"{self.label}: step seconds cannot be negative")
        if self.vram_gb < 0:
            raise ValueError(f"{self.label}: vram cannot be negative")

    @property
    def step_seconds(self) -> float:
        return self.gpu_seconds + self.reward_seconds

    @property
    def duty_cycle(self) -> float:
        """Share of wall time this run actually needs the card for.

        This is the quantity that decides packing, and it is deliberately a RATIO rather than a
        duration. Two runs whose steps differ by 100x share a card exactly as well as two runs with
        identical steps, as long as each leaves the same proportion of its cycle idle -- a step
        boundary is not a barrier between them. Comparing durations would invent a conflict between
        a fast run and a slow one that does not exist in the hardware.
        """
        total = self.step_seconds
        return 1.0 if total <= 0 else self.gpu_seconds / total

    @property
    def idle_fraction(self) -> float:
        """Share of this run's step the card spends waiting on the grader."""
        return 1.0 - self.duty_cycle


@dataclass(frozen=True)
class Placement:
    """A set of runs proposed for one set of cards, with what the placement is worth."""

    runs: tuple[RunShape, ...]
    wall_stretch: float  # how much longer each member's own step takes when shared

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(r.label for r in self.runs)

    @property
    def vram_gb(self) -> float:
        """Total resident vram this placement needs. Concurrent tenants do not timeshare memory."""
        return sum(r.vram_gb for r in self.runs)

    @property
    def throughput_gain(self) -> float:
        """Steps per card-hour when shared, over running the same runs one after another.

        N runs on one card do N times the work per unit of wall time and each pays ``wall_stretch``
        for it, so the ratio is ``N / wall_stretch``. A stretch of exactly N is break-even: the
        group finishes in the same total time the runs would have taken in sequence.
        """
        return len(self.runs) / self.wall_stretch

    @property
    def worth_sharing(self) -> bool:
        # a group of one has a gain of exactly 1.0 (one run, no stretch), so the threshold alone
        # already excludes it. the tenant count is not re-checked here.
        return self.throughput_gain >= _MIN_THROUGHPUT_GAIN

    def describe(self) -> str:
        idle = " / ".join(f"{r.idle_fraction:.0%}" for r in self.runs)
        return (
            f"{' + '.join(self.labels)}: {self.throughput_gain:.2f}x throughput "
            f"(each step {self.wall_stretch:.2f}x longer; idle {idle})"
        )


def sharing_efficiency(tenants: int) -> float:
    """What fraction of ideal packing a group of ``tenants`` retains."""
    return _PAIR_EFFICIENCY ** max(0, tenants - 1)


def wall_stretch(runs: tuple[RunShape, ...]) -> float:
    """How much longer each member's step takes with the rest of the group on the same card.

    The card serves one tenant at a time, so the group needs ``sum(duty)`` card-time per unit of
    wall time each member used to have to itself. Below a combined duty cycle of 1.0 the demands fit
    inside each other's idle and nobody is slowed -- that is the whole point of grouping a
    compute-bound run with latency-bound ones. Above 1.0 the excess is exactly the stretch.

    Divided by the group's sharing efficiency, because concurrent processes contend even when their
    phases interleave cleanly.
    """
    if not runs:
        return 1.0
    combined = sum(r.duty_cycle for r in runs)
    return max(1.0, combined) / sharing_efficiency(len(runs))


def evaluate_placement(runs: tuple[RunShape, ...]) -> Placement:
    """What co-locating ``runs`` on one set of cards is worth."""
    return Placement(runs, wall_stretch=wall_stretch(runs))


def plan_colocation(
    runs: list[RunShape],
    *,
    vram_capacity_gb: float = 0.0,
    max_tenants: int = _MAX_TENANTS,
) -> tuple[list[Placement], list[RunShape]]:
    """Pack ``runs`` onto shared cards. Returns (placements, runs left on their own).

    Greedy, and deliberately so: seed each group with the least idle run still free -- the one
    hardest to place, since it has the least to offer a group -- then repeatedly admit whichever
    remaining run most improves the group's throughput. Optimal packing is a maximum-weight set
    partition, but the ranking is built from an ESTIMATED step model, and resolving ties exactly
    would be false precision over inputs that are themselves approximate.

    ``vram_capacity_gb`` is the card's usable memory. Leave it 0 to skip the memory bound entirely
    (callers that have not sized their runs); any positive value is enforced as a hard admission
    limit, because exceeding it is an OOM rather than a slowdown.
    """
    if len({r.label for r in runs}) != len(runs):
        raise ValueError("run labels must be unique: they are what identifies a placement")
    if max_tenants < 1:
        raise ValueError("max_tenants must be at least 1")

    def fits(group: tuple[RunShape, ...]) -> bool:
        return vram_capacity_gb <= 0 or sum(r.vram_gb for r in group) <= vram_capacity_gb

    free = sorted(runs, key=lambda r: (-r.duty_cycle, r.label))
    placements: list[Placement] = []
    solo: list[RunShape] = []

    while free:
        seed = free.pop(0)
        group = (seed,)
        if not fits(group):
            # a run that alone exceeds the capacity is not placeable here; report it as solo and let
            # the caller size it a bigger card rather than silently dropping it.
            solo.append(seed)
            continue
        while len(group) < max_tenants:
            best: tuple[float, RunShape] | None = None
            for cand in free:
                trial = (*group, cand)
                if not fits(trial):
                    continue
                gain = evaluate_placement(trial).throughput_gain
                if gain > evaluate_placement(group).throughput_gain and (
                    best is None or gain > best[0]
                ):
                    best = (gain, cand)
            if best is None:
                break
            group = (*group, best[1])
            free.remove(best[1])
        placement = evaluate_placement(group)
        if placement.worth_sharing:
            placements.append(placement)
        else:
            solo.extend(group)

    placements.sort(key=lambda p: (-p.throughput_gain, p.labels))
    solo.sort(key=lambda r: r.label)
    return placements, solo
