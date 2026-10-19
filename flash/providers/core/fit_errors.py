"""How a VRAM fit failure is explained to the user.

Split from ``base`` as one cohesive group: these build the REJECTION MESSAGE, not the fit decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flash.providers.core.base import (
    GPU_CLASSES,
    GpuClass,
    gpu_capacity_shape,
    largest_rentable_count,
    providers_for,
    smallest_fitting_gpu_count,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def _shape_label(gpu: GpuClass, count: int) -> str:
    return f"{count}x {gpu.name}" if count > 1 else gpu.name


def _join_note(count: int, executed_width=None) -> str:
    """Reconcile a rented card count with the launched capacity printed beside it.

    The count names what you RENT while the capacity is what LAUNCHES, so a bare label states a
    false equation: `2x H200` next to 141 GB reads as though an H200 were a 70 GB card. Naming the
    join count keeps the two numbers reconcilable. Empty when every rented card joins, which is
    every algorithm but sft and most sft runs.

    Deliberately terse -- a message can print two of these, and the reason they differ is stated
    once by ``_batch_bound_width_note`` rather than repeated per shape.
    """
    launched = executed_width(count) if executed_width else count
    if launched == count:
        return ""
    return f", {launched} of which {'joins' if launched == 1 else 'join'} this run"


def batch_bound_width_note(*, algorithm: str = "", parenthetical: bool = False) -> str:
    """Why this run launches on fewer ranks than it rents, worded for ``algorithm``.

    THE wording for that gap, shared by every message that shows one. The operator has to learn that
    the card ceiling is not the limiter, or they raise `--gpus` and hit the identical failure.

    The reason is per-algorithm because the bounding quantity is: sft counts rows, grpo and opd count
    the sequences one step holds. Naming sft's rows to an opd user would point them at a knob opd
    rejects at parse time -- which is exactly what the pinned-class path in ``allocator`` did while it
    spelled this sentence itself.

    For grpo and opd it also states the DIRECTION, because the general knob advice appended after it
    says to lower those same knobs and here that is a dead end: a step bound by its sequence count
    cannot go below one prompt, and RAISING the count is what buys ranks. MEASURED at 32k on eight
    180 GB cards -- every width-bound catalog row fits once the step is wide enough to fill more
    ranks (27B grpo at 4, 35B grpo and both 27B/35B opd at 2), and none of them fits by shrinking.

    ``parenthetical`` when a remedy clause follows -- a trailing dash clause would swallow it, so
    "..., or lower batch_size" would read as part of the explanation rather than as the fix.
    """
    if (algorithm or "").lower() in ("grpo", "rl", "opd"):
        reason = (
            "every rank needs its own share of the step, so prompts_per_step x group_size bounds "
            "the rank count; RAISE it to buy ranks"
        )
    else:
        reason = "sft shards by data, so the batch and retained rows bound the rank count"
    return f" ({reason})" if parenthetical else f" -- {reason}"


def _batch_bound_width_note(*counts: str, algorithm: str = "", parenthetical: bool = False) -> str:
    """``batch_bound_width_note`` gated on some shape having actually printed a join count.

    A run that launches every card it rents never sees the note, so the messages below can append it
    unconditionally and let the join notes they already built decide.
    """
    if not any(counts):
        return ""
    return batch_bound_width_note(algorithm=algorithm, parenthetical=parenthetical)


def vram_knob_advice(algorithm: str) -> str:
    """Return the algorithm knobs that actually reduce its measured vram floor."""
    algorithm = (algorithm or "").lower()
    if algorithm == "grpo":
        return (
            "lower [train].max_context_tokens / [train].max_completion_tokens / "
            "[train].lora_rank to fit"
        )
    if algorithm == "opd":
        # names prompts_per_step, not batch_size: opd REJECTS batch_size at parse time, so the old
        # wording sent a user whose run did not fit straight into a config error.
        return (
            "lower [train].group_size and/or [train].prompts_per_step (rollout concurrency = "
            "prompts_per_step x group_size; distillation needs no group variance, so group_size=1 "
            "is fine) and/or [train].max_completion_tokens / [train].max_context_tokens to fit"
        )
    return "lower [train].batch_size / [train].max_context_tokens / [train].lora_rank to fit"


def rents_arbitrary_card_counts(providers: Iterable[str]) -> bool:
    """Whether some provider here sells a card count that no live catalog has to confirm."""
    from flash.providers.core.registry import get_provider

    return any(not getattr(get_provider(name), "live_capacity", False) for name in providers)


def widenable_gpu_names(
    gpu_names: tuple[str, ...] | None, providers: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    """``gpu_names`` reduced to classes a wider shape can actually be BOUGHT for."""
    pool = (
        gpu_names
        if gpu_names is not None
        else tuple(g.name for g in GPU_CLASSES if g.modal_name and g.validated)
    )
    return tuple(
        name
        for name in pool
        if rents_arbitrary_card_counts(
            providers_for(name)
            if providers is None
            else tuple(p for p in providers_for(name) if p in providers)
        )
    )


def drop_pin_hint(
    exact: str,
    unpinned_widths: tuple[int, ...],
    need: float,
    *,
    ceiling: int,
    above: int,
    executed_width=None,
) -> str:
    """Name dropping the provider pin AND the width it unlocks, or nothing if no width fits.

    Routed through ``wider_shape_remedy`` so this shares the sibling path's two rules: a width is
    only named once PROVED to fit (an oversized pin gets no hint at all, since dropping the pin
    cannot help), and the ceiling is named alongside the pin -- reaching here means the authored
    count already failed, so advice omitting it just buys a second rejection.
    """
    from flash.providers.core.base import wider_shape_remedy

    remedy = wider_shape_remedy(
        unpinned_widths, need, ceiling=ceiling, above=above, executed_width=executed_width
    )
    if not remedy:
        return ""
    # the shared helper opens with "; it fits on N cards -- ...", so splice the pin clause in
    # front of its body rather than concatenating two separately punctuated sentences.
    return f". Drop the provider pin to rent {exact!r}:{remedy.removeprefix(';')}"


def catalog_check_hint(
    exact: str,
    need: float,
    *,
    ceiling: int,
    above: int,
    offerable: bool,
    validated: bool,
    executed_width=None,
) -> str:
    """Name the width to ASK a fixed-count provider for, when no offline width can be proved."""
    if not offerable:
        return ""
    width = smallest_fitting_gpu_count(
        need,
        max_gpu_count=ceiling,
        gpu_names=(exact,) if validated else (),
        executed_width=executed_width,
    )
    if width is None or width <= above:
        return ""
    return (
        f". Their catalog may list a {width}-card {exact} instance -- raise the card ceiling "
        f"with `--gpus {width}` to check it against their catalog"
    )


def _unbuyable_width_message(
    algorithm: str,
    need: float,
    *,
    requested_gpu_count: int,
    max_gpu_count: int,
    gpu_names: tuple[str, ...] | None,
    widenable_without_pin: tuple[str, ...] | None,
    executed_width=None,
) -> str:
    """Diagnose a run that FITS at some width but cannot buy it, and say what to do about it.

    Split out from ``vram_fit_error_message`` because it is a self-contained diagnosis: it decides
    the obstacle and the remedy together, and those two must agree -- an obstacle claiming nobody
    sells the shape reads as a contradiction above a remedy offering to rent exactly that shape.
    """
    unpinned_width = smallest_fitting_gpu_count(
        need,
        max_gpu_count=max_gpu_count,
        gpu_names=widenable_without_pin or (),
        executed_width=executed_width,
    )
    if unpinned_width is not None:
        remedy = (
            (
                "Drop the provider pin and raise the card ceiling with "
                f"`--gpus {unpinned_width}` to let the allocator choose"
            )
            if unpinned_width > requested_gpu_count
            else "Drop the provider pin to let the allocator choose"
        )
    else:
        catalog_width = smallest_fitting_gpu_count(
            need,
            max_gpu_count=max_gpu_count,
            gpu_names=gpu_names,
            executed_width=executed_width,
        )
        # the identical fit check guarding this block guarantees a catalog width is present.
        remedy = (
            f"Raise the card ceiling with `--gpus {catalog_width}` to check it against their "
            "catalog, or configure a provider that rents card counts directly (Modal)"
            if catalog_width > requested_gpu_count
            else "Configure a provider that rents card counts directly (Modal)"
        )
    # the obstacle must agree with the remedy printed right after it. whenever an unpinned width
    # exists, the PIN is what blocks the run -- saying "no available provider is confirmed to
    # sell" contradicts a remedy that offers to drop the pin and rent exactly that shape. only
    # when nothing behind the pin fits is the fixed-count catalog wording the true diagnosis.
    if unpinned_width is None:
        obstacle = (
            "which fits only on a multi-card shape that no available provider is confirmed to "
            "sell: they offer fixed card counts as distinct instance types, so a wider shape "
            "exists only if their live catalog lists one"
        )
    elif unpinned_width <= (requested_gpu_count or 0):
        obstacle = (
            "which no available provider is confirmed to sell at "
            f"{requested_gpu_count} {'card' if requested_gpu_count == 1 else 'cards'}: the "
            "pinned provider's largest card is too small, and the classes that would fit are "
            "sold by providers this pin excludes"
        )
    else:
        # the pinned pool cannot reach it at any purchasable width, but the fleet behind the
        # pin rents this shape directly -- so the run is two flags from working, not unsellable.
        obstacle = (
            f"which needs {unpinned_width} cards that the pinned provider is not confirmed to "
            "sell: they offer fixed card counts as distinct instance types, while a provider "
            "this pin excludes rents that shape directly"
        )
    return (
        f"{algorithm} needs >= {need:g} GB VRAM, {obstacle}. "
        f"{remedy}, or {vram_knob_advice(algorithm)}."
    )


def vram_fit_error_message(
    algorithm: str,
    need: float,
    *,
    requested_gpu_count: int | None,
    effective_gpu_count: int,
    max_gpu_count: int,
    gpu_names: tuple[str, ...] | None = None,
    providers: tuple[str, ...] | None = None,
    widenable_without_pin: tuple[str, ...] | None = None,
    executed_width=None,
) -> str:
    """Build an actionable pinned-count or terminal vram rejection."""
    algorithm = (algorithm or "").lower()
    widenable = widenable_gpu_names(gpu_names, providers)
    fitting_count = smallest_fitting_gpu_count(
        need, max_gpu_count=max_gpu_count, gpu_names=widenable, executed_width=executed_width
    )
    if requested_gpu_count is not None and fitting_count is not None:
        # both shapes are valued at the width the SEARCH above used, or the message rejects a count
        # for holding less than it claims to provide and then recommends one that holds less still.
        provided = gpu_capacity_shape(
            effective_gpu_count, gpu_names=gpu_names, executed_width=executed_width
        )
        fitting = gpu_capacity_shape(
            fitting_count, min_vram_gb=need, gpu_names=widenable, executed_width=executed_width
        )
        if provided is not None and fitting is not None:
            provided_gpu, provided_count, provided_vram = provided
            fitting_gpu, fitting_count, fitting_vram = fitting
            # `--gpus {n}` is spelled exactly as `wider_shape_remedy` spells it, so the flag a user
            # copies out of a fit failure is the same string on every path that can reject one.
            provided_join = _join_note(provided_count, executed_width)
            fitting_join = _join_note(fitting_count, executed_width)
            return (
                f"{algorithm} needs >= {need:g} GB VRAM; gpu.count={requested_gpu_count} provides "
                f"at most {provided_vram:g} GB "
                f"({_shape_label(provided_gpu, provided_count)}{provided_join}). "
                f"Raise the card ceiling with `--gpus {fitting_count}` "
                f"({_shape_label(fitting_gpu, fitting_count)} = {fitting_vram:g} GB"
                f"{fitting_join})"
                f"{_batch_bound_width_note(provided_join, fitting_join, algorithm=algorithm, parenthetical=True)}"
                f", or "
                f"{vram_knob_advice(algorithm)}."
            )

    # a width that FITS but cannot be bought is not the same failure as a run that exceeds every
    # class: saying it "needs more than any 8-card combination" would be false, and the knob advice
    # would send the user to shrink a run that already fits. name the real obstacle instead.
    if requested_gpu_count is not None and smallest_fitting_gpu_count(
        need, max_gpu_count=max_gpu_count, gpu_names=gpu_names, executed_width=executed_width
    ):
        return _unbuyable_width_message(
            algorithm,
            need,
            requested_gpu_count=requested_gpu_count,
            max_gpu_count=max_gpu_count,
            gpu_names=gpu_names,
            widenable_without_pin=widenable_without_pin,
            executed_width=executed_width,
        )

    # the terminal message states the ceiling this run cannot clear, so the ceiling has to be the
    # memory it would actually get -- crediting idle cards understates the shortfall it is reporting.
    widest = gpu_capacity_shape(max_gpu_count, gpu_names=gpu_names, executed_width=executed_width)
    widest_count = largest_rentable_count(max_gpu_count)
    biggest = widest[2] if widest is not None else 0.0
    # the note attaches to the CARD COUNT, not to the GB figure -- "446.6 GB max, 3 of which join"
    # reads as though ranks were a subset of gigabytes.
    widest_join = _join_note(widest_count, executed_width) if widest_count > 1 else ""
    shape = (
        f"any {widest_count}-card validated GPU combination{widest_join}"
        if widest_count > 1
        else "any single validated GPU"
    )
    ceiling = (
        f"{shape} ({biggest:g} GB max){_batch_bound_width_note(widest_join, algorithm=algorithm)}"
    )
    if algorithm == "opd":
        return (
            f"opd needs >= {need:g} GB VRAM, more than {ceiling}. "
            "opd is resident-only: the trainer and the colocated vLLM student rollout engine hold "
            "two model-weight copies plus the rollout KV cache at once. "
            f"{vram_knob_advice(algorithm).capitalize()}."
        )
    return (
        f"{algorithm} needs >= {need:g} GB VRAM, more than {ceiling}. "
        f"{vram_knob_advice(algorithm).capitalize()}."
    )
