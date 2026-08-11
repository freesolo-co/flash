"""How a VRAM fit failure is explained to the user.

Split from ``base`` as one cohesive group: these build the REJECTION MESSAGE, not the fit decision.
``base`` re-exports every public name here, so ``from flash.providers.base import ...`` keeps
working; ``base`` imports this module lazily because this module imports ``base``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flash.providers.base import (
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


def vram_knob_advice(algorithm: str) -> str:
    """Return the algorithm knobs that actually reduce its measured vram floor."""
    algorithm = (algorithm or "").lower()
    if algorithm == "grpo":
        return (
            "lower [train].max_context_tokens / [train].max_completion_tokens / "
            "[train].lora_rank to fit"
        )
    if algorithm == "opd":
        return (
            "lower [train].group_size and/or [train].batch_size (rollout concurrency = "
            "batch_size x group_size; distillation needs no group variance, so group_size=1 is "
            "fine) and/or [train].max_completion_tokens / [train].max_context_tokens to fit"
        )
    return "lower [train].batch_size / [train].max_context_tokens / [train].lora_rank to fit"


def rents_arbitrary_card_counts(providers: Iterable[str]) -> bool:
    """Whether some provider here sells a card count that no live catalog has to confirm.

    Providers differ in KIND on this, as ``AllocationConstraints.max_gpu_count`` records: RunPod
    takes the count as a launch parameter, so any rentable count is purchasable from the static
    table alone, while Lambda names it in the instance type and Vast has it baked into the offer,
    so an N-card shape only exists if their live catalog happens to list one. ``live_capacity`` is
    the signal that separates the two -- a provider stocking from a live market is exactly the one
    whose per-count SKUs cannot be confirmed offline. A new provider that sells fixed counts from a
    static table would need its own flag rather than this proxy.

    Used to decide whether a suggested width can be promised; the same attribute classifies an
    empty candidate set as retryable capacity in ``allocator._raise_no_candidate_error``.
    """
    from flash.providers import get_provider

    return any(not getattr(get_provider(name), "live_capacity", False) for name in providers)


def widenable_gpu_names(
    gpu_names: tuple[str, ...] | None, providers: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    """``gpu_names`` reduced to classes a wider shape can actually be BOUGHT for.

    A class survives when some provider still in play rents card counts freely
    (see ``rents_arbitrary_card_counts``). ``providers`` is the set in play, which a pin narrows:
    B200 is carried by RunPod and Lambda, but a Lambda-pinned run may not borrow RunPod's freedom
    to rent any count. ``None`` means no pin, so every carrier of the class counts.

    Separate from the pool that answers "what could hold this run" because the two questions differ:
    a rejection still names the biggest class it found (``provides at most 180 GB (B200)``) even
    when no wider shape of it is purchasable.
    """
    pool = (
        gpu_names
        if gpu_names is not None
        else tuple(g.name for g in GPU_CLASSES if g.enum_member and g.validated)
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
) -> str:
    """Build an actionable pinned-count or terminal vram rejection.

    A ``--gpus N`` suggestion is only offered for a class whose shape at N cards can be purchased;
    see ``widenable_gpu_names``. Otherwise the run falls through to the terminal message, which
    states the shortfall without sending the user to buy a shape no provider in play sells.

    ``widenable_without_pin`` is what the pool would widen to if the provider pin were dropped
    (``widenable_gpu_names`` over the unpinned fleet), or ``None`` when nothing was pinned. It
    decides whether "drop the pin" is a remedy at all, which ``providers`` cannot answer: a
    one-provider set is a pin on some planes and the entire configured fleet on others, and a pin
    on a plane that only ever had Lambda drops to the same pool and the same rejection. Only a
    caller that kept the unpinned fleet knows the difference.
    """
    algorithm = (algorithm or "").lower()
    widenable = widenable_gpu_names(gpu_names, providers)
    fitting_count = smallest_fitting_gpu_count(
        need, max_gpu_count=max_gpu_count, gpu_names=widenable
    )
    if requested_gpu_count is not None and fitting_count is not None:
        provided = gpu_capacity_shape(effective_gpu_count, gpu_names=gpu_names)
        fitting = gpu_capacity_shape(fitting_count, min_vram_gb=need, gpu_names=widenable)
        if provided is not None and fitting is not None:
            provided_gpu, provided_count, provided_vram = provided
            fitting_gpu, fitting_count, fitting_vram = fitting
            # `--gpus {n}` is spelled exactly as `wider_shape_remedy` spells it, so the flag a user
            # copies out of a fit failure is the same string on every path that can reject one.
            return (
                f"{algorithm} needs >= {need:g} GB VRAM; gpu.count={requested_gpu_count} provides "
                f"at most {provided_vram:g} GB ({_shape_label(provided_gpu, provided_count)}). "
                f"Raise the card ceiling with `--gpus {fitting_count}` "
                f"({_shape_label(fitting_gpu, fitting_count)} = {fitting_vram:g} GB), or "
                f"{vram_knob_advice(algorithm)}."
            )

    # a width that FITS but cannot be bought is not the same failure as a run that exceeds every
    # class: saying it "needs more than any 8-card combination" would be false, and the knob advice
    # would send the user to shrink a run that already fits. name the real obstacle instead.
    if requested_gpu_count is not None and smallest_fitting_gpu_count(
        need, max_gpu_count=max_gpu_count, gpu_names=gpu_names
    ):
        # the remedy differs by WHY only fixed-count providers are in play. dropping a pin helps
        # only when the fleet BEHIND it still sells the wider shape; a pin on a Lambda-only plane
        # drops to the same pool and the same rejection, so it gets the configure-a-provider advice
        # exactly like the operator who pinned nothing at all.
        #
        # either way the authored ceiling is ALSO too small -- reaching this branch means
        # `requested_gpu_count` already failed -- so a remedy naming only the provider costs the
        # user a second rejection that finally reveals `--gpus N`. name both halves at once.
        unpinned_width = smallest_fitting_gpu_count(
            need, max_gpu_count=max_gpu_count, gpu_names=widenable_without_pin or ()
        )
        if unpinned_width is not None:
            remedy = (
                "Drop the provider pin and raise the card ceiling with "
                f"`--gpus {unpinned_width}` to let the allocator choose"
            )
        else:
            remedy = (
                "Configure a provider that rents card counts directly (RunPod) and raise the card "
                "ceiling with `--gpus`"
            )
        return (
            f"{algorithm} needs >= {need:g} GB VRAM, which fits only on a multi-card shape that "
            "no available provider sells: they offer fixed card counts as distinct instance "
            "types, so a wider shape exists only if their live catalog lists one. "
            f"{remedy}, or {vram_knob_advice(algorithm)}."
        )

    widest = gpu_capacity_shape(max_gpu_count, gpu_names=gpu_names)
    widest_count = largest_rentable_count(max_gpu_count)
    biggest = widest[2] if widest is not None else 0.0
    shape = (
        f"any {widest_count}-card validated GPU combination"
        if widest_count > 1
        else "any single validated GPU"
    )
    if algorithm == "opd":
        return (
            f"opd needs >= {need:g} GB VRAM, more than {shape} ({biggest:g} GB max). "
            "opd is resident-only: the trainer and the colocated vLLM student rollout engine hold "
            "two model-weight copies plus the rollout KV cache at once. "
            f"{vram_knob_advice(algorithm).capitalize()}."
        )
    return (
        f"{algorithm} needs >= {need:g} GB VRAM, more than {shape} ({biggest:g} GB max). "
        f"{vram_knob_advice(algorithm).capitalize()}."
    )
