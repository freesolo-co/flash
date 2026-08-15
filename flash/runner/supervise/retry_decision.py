"""Whether a failed attempt gets another one, and what the log tells the user instead.

Split out of ``flash.runner.supervise.seed_submission`` to keep that module under the file-size
limit. The two questions belong together: the decision and the line describing it read the same
bookkeeping, and when they drifted apart the log claimed a retry the run was not going to make.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flash.runner.supervise import lifecycle as _lifecycle

if TYPE_CHECKING:  # pragma: no cover - annotations only, and importing these would be circular
    from flash.runner.supervise.seed_submission import _AttemptOutcome, _SubmitContext


def _every_shape_tried(
    ctx: _SubmitContext,
    outcome: _AttemptOutcome,
    *,
    first_cache_drop: bool,
) -> bool:
    """Whether this failure's bookkeeping leaves NO fitting shape the run has not already tried.

    Derived from the sets as they will stand after this failure is recorded, not from
    ``on_last_gpu``: a cache-drop retry deliberately leaves ``tried_classes`` untouched and reuses
    the class cold, and ``on_last_gpu`` is also true when the infra budget runs out with classes
    still untried. Neither is exhaustion.
    """
    if outcome.chosen is None:
        return False
    retry_tried = (
        ctx.tried_classes
        if first_cache_drop
        else ctx.tried_classes | {_lifecycle._shape_key(outcome.chosen)}
    )
    return all(_lifecycle._shape_key(candidate) in retry_tried for candidate in outcome.candidates)


# how many times one shape has to refuse capacity before the run believes it. one refusal is a
# data point (a search flake, a momentary shortage); two is the market's answer.
_REFUSALS_BEFORE_GIVING_UP = 2


# what the log says INSTEAD of naming a retry target, when the run stops on exhausted capacity.
#
# claims only what the stop establishes: that the class this retry would land on has refused twice.
# NOT "every class refused twice" (with several fitting classes the walk is A, B, A) and not "every
# fitting class was tried" (`_select_candidate` sorts on failed provider before untried, so an
# untried class on a failed provider loses to a tried one on a healthy provider). the remedies
# cover the rest.
_EXHAUSTED_CAPACITY_ACTION = (
    "not retrying: the GPU class this retry would return to has already refused capacity twice, "
    "and no better class is left to fail over to, so re-queueing would fail the same way. Widen "
    "the search -- drop the gpu.type pin, or drop gpu.provider"
)


def _capacity_exhausted(
    ctx: _SubmitContext,
    outcome: _AttemptOutcome,
    *,
    first_cache_drop: bool,
) -> bool:
    """Whether a retry would re-ask a question the class it lands on has ALREADY refused twice.

    A ``no_capacity`` verdict is about the CLASS, not the host: unlike a stall or a preemption, which
    a fresh box can clear, re-submitting to a shape the provider refuses asks the identical question
    at up to 900s of queue grace per attempt, which is what turned an unavailable pin into a
    75-minute serial retry loop.

    The bar is a SECOND refusal, since ``no_capacity`` also covers a transient search flake and a
    market that was dry a minute ago routinely frees a card. Counted PER SHAPE rather than as a set
    of names: over an ordered pin of A and B, a membership list writes off both classes after one
    refusal each.

    Asked of the shape the retry WOULD pick, not of every candidate. "Have they all refused twice?"
    is unreachable -- ``_select_candidate`` prefers an untried shape once, then clamps onto the
    cheapest, so a dry ordered pin walks A, B, A, A... and B stays at one refusal forever, leaving
    the stop inert for exactly the multi-class pins this feature creates.

    Narrow by design: it never fires while a cache-less retry is left, because dropping the weight
    cache lifts the volume's datacenter restriction and reaches hosts the cached attempt could not.
    Every other infra failure keeps its full retry budget.
    """
    if outcome.result.failure != "no_capacity" or first_cache_drop:
        return False
    if outcome.chosen is None:
        return False
    # the tally as it will stand once this failure is recorded, so the shape that just refused is
    # counted in its own verdict rather than judged on the state before it spoke.
    refusals = dict(ctx.capacity_refusals)
    chosen_key = _lifecycle._shape_key(outcome.chosen)
    refusals[chosen_key] = refusals.get(chosen_key, 0) + 1
    # the same projection the log line reads, so the decision and its explanation cannot disagree.
    projected = _lifecycle._projected_retry_class(
        outcome.candidates,
        ctx.failed_providers,
        ctx.tried_classes,
        outcome.chosen,
        cache_drop=False,
    )
    if projected is None:
        return False
    return refusals.get(_lifecycle._shape_key(projected), 0) >= _REFUSALS_BEFORE_GIVING_UP


def _retry_target(
    ctx: _SubmitContext,
    outcome: _AttemptOutcome,
    *,
    will_retry: bool,
    first_cache_drop: bool,
) -> str:
    # name the class the retry will actually land on. on_last_gpu only says no further class
    # escalation follows -- it does not mean this class is reused: with several fitting classes
    # the picker clamps back to the cheapest already-tried one (a100 pcie, a100 sxm, then a100
    # pcie again). so re-run the picker against the sets this failure is about to update.
    #
    # projected off the current candidate list, which the next attempt rebuilds by calling
    # allocate() again. lambda/vast rebuild from live capacity and pricing, so the named class
    # can disappear or lose to a newly available one -- word it as the expected target, not a
    # guarantee. a cache-drop retry projects too: it leaves both sets untouched and so reuses
    # this class cold, which is exactly the escalation the generic line failed to describe.
    projected = None
    if (
        will_retry
        and ctx.oom_vram_floor <= 0
        and outcome.chosen is not None
        and not outcome.quote_refresh_failed
    ):
        projected = _lifecycle._projected_retry_class(
            outcome.candidates,
            ctx.failed_providers,
            ctx.tried_classes,
            outcome.chosen,
            cache_drop=first_cache_drop,
        )
    if projected is None:
        return "retrying (resume from last checkpoint)"
    same = (projected.provider, projected.gpu) == (
        outcome.chosen.provider,
        outcome.chosen.gpu,
    )
    # derived from the sets after this failure's bookkeeping, not from the flag. a
    # cache-drop retry deliberately leaves tried_classes untouched and reuses this class
    # cold, so the projected class is still untried and reading current_on_last_gpu there
    # printed "retry on h100 again, no untried gpu class fits" -- naming the untried class
    # in the same clause that denies one exists. on_last_gpu is also true when the infra
    # retry budget runs out with classes still untried, which is not exhaustion either.
    exhausted = _every_shape_tried(ctx, outcome, first_cache_drop=first_cache_drop)
    no_escalation = ", no untried GPU class fits this run" if exhausted else ""
    # a projected provider this run has ALREADY marked failed is not a failover, it is a clamp back
    # onto the pool that is failing. `_select_candidate` sorts on `provider in failed_providers`
    # first, so once every candidate's provider has failed that key is True for all of them and the
    # escape degrades to the plain cheapest-first order. the line then reads like recovery while the
    # run loops on the same substrate, which is what made a 46-attempt loop look like progress.
    # naming it is the difference between "waiting" and "unpin gpu.provider or keep another provider
    # configured as a soft fallback".
    retry_failed_providers = (
        ctx.failed_providers
        if first_cache_drop
        else ctx.failed_providers | {outcome.chosen.provider}
    )
    # what landing on a failed provider actually proves is that no UNFAILED one is left, not that
    # this is the only provider with a fitting class. with fitting candidates on two providers that
    # have both failed, sort key 1 is True for either, so "no other provider offers a fitting class"
    # would deny a provider sitting right there in the list. read the list instead of inferring from
    # the sort, and distinguish the two cases: another failed provider is still somewhere to go
    # (unpin, or wait for it to recover), whereas a single-provider candidate list is not.
    other_providers = {candidate.provider for candidate in outcome.candidates} - {
        projected.provider
    }
    no_escape = (
        (
            ", which this run has already lost an attempt on -- every provider offering a fitting "
            f"class ({', '.join(sorted(other_providers | {projected.provider}))}) has now failed"
            if other_providers
            else ", which this run has already lost an attempt on -- no other provider offers a "
            "fitting class"
        )
        if projected.provider in retry_failed_providers
        else ""
    )
    return (
        f"expecting to retry on {projected.gpu} @ {projected.provider}"
        f"{' again' if same else ''}{no_escalation}{no_escape} (resume from last checkpoint; "
        "reallocated against live capacity, so the class may change)"
    )
