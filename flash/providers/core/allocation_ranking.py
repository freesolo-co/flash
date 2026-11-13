"""Rank a fitting candidate set into one allocation, and describe the choice for run logs."""

from __future__ import annotations

from dataclasses import replace

from flash.providers.core.base import Allocation


def _cheapest_allocation(
    candidates, *, need: float, cost_per_step, provider_rank: dict[str, int]
) -> Allocation:
    """The cheapest-JOB shape from a non-empty fitting set, plus the full ranking behind it.

    Spot candidates rank by rental price, then class priority, before any cost key (see below).

    Cheapest job, not cheapest rental: rank on the dollars one step costs on each candidate (rate x
    how long that hardware takes), so a faster card wins whenever it finishes enough sooner to pay
    for itself. An authored provider preference ranks ahead of every cost key; unnamed providers
    share the final rank, so they remain eligible and retain their relative cost order. Within one
    provider rank, ties prefer fewer cards (less inter-card overhead), then combined VRAM, then class
    name. Sorting is stable, so provider and provider-local order apply only when every explicit key
    matches. A run the cost model cannot price (``cost_per_step`` is ``None``) falls back to total
    $/hr.
    """
    from flash.providers.mithril.pricing import priority as spot_priority
    from flash.providers.mithril.pricing import shape_price as spot_shape_price

    primary = cost_per_step if cost_per_step is not None else (lambda c: c.total_hourly_usd)
    unnamed_rank = len(provider_rank)
    ranked = sorted(
        candidates,
        key=lambda c: (
            provider_rank.get(c.provider, unnamed_rank),
            # Spot: cheapest rental first (fixed class rate x cards), then class priority. Ranking
            # class ahead of price chose 8x H100 ($16/hr, never under its cap) over a fitting
            # 1x H200 ($2.50/hr) that was rentable now.
            spot_shape_price(c.gpu, c.gpu_count) if c.provider == "spot" else 0,
            spot_priority(c.gpu) if c.provider == "spot" else 0,
            primary(c),
            c.total_hourly_usd,
            c.gpu_count,
            c.total_vram_gb,
            c.gpu,
        ),
    )
    # Spot prioritizes GPU class, then current price; fixed entry rates price estimates.
    from flash.providers.mithril.pricing import quote_rate as spot_quote_rate

    # Spot quotes the costliest class the run may take; billing scales to the class rented.
    ranked = [
        replace(c, hourly_usd=spot_quote_rate()) if c.provider == "spot" else c for c in ranked
    ]
    best = ranked[0]
    return Allocation(
        provider=best.provider,
        gpu=best.gpu,
        hourly_usd=best.hourly_usd,
        min_vram_gb=need,
        candidates=tuple(ranked),
        gpu_count=best.gpu_count,
    )


def allocation_summary(a: Allocation) -> str:
    shape = f"{a.gpu_count}x {a.gpu}" if a.gpu_count > 1 else a.gpu
    total = a.gpu_count * a.hourly_usd
    head = f"allocated {shape} on {a.provider} at ${total:.2f}/hr (need >= {a.min_vram_gb} GB VRAM)"
    if len(a.candidates) > 1:
        nxt = a.candidates[1]
        nxt_shape = f"{nxt.gpu_count}x {nxt.gpu}" if nxt.gpu_count > 1 else nxt.gpu
        head += f"; next-best: {nxt_shape}@{nxt.provider} ${nxt.total_hourly_usd:.2f}/hr"
    return head
