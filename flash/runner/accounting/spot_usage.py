"""Spot usage per GPU class, so each stretch of a run is billed at the class it actually ran on.

A Spot run may start on any class that fits it and may resume on a different class after a
preemption. Its quote is priced at the costliest Spot class's rate, and settlement scales the
quote by the usage-weighted rate actually rented:

    factor = sum(usage[gpu] * rate(gpu)) / (sum(usage) * max(rate))

``rate`` is the tariff recorded on the run at submission (``submitted_spot_rates``), so a later
rate change never reprices an accepted quote; runs submitted before rates were recorded use
``pricing.LEGACY_SPOT_RATES``, the tariff they were quoted under.

Usage is recorded from heartbeats as increments credited to the class the run is on at that
moment (read from its live provider handle): completed training steps for SFT/GRPO, session
seconds for custom runs. Runs without recorded usage (launched before this accounting) keep a
factor of 1 and are billed on their accepted quote.
"""

from __future__ import annotations

import math

USAGE_KEY = "spot_usage_by_gpu"
CREDITED_STEP_KEY = "spot_credited_step"


def _is_spot(status) -> bool:
    gpu = (status.spec or {}).get("gpu") if isinstance(status.spec, dict) else None
    return isinstance(gpu, dict) and (gpu.get("spot") is True or gpu.get("provider") == "spot")


def _amount(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.0
    return max(0.0, float(value))


def _rented_gpu(status) -> str:
    remote = status.remote if isinstance(status.remote, dict) else {}
    if remote.get("provider") != "spot":
        return ""
    try:
        from flash.providers.core.base import canonical_gpu

        return canonical_gpu(remote.get("gpu") or "")
    except Exception:
        return ""


def record(status, prev: dict | None, hb: dict, *, same_attempt: bool) -> None:
    """Carry the per-class usage totals forward and credit this heartbeat's increment."""
    if not _is_spot(status):
        return
    prev = prev if isinstance(prev, dict) else {}
    usage = {k: _amount(v) for k, v in (prev.get(USAGE_KEY) or {}).items() if isinstance(k, str)}
    gpu = _rented_gpu(status)
    if status.spec.get("algorithm") == "custom":
        # session seconds restart with each attempt; credit only this attempt's growth.
        seconds = hb.get("session_seconds")
        before = _amount(prev.get("session_seconds")) if same_attempt else 0.0
        delta = _amount(seconds) - before if seconds is not None else 0.0
    else:
        # steps are run-wide and may repeat after a resume; credit only newly reached steps.
        credited = _amount(prev.get(CREDITED_STEP_KEY))
        step = hb.get("step")
        delta = _amount(step) - credited if step is not None else 0.0
        hb[CREDITED_STEP_KEY] = max(credited, _amount(step))
    if delta > 0 and gpu:
        usage[gpu] = usage.get(gpu, 0.0) + delta
    if usage:
        hb[USAGE_KEY] = usage


def submitted_rates(status) -> dict[str, float]:
    """The Spot tariff this run was quoted under: recorded at submit, else the legacy table."""
    from flash.providers.mithril.pricing import LEGACY_SPOT_RATES

    recorded = getattr(status, "submitted_spot_rates", None)
    if isinstance(recorded, dict):
        rates = {gpu: _amount(rate) for gpu, rate in recorded.items() if isinstance(gpu, str)}
        rates = {gpu: rate for gpu, rate in rates.items() if rate > 0}
        if rates:
            return rates
    return {gpu: float(rate) for gpu, rate in LEGACY_SPOT_RATES.items()}


def rate_factor(status) -> float:
    """Share of a Spot quote owed for the classes actually rented (1.0 without usage)."""
    heartbeat = getattr(status, "last_heartbeat", None)
    usage = heartbeat.get(USAGE_KEY) if isinstance(heartbeat, dict) else None
    if not isinstance(usage, dict):
        return 1.0
    rates = submitted_rates(status)
    usage = {gpu: _amount(amount) for gpu, amount in usage.items() if gpu in rates}
    total = sum(usage.values())
    if total <= 0:
        return 1.0
    billed = sum(amount * rates[gpu] for gpu, amount in usage.items())
    return min(1.0, billed / (total * max(rates.values())))
