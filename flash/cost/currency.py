"""shared whole-cent rounding for customer-visible usd quotes and billing."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

_CENT = Decimal("0.01")


def usd_cents(usd: float) -> int:
    """Return nonnegative whole cents using decimal round-half-up."""
    rounded = Decimal(str(usd)).quantize(_CENT, rounding=ROUND_HALF_UP)
    return max(0, int(rounded.scaleb(2)))


def usd_amount(usd: float) -> float:
    """Return the nonnegative whole-cent usd amount stored on a quote."""
    return usd_cents(usd) / 100


def format_usd(usd: float) -> str:
    """Format usd from the same whole cents sent to billing."""
    cents = usd_cents(usd)
    dollars, remainder = divmod(cents, 100)
    return f"${dollars}.{remainder:02d}"
