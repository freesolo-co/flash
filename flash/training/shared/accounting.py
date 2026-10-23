"""Exact estimated resource-cost attribution, separate from customer billing."""

from decimal import Decimal

DENOMINATOR = 3_600_000_000_000_000_000  # nanoseconds/hour times micro-USD/USD


def estimate(elapsed_ns, rate_microusd_per_hour):
    if rate_microusd_per_hour is None:
        return None
    numerator = elapsed_ns * rate_microusd_per_hour
    return {
        "numerator": numerator,
        "denominator": DENOMINATOR,
        "usd": str(Decimal(numerator) / Decimal(DENOMINATOR)),
        "rate_microusd_per_hour": rate_microusd_per_hour,
        "basis": "observed_execution_estimate_not_invoice",
    }
