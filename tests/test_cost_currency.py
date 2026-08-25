"""shared whole-cent rounding for quotes and billing."""

from flash.cost.currency import format_usd, usd_amount, usd_cents
from flash.server.billing.charges import _cents


def test_billing_and_display_share_round_half_up_cents():
    assert _cents is usd_cents
    assert _cents(1.005) == 101
    assert usd_amount(1.005) == 1.01
    assert format_usd(1.005) == "$1.01"
