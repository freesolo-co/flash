"""tests for shared provider poll helpers."""

from __future__ import annotations

import sys

from flash.providers._lifecycle.instances.poll import _attempt_int


def test_attempt_int_rejects_oversized_ascii_decimal():
    if not hasattr(sys, "set_int_max_str_digits"):
        assert _attempt_int("9" * 5000) is None
        return
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        assert int("9" * 5000) > 0
        assert _attempt_int("9" * 5000) is None
    finally:
        sys.set_int_max_str_digits(previous)
