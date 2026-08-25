"""argparse type validators for `flash env eval`.

Separated from `eval.py` so the evaluation driver stays under the file-size limit. These reject a
bad flag value at parse time, before the command spends anything: a temperature the chat route will
refuse costs one paid rejection per case if it is discovered case by case instead.
"""

from __future__ import annotations

import argparse
import math

# ceiling on parallel model requests: enough to keep a deployment busy without opening so many
# connections that the suite is rate limited into slower wall-clock than a smaller fan-out.
_MAX_CONCURRENCY = 32


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def finite_float(value: str) -> float:
    """A temperature the chat route will accept.

    Reject non-finite or negative values before they become one paid rejection per case.
    The nonnegative floor matches `flash/schema/__init__.py`.
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {value}")
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"must be at least 0.0, got {value}")
    return parsed


def bounded_concurrency(value: str) -> int:
    parsed = positive_int(value)
    if parsed > _MAX_CONCURRENCY:
        raise argparse.ArgumentTypeError(f"must be at most {_MAX_CONCURRENCY}")
    return parsed
