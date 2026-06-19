"""Cost estimator CLI: the ``--versions`` spec parser.

No network. ``_parse_versions`` must accept comma lists, ``a-b`` ranges, and any mix
of the two (e.g. ``1,3-6``) without crashing.
"""

from __future__ import annotations

import pytest

from flash.cost.cli import _parse_versions


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("1,2,3", [1, 2, 3]),
        ("1-6", [1, 2, 3, 4, 5, 6]),
        ("3", [3]),
        # Mixed list + range -- the case that used to crash on int("1,3").
        ("1,3-6", [1, 3, 4, 5, 6]),
        ("1-3,5,7-8", [1, 2, 3, 5, 7, 8]),
        # Stray whitespace / empty tokens are tolerated.
        (" 1 , 3 - 4 ", [1, 3, 4]),
        ("1,,2", [1, 2]),
        # Overlapping tokens are de-duplicated and the result is sorted, so a repeated
        # version doesn't grade the same prompt twice and skew the experiment.
        ("1,1-3", [1, 2, 3]),
        ("3,2,1", [1, 2, 3]),
        ("2-4,3-5", [2, 3, 4, 5]),
        ("6,6,6", [6]),
    ],
)
def test_parse_versions(spec, expected):
    assert _parse_versions(spec) == expected
