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


@pytest.mark.parametrize("spec", ["x", "1,x", "6-1", "3-1", "", ",", " , "])
def test_parse_versions_rejects_malformed(spec):
    # Non-integer tokens, reversed ranges, and empty specs raise rather than silently
    # yielding garbage that crashes mid-sweep.
    with pytest.raises(ValueError, match="version"):
        _parse_versions(spec)


@pytest.mark.parametrize("spec", ["0", "99", "1,7", "0-2"])
def test_cmd_experiment_rejects_out_of_range(spec):
    # Out-of-range versions are caught before the sweep and reported as a clean CLI
    # error (exit 2), not a downstream traceback from prompt_for_version.
    from flash.cost.cli import main

    assert main(["experiment", "--offline", "--versions", spec]) == 2
