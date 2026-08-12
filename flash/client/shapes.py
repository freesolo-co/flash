"""Shape specs for control-plane response validation.

Pure stdlib: this module is on the bare-install client path (see pyproject's dependency
notes and tests/test_client_import_purity.py).
"""

from __future__ import annotations

from typing import Any

# One accepted shape for a required response value: a type, a tuple of accepted types, or a
# one-element list ``[element_spec]`` meaning "a list whose every element matches element_spec".
RequireSpec = type | tuple[type, ...] | list[Any]


def matches_require(value: object, expected: RequireSpec) -> bool:
    """True when a required response value has a shape the caller can actually read.

    ``[dict]`` exists because a bare ``list`` accepts ``{"runs": [null]}``, which then crashes on
    element access in the caller instead of surfacing as a ``ClientError``. bool subclasses int,
    so a json true/false would satisfy an int requirement and flow into arithmetic; it counts as
    malformed unless bool is itself expected.
    """
    if isinstance(expected, list):
        (element,) = expected
        return isinstance(value, list) and all(matches_require(item, element) for item in value)
    wants_bool = expected is bool or (isinstance(expected, tuple) and bool in expected)
    return isinstance(value, expected) and (wants_bool or not isinstance(value, bool))
