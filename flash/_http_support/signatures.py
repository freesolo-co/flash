"""bounded implementation signatures for private http transport."""

from __future__ import annotations

import struct
import types
import urllib.request

from flash._internal.http_refs import (
    _SNAPSHOT_ITEMS_MAX,
    _TRAVERSAL_NODES_MAX,
    _module_function_code,
)

_ABSENT_SLOT = object()


def snapshot_value(
    value: object,
    seen: set[int] | None = None,
    active: set[int] | None = None,
    budget: list[int] | None = None,
    limit: int = _TRAVERSAL_NODES_MAX,
) -> object:
    if seen is None:
        seen = set()
    if active is None:
        active = set()
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > limit:
        raise ValueError
    value_type = type(value)
    if value is None or value_type in (bool, int, str, bytes):
        return (value_type, value)
    if value_type is float:
        return (float, struct.pack("!d", value))
    if value_type in (list, tuple, dict, set, frozenset):
        value_id = id(value)
        if value_id in active:
            raise ValueError
        if value_id in seen:
            return ("alias", value_id)
        seen.add(value_id)
        active.add(value_id)
        try:
            if len(value) > _SNAPSHOT_ITEMS_MAX:
                raise ValueError
            if value_type in (list, tuple):
                snapshot = tuple(
                    snapshot_value(item, seen, active, budget, limit) for item in value
                )
            elif value_type is dict:
                snapshot = frozenset(
                    (
                        snapshot_value(key, seen, active, budget, limit),
                        snapshot_value(item, seen, active, budget, limit),
                    )
                    for key, item in value.items()
                )
            else:
                snapshot = frozenset(
                    snapshot_value(item, seen, active, budget, limit) for item in value
                )
        finally:
            active.remove(value_id)
        return (value_type, snapshot)
    return ("opaque", id(value_type), id(value))


def function_implementation_signature(function: types.FunctionType) -> tuple[object, ...]:
    code = object.__getattribute__(function, "__code__")
    defaults = object.__getattribute__(function, "__defaults__")
    kwdefaults = object.__getattribute__(function, "__kwdefaults__")
    closure = object.__getattribute__(function, "__closure__")
    if (
        type(code) is not types.CodeType
        or (defaults is not None and type(defaults) is not tuple)
        or (kwdefaults is not None and type(kwdefaults) is not dict)
        or (closure is not None and type(closure) is not tuple)
    ):
        raise TypeError
    seen: set[int] = set()
    active: set[int] = set()
    budget = [0]
    closure_values = []
    for cell in closure or ():
        try:
            value = object.__getattribute__(cell, "cell_contents")
        except ValueError:
            value = _ABSENT_SLOT
        closure_values.append(snapshot_value(value, seen, active, budget))
    return (
        code,
        snapshot_value(defaults, seen, active, budget),
        snapshot_value(kwdefaults, seen, active, budget),
        tuple(closure_values),
    )


def stdlib_function_signature(function: types.FunctionType) -> tuple[object, ...]:
    qualname = object.__getattribute__(function, "__qualname__")
    if type(qualname) is not str:
        raise TypeError
    implementation = function_implementation_signature(function)
    return (_module_function_code(urllib.request, qualname), *implementation[1:])
