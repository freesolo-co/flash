"""descriptor-safe slot inspection for private http transport."""

from __future__ import annotations

import types


def slot_names(declaration: object) -> tuple[str, ...]:
    if type(declaration) is str:
        names = (declaration,)
    elif type(declaration) in (list, tuple, set, frozenset):
        names = tuple(declaration)
    elif type(declaration) is dict:
        names = tuple(declaration.keys())
    else:
        raise TypeError
    if any(type(name) is not str for name in names):
        raise TypeError
    return names


def mangled_slot_name(owner: type, name: str) -> str:
    if name.startswith("__") and not name.endswith("__"):
        class_name = type.__getattribute__(owner, "__name__").lstrip("_")
        if class_name:
            return f"_{class_name}{name}"
    return name


def slot_entries(
    value: object,
    absent: object,
) -> tuple[tuple[int, str, types.MemberDescriptorType, object], ...]:
    found = []
    seen: set[int] = set()
    value_type = type(value)
    for owner in type.__getattribute__(value_type, "__mro__"):
        namespace = type.__getattribute__(owner, "__dict__")
        for declared_name in slot_names(namespace.get("__slots__", ())):
            if declared_name in {"__dict__", "__weakref__", "parent"}:
                continue
            slot_name = mangled_slot_name(owner, declared_name)
            descriptor = namespace.get(slot_name)
            if type(descriptor) is not types.MemberDescriptorType:
                raise TypeError
            if id(descriptor) in seen:
                continue
            seen.add(id(descriptor))
            try:
                slot_value = types.MemberDescriptorType.__get__(descriptor, value, value_type)
            except AttributeError:
                slot_value = absent
            found.append((id(owner), slot_name, descriptor, slot_value))
    return tuple(found)
