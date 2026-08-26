"""descriptor-safe callback reference analysis for private http transport."""

from __future__ import annotations

import builtins
import dis
import functools
import types

_SNAPSHOT_ITEMS_MAX = 256
_TRAVERSAL_NODES_MAX = 1024
_DYNAMIC_NAMESPACE_NAMES = frozenset(
    {"__import__", "compile", "eval", "exec", "getattr", "globals", "locals", "vars"}
)
_DYNAMIC_NAMESPACE_VALUES = tuple(
    types.ModuleType.__getattribute__(builtins, "__dict__")[name]
    for name in _DYNAMIC_NAMESPACE_NAMES
)
_ABSENT_SLOT = object()


def _getattr_type_static(owner: type, name: str, default: object) -> object:
    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    return default


def _nested_default_origins(
    instructions: tuple[dis.Instruction, ...],
    code_index: int,
    child: types.CodeType,
    origins: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    if code_index < 2:
        return {}
    build = instructions[code_index - 1]
    if build.opname != "BUILD_TUPLE" or type(build.arg) is not int or build.arg < 1:
        return {}
    sources = instructions[code_index - 1 - build.arg : code_index - 1]
    if len(sources) != build.arg:
        return {}
    child_names = object.__getattribute__(child, "co_varnames")
    child_argcount = object.__getattribute__(child, "co_argcount")
    if type(child_names) is not tuple or type(child_argcount) is not int:
        raise ValueError
    default_names = child_names[child_argcount - build.arg : child_argcount]
    if len(default_names) != len(sources):
        return {}
    mapped = {}
    for child_name, source in zip(default_names, sources, strict=True):
        if source.opname not in {"LOAD_FAST", "LOAD_DEREF"}:
            continue
        source_name = source.argval
        if source_name in origins:
            mapped[child_name] = origins[source_name]
    return mapped


def _loaded_reference_paths(
    code: types.CodeType,
    root_origins: dict[str, tuple[str, str]],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    pending = [(code, root_origins)]
    seen: set[int] = set()
    paths: set[tuple[str, str, tuple[str, ...]]] = set()
    while pending:
        current, origins = pending.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        if len(seen) > _SNAPSHOT_ITEMS_MAX:
            raise ValueError
        instructions = tuple(dis.get_instructions(current))
        for index, instruction in enumerate(instructions):
            name = instruction.argval
            if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}:
                root_kind = "global"
                root_name = name
            elif (instruction.opname in {"LOAD_DEREF", "LOAD_FAST"}) and name in origins:
                root_kind, root_name = origins[name]
            else:
                continue
            if type(name) is not str:
                raise ValueError
            attributes = []
            for following in instructions[index + 1 :]:
                if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                    break
                attribute = following.argval
                if type(attribute) is not str or attribute == "__dict__":
                    raise ValueError
                attributes.append(attribute)
            paths.add((root_kind, root_name, tuple(attributes)))
        constants = object.__getattribute__(current, "co_consts")
        if type(constants) is not tuple:
            raise ValueError
        for item in constants:
            if type(item) is not types.CodeType:
                continue
            free_names = object.__getattribute__(item, "co_freevars")
            if type(free_names) is not tuple:
                raise ValueError
            child_origins = {name: origins[name] for name in free_names if name in origins}
            code_index = next(
                index
                for index, instruction in enumerate(instructions)
                if instruction.opname == "LOAD_CONST" and instruction.argval is item
            )
            child_origins.update(_nested_default_origins(instructions, code_index, item, origins))
            pending.append((item, child_origins))
    return tuple(sorted(paths))


def _has_descriptor_get(value: object) -> bool:
    return _getattr_type_static(type(value), "__get__", _ABSENT_SLOT) is not _ABSENT_SLOT


def _static_namespace_attribute(namespace: object, name: str) -> object:
    namespace_type = type(namespace)
    if isinstance(namespace, types.ModuleType):
        if namespace_type is not types.ModuleType:
            raise ValueError
        state = types.ModuleType.__getattribute__(namespace, "__dict__")
        if type(state) is not dict or name not in state:
            raise ValueError
        return state[name]
    if isinstance(namespace, type):
        if namespace_type is not type:
            raise ValueError
        value = _getattr_type_static(namespace, name, _ABSENT_SLOT)
        if value is _ABSENT_SLOT or _has_descriptor_get(value):
            raise ValueError
        return value
    if (
        _getattr_type_static(namespace_type, "__getattribute__", _ABSENT_SLOT)
        is not object.__getattribute__
        or _getattr_type_static(namespace_type, "__getattr__", _ABSENT_SLOT) is not _ABSENT_SLOT
    ):
        raise ValueError
    class_value = _getattr_type_static(namespace_type, name, _ABSENT_SLOT)
    if (
        class_value is not _ABSENT_SLOT
        and _has_descriptor_get(class_value)
        and (
            _getattr_type_static(type(class_value), "__set__", _ABSENT_SLOT) is not _ABSENT_SLOT
            or _getattr_type_static(type(class_value), "__delete__", _ABSENT_SLOT)
            is not _ABSENT_SLOT
        )
    ):
        raise ValueError
    state = _static_object_state(namespace)
    if name not in state:
        raise ValueError
    return state[name]


def _resolve_reference_path(value: object, attributes: tuple[str, ...]) -> object:
    resolved = value
    if any(resolved is dynamic for dynamic in _DYNAMIC_NAMESPACE_VALUES):
        raise ValueError
    for attribute in attributes:
        if attribute in _DYNAMIC_NAMESPACE_NAMES:
            raise ValueError
        if type(resolved) in (list, tuple, set, frozenset, dict):
            return resolved
        resolved = _static_namespace_attribute(resolved, attribute)
        if any(resolved is dynamic for dynamic in _DYNAMIC_NAMESPACE_VALUES):
            raise ValueError
    if not attributes and type(resolved) in (types.ModuleType, type):
        raise ValueError
    return resolved


def _function_reference_roots(function: types.FunctionType) -> dict[str, dict[str, object]]:
    code = object.__getattribute__(function, "__code__")
    function_globals = object.__getattribute__(function, "__globals__")
    defaults = object.__getattribute__(function, "__defaults__")
    kwdefaults = object.__getattribute__(function, "__kwdefaults__")
    closure = object.__getattribute__(function, "__closure__")
    if (
        type(code) is not types.CodeType
        or type(function_globals) is not dict
        or (defaults is not None and type(defaults) is not tuple)
        or (kwdefaults is not None and type(kwdefaults) is not dict)
        or (closure is not None and type(closure) is not tuple)
    ):
        raise ValueError
    variable_names = object.__getattribute__(code, "co_varnames")
    free_names = object.__getattribute__(code, "co_freevars")
    if type(variable_names) is not tuple or type(free_names) is not tuple:
        raise ValueError
    positional_count = object.__getattribute__(code, "co_argcount")
    positional_only_count = object.__getattribute__(code, "co_posonlyargcount")
    keyword_only_count = object.__getattribute__(code, "co_kwonlyargcount")
    if (
        type(positional_count) is not int
        or type(positional_only_count) is not int
        or type(keyword_only_count) is not int
        or positional_only_count > positional_count
    ):
        raise ValueError
    defaults = defaults or ()
    if len(defaults) > positional_count:
        raise ValueError
    default_names = variable_names[positional_count - len(defaults) : positional_count]
    default_roots = dict(zip(default_names, defaults, strict=True))
    keyword_only_names = variable_names[positional_count : positional_count + keyword_only_count]
    if kwdefaults:
        if any(type(name) is not str for name in kwdefaults) or not set(kwdefaults).issubset(
            keyword_only_names
        ):
            raise ValueError
        default_roots.update(kwdefaults)
    binding_roots = {}
    if closure:
        if len(free_names) != len(closure):
            raise ValueError
        for name, cell in zip(free_names, closure, strict=True):
            try:
                binding_roots[name] = object.__getattribute__(cell, "cell_contents")
            except ValueError:
                continue
    return {
        "global": function_globals,
        "default": default_roots,
        "binding": binding_roots,
    }


def _function_reference_values(function: types.FunctionType) -> tuple[object, ...]:
    code = object.__getattribute__(function, "__code__")
    roots = _function_reference_roots(function)
    root_origins = {
        name: (root_kind, name) for root_kind in ("default", "binding") for name in roots[root_kind]
    }
    values = []
    for root_kind, name, attributes in _loaded_reference_paths(code, root_origins):
        if root_kind == "global" and name in _DYNAMIC_NAMESPACE_NAMES:
            raise ValueError
        root = roots[root_kind]
        if name not in root:
            continue
        root_value = root[name]
        if not attributes and type(root_value) in (types.ModuleType, type):
            continue
        values.append(_resolve_reference_path(root_value, attributes))
    return tuple(values)


def _static_object_state(value: object) -> dict[str, object]:
    for owner in type.__getattribute__(type(value), "__mro__"):
        namespace = type.__getattribute__(owner, "__dict__")
        descriptor = namespace.get("__dict__", _ABSENT_SLOT)
        if descriptor is _ABSENT_SLOT:
            continue
        if type(descriptor) is not types.GetSetDescriptorType:
            raise ValueError
        state = types.GetSetDescriptorType.__get__(descriptor, value, type(value))
        if type(state) is not dict or any(type(name) is not str for name in state):
            raise ValueError
        return state
    raise ValueError


def _references_target(
    value: object,
    targets: tuple[object, ...],
    seen: set[int] | None = None,
    active: set[int] | None = None,
    budget: list[int] | None = None,
    inspect_global_object: bool = False,
    max_nodes: int = _TRAVERSAL_NODES_MAX,
) -> bool:
    if seen is None:
        seen = set()
    if active is None:
        active = set()
    if budget is None:
        budget = [0]
    if any(value is target for target in targets):
        return True
    value_type = type(value)
    if value is None or value_type in (bool, int, float, str, bytes, type):
        budget[0] += 1
        if budget[0] > max_nodes:
            raise ValueError
        return False
    value_id = id(value)
    if value_id in active:
        raise ValueError
    if value_id in seen:
        return False
    seen.add(value_id)
    budget[0] += 1
    if budget[0] > max_nodes:
        raise ValueError
    active.add(value_id)
    reference_values: tuple[object, ...] = ()
    try:
        if value_type is types.MethodType:
            method_self = object.__getattribute__(value, "__self__")
            if any(method_self is target for target in targets):
                return True
            related = (object.__getattribute__(value, "__func__"),)
        elif value_type is functools.partial:
            related = (
                object.__getattribute__(value, "func"),
                object.__getattribute__(value, "args"),
                object.__getattribute__(value, "keywords"),
            )
        elif value_type is types.FunctionType:
            reference_values = _function_reference_values(value)
            related = [
                object.__getattribute__(value, "__defaults__"),
                object.__getattribute__(value, "__kwdefaults__"),
                *reference_values,
            ]
            closure = object.__getattribute__(value, "__closure__") or ()
            for cell in closure:
                try:
                    related.append(object.__getattribute__(cell, "cell_contents"))
                except ValueError:
                    continue
        elif value_type in (list, tuple, set, frozenset):
            if len(value) > _SNAPSHOT_ITEMS_MAX:
                raise ValueError
            related = value
        elif value_type is dict:
            if len(value) > _SNAPSHOT_ITEMS_MAX:
                raise ValueError
            related = (item for pair in value.items() for item in pair)
        elif value_type is types.ModuleType:
            return False
        elif isinstance(value, (list, tuple, set, frozenset, dict, functools.partial)) or callable(
            value
        ):
            raise ValueError
        elif inspect_global_object:
            state = _static_object_state(value)
            if len(state) > _SNAPSHOT_ITEMS_MAX:
                raise ValueError
            related = state.values()
        else:
            return False
        found = False
        for item in related:
            if item is not None:
                item_is_global = any(
                    item is reference_value for reference_value in reference_values
                )
                found = (
                    _references_target(
                        item,
                        targets,
                        seen,
                        active,
                        budget,
                        inspect_global_object or item_is_global,
                        max_nodes,
                    )
                    or found
                )
        return found
    finally:
        active.remove(value_id)
