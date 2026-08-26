"""descriptor-safe callback reference analysis for private http transport."""

from __future__ import annotations

import _thread
import builtins
import dis
import functools
import importlib.machinery
import itertools
import operator
import types
import zipimport

_SNAPSHOT_ITEMS_MAX = 256
_TRAVERSAL_NODES_MAX = 1024
_DYNAMIC_NAMESPACE_NAMES = frozenset(
    {"__import__", "compile", "eval", "exec", "getattr", "globals", "locals", "vars"}
)
_DYNAMIC_NAMESPACE_VALUES = (
    *(
        types.ModuleType.__getattribute__(builtins, "__dict__")[name]
        for name in _DYNAMIC_NAMESPACE_NAMES
    ),
    operator.attrgetter,
    operator.itemgetter,
    operator.methodcaller,
)
_ABSENT_SLOT = object()
_STATIC_CONSTANT = object()
_STATIC_MAPPING = object()
_STATIC_ORIGIN = object()
_STATIC_SEQUENCE = object()
_STATIC_UNKNOWN = object()
_BRANCH_OPCODES = frozenset((*dis.hasjabs, *dis.hasjrel))
_FAST_LOAD_OPNAMES = frozenset({"LOAD_FAST", "LOAD_FAST_CHECK"})
_STATELESS_TERMINAL_TYPES = (object, _thread.LockType, _thread.RLock)


def _is_registered_callback_name(name: str) -> bool:
    if name in {"redirect_request", "do_open", "proxy_open"} or "_" not in name:
        return False
    condition = name.split("_", 1)[1]
    return (
        condition == "open"
        or condition == "request"
        or condition == "response"
        or condition.startswith("error")
    )


def _module_function_code(module: types.ModuleType, qualname: str) -> types.CodeType:
    spec = types.ModuleType.__getattribute__(module, "__spec__")
    if type(spec) is not importlib.machinery.ModuleSpec:
        raise TypeError
    loader = object.__getattribute__(spec, "loader")
    name = types.ModuleType.__getattribute__(module, "__name__")
    loader_type = type(loader)
    if loader_type is importlib.machinery.SourceFileLoader:
        root = importlib.machinery.SourceFileLoader.get_code(loader, name)
    elif loader_type is importlib.machinery.SourcelessFileLoader:
        root = importlib.machinery.SourcelessFileLoader.get_code(loader, name)
    elif loader_type is zipimport.zipimporter:
        root = zipimport.zipimporter.get_code(loader, name)
    elif loader is importlib.machinery.FrozenImporter:
        root = importlib.machinery.FrozenImporter.get_code(name)
    else:
        raise TypeError
    if type(root) is not types.CodeType:
        raise TypeError
    pending = [root]
    matches = []
    while pending:
        code = pending.pop()
        if object.__getattribute__(code, "co_qualname") == qualname:
            matches.append(code)
        pending.extend(
            item
            for item in object.__getattribute__(code, "co_consts")
            if type(item) is types.CodeType
        )
    if len(matches) != 1:
        raise ValueError
    return matches[0]


def _slot_names(declaration: object) -> tuple[str, ...]:
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


def _mangled_slot_name(owner: type, name: str) -> str:
    if name.startswith("__") and not name.endswith("__"):
        class_name = type.__getattribute__(owner, "__name__").lstrip("_")
        if class_name:
            return f"_{class_name}{name}"
    return name


def _slot_entries(
    value: object,
    absent: object,
) -> tuple[tuple[int, str, types.MemberDescriptorType, object], ...]:
    found = []
    seen: set[int] = set()
    value_type = type(value)
    for owner in type.__getattribute__(value_type, "__mro__"):
        namespace = type.__getattribute__(owner, "__dict__")
        for declared_name in _slot_names(namespace.get("__slots__", ())):
            if declared_name in {"__dict__", "__weakref__", "parent"}:
                continue
            slot_name = _mangled_slot_name(owner, declared_name)
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


def _getattr_type_static(owner: type, name: str, default: object) -> object:
    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    return default


def _stack_expression_start(
    instructions: tuple[dis.Instruction, ...],
    end: int,
) -> int:
    required = 1
    for index in range(end - 1, -1, -1):
        instruction = instructions[index]
        if instruction.opcode in _BRANCH_OPCODES:
            raise ValueError
        if instruction.arg is None:
            effect = dis.stack_effect(instruction.opcode)
        else:
            effect = dis.stack_effect(instruction.opcode, instruction.arg)
        required -= effect
        if required == 0:
            return index
        if required < 0:
            raise ValueError
    raise ValueError


def _function_metadata_spans(
    instructions: tuple[dis.Instruction, ...],
    code_index: int,
) -> dict[int, tuple[int, int]]:
    if code_index + 1 >= len(instructions):
        return {}
    make_function = instructions[code_index + 1]
    flags = make_function.arg
    if make_function.opname != "MAKE_FUNCTION" or type(flags) is not int or flags & ~0x0F:
        return {}
    spans = {}
    cursor = code_index
    for flag in (0x08, 0x04, 0x02, 0x01):
        if flags & flag:
            start = _stack_expression_start(instructions, cursor)
            spans[flag] = (start, cursor)
            cursor = start
    return spans


def _loaded_origin(
    instruction: dis.Instruction,
    origins: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    source_name = instruction.argval
    if type(source_name) is not str:
        raise ValueError
    if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}:
        return ("global", source_name)
    if instruction.opname in _FAST_LOAD_OPNAMES | {"LOAD_DEREF"}:
        return origins.get(source_name)
    return None


def _contains_static_origin(value: tuple[object, ...]) -> bool:
    tag = value[0]
    if tag is _STATIC_ORIGIN:
        return True
    if tag is _STATIC_SEQUENCE:
        return any(_contains_static_origin(item) for item in value[1])
    if tag is _STATIC_MAPPING:
        return any(_contains_static_origin(item) for pair in value[1] for item in pair)
    return False


def _static_subscript(
    container: tuple[object, ...],
    key: tuple[object, ...],
) -> tuple[object, ...]:
    if key[0] is not _STATIC_CONSTANT:
        return (_STATIC_UNKNOWN,)
    key_value = key[1]
    if container[0] is _STATIC_SEQUENCE and type(key_value) is int:
        values = container[1]
        if -len(values) <= key_value < len(values):
            return values[key_value]
    elif container[0] is _STATIC_MAPPING:
        for candidate, value in reversed(container[1]):
            if (
                candidate[0] is _STATIC_CONSTANT
                and type(candidate[1]) is type(key_value)
                and candidate[1] == key_value
            ):
                return value
    return (_STATIC_UNKNOWN,)


def _static_constant(value: object) -> tuple[object, ...]:
    if value is None or type(value) in (bool, int, float, str, bytes):
        return (_STATIC_CONSTANT, value)
    if type(value) is tuple:
        items = tuple(_static_constant(item) for item in value)
        if all(item[0] is _STATIC_CONSTANT for item in items):
            return (_STATIC_CONSTANT, value)
    return (_STATIC_UNKNOWN,)


def _has_ambiguous_default_control_flow(
    instructions: tuple[dis.Instruction, ...],
    start: int,
    end: int,
    origins: dict[str, tuple[str, str]],
) -> bool:
    offset_indices = {instruction.offset: index for index, instruction in enumerate(instructions)}
    for index, instruction in enumerate(instructions[:end]):
        if instruction.opcode not in _BRANCH_OPCODES:
            continue
        target = instruction.argval
        target_index = offset_indices.get(target) if type(target) is int else None
        if target_index is None or not start <= target_index <= end:
            continue
        branch_start = _stack_expression_start(instructions, index)
        if any(
            candidate.opname in {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF"} | _FAST_LOAD_OPNAMES
            and _loaded_origin(candidate, origins) is not None
            for candidate in instructions[branch_start:end]
        ):
            return True
    return False


def _static_default_origin(
    instructions: tuple[dis.Instruction, ...],
    span: tuple[int, int],
    origins: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    start, end = span
    if _has_ambiguous_default_control_flow(instructions, start, end, origins):
        raise ValueError
    expression = instructions[start:end]
    has_origin = any(
        instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF"} | _FAST_LOAD_OPNAMES
        and _loaded_origin(instruction, origins) is not None
        for instruction in expression
    )
    stack: list[tuple[object, ...]] = []
    for instruction in expression:
        if instruction.opname == "LOAD_CONST":
            stack.append(_static_constant(instruction.argval))
        elif instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF"} | _FAST_LOAD_OPNAMES:
            origin = _loaded_origin(instruction, origins)
            stack.append((_STATIC_ORIGIN, origin) if origin is not None else (_STATIC_UNKNOWN,))
        elif instruction.opname in {"BUILD_TUPLE", "BUILD_LIST"}:
            count = instruction.arg
            if type(count) is not int or count < 0 or len(stack) < count:
                raise ValueError
            values = tuple(stack[-count:]) if count else ()
            if count:
                del stack[-count:]
            stack.append((_STATIC_SEQUENCE, values))
        elif instruction.opname == "LIST_TO_TUPLE":
            if not stack:
                raise ValueError
        elif instruction.opname == "BUILD_CONST_KEY_MAP":
            count = instruction.arg
            if type(count) is not int or count < 1 or len(stack) < count + 1:
                raise ValueError
            keys = stack.pop()
            values = stack[-count:]
            del stack[-count:]
            if keys[0] is not _STATIC_CONSTANT or type(keys[1]) is not tuple:
                stack.append((_STATIC_UNKNOWN,))
            else:
                stack.append(
                    (
                        _STATIC_MAPPING,
                        tuple(
                            ((_STATIC_CONSTANT, key), value)
                            for key, value in zip(keys[1], values, strict=True)
                        ),
                    )
                )
        elif instruction.opname == "BUILD_MAP":
            count = instruction.arg
            if type(count) is not int or count < 0 or len(stack) < 2 * count:
                raise ValueError
            items = stack[-2 * count :] if count else []
            if count:
                del stack[-2 * count :]
            stack.append((_STATIC_MAPPING, tuple(zip(items[::2], items[1::2], strict=True))))
        elif instruction.opname == "BINARY_SUBSCR":
            if len(stack) < 2:
                raise ValueError
            key = stack.pop()
            container = stack.pop()
            stack.append(_static_subscript(container, key))
        elif instruction.opname == "COPY":
            depth = instruction.arg
            if type(depth) is not int or depth < 1 or depth > len(stack):
                raise ValueError
            stack.append(stack[-depth])
        elif instruction.opname == "SWAP":
            depth = instruction.arg
            if type(depth) is not int or depth < 2 or depth > len(stack):
                raise ValueError
            stack[-1], stack[-depth] = stack[-depth], stack[-1]
        elif has_origin:
            raise ValueError
        else:
            return None
    if len(stack) != 1:
        raise ValueError
    result = stack[0]
    if result[0] is _STATIC_ORIGIN:
        return result[1]
    if _contains_static_origin(result):
        raise ValueError
    if has_origin:
        raise ValueError
    return None


def _preceding_value_spans(
    instructions: tuple[dis.Instruction, ...],
    start: int,
    end: int,
    count: int,
) -> tuple[tuple[int, int], ...]:
    values = []
    cursor = end
    for _ in range(count):
        value_start = _stack_expression_start(instructions, cursor)
        values.append((value_start, cursor))
        cursor = value_start
    if cursor != start:
        return ()
    values.reverse()
    return tuple(values)


def _nested_default_origins(
    instructions: tuple[dis.Instruction, ...],
    code_index: int,
    child: types.CodeType,
    origins: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    child_names = object.__getattribute__(child, "co_varnames")
    child_argcount = object.__getattribute__(child, "co_argcount")
    child_kwonlyargcount = object.__getattribute__(child, "co_kwonlyargcount")
    if (
        type(child_names) is not tuple
        or type(child_argcount) is not int
        or type(child_kwonlyargcount) is not int
    ):
        raise ValueError
    spans = _function_metadata_spans(instructions, code_index)
    mapped = {}
    positional_span = spans.get(0x01)
    if positional_span is not None:
        start, end = positional_span
        build = instructions[end - 1]
        count = build.arg
        if build.opname == "BUILD_TUPLE" and type(count) is int and 1 <= count <= child_argcount:
            sources = _preceding_value_spans(instructions, start, end - 1, count)
            default_names = child_names[child_argcount - count : child_argcount]
            if len(sources) == len(default_names):
                for child_name, source in zip(default_names, sources, strict=True):
                    origin = _static_default_origin(instructions, source, origins)
                    if origin is not None:
                        mapped[child_name] = origin
    keyword_span = spans.get(0x02)
    if keyword_span is not None:
        start, end = keyword_span
        build = instructions[end - 1]
        count = build.arg
        if build.opname == "BUILD_CONST_KEY_MAP" and type(count) is int and count >= 1:
            key_start = _stack_expression_start(instructions, end - 1)
            if key_start + 1 == end - 1:
                keys = instructions[key_start].argval
                if (
                    type(keys) is tuple
                    and len(keys) == count
                    and all(type(name) is str for name in keys)
                ):
                    values = _preceding_value_spans(
                        instructions,
                        start,
                        key_start,
                        count,
                    )
                    keyword_names = child_names[
                        child_argcount : child_argcount + child_kwonlyargcount
                    ]
                    if set(keys).issubset(keyword_names) and len(values) == len(keys):
                        for child_name, source in zip(keys, values, strict=True):
                            origin = _static_default_origin(instructions, source, origins)
                            if origin is not None:
                                mapped[child_name] = origin
    return mapped


def _loaded_reference_paths(
    code: types.CodeType,
    root_origins: dict[str, tuple[str, str]],
    safe_parent_offsets: frozenset[tuple[int, int]] = frozenset(),
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
        current_origins = dict(origins)
        changed = True
        while changed:
            changed = False
            for source, target in itertools.pairwise(instructions):
                if target.opname not in {"STORE_FAST", "STORE_DEREF"}:
                    continue
                source_name = source.argval
                if source.opname in {"LOAD_GLOBAL", "LOAD_NAME"}:
                    origin = ("global", source_name)
                elif source.opname in _FAST_LOAD_OPNAMES | {"LOAD_DEREF"}:
                    origin = current_origins.get(source_name)
                else:
                    continue
                target_name = target.argval
                if type(source_name) is not str or type(target_name) is not str:
                    raise ValueError
                if origin is not None and current_origins.get(target_name) != origin:
                    current_origins[target_name] = origin
                    changed = True
        for index, instruction in enumerate(instructions):
            if instruction.opname in {"IMPORT_NAME", "IMPORT_FROM", "IMPORT_STAR"}:
                raise ValueError
            name = instruction.argval
            if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}:
                root_kind = "global"
                root_name = name
            elif (
                instruction.opname in _FAST_LOAD_OPNAMES | {"LOAD_DEREF"}
                and name in current_origins
            ):
                root_kind, root_name = current_origins[name]
            elif instruction.opname == "LOAD_CLOSURE" and name in current_origins:
                if current_origins[name][0] == "bound":
                    raise ValueError
                continue
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
            if root_kind == "bound" and not attributes:
                raise ValueError
            if (
                root_kind == "bound"
                and attributes == ["parent"]
                and (current_id, instruction.offset) not in safe_parent_offsets
            ):
                raise ValueError
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
            child_origins = {
                name: current_origins[name]
                for name in free_names
                if name in current_origins and current_origins[name][0] != "bound"
            }
            code_index = next(
                index
                for index, instruction in enumerate(instructions)
                if instruction.opname == "LOAD_CONST" and instruction.argval is item
            )
            child_origins.update(
                _nested_default_origins(instructions, code_index, item, current_origins)
            )
            pending.append((item, child_origins))
    return tuple(sorted(paths))


def _has_descriptor_get(value: object) -> bool:
    return _getattr_type_static(type(value), "__get__", _ABSENT_SLOT) is not _ABSENT_SLOT


def _has_descriptor_set(value: object) -> bool:
    descriptor_type = type(value)
    return (
        _getattr_type_static(descriptor_type, "__set__", _ABSENT_SLOT) is not _ABSENT_SLOT
        or _getattr_type_static(descriptor_type, "__delete__", _ABSENT_SLOT) is not _ABSENT_SLOT
    )


def _static_instance_attribute(
    namespace: object,
    name: str,
    class_value: object,
) -> tuple[bool, object]:
    namespace_type = type(namespace)
    if (
        _getattr_type_static(namespace_type, "__getattribute__", _ABSENT_SLOT)
        is not object.__getattribute__
        or _getattr_type_static(namespace_type, "__getattr__", _ABSENT_SLOT) is not _ABSENT_SLOT
    ):
        raise ValueError
    if type(class_value) is types.MemberDescriptorType:
        try:
            return True, types.MemberDescriptorType.__get__(
                class_value,
                namespace,
                namespace_type,
            )
        except AttributeError as exc:
            raise ValueError from exc
    if class_value is not _ABSENT_SLOT and _has_descriptor_set(class_value):
        raise ValueError
    state = _static_object_state(namespace)
    if state is not None and name in state:
        return True, state[name]
    return False, _ABSENT_SLOT


def _static_namespace_attribute(namespace: object, name: str) -> object:
    namespace_type = type(namespace)
    namespace_mro = type.__getattribute__(namespace_type, "__mro__")
    if any(owner is types.ModuleType for owner in namespace_mro):
        if namespace_type is not types.ModuleType:
            raise ValueError
        state = types.ModuleType.__getattribute__(namespace, "__dict__")
        if type(state) is not dict or name not in state:
            raise ValueError
        return state[name]
    if any(owner is type for owner in namespace_mro):
        if namespace_type is not type:
            raise ValueError
        value = _getattr_type_static(namespace, name, _ABSENT_SLOT)
        if value is _ABSENT_SLOT or _has_descriptor_get(value):
            raise ValueError
        return value
    class_value = _getattr_type_static(namespace_type, name, _ABSENT_SLOT)
    found, value = _static_instance_attribute(namespace, name, class_value)
    if found:
        return value
    if class_value is _ABSENT_SLOT or _has_descriptor_get(class_value):
        raise ValueError
    return class_value


def _has_rebound_parent_field(value: object) -> bool:
    state = _static_object_state(value)
    if state is None or "parent" not in state:
        return False
    class_value = _getattr_type_static(type(value), "parent", _ABSENT_SLOT)
    if class_value is not _ABSENT_SLOT:
        return False
    found, resolved = _static_instance_attribute(value, "parent", class_value)
    return found and resolved is state["parent"]


def _resolve_reference_path(value: object, attributes: tuple[str, ...]) -> object:
    resolved = value
    if any(resolved is dynamic for dynamic in _DYNAMIC_NAMESPACE_VALUES):
        raise ValueError
    for attribute in attributes:
        if attribute in _DYNAMIC_NAMESPACE_NAMES:
            raise ValueError
        if resolved is None or type(resolved) in (
            object,
            bool,
            int,
            float,
            str,
            bytes,
            list,
            tuple,
            set,
            frozenset,
            dict,
        ):
            return resolved
        resolved = _static_namespace_attribute(resolved, attribute)
        if any(resolved is dynamic for dynamic in _DYNAMIC_NAMESPACE_VALUES):
            raise ValueError
    if not attributes and type(resolved) in (types.ModuleType, type):
        raise ValueError
    return resolved


def _function_reference_roots(
    function: types.FunctionType,
    bound_self: object = _ABSENT_SLOT,
) -> dict[str, dict[str, object]]:
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
    bound_roots = {}
    if bound_self is not _ABSENT_SLOT:
        if positional_count < 1:
            raise ValueError
        bound_roots[variable_names[0]] = bound_self
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
        "bound": bound_roots,
    }


def _function_capture_values(function: types.FunctionType) -> tuple[object, ...]:
    roots = _function_reference_roots(function)
    return (*roots["default"].values(), *roots["binding"].values())


def _function_global_reference_values(function: types.FunctionType) -> tuple[object, ...]:
    code = object.__getattribute__(function, "__code__")
    roots = _function_reference_roots(function)
    values = []
    for root_kind, name, attributes in _loaded_reference_paths(code, {}):
        if root_kind != "global":
            continue
        if name in _DYNAMIC_NAMESPACE_NAMES:
            raise ValueError
        if name not in roots["global"]:
            continue
        values.append(_resolve_reference_path(roots["global"][name], attributes))
    return tuple(values)


def _function_bound_reference_values(
    function: types.FunctionType,
    bound_self: object,
) -> tuple[object, ...]:
    code = object.__getattribute__(function, "__code__")
    roots = _function_reference_roots(function, bound_self)
    root_origins = {name: ("bound", name) for name in roots["bound"]}
    safe_parent_offsets = frozenset()
    values = []
    for root_kind, name, attributes in _loaded_reference_paths(
        code, root_origins, safe_parent_offsets
    ):
        if root_kind != "bound":
            continue
        root_value = roots["bound"][name]
        if attributes[:1] == ("parent",):
            if not _has_rebound_parent_field(root_value) or len(attributes) != 1:
                raise ValueError
            continue
        values.append(_resolve_reference_path(root_value, attributes))
    return tuple(values)


def _function_reference_values(
    function: types.FunctionType,
    bound_self: object = _ABSENT_SLOT,
) -> tuple[object, ...]:
    code = object.__getattribute__(function, "__code__")
    roots = _function_reference_roots(function, bound_self)
    root_origins = {
        name: (root_kind, name)
        for root_kind in ("default", "binding", "bound")
        for name in roots[root_kind]
    }
    safe_parent_offsets = frozenset()
    values = [*roots["default"].values(), *roots["binding"].values()]
    for root_kind, name, attributes in _loaded_reference_paths(
        code, root_origins, safe_parent_offsets
    ):
        if root_kind == "global" and name in _DYNAMIC_NAMESPACE_NAMES:
            raise ValueError
        root = roots[root_kind]
        if name not in root:
            continue
        root_value = root[name]
        if root_kind == "bound" and attributes[:1] == ("parent",):
            if not _has_rebound_parent_field(root_value) or len(attributes) != 1:
                raise ValueError
            continue
        if not attributes and type(root_value) in (types.ModuleType, type):
            continue
        values.append(_resolve_reference_path(root_value, attributes))
    return tuple(values)


def _static_object_state(value: object) -> dict[str, object] | None:
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
    return None


def _declared_slot_descriptors(owner: type, namespace: dict[str, object]) -> tuple[object, ...]:
    raw_slots = namespace.get("__slots__", _ABSENT_SLOT)
    if raw_slots is _ABSENT_SLOT:
        return ()
    if type(raw_slots) is str:
        slot_names = (raw_slots,)
    elif type(raw_slots) in (tuple, list, set, frozenset):
        slot_names = tuple(raw_slots)
    elif type(raw_slots) is dict:
        slot_names = tuple(raw_slots.keys())
    else:
        raise ValueError
    if any(type(name) is not str for name in slot_names):
        raise ValueError
    owner_name = type.__getattribute__(owner, "__name__")
    if type(owner_name) is not str:
        raise ValueError
    descriptors = []
    for name in slot_names:
        if name in ("__dict__", "__weakref__"):
            descriptor = namespace.get(name, _ABSENT_SLOT)
            if type(descriptor) is not types.GetSetDescriptorType:
                raise ValueError
            continue
        storage_name = name
        class_name = owner_name.lstrip("_")
        if class_name and name.startswith("__") and not name.endswith("__"):
            storage_name = f"_{class_name}{name}"
        descriptor = namespace.get(storage_name, _ABSENT_SLOT)
        if type(descriptor) is not types.MemberDescriptorType:
            raise ValueError
        descriptors.append(descriptor)
    return tuple(descriptors)


def _static_object_values(value: object) -> tuple[object, ...]:
    state = _static_object_state(value)
    values = list(state.values()) if state is not None else []
    if len(values) > _SNAPSHOT_ITEMS_MAX:
        raise ValueError
    has_slot_declaration = False
    descriptors = []
    for owner in type.__getattribute__(type(value), "__mro__"):
        namespace = type.__getattribute__(owner, "__dict__")
        if "__slots__" in namespace:
            has_slot_declaration = True
        declared = _declared_slot_descriptors(owner, namespace)
        descriptors.extend(declared)
        if any(
            type(descriptor) is types.MemberDescriptorType
            and not any(descriptor is declared_descriptor for declared_descriptor in declared)
            for descriptor in namespace.values()
        ):
            raise ValueError
    if (
        state is None
        and not has_slot_declaration
        and not any(type(value) is terminal_type for terminal_type in _STATELESS_TERMINAL_TYPES)
    ):
        raise ValueError
    for descriptor in descriptors:
        try:
            slot_value = types.MemberDescriptorType.__get__(descriptor, value, type(value))
        except AttributeError:
            continue
        values.append(slot_value)
        if len(values) > _SNAPSHOT_ITEMS_MAX:
            raise ValueError
    return tuple(values)


def _references_target(
    value: object,
    targets: tuple[object, ...],
    seen: set[int] | None = None,
    active: set[int] | None = None,
    budget: list[int] | None = None,
    inspect_global_object: bool = False,
    inspect_method_receiver: bool = False,
    max_nodes: int = _TRAVERSAL_NODES_MAX,
    trusted_objects: tuple[object, ...] = (),
) -> bool:
    if seen is None:
        seen = set()
    if active is None:
        active = set()
    if budget is None:
        budget = [0]
    if any(value is target for target in targets):
        return True
    if any(value is dynamic for dynamic in _DYNAMIC_NAMESPACE_VALUES):
        raise ValueError
    if any(value is trusted for trusted in trusted_objects):
        return False
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
            method_function = object.__getattribute__(value, "__func__")
            related = (method_function, method_self)
            if inspect_method_receiver:
                related = (
                    *related,
                    *_function_bound_reference_values(method_function, method_self),
                )
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
        elif value_type is types.BuiltinMethodType:
            related = (object.__getattribute__(value, "__self__"),)
        elif value_type is types.ModuleType:
            if inspect_global_object:
                return False
            state = types.ModuleType.__getattribute__(value, "__dict__")
            if type(state) is not dict or len(state) > _SNAPSHOT_ITEMS_MAX:
                raise ValueError
            related = state.values()
        elif (
            any(
                owner is container_type
                for owner in type.__getattribute__(value_type, "__mro__")
                for container_type in (list, tuple, set, frozenset, dict, functools.partial)
            )
            or _getattr_type_static(value_type, "__call__", _ABSENT_SLOT) is not _ABSENT_SLOT
        ):
            raise ValueError
        else:
            if (
                _getattr_type_static(value_type, "__getattribute__", _ABSENT_SLOT)
                is not object.__getattribute__
                or _getattr_type_static(value_type, "__getattr__", _ABSENT_SLOT) is not _ABSENT_SLOT
            ):
                raise ValueError
            related = _static_object_values(value)
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
                        inspect_method_receiver and value_type is not types.MethodType,
                        max_nodes,
                        trusted_objects,
                    )
                    or found
                )
        return found
    finally:
        active.remove(value_id)
