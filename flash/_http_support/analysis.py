"""static namespace-root classification for private http transport."""

from __future__ import annotations

import dis
import types


def namespace_origins(
    roots: dict[str, dict[str, object]],
) -> frozenset[tuple[str, str]]:
    found = set()
    for root_kind, values in roots.items():
        for name, value in values.items():
            value_mro = type.__getattribute__(type(value), "__mro__")
            if types.ModuleType in value_mro or type in value_mro:
                found.add((root_kind, name))
    return frozenset(found)


def has_ambiguous_control_flow(
    instructions: tuple[dis.Instruction, ...],
    start: int,
    end: int,
    origins: dict[str, tuple[str, str]],
    branch_opcodes: frozenset[int],
    fast_load_opnames: frozenset[str],
    expression_start,
    strict_globals: frozenset[str] | None = None,
) -> bool:
    offset_indices = {instruction.offset: index for index, instruction in enumerate(instructions)}
    for index, instruction in enumerate(instructions[:end]):
        if instruction.opcode not in branch_opcodes:
            continue
        target = instruction.argval
        target_index = offset_indices.get(target) if type(target) is int else None
        if target_index is None or not start <= target_index <= end:
            continue
        branch_start = expression_start(instructions, index)
        while instructions[branch_start].opname == "COPY":
            branch_start = expression_start(instructions, branch_start)
        for candidate_index in range(branch_start, end):
            candidate = instructions[candidate_index]
            direct_global = (
                candidate.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
                and (strict_globals is None or candidate.argval in strict_globals)
                and (
                    candidate_index + 1 == len(instructions)
                    or instructions[candidate_index + 1].opname not in {"LOAD_ATTR", "LOAD_METHOD"}
                )
            )
            local_origin = (
                candidate.opname in fast_load_opnames | {"LOAD_DEREF"}
                and candidate.argval in origins
            )
            if direct_global or local_origin:
                return True
    return False


def is_direct_namespace_global(
    instructions: tuple[dis.Instruction, ...],
    index: int,
    origins: frozenset[tuple[str, str]],
) -> bool:
    instruction = instructions[index]
    return (
        instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        and ("global", instruction.argval) in origins
        and (
            index + 1 == len(instructions)
            or instructions[index + 1].opname not in {"LOAD_ATTR", "LOAD_METHOD"}
        )
    )
