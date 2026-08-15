from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


def _gate_module():
    path = Path(__file__).resolve().parents[1] / "scripts/check_function_size.py"
    spec = importlib.util.spec_from_file_location("check_function_size", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_function_size_excludes_nested_definitions_but_checks_them_independently():
    gate = _gate_module()
    nested_body = "\n".join(f"        value_{index} = {index}" for index in range(151))
    tree = ast.parse(
        "def outer():\n"
        "    before = 1\n"
        "    @staticmethod\n"
        "    def inner():\n"
        f"{nested_body}\n"
        "        return value_150\n"
        "    return before + inner()\n"
    )
    definitions = dict(gate._walk_defs(tree))

    assert gate._length(definitions["outer"]) == 3
    assert gate._length(definitions["outer.inner"]) > gate.FUNCTION_MAX


def test_function_size_checks_a_nested_method_without_inflating_the_owner():
    gate = _gate_module()
    method_body = "\n".join(f"            value_{index} = {index}" for index in range(151))
    tree = ast.parse(
        "def outer():\n"
        "    before = 1\n"
        "    class Nested:\n"
        "        def method(self):\n"
        f"{method_body}\n"
        "            return value_150\n"
        "    return before\n"
    )
    definitions = dict(gate._walk_defs(tree))

    assert gate._length(definitions["outer"]) == 4
    assert gate._length(definitions["outer.Nested.method"]) > gate.FUNCTION_MAX


def test_function_size_charges_nested_class_level_body_to_the_owner():
    gate = _gate_module()
    class_body = "\n".join(f"        value_{index} = {index}" for index in range(151))
    tree = ast.parse(f"def outer():\n    class Nested:\n{class_body}\n    return Nested\n")
    definitions = dict(gate._walk_defs(tree))

    assert gate._length(definitions["outer"]) > gate.FUNCTION_MAX
    assert set(definitions) == {"outer"}


def test_function_size_gate_has_no_source_shipped_exemption():
    gate = _gate_module()

    assert not hasattr(gate, "SOURCE_SHIPPED")
