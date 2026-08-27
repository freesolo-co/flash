"""Fresh-process import-order regressions for training dependency leaves."""

from __future__ import annotations

import ast
import itertools
import os
import pathlib
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_VERL_MODULES = (
    "flash.engine.worker.train.entry.backend_common",
    "flash.engine.worker.verl.capabilities",
    "flash.engine.worker.verl.install",
)
_SFT_MODULES = (
    "flash.engine.worker.train.entry.sft_train",
    "flash.engine.worker.train.entry.sft_train_runner",
    "flash.engine.worker.train.sft.orchestration",
)
_NEUTRAL_LEAVES = (
    "flash.engine.worker.train.sft.orchestration",
    "flash.engine.worker.train.sft.setup.checkpoints",
    "flash.engine.worker.train.sft.setup.config",
    "flash.engine.worker.verl.capabilities",
    "flash.engine.worker.verl.checkpoints",
    "flash.engine.worker.verl.child_io",
    "flash.engine.worker.verl.diagnostics",
    "flash.engine.worker.verl.install",
    "flash.engine.worker.verl.process",
)


def _cyclic_orders(modules: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """bounded orders with every module first and last and every pair in both directions."""
    return tuple(modules[index:] + modules[:index] for index in range(len(modules)))


@pytest.mark.parametrize("order", tuple(itertools.permutations(_VERL_MODULES)))
def test_verl_modules_import_in_every_order_in_a_fresh_process(order):
    _assert_fresh_import_order(order)


@pytest.mark.parametrize("order", tuple(itertools.permutations(_SFT_MODULES)))
def test_sft_modules_import_in_every_order_in_a_fresh_process(order):
    _assert_fresh_import_order(order)


@pytest.mark.parametrize("order", _cyclic_orders(_NEUTRAL_LEAVES))
def test_neutral_leaves_import_in_pairwise_orders_in_a_fresh_process(order):
    _assert_fresh_import_order(order)


def test_training_dependencies_point_from_facades_to_leaves_only():
    imports = {
        module: _imported_modules(module)
        for module in (*_VERL_MODULES, *_SFT_MODULES, *_NEUTRAL_LEAVES)
    }
    assert (
        "flash.engine.worker.train.entry.sft_train"
        not in imports["flash.engine.worker.train.entry.sft_train_runner"]
    )
    for leaf in _NEUTRAL_LEAVES:
        forbidden = sorted(
            imported
            for imported in imports[leaf]
            if imported == "flash.engine.worker.train.entry"
            or imported.startswith("flash.engine.worker.train.entry.")
        )
        assert not forbidden, f"neutral leaf {leaf} imports training facade(s): {forbidden}"


def test_backend_facade_reexports_neutral_owner_identities():
    from flash.engine.worker.train.entry import backend_common, sft_train
    from flash.engine.worker.train.sft.setup import checkpoints as sft_checkpoints
    from flash.engine.worker.verl import capabilities, process

    assert backend_common.fused_ce_backend is capabilities.fused_ce_backend
    assert backend_common.run_verl_training is process.run_verl_training
    assert sft_train._export_checkpoint_adapter is sft_checkpoints._export_checkpoint_adapter


def _module_source(module: str) -> pathlib.Path:
    """the module's own source, whether it is a plain module or a package marker."""
    base = _REPO_ROOT.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"no source file for {module}")


def _imported_modules(module: str) -> set[str]:
    tree = ast.parse(_module_source(module).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _assert_fresh_import_order(order: tuple[str, ...]) -> None:
    script = "\n".join(f"import {module}" for module in order)
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    assert completed.returncode == 0, f"import order {order!r} failed:\n{output}"
    assert "partially initialized module" not in output
