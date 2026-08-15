"""focused contracts for verl child plugin delivery."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from flash.engine.worker import opd_train, rl_train, sft_train
from flash.engine.worker.train.core.child import runtime
from flash.engine.worker.train.opd.child import entry as opd_entry
from flash.engine.worker.train.rl.child import entry as grpo_entry
from flash.engine.worker.train.sft.child import entry as sft_entry
from flash.engine.worker.verl.child_io import render_sitecustomize_bootstrap


@pytest.mark.parametrize(
    ("entry", "plugin_name"),
    [
        (sft_entry, "flash_sft_plugin"),
        (grpo_entry, "flash_grpo_plugin"),
        (opd_entry, "flash_opd_plugin"),
    ],
)
def test_strict_entries_reject_a_missing_external_module(monkeypatch, entry, plugin_name):
    monkeypatch.setitem(sys.modules, "verl", types.ModuleType("verl"))
    monkeypatch.setenv("VERL_USE_EXTERNAL_MODULES", plugin_name)
    monkeypatch.delitem(sys.modules, plugin_name, raising=False)

    with pytest.raises(RuntimeError, match="was not loaded by verl"):
        entry.main()


def test_each_runner_configures_exactly_one_algorithm_plugin():
    sft_source = inspect.getsource(sft_train._prepare_sft_child)
    grpo_source = inspect.getsource(rl_train._build_rl_child_env)
    opd_source = inspect.getsource(opd_train._build_opd_child_env)

    assert 'child_env["VERL_USE_EXTERNAL_MODULES"] = "flash_sft_plugin"' in sft_source
    assert 'env_for_verl["VERL_USE_EXTERNAL_MODULES"] = "flash_grpo_plugin"' in grpo_source
    assert '"VERL_USE_EXTERNAL_MODULES": "flash_opd_plugin"' in opd_source
    assert sft_source.count("VERL_USE_EXTERNAL_MODULES") == 1
    assert grpo_source.count("VERL_USE_EXTERNAL_MODULES") == 1
    assert opd_source.count("VERL_USE_EXTERNAL_MODULES") == 1


def test_large_grpo_plugin_config_uses_a_file_and_launches_the_child(tmp_path):
    description = "x" * 140_000
    structured_outputs = {
        "json": {
            "type": "object",
            "properties": {
                "payload": {"type": "string", "description": description},
            },
        }
    }
    files = {
        "shim_dir": str(tmp_path),
        "shim_py": str(tmp_path / "sitecustomize.py"),
        "shim_markers": str(tmp_path / "applied_shims.txt"),
        "rank_device_claims": str(tmp_path / "rank_device_claims.txt"),
        "plugin_config_path": str(tmp_path / "flash_grpo_plugin_config.json"),
    }
    inp = {
        "dp_cards": 1,
        "reentrant_checkpointing": False,
        "multimodal": False,
        "entropy_quantile": None,
        "per_turn_credit": False,
        "stop_sequences": (),
        "image_pad_token_id": None,
        "structured_outputs": structured_outputs,
        "save_at_steps": (),
        "steps": 1,
        "warmstart_adapter": None,
        "kl_coef": 0.0,
        "multi_turn": False,
    }
    rl_train._write_rl_shim(inp, files)
    rl_train._write_rl_plugin_config(inp, files, gdn_reset_arch=None, loggers=[])

    config_path = Path(files["plugin_config_path"])
    assert config_path.stat().st_size > 128 * 1024
    env = rl_train._build_rl_child_env(inp, files, [], "http://127.0.0.1:9/")
    assert "FLASH_GRPO_PLUGIN_CONFIG" not in env
    assert env["FLASH_GRPO_PLUGIN_CONFIG_PATH"] == str(config_path)
    assert len(env["FLASH_GRPO_PLUGIN_CONFIG_PATH"]) < 4096

    script = """
import flash_verl_runtime as runtime
config = runtime.load_plugin_config_file("FLASH_GRPO_PLUGIN_CONFIG_PATH")
schema = config["structured_outputs"]["json"]
assert schema["type"] == "object"
assert len(schema["properties"]["payload"]["description"]) == 140000
print("config-loaded")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "config-loaded" in result.stdout.splitlines()


def test_grpo_external_plugin_arms_without_importing_cuda_sensitive_targets(tmp_path):
    rl_train.copy_grpo_child_modules(str(tmp_path))
    config_path = tmp_path / "flash_grpo_plugin_config.json"
    config_path.write_text(
        json.dumps(
            {
                "marker_file": str(tmp_path / "markers"),
                "dp_cards": 2,
                "reentrant_checkpointing": True,
                "multimodal": True,
                "entropy_quantile": 0.2,
                "per_turn_credit": True,
                "stop_sequences": ["</answer>"],
                "image_pad_token_id": 151655,
                "structured_outputs": {"json": {"type": "object"}},
                "save_at_steps": [3],
                "total_steps": 10,
                "kl_ref_adapter": True,
                "multi_turn": True,
                "gdn_model_type": None,
                "wandb": False,
            }
        )
    )
    script = """
import sys
import flash_grpo_plugin
forbidden = sorted(
    name for name in sys.modules
    if name == "torch" or name == "verl" or name == "vllm"
    or name.startswith("torch.") or name.startswith("verl.") or name.startswith("vllm.")
)
assert forbidden == [], forbidden
targets = [
    finder.target for finder in sys.meta_path
    if type(finder).__name__ == "_DeferredFinder"
]
assert "verl.single_controller.base.worker" in targets
assert targets.count("verl.experimental.agent_loop.agent_loop") == 4
print("plugin-armed")
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        env={
            "PYTHONPATH": str(tmp_path),
            "FLASH_GRPO_PLUGIN_CONFIG_PATH": str(config_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["plugin-armed"]


def test_sft_entry_calls_the_canonical_trainer_once_without_runpy_warning(monkeypatch, capsys):
    calls = []
    verl = types.ModuleType("verl")
    trainer_package = types.ModuleType("verl.trainer")
    trainer_package.__path__ = []
    trainer = types.ModuleType("verl.trainer.sft_trainer")
    trainer.main = lambda: calls.append("main")
    plugin = types.ModuleType("flash_sft_plugin")
    plugin.PLUGIN_LOADED_EXTERNALLY = True

    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.trainer", trainer_package)
    monkeypatch.setitem(sys.modules, "verl.trainer.sft_trainer", trainer)
    monkeypatch.setitem(sys.modules, "flash_sft_plugin", plugin)
    monkeypatch.setenv("VERL_USE_EXTERNAL_MODULES", "flash_sft_plugin")

    sft_entry.main()

    assert calls == ["main"]
    assert "found in sys.modules" not in capsys.readouterr().err
    runner_source = inspect.getsource(sft_train._prepare_sft_child)
    assert '"flash_sft_entry"' in runner_source
    assert '"verl.trainer.sft_trainer"' not in runner_source


def test_sitecustomize_contains_only_startup_behavior():
    source = render_sitecustomize_bootstrap()
    compile(source, "sitecustomize.py", "exec")
    assert "tf32" in source.lower()
    assert "tilelang libcudart" in source
    for forbidden in ("import verl", "transformers.models", "vllm_rollout", "wandb.init"):
        assert forbidden not in source


def _clear_deferred_target(target: str) -> None:
    sys.modules.pop(target, None)
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not isinstance(finder, runtime._DeferredFinder) or finder.target != target
    ]


@pytest.mark.parametrize("names", [("gdn-varlen", "flashqla-gdn"), ("flashqla-gdn", "gdn-varlen")])
def test_deferred_finders_stack_in_both_orders_and_mark_after_application(
    tmp_path, monkeypatch, names
):
    target = "flash_deferred_target"
    module_file = tmp_path / f"{target}.py"
    module_file.write_text("loaded = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    marker_file = str(tmp_path / "applied_shims.txt")
    applied = []
    _clear_deferred_target(target)

    try:
        for name in names:
            runtime._arm_deferred(
                name=name,
                marker_file=marker_file,
                target=target,
                patch=lambda module, patch_name=name: applied.append(patch_name),
                required=True,
            )
        assert target not in sys.modules
        assert not Path(marker_file).exists()

        imported = importlib.import_module(target)

        assert imported.loaded is True
        assert applied == list(names)
        assert Path(marker_file).read_text().splitlines() == list(names)
        assert not [
            finder
            for finder in sys.meta_path
            if isinstance(finder, runtime._DeferredFinder) and finder.target == target
        ]
    finally:
        _clear_deferred_target(target)


def test_gdn_flashqla_and_lora_installers_do_not_import_targets_when_armed(tmp_path):
    marker_file = str(tmp_path / "applied_shims.txt")
    targets = (
        "transformers.models.flash_fake.modeling_flash_fake",
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
    )
    for target in targets:
        _clear_deferred_target(target)
    try:
        runtime.install_deferred_gdn("flash_fake", marker_file)
        runtime.install_deferred_flash_qla("flash_fake", marker_file)
        runtime.install_deferred_lora_rollout_guard(marker_file)
        assert all(target not in sys.modules for target in targets)
        assert not Path(marker_file).exists()
    finally:
        for target in targets:
            _clear_deferred_target(target)


def test_required_installer_failure_exits_with_97(monkeypatch, tmp_path):
    class RequiredExit(RuntimeError):
        pass

    seen = []

    def fake_exit(code):
        seen.append(code)
        raise RequiredExit

    def fail():
        raise ValueError("incompatible target")

    monkeypatch.setattr(runtime.os, "_exit", fake_exit)
    with pytest.raises(RequiredExit):
        runtime.install_required("required", str(tmp_path / "markers"), fail)
    assert seen == [runtime.SHIM_FRAGMENT_FAILED_EXIT_CODE]


def test_opd_plugin_installs_runtime_before_verl_extensions():
    source = Path(
        "/home/azureuser/benchmark/flash-pristine-base/flash/engine/worker/train/opd/child/plugin.py"
    ).read_text()
    bridge_at = source.index("from flash_opd_bridge import")
    runtime_at = source.index("flash_opd_runtime.install(")
    extensions_at = source.rindex("_install_verl_extensions()")
    assert bridge_at < runtime_at < extensions_at


def test_copied_flat_modules_have_no_importerror_fallback_to_flash():
    roots = (
        Path("/home/azureuser/benchmark/flash-pristine-base/flash/engine/worker/train/core/child"),
        Path("/home/azureuser/benchmark/flash-pristine-base/flash/engine/worker/train/sft/child"),
        Path("/home/azureuser/benchmark/flash-pristine-base/flash/engine/worker/train/rl/child"),
        Path("/home/azureuser/benchmark/flash-pristine-base/flash/engine/worker/train/opd/child"),
    )
    for root in roots:
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                for handler in node.handlers:
                    fallback_imports = [
                        child
                        for child in ast.walk(handler)
                        if isinstance(child, ast.ImportFrom)
                        and (child.module or "").startswith("flash.")
                    ]
                    assert not fallback_imports, path
