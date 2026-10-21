"""contracts for packaged training runtimes (replaces generated plugin loading)."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.engine.worker.train.core.child.launch import load_launch
from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.core.child.readiness import publish_ready
from flash.engine.worker.train.entry.sft_train import _build_verl_child_env
from flash.engine.worker.verl.child_io import verify_runtime_ready

ROOT = Path(__file__).resolve().parents[1]


def _modules(monkeypatch, values):
    """supply CPU stand-ins only for dependency boundaries, then import the real recipe."""
    if "verl.workers.engine.fsdp.transformer_impl" in values:
        values = {
            "verl.workers.engine": {
                "EngineRegistry": SimpleNamespace(register=lambda **kw: lambda cls: cls)
            },
            **values,
        }
    for name, attrs in values.items():
        if name.startswith("flash."):
            importlib.import_module(name.rsplit(".", 1)[0])
        for depth in range(1, len(name.split(".")) + 1):
            package = ".".join(name.split(".")[:depth])
            if package not in sys.modules:
                module = types.ModuleType(package)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, package, module)
        module = types.ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)


def _import_recipe(monkeypatch, name):
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules.pop(name)
    return importlib.import_module(name)


@pytest.mark.parametrize(
    ("algorithm", "directory"), [("sft", "sft"), ("grpo", "rl"), ("opd", "opd")]
)
def test_entry_help_runs_from_an_unrelated_directory_without_verl(tmp_path, algorithm, directory):
    result = subprocess.run(
        [sys.executable, "-m", f"flash.engine.worker.train.{directory}.child.entry", "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--flash-config" in result.stdout
    assert algorithm in result.stdout


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_launch_config_is_data_and_can_exceed_environment_limits(tmp_path, algorithm):
    path = tmp_path / "launch.json"
    payload = {
        "total_steps": 9,
        "readiness_path": str(tmp_path / "ready"),
        "structured_outputs": {"json": {"description": "large" * 50000}},
    }
    path.write_text(json.dumps(payload))
    options, overrides = load_launch(
        algorithm, ["--flash-config", str(path), "trainer.total_epochs=2"]
    )
    assert options.algorithm == algorithm
    assert options.structured_outputs == payload["structured_outputs"]
    assert overrides == ["trainer.total_epochs=2"]


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": 1},
        {"total_steps": 0},
        {"total_steps": True},
        {"save_at_steps": [0]},
        {"save_at_steps": [10]},
        {"algorithm": "unknown"},
        {"no_signal_attempts": 0},
    ],
)
def test_launch_rejects_invalid_policy(changes):
    value = {"algorithm": "sft", "total_steps": 9, "readiness_path": "ready"}
    value.update(changes)
    with pytest.raises(ValueError, match=r"unknown|positive|horizon"):
        RuntimeOptions.decode(value)


@pytest.mark.parametrize(("schedule", "expected"), [([], set(range(1, 8))), ([3, 5], {3, 5, 7})])
def test_exact_checkpoint_schedule_includes_final_save(schedule, expected):
    options = RuntimeOptions("sft", 7, "ready", save_at_steps=tuple(schedule))
    assert {step for step in range(1, 8) if options.should_save(step)} == expected


def test_readiness_is_required_and_atomically_published(tmp_path):
    path = str(tmp_path / "ready")
    with pytest.raises(RuntimeError, match="did not finish initialization"):
        verify_runtime_ready(path, ("runtime-ready",))
    publish_ready(RuntimeOptions("sft", 7, path))
    verify_runtime_ready(path, ("runtime-ready",))
    assert [item.name for item in tmp_path.iterdir()] == ["ready"]


def test_child_uses_only_its_verified_package_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "obsolete-shims"))
    monkeypatch.setenv("VERL_USE_EXTERNAL_MODULES", "old_plugin")
    env = _build_verl_child_env(wandb_enabled=False)
    assert env["PYTHONPATH"] == str(ROOT)
    assert "VERL_USE_EXTERNAL_MODULES" not in env


def test_packaged_runtime_has_no_import_interception_or_dependency_method_rebinding():
    child_root = ROOT / "flash/engine/worker/train"
    forbidden = [
        "sys.meta_path",
        "types.FunctionType",
        "types.MethodType",
        "sitecustomize",
        "VERL_USE_EXTERNAL_MODULES",
        "FSDPEngine._build",
        "AgentLoopWorker._compute_teacher_logprobs =",
        "wandb.init =",
        "CheckpointHandler.save_checkpoint =",
        ".chunk_gated_delta_rule =",
    ]
    for path in child_root.glob("*/child/*.py"):
        source = path.read_text()
        for fragment in forbidden:
            assert fragment not in source, (path, fragment)
    assert not (child_root / "core/child/runtime.py").exists()
    assert not (child_root / "rl/child/patches.py").exists()


@pytest.mark.parametrize("enabled", [True, False])
def test_model_options_are_owned_by_each_engine(monkeypatch, enabled):
    calls = []

    class BaseEngine:
        def __init__(self, **kwargs):
            self.model_config = kwargs["model_config"]

        def _build_module(self):
            return self.model_config.module

        def _build_lr_scheduler(self, optimizer):
            return ("upstream", optimizer)

    _modules(
        monkeypatch,
        {"verl.workers.engine.fsdp.transformer_impl": {"FSDPEngineWithLMHead": BaseEngine}},
    )
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.core.child.engine")
    monkeypatch.setattr(recipe, "initialize_process", lambda: None)
    monkeypatch.setattr(recipe, "install_vision_input_grads", lambda module: calls.append("vision"))
    module = SimpleNamespace(
        enable_input_require_grads=lambda: calls.append("input"),
        gradient_checkpointing_enable=lambda **kw: calls.append(kw),
    )
    runtime = RuntimeOptions(
        "sft", 7, "ready", reentrant_gradient_checkpointing=True, multimodal=True
    )
    engine = recipe.FlashFSDPEngine(
        model_config=SimpleNamespace(
            runtime_options=runtime.encode(), module=module, enable_gradient_checkpointing=enabled
        ),
        engine_config=SimpleNamespace(native_config=lambda: "native"),
    )
    assert engine._build_module() is module
    expected = (
        ["input", {"gradient_checkpointing_kwargs": {"use_reentrant": True}}, "vision"]
        if enabled
        else []
    )
    assert calls == expected
    assert (
        BaseEngine._build_module(SimpleNamespace(model_config=SimpleNamespace(module=module)))
        is module
    )
    assert calls == expected


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_every_engine_delegates_model_loading_to_upstream(monkeypatch, algorithm):
    calls = []
    model = object()

    class BaseEngine:
        def _build_module(self):
            calls.append(self.model_config)
            return model

    _modules(
        monkeypatch,
        {"verl.workers.engine.fsdp.transformer_impl": {"FSDPEngineWithLMHead": BaseEngine}},
    )
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.core.child.engine")
    engine = recipe.FlashFSDPEngine.__new__(recipe.FlashFSDPEngine)
    engine.model_config = SimpleNamespace(enable_gradient_checkpointing=False)
    engine.runtime = RuntimeOptions(algorithm, 1, "ready")
    assert engine._build_module() is model
    assert calls == [engine.model_config]


def test_sft_uses_linear_scheduler_without_replacing_upstream(monkeypatch):
    class BaseEngine:
        def _build_lr_scheduler(self, optimizer):
            return "upstream"

    calls = []
    _modules(
        monkeypatch,
        {
            "verl.workers.engine.fsdp.transformer_impl": {"FSDPEngineWithLMHead": BaseEngine},
            "transformers": {
                "get_linear_schedule_with_warmup": lambda opt, **kw: calls.append((opt, kw))
            },
        },
    )
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.core.child.engine")
    engine = recipe.FlashFSDPEngine.__new__(recipe.FlashFSDPEngine)
    engine.runtime = RuntimeOptions("sft", 10, "ready")
    engine.optimizer_config = SimpleNamespace(
        lr_warmup_steps=0, lr_warmup_steps_ratio=0.2, total_training_steps=10
    )
    optimizer = object()
    engine._build_lr_scheduler(optimizer)
    assert calls == [(optimizer, {"num_warmup_steps": 2, "num_training_steps": 10})]
    engine.runtime = RuntimeOptions("grpo", 10, "ready")
    assert engine._build_lr_scheduler(optimizer) == "upstream"
    assert BaseEngine()._build_lr_scheduler(optimizer) == "upstream"


@pytest.mark.parametrize("fail", [False, True])
def test_tracking_closes_the_owned_run_with_the_actual_outcome(monkeypatch, capsys, fail):
    finishes = []
    run = SimpleNamespace(
        url="https://wandb.ai/run", id="abc", finish=lambda **kw: finishes.append(kw)
    )

    class Tracking:
        def __init__(self, **kwargs):
            self.logger = {"wandb": SimpleNamespace(run=run)}

        def finish(self, exit_code=0):
            assert "wandb" not in self.logger

    _modules(monkeypatch, {"verl.utils.tracking": {"Tracking": Tracking}})
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.core.child.tracking")

    tracking = recipe.OwnedTracking(project_name="flash")
    tracking.finish(exit_code=int(fail))
    tracking.finish(exit_code=int(fail))
    assert finishes == [{"exit_code": int(fail)}]
    assert json.loads(capsys.readouterr().out.removeprefix("FLASH_WANDB_LINK ")) == {
        "wandb_url": run.url,
        "wandb_id": "abc",
    }


def _sft_recipe(monkeypatch, *, resume_step=0):
    class BaseTrainer:
        def __init__(self, config):
            self.config = config
            self.train_dataset = list(range(10))
            self.engine = SimpleNamespace(
                get_data_parallel_size=lambda: 1, get_data_parallel_rank=lambda: 0
            )
            self._build_dataloader()
            self.original_loader = self.train_dataloader
            self.steps_per_epoch = len(self.train_dataloader)
            self.total_training_steps = 9
            self.resume_global_step = resume_step
            self.ckpt_handler = SimpleNamespace(train_dataloader=self.train_dataloader)

    class Checkpoint:
        def __init__(self, **kwargs):
            self.saved = []

        def save_checkpoint(self, step):
            self.saved.append(step)
            return step

    _modules(
        monkeypatch,
        {
            "verl.trainer.sft_trainer": {"SFTTrainer": BaseTrainer},
            "flash.engine.worker.train.sft.child.loop": {"fit_sft": lambda _: None},
            "verl.utils.checkpoint": {"CheckpointHandler": Checkpoint},
            "verl.utils.dataset.dataset_utils": {"SFTTensorCollator": lambda _: list},
            "verl.utils.tracking": {"Tracking": object},
        },
    )
    _import_recipe(monkeypatch, "flash.engine.worker.train.core.child.tracking")
    return _import_recipe(monkeypatch, "flash.engine.worker.train.sft.child.trainer")


@pytest.mark.parametrize("resume_step", [0, 1, 3, 6, 9])
def test_sft_exact_order_partial_batch_and_epoch_boundary_resume(monkeypatch, resume_step):
    pytest.importorskip("torchdata")
    recipe = _sft_recipe(monkeypatch, resume_step=resume_step)
    trainer = recipe.FlashSFTTrainer(
        SimpleNamespace(
            model=SimpleNamespace(runtime_options=RuntimeOptions("sft", 9, "ready").encode()),
            data=SimpleNamespace(train_batch_size=4, pad_mode="right", num_workers=0),
        )
    )
    assert list(trainer.train_dataloader) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert trainer.train_sampler.seed == 42
    assert (trainer.train_dataloader is not trainer.original_loader) == (resume_step in (3, 6))
    assert trainer.ckpt_handler.train_dataloader is trainer.train_dataloader


def test_sft_required_checkpoint_waits_for_publication_before_retention(monkeypatch, tmp_path):
    pytest.importorskip("torchdata")
    recipe = _sft_recipe(monkeypatch)
    handler = recipe.FlashCheckpointHandler(
        runtime=RuntimeOptions(
            "sft",
            7,
            "ready",
            save_at_steps=(2, 4),
            checkpoint_published_dir=str(tmp_path),
        )
    )
    assert handler.save_checkpoint(1) is None
    assert handler.save_checkpoint(2) == 2
    waits = []

    def publish_during_wait(_delay):
        assert handler.saved == [2]
        waits.append(True)
        (tmp_path / "step-2").touch()

    monkeypatch.setattr(recipe.time, "sleep", publish_during_wait)
    assert handler.save_checkpoint(4) == 4
    assert waits == [True]
    (tmp_path / "step-4").touch()
    assert handler.save_checkpoint(7) == 7
    assert handler.saved == [2, 4, 7]


def test_gdn_boundaries_are_confined_to_the_constructed_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from flash.engine.worker.train.core.child.model import configure_gdn

    class ExampleTextModel(torch.nn.Module):
        def forward(self, *, position_ids, **kwargs):
            return kwargs

    ExampleTextModel.__module__ = "transformers.models.test_gdn.modeling_test_gdn"
    configured, untouched = ExampleTextModel(), ExampleTextModel()
    configure_gdn(configured, "test_gdn")
    positions = torch.tensor([[0, 1, 2, 0, 1]])
    kwargs = configured(position_ids=positions)
    assert kwargs["seq_idx"].tolist() == [[0, 0, 0, 1, 1]]
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 3, 5]
    assert untouched(position_ids=positions) == {}
    assert configured(position_ids=torch.tensor([[0, 1, 2]])) == {}
    explicit = torch.tensor([0, 5])
    assert configured(position_ids=positions, cu_seq_lens_q=explicit)["cu_seq_lens_q"] is explicit


def test_optimizer_mutation_notice_precedes_real_update_and_is_once_per_optimizer(monkeypatch):
    torch = pytest.importorskip("torch")
    from flash.engine.worker.train.opd.child import bridge

    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.25)
    before = []
    monkeypatch.setattr(
        bridge, "_coordinate_first_mutation_notice", lambda *_: before.append(parameter.item())
    )
    bridge._wrap_optimizer_with_mutation_notice(optimizer, "url", "token")
    for _ in range(2):
        parameter.grad = torch.tensor(1.0)
        optimizer.step()
    assert before == [2.0]
    assert parameter.item() == 1.5


def test_physical_device_claims_reject_cross_rank_collisions(monkeypatch, tmp_path):
    _modules(monkeypatch, {"verl.workers.engine_workers": {"ActorRolloutRefWorker": object}})
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.core.child.worker")
    path = str(tmp_path / "claims")
    recipe.claim_device(0, path, "gpu-a")
    recipe.claim_device(1, path, "gpu-b")
    recipe.claim_device(0, path, "gpu-a")
    with pytest.raises(RuntimeError, match="SAME physical gpu"):
        recipe.claim_device(2, path, "gpu-a")


@pytest.mark.parametrize("mask", [[[0, 0]], [[0, 0], [1, 0]]])
def test_grpo_rejects_empty_response_batches_before_advantages(monkeypatch, mask):
    torch = pytest.importorskip("torch")
    calls = []

    class BaseTrainer:
        def _compute_advantage(self, batch, metrics):
            calls.append(batch)
            return batch

    _modules(
        monkeypatch,
        {
            "transfer_queue": {"kv_batch_get": lambda **kw: {"response_mask": torch.tensor(mask)}},
            "tensordict": {"TensorDict": object},
            "verl": {"DataProto": object},
            "verl.workers.utils.padding": {"response_to_nested": object},
            "flash.engine.worker.train.core.child.trainer": {"FlashSyncTrainer": BaseTrainer},
        },
    )
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.rl.child.trainer")
    trainer = recipe.FlashGrpoTrainer.__new__(recipe.FlashGrpoTrainer)
    trainer.runtime = RuntimeOptions("grpo", 1, "ready")
    batch = SimpleNamespace(keys=["a"], partition_id="train")
    if len(mask) == 1:
        with pytest.raises(RuntimeError, match="no trainable response tokens"):
            trainer._compute_advantage(batch, {})
        assert calls == []
    else:
        assert trainer._compute_advantage(batch, {}) is batch
        assert calls == [batch]
