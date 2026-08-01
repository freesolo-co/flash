"""cpu contracts for the sft to verl migration."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest

from flash.engine.worker.backend_common import parse_verl_metric
from flash.engine.worker.sft import _pretokenize_completion_only
from flash.engine.worker.sft_train import (
    _LORAPLUS_READY_MARKER,
    _MAX_ZERO_GRAD_STEPS,
    _VERL_OPTIMIZER_IMPL,
    _VERL_OPTIMIZER_NAME,
    _build_verl_child_env,
    _render_sft_dataset_module,
    _render_sft_sitecustomize,
    _serialize_multimodal_inputs,
    _write_sft_parquet,
    build_sft_overrides,
    render_loraplus_shim,
)


def _cfg(**over):
    base = {
        "train_files": "/w/train.parquet",
        "val_files": "/w/val.parquet",
        "train_batch_size": 32,
        "max_length": 32768,
        "micro_batch": 1,
        "max_token_len_per_gpu": 8192,
        "custom_dataset_path": "/w/flash_verl_sft_dataset.py",
        "model_path": "Qwen/Qwen3-4B",
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "ulysses_sp_size": 2,
        "lr": 1e-4,
        "warmup_ratio": 0.03,
        "optimizer_impl": "bitsandbytes.optim",
        "optimizer_name": "PagedAdamW8bit",
        "optimizer_kwargs": {},
        "local_dir": "/w/ckpt",
        "save_freq": 50,
        "n_gpus_per_node": 2,
        "seed": 42,
        "project_name": "flash-sft",
        "experiment_name": "run-xyz",
        "loop_epochs": 4,
        "total_training_steps": 120,
    }
    base.update(over)
    return base


def _as_map(overrides):
    return dict(override.split("=", 1) for override in overrides)


def test_overrides_match_verl_0_8_sft_and_fsdp_config_surface():
    overrides = _as_map(build_sft_overrides(_cfg()))
    assert overrides == {
        "data.train_files": "/w/train.parquet",
        "data.val_files": "/w/val.parquet",
        "data.train_batch_size": "32",
        "data.max_length": "32768",
        "data.micro_batch_size_per_gpu": "1",
        "data.use_dynamic_bsz": "true",
        "data.max_token_len_per_gpu": "8192",
        "data.truncation": "right",
        "data.num_workers": "4",
        "data.ignore_input_ids_mismatch": "false",
        "data.custom_cls.path": "/w/flash_verl_sft_dataset.py",
        "data.custom_cls.name": "FlashTokenizedSFTDataset",
        "model.path": "Qwen/Qwen3-4B",
        "model.trust_remote_code": "true",
        "model.lora_rank": "16",
        "model.lora_alpha": "32",
        "model.target_modules": "all-linear",
        "model.lora_adapter_path": "null",
        "model.use_remove_padding": "true",
        "model.use_liger": "true",
        "model.enable_gradient_checkpointing": "true",
        "engine.strategy": "fsdp2",
        "engine.model_dtype": "bfloat16",
        "engine.seed": "42",
        "engine.ulysses_sequence_parallel_size": "2",
        "optim.lr": "0.0001",
        "optim.lr_warmup_steps_ratio": "0.03",
        "optim.optimizer_impl": "bitsandbytes.optim",
        "optim.optimizer": "PagedAdamW8bit",
        "optim.weight_decay": "0.0",
        "optim.betas": "[0.9,0.999]",
        "optim.override_optimizer_config": "{eps:0.00000001}",
        "trainer.default_local_dir": "/w/ckpt",
        "trainer.save_freq": "50",
        "trainer.n_gpus_per_node": "2",
        "trainer.nnodes": "1",
        "trainer.seed": "42",
        "trainer.logger": "[console]",
        "trainer.project_name": "flash-sft",
        "trainer.experiment_name": "run-xyz",
        "trainer.total_epochs": "4",
        "trainer.test_freq": "-1",
        "trainer.resume_mode": "auto",
        "model.use_fused_kernels": "true",
        "model.fused_kernel_options.impl_backend": "torch",
        "trainer.max_ckpt_to_keep": "1",
        "trainer.total_training_steps": "120",
    }
    assert "optim.eps" not in overrides
    assert "optim.lr_scheduler_type" not in overrides
    assert "data.messages_key" not in overrides


def test_verl_sft_optimizer_is_dtensor_safe():
    """the fsdp2 engine hands DTensor params to the optimizer.

    bitsandbytes' 8-bit blockwise kernel is not a distributed operator and raises
    "got mixed torch.Tensor and DTensor" on the first step, so the verl SFT path must
    never select an 8-bit optimizer regardless of what the TRL memory profile prefers.
    """
    assert (_VERL_OPTIMIZER_IMPL, _VERL_OPTIMIZER_NAME) == ("torch.optim", "AdamW")
    assert "8bit" not in _VERL_OPTIMIZER_NAME.lower()
    assert "bitsandbytes" not in _VERL_OPTIMIZER_IMPL


def test_sft_engine_strategy_stays_fsdp2():
    """LoRA+ groups parameters by name ("lora_B" in name).

    fsdp1 flattens parameters into a 1-D flat_param, which would route every parameter
    into the 16x group B and silently corrupt the learning rates, so the DTensor problem
    above must not be "fixed" by downgrading the strategy.
    """
    assert _as_map(build_sft_overrides(_cfg()))["engine.strategy"] == "fsdp2"


def test_optimizer_eps_merges_into_override_config():
    overrides = _as_map(build_sft_overrides(_cfg(optimizer_kwargs={"amsgrad": True}, eps=1e-6)))
    assert overrides["optim.override_optimizer_config"] == "{amsgrad:true,eps:0.000001}"


def test_small_lr_renders_fixed_point_not_scientific():
    overrides = _as_map(build_sft_overrides(_cfg(lr=5e-5)))
    assert overrides["optim.lr"] == "0.00005"


def test_steps_xor_epochs_is_enforced():
    with pytest.raises(ValueError, match="exactly one"):
        build_sft_overrides(_cfg(total_training_steps=120, total_epochs=3))
    with pytest.raises(ValueError, match="exactly one"):
        build_sft_overrides(_cfg(total_training_steps=None, total_epochs=None))


class _ExactTokenizer:
    eos_token = "!"
    all_special_ids = (0,)

    def __call__(self, texts, *, truncation, max_length):
        assert truncation is True
        return {"input_ids": [[ord(char) for char in text][:max_length] for text in texts]}


def test_exact_mask_keeps_prompt_assistant_history_masked_and_full_target_active():
    tokenizer = _ExactTokenizer()
    prompt = "<user>q</user><assistant>history</assistant><assistant>"
    full = prompt + "first</assistant><user>tool</user><assistant>second</assistant>"
    texts = [{"text": full, "prompt_text": prompt}]

    kept, rows, dropped = _pretokenize_completion_only(texts, tokenizer, max_length=512)

    assert kept == texts
    assert dropped == 0
    split = len(prompt)
    assert rows[0]["completion_mask"][:split] == [0] * split
    assert all(rows[0]["completion_mask"][split:])
    assert len(rows[0]["input_ids"]) == len(rows[0]["completion_mask"])


def test_exact_mask_drops_right_truncated_completion_and_handles_thinking_prefix():
    tokenizer = _ExactTokenizer()
    prompt = "<think>prompt<assistant>"
    full = "<think>prompt<assistant>answer"
    kept, rows, dropped = _pretokenize_completion_only(
        [{"text": full, "prompt_text": prompt}],
        tokenizer,
        max_length=len(prompt),
    )
    assert kept == []
    assert rows == []
    assert dropped == 1

    kept, rows, dropped = _pretokenize_completion_only(
        [{"text": full, "prompt_text": prompt}],
        tokenizer,
        max_length=512,
    )
    assert kept
    assert dropped == 0
    assert any(rows[0]["completion_mask"])


def _load_custom_dataset_module(tmp_path):
    path = tmp_path / "flash_verl_sft_dataset.py"
    path.write_text(_render_sft_dataset_module())
    spec = importlib.util.spec_from_file_location("flash_verl_sft_dataset_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("image_first", [False, True])
def test_explicit_schema_and_custom_dataset_read_text_image_orders(
    monkeypatch, tmp_path, image_first
):
    np = pytest.importorskip("numpy")

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def __len__(self):
            return len(self.value)

        def tolist(self):
            return self.value.tolist()

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.value, axis))

    fake_torch = _module("torch")
    fake_torch.long = np.int64
    fake_torch.tensor = lambda value, dtype=None: FakeTensor(value)
    fake_torch.arange = lambda length, dtype=None: FakeTensor(np.arange(length))
    fake_torch.ones_like = lambda value: FakeTensor(np.ones_like(value.value))
    fake_torch.from_numpy = FakeTensor
    fake_torch.cat = lambda values, dim=0: FakeTensor(
        np.concatenate([value.value for value in values], axis=dim)
    )
    image_row = {
        "input_ids": [20, 21, 22],
        "loss_mask": [0, 1, 1],
        "images": ["file:///tmp/image.png"],
        "multimodal_inputs": _serialize_multimodal_inputs(
            {"image_grid_thw": np.asarray([[1, 2, 3]], dtype=np.int64)}
        ),
    }
    text_row = {
        "input_ids": [10, 11],
        "loss_mask": [0, 1],
        "images": [],
        "multimodal_inputs": b"",
    }
    rows = [image_row, text_row] if image_first else [text_row, image_row]
    parquet = tmp_path / "mixed.parquet"
    _write_sft_parquet(rows, str(parquet))

    datasets = pytest.importorskip("datasets")
    raw = datasets.Dataset.from_parquet(str(parquet))
    assert raw.column_names == ["input_ids", "loss_mask", "images", "multimodal_inputs"]
    assert raw[0]["images"] == rows[0]["images"]
    assert raw[1]["images"] == rows[1]["images"]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    qwen_module = _module(
        "verl.models.transformers.qwen2_vl",
        get_rope_index=lambda processor, input_ids, **kwargs: FakeTensor(
            np.zeros((3, len(input_ids)), dtype=np.int64)
        ),
    )
    for name, injected in {
        "verl": _module("verl"),
        "verl.models": _module("verl.models"),
        "verl.models.transformers": _module("verl.models.transformers"),
        "verl.models.transformers.qwen2_vl": qwen_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, injected)

    class Qwen2VLImageProcessorFake:
        pass

    module = _load_custom_dataset_module(tmp_path)
    dataset = module.FlashTokenizedSFTDataset(
        parquet_files=str(parquet),
        tokenizer=SimpleNamespace(),
        processor=SimpleNamespace(image_processor=Qwen2VLImageProcessorFake()),
        config={"max_length": 8, "truncation": "right", "ignore_input_ids_mismatch": False},
    )
    first = dataset[0]
    second = dataset[1]
    assert first["input_ids"].tolist() == rows[0]["input_ids"]
    assert first["loss_mask"].tolist() == rows[0]["loss_mask"]
    assert second["input_ids"].tolist() == rows[1]["input_ids"]
    assert second["loss_mask"].tolist() == rows[1]["loss_mask"]
    assert len(first["position_ids"].tolist()) == 4
    assert len(second["position_ids"].tolist()) == 4
    image_item = first if image_first else second
    assert image_item["multi_modal_inputs"]["image_grid_thw"].tolist() == [[1, 2, 3]]


@pytest.mark.parametrize("shape", ["listconfig", "list", "str"])
def test_custom_dataset_accepts_every_parquet_files_shape(tmp_path, shape):
    """verl passes data.train_files through from hydra, so the dataset sees a ListConfig."""
    module = _load_custom_dataset_module(tmp_path)
    parquet = tmp_path / "rows.parquet"
    rows = [{"input_ids": [1, 2], "loss_mask": [0, 1], "images": [], "multimodal_inputs": b""}]
    _write_sft_parquet(rows, str(parquet))

    if shape == "listconfig":
        # omegaconf is not a flash dependency (it lives in the verl venv), so stand in for
        # ListConfig with the property that actually broke: a sequence that is not a list/tuple.
        class _ListConfig:
            def __init__(self, items):
                self._items = list(items)

            def __iter__(self):
                return iter(self._items)

            def __len__(self):
                return len(self._items)

        assert not isinstance(_ListConfig([]), (list, tuple))
        parquet_files = _ListConfig([str(parquet)])
    elif shape == "list":
        parquet_files = [str(parquet)]
    else:
        parquet_files = str(parquet)

    dataset = module.FlashTokenizedSFTDataset(
        parquet_files=parquet_files,
        tokenizer=SimpleNamespace(),
        config={"max_length": 8, "truncation": "right", "ignore_input_ids_mismatch": False},
    )
    # constructing is the assertion: the ListConfig shape used to die here reading the parquet.
    assert len(dataset) == 1
    assert list(dataset.dataframe["input_ids"].iloc[0]) == [1, 2]


def test_custom_dataset_rejects_suppressed_input_id_checks(tmp_path):
    module = _load_custom_dataset_module(tmp_path)
    parquet = tmp_path / "one.parquet"
    _write_sft_parquet(
        [{"input_ids": [1], "loss_mask": [1], "images": [], "multimodal_inputs": b""}],
        str(parquet),
    )
    with pytest.raises(ValueError, match="mismatch checks"):
        module.FlashTokenizedSFTDataset(
            parquet_files=str(parquet),
            tokenizer=SimpleNamespace(),
            config={"max_length": 8, "ignore_input_ids_mismatch": True},
        )


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_generated_sitecustomize_installs_linear_scheduler_and_required_loraplus(
    monkeypatch, capsys
):
    class FakeLoader:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class FakeTrainer:
        def _build_dataloader(self):
            self.train_sampler = SimpleNamespace(seed=None)
            self.val_sampler = None
            return "loader"

    class FakeCheckpointHandler:
        def save_checkpoint(self, step):
            return step

    class FakeEngine:
        rank = 0

        def _build_optimizer(self, module):
            return "plain"

        def _build_lr_scheduler(self, optimizer):
            return "constant"

        def _build_module(self):
            return SimpleNamespace(gradient_checkpointing_enable=lambda **kwargs: None)

    fake_torch = _module("torch")
    fake_torch.manual_seed = lambda seed: None
    fake_torch.set_float32_matmul_precision = lambda value: None
    fake_torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        cudnn=SimpleNamespace(allow_tf32=False),
    )
    fake_numpy = _module("numpy", random=SimpleNamespace(seed=lambda seed: None))
    scheduler_calls = []
    fake_transformers = _module(
        "transformers",
        get_linear_schedule_with_warmup=lambda optimizer, **kwargs: (
            scheduler_calls.append((optimizer, kwargs)) or "linear"
        ),
    )
    fake_sft_module = _module(
        "verl.trainer.sft_trainer",
        StatefulDataLoader=FakeLoader,
        SFTTrainer=FakeTrainer,
    )
    fake_checkpoint_module = _module(
        "verl.utils.checkpoint.checkpoint_handler",
        CheckpointHandler=FakeCheckpointHandler,
    )
    fake_fsdp_module = _module(
        "verl.workers.engine.fsdp.transformer_impl",
        FSDPEngine=FakeEngine,
    )
    optimizer_calls = []
    fake_peft = _module(
        "peft.optimizers",
        create_loraplus_optimizer=lambda **kwargs: optimizer_calls.append(kwargs) or "lora+",
    )
    fake_optimizer_module = _module("fake_optimizer", AdamW=type("AdamW", (), {}))

    modules = {
        "torch": fake_torch,
        "numpy": fake_numpy,
        "transformers": fake_transformers,
        "verl": _module("verl"),
        "verl.trainer": _module("verl.trainer", sft_trainer=fake_sft_module),
        "verl.trainer.sft_trainer": fake_sft_module,
        "verl.utils": _module("verl.utils"),
        "verl.utils.checkpoint": _module("verl.utils.checkpoint"),
        "verl.utils.checkpoint.checkpoint_handler": fake_checkpoint_module,
        "verl.workers": _module("verl.workers"),
        "verl.workers.engine": _module("verl.workers.engine"),
        "verl.workers.engine.fsdp": _module("verl.workers.engine.fsdp"),
        "verl.workers.engine.fsdp.transformer_impl": fake_fsdp_module,
        "peft": _module("peft"),
        "peft.optimizers": fake_peft,
        "fake_optimizer": fake_optimizer_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = _render_sft_sitecustomize(
        seed=43,
        loraplus_ratio=16,
        save_at_steps=(3, 7),
        total_steps=9,
        reentrant_gradient_checkpointing=False,
    )
    exec(compile(source, "sitecustomize.py", "exec"), {})

    engine = FakeEngine()
    engine.optimizer_config = SimpleNamespace(
        optimizer_impl="fake_optimizer",
        optimizer="AdamW",
        lr=5e-5,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        override_optimizer_config={"eps": 1e-8},
        lr_warmup_steps=-1,
        lr_warmup_steps_ratio=0.1,
        total_training_steps=20,
    )
    assert engine._build_optimizer(SimpleNamespace()) == "lora+"
    assert _LORAPLUS_READY_MARKER in capsys.readouterr().out
    assert optimizer_calls[0]["optimizer_kwargs"]["eps"] == 1e-8
    assert engine._build_lr_scheduler("optimizer") == "linear"
    assert scheduler_calls == [("optimizer", {"num_warmup_steps": 2, "num_training_steps": 20})]


def test_reentrant_checkpointing_enables_input_grads_before_enabling_checkpointing():
    """GRAD-001: lora freezes the embeddings, so nothing entering the first checkpointed layer
    requires grad and reentrant checkpointing returns no gradient at all. the shim must call
    enable_input_require_grads() BEFORE gradient_checkpointing_enable(), or every sft run
    trains nothing while reporting done and billing."""
    calls = []

    class FakeModule:
        def enable_input_require_grads(self):
            calls.append("require_grads")

        def gradient_checkpointing_enable(self, **kwargs):
            calls.append(("gc_enable", kwargs))

    class FakeEngine:
        def _build_module(self):
            return FakeModule()

    source = _render_sft_sitecustomize(
        seed=1,
        loraplus_ratio=16,
        save_at_steps=(),
        total_steps=4,
        reentrant_gradient_checkpointing=True,
    )
    # execute only the reentrant block: the surrounding shim imports verl/torch at module scope.
    block = source[source.index("def _flash_build_reentrant_module") :]
    block = block[: block.index("_FlashFSDPEngine._build_module = _flash_build_reentrant_module")]
    namespace = {"_flash_original_build_module": FakeEngine._build_module}
    exec(compile(block, "shim.py", "exec"), namespace)

    namespace["_flash_build_reentrant_module"](FakeEngine())

    # order matters: enabling checkpointing first would capture the graph before any input
    # requires grad, so asserting mere presence would pass on a broken shim.
    assert calls[0] == "require_grads"
    assert calls[1] == ("gc_enable", {"gradient_checkpointing_kwargs": {"use_reentrant": True}})

    # the non-reentrant path never patches _build_module at all, so it must not appear.
    non_reentrant = _render_sft_sitecustomize(
        seed=1,
        loraplus_ratio=16,
        save_at_steps=(),
        total_steps=4,
        reentrant_gradient_checkpointing=False,
    )
    assert "enable_input_require_grads" not in non_reentrant


def test_zero_grad_norm_at_nonzero_lr_fails_the_run():
    """GRAD-001: four runs reported done and charged while grad_norm was 0.0 on every step.
    the number was parsed and recorded, never read. replay the real g4 log lines."""

    def replay(lines):
        zero_grad_steps: list[int] = []
        for step, line in enumerate(lines):
            grad_norm = parse_verl_metric(line, "train/grad_norm")
            learning_rate_value = parse_verl_metric(line, "train/lr")
            if grad_norm is None:
                continue
            if grad_norm == 0.0 and (learning_rate_value is None or learning_rate_value > 0.0):
                zero_grad_steps.append(step)
                if len(zero_grad_steps) >= _MAX_ZERO_GRAD_STEPS:
                    raise RuntimeError(
                        "verl reported train/grad_norm=0.0 with a nonzero learning rate on "
                        f"{len(zero_grad_steps)} consecutive steps: no gradient is reaching "
                        "the lora parameters, so this run would train nothing. see GRAD-001"
                    )
            else:
                zero_grad_steps.clear()

    # verbatim shape of the g4 gsm8k lines (flash-1785592071-e56cf3c6): loss barely moves on a
    # replayed identical batch because nothing is learning.
    broken = [
        "step:1 - train/loss:0.5470 - train/grad_norm:0.0 - train/lr:0.0001",
        "step:2 - train/loss:0.5437 - train/grad_norm:0.0 - train/lr:0.0001",
        "step:3 - train/loss:0.5474 - train/grad_norm:0.0 - train/lr:0.0001",
        "step:4 - train/loss:0.5444 - train/grad_norm:0.0 - train/lr:0.0001",
    ]
    with pytest.raises(RuntimeError, match="grad_norm=0.0"):
        replay(broken)

    # a healthy run must not trip the guard.
    replay(
        [
            "step:1 - train/loss:0.9 - train/grad_norm:1.4 - train/lr:0.0001",
            "step:2 - train/loss:0.7 - train/grad_norm:0.9 - train/lr:0.0001",
        ]
    )

    # an isolated zero is a legitimately fully-masked micro-batch, not a severed graph.
    replay(
        [
            "step:1 - train/loss:0.9 - train/grad_norm:0.0 - train/lr:0.0001",
            "step:2 - train/loss:0.7 - train/grad_norm:1.1 - train/lr:0.0001",
            "step:3 - train/loss:0.6 - train/grad_norm:0.0 - train/lr:0.0001",
        ]
    )

    # a decayed schedule reaching lr 0.0 produces a legitimate zero grad norm.
    replay(
        [
            "step:1 - train/loss:0.5 - train/grad_norm:0.0 - train/lr:0.0",
            "step:2 - train/loss:0.5 - train/grad_norm:0.0 - train/lr:0.0",
            "step:3 - train/loss:0.5 - train/grad_norm:0.0 - train/lr:0.0",
        ]
    )


def test_loraplus_shim_has_no_plain_lora_fallback():
    source = render_loraplus_shim(16)
    assert _LORAPLUS_READY_MARKER in source
    assert "falling back" not in source
    assert "_flash_original_build_optimizer" not in source
    assert render_loraplus_shim(1) == ""


def test_child_environment_excludes_provider_and_control_plane_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    monkeypatch.setenv("HF_HOME", "/cache/hf")
    monkeypatch.setenv("FLASH_VERL_PYTHON", "/verl/python")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("FIREWORKS_API_KEY", "teacher-secret")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "control-secret")
    monkeypatch.setenv("RUNPOD_API_KEY", "provider-secret")
    monkeypatch.setenv("HF_TOKEN", "hub-secret")

    without_wandb = _build_verl_child_env(shim_dir=str(tmp_path), wandb_enabled=False)
    assert without_wandb["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert without_wandb["NCCL_DEBUG"] == "WARN"
    assert without_wandb["HF_HOME"] == "/cache/hf"
    assert without_wandb["FLASH_VERL_PYTHON"] == "/verl/python"
    assert "WANDB_API_KEY" not in without_wandb
    for secret in (
        "FIREWORKS_API_KEY",
        "FREESOLO_INTERNAL_KEY",
        "RUNPOD_API_KEY",
        "HF_TOKEN",
    ):
        assert secret not in without_wandb

    with_wandb = _build_verl_child_env(shim_dir=str(tmp_path), wandb_enabled=True)
    assert with_wandb["WANDB_API_KEY"] == "wandb-secret"


def test_checkpoint_watcher_exports_and_uploads_required_step(monkeypatch, tmp_path):
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    checkpoint_dir = tmp_path / "checkpoints" / "global_step_5"
    actor_dir = checkpoint_dir / "actor"
    (actor_dir / "huggingface").mkdir(parents=True)
    exported = []
    published = []
    uploaded = []

    def fake_export(actor, adapter, **kwargs):
        exported.append((actor, adapter, kwargs))
        os.makedirs(adapter, exist_ok=True)

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", fake_export)
    monkeypatch.setattr(
        worker,
        "publish_deployable_checkpoint",
        lambda adapter, step, **kwargs: published.append((adapter, step, kwargs)),
    )

    def fake_upload(step, checkpoint, **kwargs):
        kwargs["before_upload"]()
        uploaded.append((step, checkpoint))
        return True

    monkeypatch.setattr(worker, "upload_resume_checkpoint", fake_upload)
    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(tmp_path / "checkpoints"),
        export_root=str(tmp_path / "exports"),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(5,),
    )

    watcher._publish(5, str(checkpoint_dir))

    assert exported[0][0] == str(actor_dir)
    assert published[0][1] == 5
    assert published[0][2]["required"] is True
    assert uploaded == [(5, str(checkpoint_dir))]
    assert watcher.processed_steps == {5}


def test_checkpoint_watcher_exports_the_sft_layout(monkeypatch, tmp_path):
    # this is the layout verl's sft trainer actually writes: shards + huggingface/ directly under
    # global_step_N. exporting <dir>/actor here hands the merger a path that does not exist.
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    checkpoint_dir = tmp_path / "checkpoints" / "global_step_5"
    (checkpoint_dir / "huggingface").mkdir(parents=True)
    exported = []

    def fake_export(actor, adapter, **kwargs):
        if not os.path.isdir(actor):
            raise AssertionError(f"exported a checkpoint dir that does not exist: {actor}")
        exported.append(actor)
        os.makedirs(adapter, exist_ok=True)

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", fake_export)
    monkeypatch.setattr(
        worker, "publish_deployable_checkpoint", lambda adapter, step, **kwargs: None
    )
    monkeypatch.setattr(
        worker,
        "upload_resume_checkpoint",
        lambda step, checkpoint, **kwargs: (kwargs["before_upload"](), True)[1],
    )
    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(tmp_path / "checkpoints"),
        export_root=str(tmp_path / "exports"),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(5,),
    )

    watcher._publish(5, str(checkpoint_dir))

    assert exported == [str(checkpoint_dir)]


def test_resume_credits_only_required_saves_that_are_durable(monkeypatch):
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    class Api:
        def file_exists(self, *, filename, **kwargs):
            return "/step-3/" in filename

    monkeypatch.setattr(worker, "HF_REPO", "owner/artifacts")
    monkeypatch.setattr(worker, "hf_prefix", lambda: "sft/run")
    monkeypatch.setattr(worker, "hf_api", Api)

    assert sft_train._durable_required_save_steps((3, 5, 9), 5) == {3}


def test_run_sft_train_orchestrates_exact_dataset_and_resume_accounting(monkeypatch):
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    spec = SimpleNamespace(
        model="Qwen/Qwen3.5-0.8B",
        model_revision="",
        gpu=SimpleNamespace(type="RTX 4090", exact_type="", count=2),
        train=SimpleNamespace(
            epochs=1,
            learning_rate=5e-5,
            batch_size=2,
            max_context_tokens=1024,
            max_examples=2,
            max_steps=2,
            save_at_steps=(),
            save_every=50,
            init_from_adapter="",
        ),
        wandb=SimpleNamespace(project=None, run_name=None),
    )

    class Env:
        id = "owner/env"
        package_root = None
        multi_turn = False

        def dataset(self):
            return [{"prompt": "one"}, {"prompt": "two"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["prompt"]}]

        def sft_completion(self, example):
            return [{"role": "assistant", "content": "answer"}]

    class Tokenizer(_ExactTokenizer):
        pad_token = None

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            enable_thinking,
        ):
            assert tokenize is False
            rendered = "".join(
                f"<{message['role']}>{message['content']}</{message['role']}>"
                for message in messages
            )
            if add_generation_prompt:
                rendered += "<assistant>"
            return rendered

    class LoraConfig:
        r = 16
        lora_alpha = 32
        target_modules = "all-linear"

    class PeakSampler:
        def start(self):
            return self

        def stop_gb(self):
            return 12.5

    class Watcher:
        def __init__(self, **kwargs):
            self.processed_steps = set()

        def start(self):
            return None

        def stop(self, *, require_complete):
            assert require_complete is True

        def raise_if_failed(self):
            return None

    captured = {"heartbeats": [], "published": [], "uploads": []}
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(worker, "SEED", 7)
    monkeypatch.setattr(worker, "RUN_ID", "test-sft-verl-orchestration")
    monkeypatch.setattr(worker, "THINKING", False)
    monkeypatch.setattr(worker, "JOB_SPEC", spec)
    monkeypatch.setattr(worker, "require_active_env", Env)
    monkeypatch.setattr(
        worker,
        "heartbeat",
        lambda stage, **fields: captured["heartbeats"].append((stage, fields)),
    )
    monkeypatch.setattr(worker, "gpu_diagnostics", lambda **kwargs: {})
    monkeypatch.setattr(worker, "prefetch_model", lambda *args, **kwargs: 1.25)
    monkeypatch.setattr(worker, "load_tokenizer", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(worker, "make_lora", lambda model_id: LoraConfig())
    monkeypatch.setattr(worker, "grad_checkpointing_on", lambda *args, **kwargs: True)
    monkeypatch.setattr(worker, "grpo_use_reentrant", lambda model_id: False)
    monkeypatch.setattr(worker, "backend_seed", lambda seed: seed)
    monkeypatch.setattr(worker, "wandb_run_name", lambda: "flash-sft-test")
    monkeypatch.setattr(
        worker,
        "hf_upload_folder",
        lambda local, remote, required=False: captured["uploads"].append((local, remote, required)),
    )
    monkeypatch.setattr(
        worker,
        "publish_deployable_checkpoint",
        lambda adapter, step, **kwargs: captured["published"].append((adapter, step)),
    )
    monkeypatch.setattr(
        worker,
        "write_train_meta",
        lambda **kwargs: captured.__setitem__("meta", kwargs),
    )
    monkeypatch.setattr(
        sft_train,
        "liveness_heartbeat",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        sft_train,
        "_probe_gpu_in_subprocess",
        lambda *args, **kwargs: {"memory_gb": 24, "capability": [8, 9]},
    )
    monkeypatch.setattr(sft_train, "_model_arch_dims", lambda *args, **kwargs: (1024, 24))
    monkeypatch.setattr(sft_train, "resolve_verl_python", lambda *a, **k: "/venv/bin/python")
    monkeypatch.setattr(sft_train, "resolve_verl_loggers", lambda python_bin: ["console"])
    # torch is not installed in this test env; the real seeding is covered in test_training_controls.
    monkeypatch.setattr(sft_train, "seed_training_rngs", lambda seed: None)
    monkeypatch.setattr(sft_train, "_cached_model_path", lambda model, revision: model)
    monkeypatch.setattr(sft_train, "_restore_verl_resume", lambda local_dir: 1)
    monkeypatch.setattr(sft_train, "_VerlCheckpointWatcher", Watcher)
    monkeypatch.setattr(sft_train, "_NvidiaSmiPeakSampler", PeakSampler)
    monkeypatch.setattr(
        sft_train,
        "latest_global_step_dir",
        lambda local_dir: (os.path.join(local_dir, "global_step_2"), 2),
    )

    def fake_export(actor_dir, adapter_dir, **kwargs):
        os.makedirs(adapter_dir, exist_ok=True)
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as file:
            file.write("{}")
        with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "wb") as file:
            file.write(b"adapter")

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", fake_export)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        captured["command"] = command
        captured["child_env"] = env
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        on_line("step:2 - train/loss:1.0 - train/global_tokens:8\n")
        on_step(2)
        heartbeat()
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    sft_train.run_sft_train(spec)

    assert captured["command"][:3] == ["/venv/bin/python", "-m", "torch.distributed.run"]
    assert "--nproc-per-node=2" in captured["command"]
    assert "verl.trainer.sft_trainer" in captured["command"]
    custom_path = next(
        value.split("=", 1)[1]
        for value in captured["command"]
        if value.startswith("data.custom_cls.path=")
    )
    assert os.path.isfile(custom_path)
    assert captured["child_env"]["PYTHONPATH"].split(os.pathsep)[0].endswith("/shim")
    assert captured["child_env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["uploads"][0][1:] == ("adapter", True)
    assert captured["published"][0][1] == 2
    assert captured["meta"]["step"] == 2
    assert captured["meta"]["train_tokens"] > 0
    assert captured["meta"]["notes"]["loss_curve"] == [1.0]
    assert captured["meta"]["notes"]["loraplus_applied"] is True
    assert captured["meta"]["notes"]["realized_max_length"] > 0


def test_overrides_enable_fused_linear_ce_for_long_context():
    # 32k contexts must not materialize [tokens, vocab] logits; the fused torch-backend
    # linear-CE computes loss from hidden states in chunks (numerically exact CE).
    o = build_sft_overrides(_cfg(max_length=32768))
    assert "model.use_fused_kernels=true" in o
    assert "model.fused_kernel_options.impl_backend=torch" in o
    assert "data.max_length=32768" in o


def test_sft_line_handler_reads_metrics_through_the_shared_parser():
    """sft shares OPD's numpy-2-aware parser instead of keeping its own float() copy.

    sft's three metrics reach the logger as plain python floats today (engine_workers.py
    returns loss/grad_norm via .item() and lr via get_last_lr()), so unlike OPD's
    Metric(SUM) they do not currently print in numpy's np.float64(...) spelling. the
    duplicate parser was still removed: one upstream metric-type change would have
    reintroduced the same silent drop, and the shared helper additionally rejects nan/inf,
    which would otherwise serialize into the heartbeat as bare NaN.
    """
    import ast
    import inspect
    import textwrap

    import flash.engine.worker.sft_train as sv

    source = textwrap.dedent(inspect.getsource(sv.run_sft_train))
    handler = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "on_line"
    )
    calls = [
        node.func.id
        for node in ast.walk(handler)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("parse_verl_metric") == 3
    assert "_metric_value" not in calls
    # the duplicated helper and its regex are gone, not merely unused.
    assert not hasattr(sv, "_metric_value")
    assert not hasattr(sv, "_VERL_METRIC_RE")


def test_sft_drops_a_non_finite_loss_instead_of_poisoning_the_heartbeat():
    """a nan loss serializes as bare NaN, which strict json consumers reject."""
    import flash.engine.worker.sft_train as sv

    assert sv.parse_verl_metric("step:2 - train/loss:nan - train/lr:1e-05", "train/loss") is None
    assert sv.parse_verl_metric("step:2 - train/loss:inf", "train/loss") is None
    # a finite value on the same line is unaffected.
    assert sv.parse_verl_metric("step:2 - train/loss:nan - train/lr:1e-05", "train/lr") == 1e-05
