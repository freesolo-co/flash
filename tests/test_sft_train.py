"""cpu contracts for the sft to verl migration."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import pathlib
import re
import shutil
import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flash.engine.sft_workload import _serialize_multimodal_inputs
from flash.engine.worker.backend_common import parse_verl_metric, verl_step_number
from flash.engine.worker.sft import _pretokenize_completion_only
from flash.engine.worker.sft_train import (
    _LORAPLUS_READY_MARKER,
    _MAX_ZERO_GRAD_STEPS,
    _VERL_OPTIMIZER_IMPL,
    _VERL_OPTIMIZER_NAME,
    _build_verl_child_env,
    _render_sft_dataset_module,
    _render_sft_sitecustomize,
    _write_sft_parquet,
    build_sft_overrides,
    render_exact_sft_dataloader_shim,
    render_loraplus_shim,
)

# distinct from `flash.__version__` on purpose: the worker resolves that to "0+unknown" (no flash
# distribution is installed there), so a fixture built from it could not catch a worker that
# re-derives the producer version instead of reading the one carried on the spec.
_PROFILE_PRODUCER_VERSION = "9.9.9"


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
        "model.use_liger": "false",
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


def test_overrides_carry_fused_expert_target_parameters():
    overrides = _as_map(
        build_sft_overrides(
            _cfg(
                target_parameters=[
                    "mlp.experts.gate_up_proj",
                    "mlp.experts.down_proj",
                ]
            )
        )
    )

    assert overrides["++model.target_parameters"] == (
        "[mlp.experts.gate_up_proj,mlp.experts.down_proj]"
    )


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


def test_verl_packs_every_batch_so_the_batch_size_is_the_isolation_boundary():
    """verl concatenates a batch into one sequence, so `train_batch_size` decides what shares it.

    The worker sends `model.use_remove_padding=true` and leaves `data.pad_mode` at verl's
    `no_padding` default, which together make the fsdp engine hand the model a single
    ``(1, total_nnz)`` row with ``attention_mask=None`` and per-example ``position_ids`` restarts.
    Attention recovers its boundaries from those restarts; GatedDeltaNet layers do not, because
    they read ``seq_idx`` and ``cu_seq_lens_q`` out of kwargs the fsdp engine never sends. So on a
    gdn hybrid -- which every catalog model is -- the batch size is the only thing standing between
    one example and the next example's carried state, and a profile that grouped examples for
    costing convenience would silently corrupt training. Pin both halves of that coupling: neither
    override may drift without this failing.
    """
    overrides = _as_map(build_sft_overrides(_cfg(train_batch_size=1)))

    assert overrides["model.use_remove_padding"] == "true"
    assert "data.pad_mode" not in overrides
    assert overrides["data.train_batch_size"] == "1"


def test_remove_padding_is_unconditional():
    """Nothing may gate `use_remove_padding`: it is the only layout verl's sft_loss can consume.

    verl leaves `data.pad_mode` at its `no_padding` default, and that loss path reads
    ``log_prob.values()`` -- which only exists on the nested tensor the remove-padding branch
    builds. Send `use_remove_padding=false` and the fsdp engine hands the loss a strided tensor
    instead, so the first optimizer step dies with "values expected sparse tensor layout but got
    Strided" before a single update lands.

    Switching to verl's padded mode is not an escape either: `pad_mode: right` collates with
    ``default_collate`` (uniform rows only) and its loss branch reads ``response_mask``, and
    ``FlashTokenizedSFTDataset`` emits neither -- it yields variable-length rows carrying
    input_ids/position_ids/loss_mask. So `no_padding` + remove-padding is the only combination
    this dataset fits, and the flag has no legitimate false case.

    Two revisions learned this the hard way, each gating the flag on something real but
    irrelevant: first `packing_mode == "packed"` (the profile), then `not gdn_hybrid or
    gdn_boundary_resets` (the child probe). Both reasoned that a gdn hybrid without child-side
    resets must not pack -- true, but it never does: `_packing_mode` answers "exact-unpacked" for
    every gdn model, which pins `examples_per_update` to 1, so a gdn run has no packed neighbour
    to be contaminated by. Batch size is the isolation lever; this flag never was.

    Read the source rather than the rendered overrides: `build_sft_overrides` takes the flag
    already computed, so a test driving it through a cfg dict asserts on its own fixture and stays
    green no matter what the derivation does.
    """
    import inspect

    from flash.engine.worker import sft_train

    src = inspect.getsource(sft_train.run_sft_train)
    line = next(
        ln.strip() for ln in src.splitlines() if ln.strip().startswith("use_remove_padding =")
    )

    assert line == "use_remove_padding = True", line


@pytest.mark.parametrize(
    ("support", "expected_mode"),
    [
        (("gdn-hybrid", False), "exact-unpacked"),
        (("unsupported", False), "exact-unpacked"),
        (("pure-attention", True), "packed"),
    ],
)
def test_batch_is_the_isolation_lever_that_replaces_the_removed_flag(support, expected_mode):
    """Deleting the gate is only safe because no unsupported architecture ever packs.

    `use_remove_padding` used to be the (broken) isolation lever. The real one is the batch size,
    and it holds one step earlier: an architecture that cannot reset state at packed boundaries
    profiles as `exact-unpacked`, which pins `examples_per_update` to 1, so the run has no packed
    neighbour whose residue could bleed across. Pin the chain at its source -- if a future change
    let a gdn hybrid profile as "packed", the deleted flag would no longer be there to catch it.

    Read `_packing_mode` and the `examples_per_update` derivation directly rather than asserting
    on rendered overrides: `build_sft_overrides` defaults `use_remove_padding` to True when the key
    is absent, so an override-level assertion reads that default and passes no matter what the
    worker computed.
    """
    from flash.engine.sft_workload import _packing_mode

    packing_mode, architecture_mode = _packing_mode(
        "Qwen/Qwen3.5-4B",
        "rev",
        multimodal=False,
        allow_packing=True,
        packing_support=lambda _m, _r: support,
    )

    assert (packing_mode, architecture_mode) == (expected_mode, support[0])
    # an architecture that cannot pack must not be able to reach a batch > 1
    assert (expected_mode == "packed") is support[1]


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


def _exec_dataloader_shim(monkeypatch):
    """Execute the real rendered dataloader shim against fake torch classes, and return them.

    The shim is a string concatenated into the shipped sitecustomize, so nothing else in the suite
    would notice it breaking: an import it cannot satisfy, or a patch that silently stops applying,
    surfaces on a rented gpu as rows arriving in a different order than the profile measured.
    """
    sampler_calls: list[dict] = []
    loader_calls: list[dict] = []

    class FakeDistributedSampler:
        def __init__(self, dataset, **kwargs):
            sampler_calls.append(kwargs)

    class FakeStatefulDataLoader:
        def __init__(self, dataset, **kwargs):
            loader_calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "torch.utils.data.distributed",
        _module("torch.utils.data.distributed", DistributedSampler=FakeDistributedSampler),
    )
    monkeypatch.setitem(
        sys.modules,
        "torchdata.stateful_dataloader",
        _module("torchdata.stateful_dataloader", StatefulDataLoader=FakeStatefulDataLoader),
    )

    exec(compile(render_exact_sft_dataloader_shim(), "sitecustomize.py", "exec"), {})
    return FakeDistributedSampler, FakeStatefulDataLoader, sampler_calls, loader_calls


def test_dataloader_shim_forces_the_row_order_the_profile_measured(monkeypatch):
    """Shuffle off and drop_last off, whatever verl asked for.

    Both are overrides, not defaults: verl passes shuffle=True and drop_last=True itself, so a shim
    that merely supplied them when absent would change nothing. Shuffling would reorder the rows
    the profile tokenized in a fixed order, and dropping the last partial batch would train on
    fewer examples than were measured and quoted -- neither of which fails loudly.
    """
    sampler, loader, sampler_calls, loader_calls = _exec_dataloader_shim(monkeypatch)

    sampler(["row"], shuffle=True, num_replicas=1, rank=0)
    loader(["row"], drop_last=True, batch_size=2)

    assert sampler_calls == [{"shuffle": False, "num_replicas": 1, "rank": 0}]
    assert loader_calls == [{"drop_last": False, "batch_size": 2}]


def test_dataloader_shim_sets_both_flags_when_the_caller_omits_them(monkeypatch):
    """A caller that passes neither must still get the exact-order behaviour.

    verl currently passes both, but the shim's guarantee cannot depend on that: a verl release that
    started relying on the library defaults (shuffle=True, drop_last=False for the sampler) would
    silently reintroduce shuffling through a shim that only rewrote what it was given.
    """
    sampler, loader, sampler_calls, loader_calls = _exec_dataloader_shim(monkeypatch)

    sampler(["row"])
    loader(["row"])

    assert sampler_calls == [{"shuffle": False}]
    assert loader_calls == [{"drop_last": False}]


def test_dataloader_shim_patches_the_classes_verl_imports(monkeypatch):
    """The patch must land on the shared class objects, not on a copy the shim made.

    A shim that rebound its own local name would execute cleanly and assert nothing -- verl
    constructs the sampler through its own import of the same class, so the identity of the patched
    attribute is the whole mechanism.
    """
    sampler, loader, _sampler_calls, _loader_calls = _exec_dataloader_shim(monkeypatch)

    from torch.utils.data.distributed import DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    assert DistributedSampler is sampler
    assert StatefulDataLoader is loader
    assert DistributedSampler.__init__.__name__ == "_flash_exact_sampler_init"
    assert StatefulDataLoader.__init__.__name__ == "_flash_exact_loader_init"


def test_shipped_shim_carries_the_exact_dataloader_patch(monkeypatch):
    """The rendered fragment is only worth testing if it reaches the file verl imports.

    ``run_sft_train`` concatenates it onto the sitecustomize source; asserting that here keeps the
    three tests above from passing against a fragment nothing writes out.
    """
    import pathlib

    import flash.engine.worker.sft_train as sft_train_module

    fragment = render_exact_sft_dataloader_shim()
    source = "".join(pathlib.Path(sft_train_module.__file__).read_text().split())
    assert "shim_source+=render_exact_sft_dataloader_shim()" in source
    assert "shuffle" in fragment
    assert "drop_last" in fragment


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


class _TolerantWatcher:
    """a watcher that permits an incomplete run, for tests where a guard is meant to raise.

    the default watcher in _stub_sft_run asserts require_complete is True, which is right for the
    happy path but masks the assertion under test here: a guard that raises from on_line unwinds
    before return_code is assigned, so the run is legitimately incomplete.
    """

    def __init__(self, **kwargs):
        self.processed_steps = set()

    def start(self):
        return None

    def raise_if_failed(self):
        return None

    def stop(self, *, require_complete):
        return None


@pytest.mark.parametrize(
    "lines",
    [
        # verbatim shape of the g4 gsm8k lines (flash-1785592071-e56cf3c6): loss barely moves on a
        # replayed identical batch because nothing is learning.
        pytest.param(
            [
                "step:1 - train/loss:0.5470 - train/grad_norm:0.0 - train/lr:0.0001",
                "step:2 - train/loss:0.5437 - train/grad_norm:0.0 - train/lr:0.0001",
                "step:3 - train/loss:0.5474 - train/grad_norm:0.0 - train/lr:0.0001",
                "step:4 - train/loss:0.5444 - train/grad_norm:0.0 - train/lr:0.0001",
            ],
            id="every-step-zero",
        ),
        # VERL-138: the same defect on a 2-step run, where the schedule decays lr to 0.0 on the
        # final step. the lr is not why the gradient is zero -- verl measures grad_norm off p.grad
        # before the optimizer and the scheduler run -- so this must fail exactly like the above.
        pytest.param(
            [
                "step:1 - train/loss:0.5464 - train/grad_norm:0.0 - train/lr:5e-05",
                "step:2 - train/loss:0.5477 - train/grad_norm:0.0 - train/lr:0.0",
            ],
            id="lr-decays-to-zero",
        ),
    ],
)
def test_zero_grad_norm_fails_the_run(monkeypatch, lines):
    """GRAD-001: four runs reported done and charged while grad_norm was 0.0 on every step.

    the number was parsed and recorded, never read. driven through run_sft_train so the assertion
    lands on the shipped guard -- this test used to define its own copy of the guard body and
    assert against that, which meant it could not fail no matter what the worker did.
    """
    from flash.engine.worker import sft_train

    spec, _ = _stub_sft_run(monkeypatch, watcher_cls=_TolerantWatcher)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        for step, line in enumerate(lines, start=1):
            on_line(line + "\n")
            on_step(step)
        raise AssertionError("the zero-grad guard should have stopped the run before this")

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    with pytest.raises(RuntimeError, match=re.escape("grad_norm=0.0")):
        sft_train.run_sft_train(spec)


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param(
            [
                "step:1 - train/loss:0.9 - train/grad_norm:1.4 - train/lr:0.0001",
                "step:2 - train/loss:0.7 - train/grad_norm:0.9 - train/lr:0.0001",
            ],
            id="healthy",
        ),
        # an isolated zero is a legitimately fully-masked micro-batch, not a severed graph.
        pytest.param(
            [
                "step:1 - train/loss:0.9 - train/grad_norm:0.0 - train/lr:0.0001",
                "step:2 - train/loss:0.7 - train/grad_norm:1.1 - train/lr:0.0001",
                "step:3 - train/loss:0.6 - train/grad_norm:0.0 - train/lr:0.0001",
            ],
            id="isolated-zeros",
        ),
    ],
)
def test_healthy_grad_norms_do_not_trip_the_guard(monkeypatch, lines):
    """the guard must not fail a run that is training: any nonzero norm resets the count."""
    from flash.engine.worker import sft_train

    spec, _ = _stub_sft_run(monkeypatch)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        for step, line in enumerate(lines, start=1):
            on_line(line + "\n")
            on_step(step)
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    sft_train.run_sft_train(spec)


def test_step_gate_admits_a_line_a_tqdm_bar_was_flushed_in_front_of():
    """VERL-134: the guard above never armed on the real run because on_line returned early.

    on_line gates every metric read on ``verl_step_number(line)``, and verl's LocalLogger shares its
    stream with tqdm, whose bar ends in "]" with no trailing newline. anchoring the left edge on
    whitespace matched step 1 and missed steps 2-4, so ``flash-1785598982-21827245`` reported done
    with train/grad_norm 0.0 on every step while the guard's counter sat at 1.

    the guard test above replays its own loop, so it cannot see this: the defect is in the gate the
    real on_line runs first, not in the counting.
    """
    # verbatim from the run log, tqdm prefix included.
    glued = (
        "Epoch 1/1:  25%|##        | 1/4 [01:21<04:04, 81.49s/it]"
        "step:2 - train/loss:1.0206047296524048 - train/grad_norm:0.0 - train/lr:5e-05"
    )

    assert verl_step_number(glued) == 2, "on_line would return before ever reading grad_norm"
    # and the metrics behind the gate are the ones the guard needs.
    assert parse_verl_metric(glued, "train/grad_norm") == 0.0
    assert parse_verl_metric(glued, "train/lr") == 5e-05


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
    monkeypatch.setenv("PARASAIL_API_KEY", "teacher-secret")
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
        "PARASAIL_API_KEY",
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

    # both read the STAGED copy rather than verl's directory (see _staged_source); what matters is
    # that they agree on one source and that it carries the `actor/` level this layout puts there.
    assert os.path.basename(exported[0][0]) == "actor"
    assert exported[0][0] == os.path.join(uploaded[0][1], "actor")
    assert published[0][1] == 5
    assert published[0][2]["required"] is True
    assert [step for step, _ in uploaded] == [5]
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

    # the export reads a STAGED copy, not verl's own directory (see _staged_source), so assert the
    # layout this test is about -- global_step_N itself, no `actor/` level -- rather than the path.
    assert [os.path.basename(path) for path in exported] == ["global_step_5"]


def test_a_required_save_survives_verl_pruning_it_mid_publish(monkeypatch, tmp_path):
    """verl deleting global_step_N while it is being published must not fail the run.

    the watcher publishes on its own thread, so the merge and the multi-GB upload overlap training.
    under `trainer.max_ckpt_to_keep=1` verl's `register_checkpoint` rmtree's global_step_N from the
    CHILD process the moment N+1 finishes saving, and the parent has no way to pin it. with close
    `save_at_steps` (here [2, 3]) that lands squarely inside the publish window.

    this reproduces the interleave rather than asserting on the staging mechanism: the fake upload
    deletes verl's directory the way verl would, then reads the source it was handed. before staging,
    that read fails and takes the whole paid run down with a required-save error.
    """
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    local_dir = tmp_path / "checkpoints"
    checkpoint_dir = local_dir / "global_step_2"
    (checkpoint_dir / "actor" / "huggingface").mkdir(parents=True)
    (checkpoint_dir / "actor" / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(
        sft_train,
        "_export_checkpoint_adapter",
        lambda actor, adapter, **kwargs: os.makedirs(adapter, exist_ok=True),
    )
    monkeypatch.setattr(
        worker, "publish_deployable_checkpoint", lambda adapter, step, **kwargs: None
    )
    read_back = {}

    def fake_upload(step, checkpoint, **kwargs):
        kwargs["before_upload"]()
        # verl prunes global_step_2 because global_step_3 just landed. this is the whole race.
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        # the upload streams the folder AFTER that, so it must still have something to read.
        read_back["payload"] = (
            pathlib.Path(checkpoint, "actor", "model.safetensors").read_bytes()
            if os.path.isdir(checkpoint)
            else None
        )
        return True

    monkeypatch.setattr(worker, "upload_resume_checkpoint", fake_upload)
    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(local_dir),
        export_root=str(tmp_path / "exports"),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(2, 3),
    )

    watcher._publish(2, str(checkpoint_dir))

    assert read_back["payload"] == b"weights", (
        "the upload lost its source when verl pruned the checkpoint, so a required save would fail "
        "an otherwise successful paid run"
    )
    assert watcher.processed_steps == {2}
    # the staging links are transient: they must not outlive the publish and accumulate on the pod.
    assert not os.path.exists(os.path.join(str(tmp_path / "exports"), "_staging", "global_step_2"))


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


def _stub_sft_run(monkeypatch, *, save_at_steps=(), watcher_cls=None):
    """monkeypatch every out-of-process dependency of run_sft_train and return (spec, captured).

    the caller supplies its own ``run_verl_training`` fake, which is the only remaining seam.
    """
    import flash.catalog as catalog
    import flash.engine.vram as vram
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    monkeypatch.setattr(catalog, "resolve_vocab_size", lambda *_args, **_kwargs: 151936)

    spec = SimpleNamespace(
        model="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        algorithm="sft",
        seed=7,
        thinking=False,
        worker_env={},
        workload_profile_input_digest="",
        workload_profile_producer_version=_PROFILE_PRODUCER_VERSION,
        workload_profile={},
        environment=SimpleNamespace(
            id="owner/env",
            resolved_sha="b" * 40,
            params={},
        ),
        gpu=SimpleNamespace(type="RTX 4090", exact_type="", count=2),
        train=SimpleNamespace(
            epochs=1,
            learning_rate=5e-5,
            batch_size=2,
            max_context_tokens=1024,
            max_examples=2,
            max_steps=2,
            save_at_steps=save_at_steps,
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

    from flash.engine.sft_workload import prepare_sft_workload
    from flash.workload_profile import sft_profile_input_digest

    # deliberately NOT `flash.__version__`: the worker has no flash distribution installed and
    # resolves that to "0+unknown", so building both sides from it would make this fixture agree
    # with a worker that re-derives the version -- the defect it must instead be able to catch.
    spec.workload_profile_input_digest = sft_profile_input_digest(
        spec,
        tokenizer_revision=spec.model_revision,
        producer_version=_PROFILE_PRODUCER_VERSION,
    )
    spec.workload_profile = prepare_sft_workload(
        spec,
        Env(),
        tokenizer_loader=lambda _model, _revision: Tokenizer(),
        producer_version=_PROFILE_PRODUCER_VERSION,
        allow_packing=False,
        packing_support=lambda _model, _revision: ("unsupported", False),
    ).profile.to_dict()
    # the fixture pins the architecture above, but the WORKER re-derives it through the live probes,
    # which read the model config off the hub. pin them to the same answer so the parity check under
    # test compares workloads rather than network reachability -- an unresolvable probe now fails
    # closed instead of quietly labelling the model "unsupported".
    from flash.engine import sft_workload as _sft_workload

    monkeypatch.setattr(_sft_workload, "probe_is_pure_attention", lambda _m, revision="": False)
    monkeypatch.setattr(_sft_workload, "probe_is_gdn_hybrid", lambda _m, revision="": False)

    class LoraConfig:
        r = 16
        lora_alpha = 32
        target_modules = "all-linear"

    class PeakSampler:
        def start(self):
            return self

        def stop_gb(self):
            return 12.5

    class _DefaultWatcher:
        def __init__(self, **kwargs):
            self.processed_steps = set()

        def start(self):
            return None

        def stop(self, *, require_complete):
            assert require_complete is True

        def raise_if_failed(self):
            return None

    Watcher = watcher_cls or _DefaultWatcher

    captured = {"heartbeats": [], "published": [], "uploads": []}
    real_sft_grad_accum = vram.sft_grad_accum

    def capture_sft_grad_accum(batch_size, **kwargs):
        captured["sft_grad_accum"] = {"batch_size": batch_size, **kwargs}
        return real_sft_grad_accum(batch_size, **kwargs)

    def capture_grad_checkpointing(model_id, max_length, **kwargs):
        captured["grad_checkpointing"] = {
            "model_id": model_id,
            "max_length": max_length,
            **kwargs,
        }
        return True

    monkeypatch.setattr(vram, "sft_grad_accum", capture_sft_grad_accum)
    # run_sft_train imports AutoProcessor at data-loading time and transformers is not installed in
    # the cpu test env. this used to pass only because some EARLIER test module left a transformers
    # stub in sys.modules, so running this file alone failed -- stub it here so the test stands on
    # its own. monkeypatch.setitem restores whatever was there (real module or nothing) afterwards.
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        sys.modules.get("transformers")
        or _module(
            "transformers",
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: None),
            # datasets' dill serializer issubclass()-checks against this while writing the
            # parquet, so it has to be a real class rather than a namespace attribute.
            PreTrainedTokenizerBase=type("PreTrainedTokenizerBase", (), {}),
        ),
    )
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
    monkeypatch.setattr(worker, "grad_checkpointing_on", capture_grad_checkpointing)
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

    return spec, captured


def test_run_sft_train_orchestrates_exact_dataset_and_resume_accounting(monkeypatch):
    from flash.engine.worker import sft_train

    spec, captured = _stub_sft_run(monkeypatch)

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
    notes = captured["meta"]["notes"]
    realized_max_length = notes["realized_max_length"]
    assert 0 < realized_max_length < notes["configured_max_length"]
    assert notes["runtime_max_length"] == realized_max_length
    assert captured["sft_grad_accum"]["seq_len"] == realized_max_length
    assert captured["sft_grad_accum"]["fused"] is True
    assert captured["grad_checkpointing"]["max_length"] == realized_max_length
    assert captured["grad_checkpointing"]["fused_ce"] is True
    assert "data.max_length=1024" in captured["command"]
    assert (
        f"data.max_token_len_per_gpu={realized_max_length * notes['per_device_train_batch_size']}"
        in captured["command"]
    )


def test_a_workload_that_moved_under_the_frozen_quote_stops_before_training(monkeypatch):
    """The worker re-derives the workload and refuses to train on one the quote never priced.

    The environment is pinned by SHA, so this is not the ordinary case: it is the one where the
    pinned inputs still produce different rows (a non-deterministic dataset build, a tokenizer
    resolving differently). Training anyway would bill a run against a quote measured on other
    data, so the profile is evidence to check rather than metadata to carry.

    The drift has to come from the workload, not from the artifact. A profile carries its own
    content digest, so an edited one is rejected as corrupt before this guard is reached; only a
    re-derivation that legitimately disagrees can exercise it.
    """
    from flash.engine import sft_workload
    from flash.engine.worker import sft_train

    spec, _captured = _stub_sft_run(monkeypatch)
    honest = sft_workload.prepare_sft_workload

    def drifted(*args, **kwargs):
        prepared = honest(*args, **kwargs)
        moved = replace(
            prepared.profile, realized_max_length=prepared.profile.realized_max_length - 1
        )
        return replace(prepared, profile=moved)

    monkeypatch.setattr(sft_train, "prepare_sft_workload", drifted)

    def unreachable_training(command, *, env, on_step, on_line, heartbeat):
        raise AssertionError("training must not start on a workload the quote did not price")

    monkeypatch.setattr(sft_train, "run_verl_training", unreachable_training)

    with pytest.raises(ValueError, match="workload changed after the quote was frozen"):
        sft_train.run_sft_train(spec)


def test_a_guard_failure_is_not_replaced_by_the_watcher_completeness_error(monkeypatch):
    """the zero-grad diagnosis must survive the finally block, not be overwritten by it.

    an on_line guard raises from INSIDE run_verl_training, so return_code is never assigned and
    keeps its initial 0. deriving require_complete from it would then demand every save_at_steps
    entry from a run that died at step 2, and the watcher's "required saves were not durably
    published" would unwind out of the finally in place of the real cause -- turning the one
    error GRAD-001 exists to surface into a checkpointing red herring.
    """
    from flash.engine.worker import sft_train

    stopped: list[bool] = []

    class Watcher:
        def __init__(self, **kwargs):
            self.processed_steps = set()
            self.required_steps = frozenset(kwargs.get("required_steps", ()))

        def start(self):
            return None

        def raise_if_failed(self):
            return None

        def stop(self, *, require_complete):
            stopped.append(require_complete)
            if require_complete:
                missing = sorted(self.required_steps - self.processed_steps)
                if missing:
                    raise RuntimeError(f"required saves were not durably published: {missing}")

    # a required save the run never durably publishes: the guard raises on the same step, before
    # the watcher has processed it. (the step is inside the 2-update horizon because
    # validate_save_steps rejects anything beyond it at config time.)
    spec, _ = _stub_sft_run(monkeypatch, save_at_steps=(2,), watcher_cls=Watcher)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        for step in range(1, _MAX_ZERO_GRAD_STEPS + 1):
            on_line(
                f"step:{step} - train/loss:1.0 - train/grad_norm:0.0 - train/lr:5e-05 "
                "- train/global_tokens:8\n"
            )
        raise AssertionError("the zero-grad guard should have stopped the run before this")

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    with pytest.raises(RuntimeError, match=re.escape("grad_norm=0.0")):
        sft_train.run_sft_train(spec)

    # the watcher still gets stopped -- it just is not asked to prove completeness for a run that
    # never finished. without this, stop() raises and the grad_norm error never reaches the caller.
    assert stopped == [False]


def test_zero_grad_guard_survives_an_lr_that_decays_to_zero(monkeypatch):
    """VERL-138: a decayed lr must not launder a run that trained nothing.

    replays the real 2-step shape of flash-1785606382-389d4630, which reported done and billed with
    grad_norm 0.0 on every step. the scheduler puts lr at 0.0 on the final step, so a guard that
    treats an lr of 0.0 as an excuse for a zero gradient never fires on the second step and the run
    bills for an adapter that learned nothing.

    the lr cannot cause this: verl computes grad_norm in optimizer_step (transformer_impl.py:683)
    by clipping over p.grad, before optimizer.step() and before lr_scheduler_step() advances the
    schedule. driven through run_sft_train rather than a local copy of the guard, so the assertion
    is about the shipped code and not about the test's own reimplementation of it.
    """
    from flash.engine.worker import sft_train

    spec, _ = _stub_sft_run(monkeypatch, watcher_cls=_TolerantWatcher)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        on_line(
            "step:1 - train/loss:0.5464 - train/grad_norm:0.0 - train/lr:5e-05 "
            "- train/global_tokens:6588\n"
        )
        on_line(
            "step:2 - train/loss:0.5477 - train/grad_norm:0.0 - train/lr:0.0 "
            "- train/global_tokens:6274\n"
        )
        raise AssertionError("the zero-grad guard should have stopped the run before this")

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    with pytest.raises(RuntimeError, match=re.escape("grad_norm=0.0")):
        sft_train.run_sft_train(spec)


def test_zero_grad_guard_clears_on_a_recovered_step(monkeypatch):
    """the guard must count consecutive steps, not keep a run-lifetime tally.

    a nonzero grad norm is proof the graph is intact, so evidence collected before it is stale and
    must be discarded. without this, one isolated zero-grad step early plus another much later
    would fail a run that is training normally in between.
    """
    from flash.engine.worker import sft_train

    spec, captured = _stub_sft_run(monkeypatch)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        # zero, healthy (clears), zero, healthy (clears). two zero-grad steps in total but never
        # two in a row, so the run must survive.
        for step, grad in enumerate([0.0, 1.4, 0.0, 0.9], start=1):
            on_line(
                f"step:{step} - train/loss:1.0 - train/grad_norm:{grad} - train/lr:5e-05 "
                "- train/global_tokens:8\n"
            )
            on_step(step)
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    sft_train.run_sft_train(spec)
    assert captured["meta"]["notes"]["loss_curve"] == [1.0, 1.0, 1.0, 1.0]


def test_a_single_step_run_with_no_gradient_is_rejected(monkeypatch):
    """Regression (codex[bot], sft_train.py): the consecutive-run guard needs _MAX_ZERO_GRAD_STEPS
    steps to fire, so a horizon SHORTER than that could not trip it at all.

    a one-update horizon is ordinary (the retained rows fit a single batch). such a run appended its
    single grad_norm 0.0, never reached the threshold, and then published and billed an adapter
    identical to the base weights -- exactly the GRAD-001 outcome the guard exists to prevent, on the
    one horizon it could not see. driven through run_sft_train so the assertion is about the shipped
    code path, not a local reimplementation of the check.
    """
    from flash.engine.worker import sft_train

    spec, _ = _stub_sft_run(monkeypatch, watcher_cls=_TolerantWatcher)
    # a fresh run, not a resume: the guard abstains on a resume because the restored weights carry
    # earlier updates this session never observed.
    monkeypatch.setattr(sft_train, "_restore_verl_resume", lambda local_dir: 0)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        # both updates report a dead gradient. the in-loop guard would also fire on two consecutive
        # zeros, so the horizon is cut to one below to isolate the end-of-run check.
        on_line(
            "step:1 - train/loss:0.5464 - train/grad_norm:0.0 - train/lr:5e-05 "
            "- train/global_tokens:6588\n"
        )
        on_step(1)
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    assert _MAX_ZERO_GRAD_STEPS > 1, "this regression only exists while the in-loop guard needs >1"
    with pytest.raises(RuntimeError, match=re.escape("grad_norm=0.0")):
        sft_train.run_sft_train(spec)


@pytest.mark.parametrize("grads", [[1.4], [0.0, 1.4], [1.4, 0.0]])
def test_a_fresh_run_with_any_real_gradient_still_completes(monkeypatch, grads):
    """The end-of-run check must reject only an ALL-zero session, never one that trained.

    pairs with the test above, and every case here is a FRESH run, so the guard is actually reached
    -- on a resume it abstains and the test could not fail however the check is spelled. `[1.4]`
    covers the short healthy horizon; the mixed cases pin the boundary, because the obvious
    over-broad spelling (`not all`, i.e. reject on any zero at all) fails a run that demonstrably
    trained. an isolated zero inside a longer run stays tolerated by contract.
    """
    from flash.engine.worker import sft_train

    spec, captured = _stub_sft_run(monkeypatch)
    monkeypatch.setattr(sft_train, "_restore_verl_resume", lambda local_dir: 0)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        for step, grad in enumerate(grads, start=1):
            on_line(
                f"step:{step} - train/loss:1.0 - train/grad_norm:{grad} - train/lr:5e-05 "
                "- train/global_tokens:8\n"
            )
            on_step(step)
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    sft_train.run_sft_train(spec)
    assert captured["meta"]["notes"]["loss_curve"] == [1.0] * len(grads)


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


def test_sft_never_enables_liger_because_it_zeroes_the_lora_gradient():
    """liger is off on the sft path, and the emitted verl override must say so.

    GRAD-001: a matched two-arm a/b on Qwen3.5-9B (identical data, seed, hardware and code,
    differing only in `model.use_liger`) measured train/grad_norm 0.0 with liger on and 7.02
    with liger off, at a loss identical to four decimal places. liger silently severed the
    gradient to the lora params under this fsdp2 + peft + gradient-checkpointing composition,
    so sft trained nothing while looking healthy. the grpo path never sets the key (verl
    defaults it false), which is why only sft was affected.

    this asserts the RENDERED override rather than the config literal, so re-enabling liger
    anywhere between the dict and the command line fails the test.
    """
    from flash.engine.worker.sft_train import build_sft_overrides

    base = {
        "train_files": "/w/train.parquet",
        "val_files": "/w/val.parquet",
        "train_batch_size": 8,
        "max_length": 1024,
        "micro_batch": 1,
        "max_token_len_per_gpu": 1024,
        "custom_dataset_path": "/w/ds.py",
        "model_path": "Qwen/Qwen3.5-9B",
        "lora_rank": 32,
        "lora_alpha": 64,
        "target_modules": "all-linear",
        "lora_adapter_path": None,
        "ulysses_sp_size": 1,
        "lr": 1e-4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "optimizer_impl": "bitsandbytes.optim",
        "optimizer_name": "PagedAdamW8bit",
        "optimizer_kwargs": None,
        "local_dir": "/w/ckpt",
        "save_freq": 50,
        "n_gpus_per_node": 1,
        "seed": 42,
        "project_name": "p",
        "experiment_name": "e",
        "loop_epochs": 1,
        "gradient_checkpointing": True,
        "total_training_steps": None,
        "total_epochs": 1,
        "loggers": ["console"],
    }

    # the default must be off: a caller that omits the key must not get liger.
    assert "model.use_liger=false" in build_sft_overrides(dict(base))
    assert "model.use_liger=true" not in build_sft_overrides(dict(base))

    # the dense-logit-free loss comes from fused kernels, not liger, so it survives.
    overrides = build_sft_overrides(dict(base))
    assert "model.use_fused_kernels=true" in overrides
    assert "model.fused_kernel_options.impl_backend=torch" in overrides


def test_drain_join_waits_out_a_slow_upload_until_the_run_deadline(monkeypatch):
    """VERL-131: a checkpoint drain is bounded by the RUN's wall deadline, not a constant.

    The measured failure was a 35B-A3B full-state upload that needed 607.6s against a fixed 600s
    join. It was healthy and still uploading -- it emitted another `checkpoint_uploading` heartbeat
    9s AFTER the join gave up -- and the timeout converted a run that had already trained and
    published into `failed`.

    The bound deliberately does NOT try to sample upload progress. `_HB_LAST_PROGRESS_TS` looks like
    a progress signal but is stamped unconditionally every 30s by the upload's own
    `liveness_heartbeat(keepalive=True)` daemon (heartbeat.py: `liveness=... and not keepalive`),
    so it advances whether or not bytes move -- a no-progress window keyed to it could never fire.
    The upload is already bounded from the inside by its retry budget and per-attempt deadline
    checks, so the only correct job here is to not impose a second, tighter deadline on top.
    """
    import threading
    import time as real_time

    # NB: the worker package rebinds the name `heartbeat` to the re-exported heartbeat
    # FUNCTION, so `import ... as hb` yields that function rather than this module.
    # import_module returns the real module object.
    hb = importlib.import_module("flash.engine.worker.heartbeat")
    from flash.engine.worker._pkg import W as _w

    # virtual clock: the test must not actually take an hour.
    now = [0.0]
    monkeypatch.setattr(hb.time, "monotonic", lambda: now[0])

    class _SlowUpload:
        """alive for 3600 virtual seconds -- six times the old fixed deadline."""

        def __init__(self) -> None:
            self.elapsed = 0.0

        def is_alive(self) -> bool:
            return self.elapsed < 3600.0

        def join(self, timeout=None) -> None:
            step = timeout or 5.0
            self.elapsed += step
            now[0] += step

    # the run still has budget left, so the drain must be allowed to finish. this raised under the
    # old fixed 600s join, which is exactly the reported failure.
    monkeypatch.setattr(_w, "_remaining_worker_wall_seconds", lambda: 7200.0, raising=False)
    hb.join_while_draining(_SlowUpload(), "slow uploader")

    # and the converse: once the RUN is out of time the drain must be cut off, or a wedged upload
    # holds the worker open past its own deadline.
    now[0] = 0.0
    budget = [120.0]

    class _Wedged:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout=None) -> None:
            step = timeout or 5.0
            now[0] += step
            budget[0] -= step

    monkeypatch.setattr(_w, "_remaining_worker_wall_seconds", lambda: budget[0], raising=False)
    with pytest.raises(RuntimeError, match="wall deadline expired"):
        hb.join_while_draining(_Wedged(), "wedged uploader")

    # a real finished thread returns immediately rather than waiting out a window.
    monkeypatch.setattr(_w, "_remaining_worker_wall_seconds", lambda: 7200.0, raising=False)
    done = threading.Thread(target=lambda: None)
    done.start()
    done.join()
    started = real_time.monotonic()
    hb.join_while_draining(done, "finished uploader")
    assert real_time.monotonic() - started < 5.0
