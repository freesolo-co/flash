from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from flash.engine.worker import sft as sft_mod
from flash.engine.worker.sft_openrlhf import (
    _resolve_immutable_model_revision,
    _serialize_multimodal_inputs,
    _training_batch_shape,
    _validate_gdn_realized_length,
    build_openrlhf_sft_child_env,
    build_sft_openrlhf_args,
    build_text_openrlhf_rows,
    filter_openrlhf_sft_rows,
    render_openrlhf_sft_runtime,
    validate_openrlhf_warmstart_adapter,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed in offline CI",
)


class _CharTokenizer:
    eos_token = "<eos>"
    eos_token_id = 999
    pad_token_id = 0
    all_special_ids = (0, 999)

    def __call__(self, texts, *, truncation=False, max_length=None, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            ids = [ord(char) + 1 for char in text]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            rows.append(ids)
        return {"input_ids": rows}


class _FixtureTokenizer:
    eos_token = "<eos>"
    eos_token_id = 999
    pad_token_id = 0
    all_special_ids = (0, 999)

    def __init__(self):
        self._rows = {
            "single-prompt": (11, 12),
            "single-full<eos>": (11, 12, 21, 22, 999),
            "multi-prompt": (31, 32),
            "multi-full<eos>": (31, 32, 41, 42, 51, 52, 999),
            "thinking-prompt": (61, 62, 63),
            "thinking-full<eos>": (61, 62, 71, 72, 999),
        }

    def __call__(self, texts, *, truncation=False, max_length=None, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = [list(self._rows[text]) for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        return {"input_ids": rows}


def _arg_config(tmp_path):
    return {
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "dataset_path": str(tmp_path / "dataset"),
        "epochs": 3,
        "gradient_checkpointing": True,
        "gradient_checkpointing_reentrant": True,
        "learning_rate": 2e-5,
        "lora_alpha": 64,
        "lora_rank": 32,
        "max_length": 4096,
        "max_num_checkpoints": 5,
        "micro_batch_size": 2,
        "model_path": str(tmp_path / "model"),
        "output_dir": str(tmp_path / "output"),
        "resume_enabled": False,
        "row_count": 17,
        "seed": 123,
        "train_batch_size": 8,
        "wandb_enabled": False,
    }


def test_run_sft_dispatches_to_openrlhf_without_entering_trl(monkeypatch):
    calls = []
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    monkeypatch.setenv("FLASH_SFT_BACKEND", "openrlhf")
    monkeypatch.setattr(openrlhf_mod, "run_sft_openrlhf", lambda: calls.append("openrlhf"))

    sft_mod.run_sft()

    assert calls == ["openrlhf"]


def test_run_sft_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("FLASH_SFT_BACKEND", "megatron")

    with pytest.raises(ValueError, match="not a known sft backend"):
        sft_mod.run_sft()


def test_build_sft_openrlhf_args_maps_zero3_lora_gc_and_dataset(tmp_path):
    args = build_sft_openrlhf_args(_arg_config(tmp_path))

    assert args[args.index("--model.model_name_or_path") + 1] == str(tmp_path / "model")
    assert args[args.index("--data.dataset") + 1] == str(tmp_path / "dataset")
    assert args[args.index("--ds.zero_stage") + 1] == "3"
    assert args[args.index("--ds.param_dtype") + 1] == "bf16"
    assert args[args.index("--ds.lora.rank") + 1] == "32"
    assert args[args.index("--ds.lora.alpha") + 1] == "64"
    assert "--model.gradient_checkpointing_enable" in args
    assert "--model.gradient_checkpointing_reentrant" in args
    assert "--ds.lora.target_modules" not in args
    assert args[args.index("--lr_scheduler") + 1] == "linear"
    assert "--ckpt.save_hf" in args
    assert "--ckpt.load_enable" not in args


def test_build_sft_openrlhf_args_enables_full_state_resume(tmp_path):
    config = _arg_config(tmp_path)
    config["resume_enabled"] = True

    args = build_sft_openrlhf_args(config)

    assert "--ckpt.load_enable" in args
    assert args[args.index("--ckpt.path") + 1] == str(tmp_path / "checkpoints")


def test_build_sft_openrlhf_args_requires_positive_lora_rank(tmp_path):
    config = _arg_config(tmp_path)
    config["lora_rank"] = 0

    with pytest.raises(ValueError, match="positive LoRA rank"):
        build_sft_openrlhf_args(config)


@pytest.mark.parametrize(
    ("prompt", "full", "expected_ids", "expected_mask"),
    [
        ("single-prompt", "single-full", [11, 12, 21, 22, 999], [0, 0, 1, 1, 1]),
        (
            "multi-prompt",
            "multi-full",
            [31, 32, 41, 42, 51, 52, 999],
            [0, 0, 1, 1, 1, 1, 1],
        ),
        ("thinking-prompt", "thinking-full", [61, 62, 71, 72, 999], [0, 0, 1, 1, 1]),
    ],
)
def test_exact_mask_rows_use_literal_single_multiturn_and_thinking_fixtures(
    prompt,
    full,
    expected_ids,
    expected_mask,
):
    rows, dropped = build_text_openrlhf_rows(
        [{"prompt_text": prompt, "text": full}],
        _FixtureTokenizer(),
        4096,
    )

    assert dropped == 0
    assert rows == [
        {
            "input_ids": expected_ids,
            "loss_mask": expected_mask,
            "multimodal_inputs": b"",
        }
    ]


@requires_torch
def test_shifted_loss_mask_selects_literal_predicted_completion_tokens():
    import torch

    input_ids = torch.tensor([[11, 12, 21, 22, 999]])
    loss_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)

    predicted_token_ids = input_ids[:, 1:][loss_mask[:, 1:]]

    assert predicted_token_ids.tolist() == [21, 22, 999]


def test_filter_openrlhf_sft_rows_drops_special_only_target_and_fails_if_empty():
    rows = [
        {"input_ids": [10, 99], "loss_mask": [0, 1], "multimodal_inputs": b""},
        {"input_ids": [10, 11], "loss_mask": [0, 1], "multimodal_inputs": b""},
    ]

    kept, dropped = filter_openrlhf_sft_rows(rows, {99})

    assert kept == [rows[1]]
    assert dropped == 1
    with pytest.raises(ValueError, match="every SFT example"):
        filter_openrlhf_sft_rows(rows[:1], {99})


def test_validate_openrlhf_warmstart_adapter_checks_rank_model_and_revision(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 32,
                "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                "revision": "a" * 40,
            }
        ),
        encoding="utf-8",
    )

    validate_openrlhf_warmstart_adapter(
        str(adapter),
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        expected_rank=32,
    )

    with pytest.raises(ValueError, match="immutable target model revision"):
        validate_openrlhf_warmstart_adapter(
            str(adapter),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="",
            expected_rank=32,
        )
    with pytest.raises(ValueError, match="rank"):
        validate_openrlhf_warmstart_adapter(
            str(adapter),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="a" * 40,
            expected_rank=16,
        )
    with pytest.raises(ValueError, match="revision"):
        validate_openrlhf_warmstart_adapter(
            str(adapter),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="b" * 40,
            expected_rank=32,
        )
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    config["revision"] = None
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        validate_openrlhf_warmstart_adapter(
            str(adapter),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="a" * 40,
            expected_rank=32,
        )


def test_validate_openrlhf_warmstart_adapter_accepts_matching_snapshot_path(tmp_path):
    revision = "c" * 40
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.bin").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "base_model_name_or_path": str(
                    tmp_path / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / revision
                ),
                "revision": None,
            }
        ),
        encoding="utf-8",
    )

    validate_openrlhf_warmstart_adapter(
        str(adapter),
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision=revision,
        expected_rank=16,
    )


def test_resolve_immutable_model_revision_uses_prefetched_snapshot(monkeypatch):
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    commit = "d" * 40
    monkeypatch.setattr(
        openrlhf_mod,
        "resolve_cached_model_commit",
        lambda model_id, revision: commit,
    )

    assert _resolve_immutable_model_revision("org/model", "") == commit


def test_resolve_immutable_model_revision_fails_before_training_when_cache_is_unresolved(
    monkeypatch,
):
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    monkeypatch.setattr(
        openrlhf_mod,
        "resolve_cached_model_commit",
        lambda model_id, revision: "",
    )

    with pytest.raises(RuntimeError, match="immutable commit"):
        _resolve_immutable_model_revision("org/model", "")


def test_gdn_32k_gate_uses_realized_rows_and_allows_short_rows(monkeypatch):
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    monkeypatch.setattr(openrlhf_mod, "model_is_gdn_hybrid", lambda *args, **kwargs: True)

    _validate_gdn_realized_length("org/gdn", "e" * 40, 2048)
    with pytest.raises(ValueError, match="matched real-GPU validation"):
        _validate_gdn_realized_length("org/gdn", "e" * 40, 32768)


def test_openrlhf_child_env_excludes_training_and_provider_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("FIREWORKS_API_KEY", "teacher-secret")
    monkeypatch.setenv("RUNPOD_API_KEY", "provider-secret")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    child = build_openrlhf_sft_child_env(shim_dir=str(tmp_path), wandb_enabled=False)

    assert child["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "HF_TOKEN" not in child
    assert "FIREWORKS_API_KEY" not in child
    assert "RUNPOD_API_KEY" not in child
    assert "FREESOLO_INTERNAL_KEY" not in child
    assert "WANDB_API_KEY" not in child
    assert child["FLASH_OPENRLHF_SFT_CONFIG"] == str(tmp_path / "flash_sft_runtime.json")


def test_openrlhf_child_env_forwards_only_wandb_secrets_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")

    child = build_openrlhf_sft_child_env(shim_dir=str(tmp_path), wandb_enabled=True)

    assert child["WANDB_API_KEY"] == "wandb-secret"
    assert child["WANDB_MODE"] == "offline"
    assert "HF_TOKEN" not in child


def test_openrlhf_runtime_installs_fail_loud_loraplus_and_warmstart_checks():
    source = render_openrlhf_sft_runtime()

    compile(source, "flash_openrlhf_sft_runtime.py", "exec")
    assert "create_loraplus_optimizer" in source
    assert "PagedAdamW8bit" in source
    assert "zero_allow_untested_optimizer" in source
    assert "assert_adapter_load_clean" in source
    assert "assert_adapter_delta_nonzero" in source
    assert "FLASH_OPENRLHF_LORAPLUS_READY" in source
    assert "loss_mask[:, 1:]" in source
    assert "loss_mask[:, :-1]" not in source
    assert "falling back" not in source.lower()


def test_restored_client_state_reaches_runtime_trainer_state(monkeypatch, tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps({"resume_step": 7}), encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    namespace = {"__name__": "flash_openrlhf_sft_resume_test"}
    exec(compile(render_openrlhf_sft_runtime(), "runtime.py", "exec"), namespace)

    states = {
        "consumed_samples": 31,
        "global_step": 7,
        "loss_curve": [2.5, 1.75],
        "token_count": 4096,
    }

    class FakeDeepspeedStrategy:
        def setup_dataloader(self, *args, **kwargs):
            return args, kwargs

        def prepare(self, *args):
            return args

        def load_ckpt(self, *args, **kwargs):
            return "/checkpoints/global_step7", states

    openrlhf = ModuleType("openrlhf")
    openrlhf.__path__ = []
    utils = ModuleType("openrlhf.utils")
    utils.__path__ = []
    deepspeed_package = ModuleType("openrlhf.utils.deepspeed")
    deepspeed_package.__path__ = []
    deepspeed_module = ModuleType("openrlhf.utils.deepspeed.deepspeed")
    deepspeed_module.DeepspeedStrategy = FakeDeepspeedStrategy
    monkeypatch.setitem(sys.modules, "openrlhf", openrlhf)
    monkeypatch.setitem(sys.modules, "openrlhf.utils", utils)
    monkeypatch.setitem(sys.modules, "openrlhf.utils.deepspeed", deepspeed_package)
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.utils.deepspeed.deepspeed",
        deepspeed_module,
    )

    namespace["_install_dataloader_and_scheduler_patches"]()
    load_path, loaded_states = FakeDeepspeedStrategy().load_ckpt(object(), "/checkpoints")
    trainer_state = namespace["_resume_training_state"](loaded_states["consumed_samples"])

    assert load_path == "/checkpoints/global_step7"
    assert namespace["CONFIG"]["_resume_states"] == states
    assert trainer_state == (7, 31, [2.5, 1.75], 4096)


@requires_torch
def test_torchrun_sitecustomize_patches_dataset_before_train_sft_import(tmp_path):
    shim_dir = tmp_path / "shim"
    package_dir = tmp_path / "packages"
    shim_dir.mkdir()
    (package_dir / "openrlhf" / "cli").mkdir(parents=True)
    (package_dir / "openrlhf" / "datasets").mkdir(parents=True)
    (package_dir / "openrlhf" / "models").mkdir(parents=True)
    (package_dir / "openrlhf" / "trainer").mkdir(parents=True)
    (package_dir / "openrlhf" / "utils" / "deepspeed").mkdir(parents=True)
    (package_dir / "peft").mkdir(parents=True)

    config_path = shim_dir / "flash_sft_runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    (shim_dir / "flash_openrlhf_sft_runtime.py").write_text(
        render_openrlhf_sft_runtime(),
        encoding="utf-8",
    )
    (shim_dir / "sitecustomize.py").write_text(
        "from flash_openrlhf_sft_runtime import apply_flash_openrlhf_sft_patches\n"
        "apply_flash_openrlhf_sft_patches()\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "openrlhf" / "cli" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "openrlhf" / "datasets" / "__init__.py").write_text(
        "class SFTDataset:\n    pass\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "cli" / "train_sft.py").write_text(
        "from openrlhf.datasets import SFTDataset\n"
        "assert SFTDataset.__name__ == 'FlashTokenizedSFTDataset'\n"
        "print('FLASH_DATASET_PATCHED_BEFORE_TRAIN_SFT', flush=True)\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "models" / "__init__.py").write_text(
        "class SFTLoss:\n    pass\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "trainer" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "openrlhf" / "trainer" / "sft_trainer.py").write_text(
        "class SFTTrainer:\n    pass\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "openrlhf" / "utils" / "deepspeed" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "utils" / "deepspeed" / "deepspeed.py").write_text(
        "class DeepspeedStrategy:\n"
        "    def setup_dataloader(self, *args, **kwargs):\n        pass\n"
        "    def prepare(self, *args):\n        pass\n"
        "    def load_ckpt(self, *args, **kwargs):\n        return None, {}\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "utils" / "distributed_sampler.py").write_text(
        "class DistributedSampler:\n    pass\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "utils" / "loss_utils.py").write_text(
        "def _optimizer_step_loss_norm(*args, **kwargs):\n    return {}\n",
        encoding="utf-8",
    )
    (package_dir / "deepspeed.py").write_text(
        "def initialize(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (package_dir / "peft" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "peft" / "optimizers.py").write_text(
        "def create_loraplus_optimizer(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["FLASH_OPENRLHF_SFT_CONFIG"] = str(config_path)
    project_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(shim_dir), str(package_dir), str(project_root)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=1",
            "-m",
            "openrlhf.cli.train_sft",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FLASH_DATASET_PATCHED_BEFORE_TRAIN_SFT" in result.stdout


def test_training_batch_shape_respects_gpu_and_device_limits():
    micro, accumulation, global_batch = _training_batch_shape(
        row_count=100,
        effective_batch=32,
        per_device_limit=4,
        gpu_count=2,
    )

    assert (micro, accumulation, global_batch) == (4, 4, 32)


@requires_torch
def test_rendered_dataset_returns_exact_input_ids_and_loss_mask(monkeypatch, tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    namespace = {"__name__": "flash_openrlhf_sft_runtime_test"}
    exec(compile(render_openrlhf_sft_runtime(), "runtime.py", "exec"), namespace)
    dataset_class = namespace["FlashTokenizedSFTDataset"]
    row = {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1], "multimodal_inputs": b""}
    tokenizer = SimpleNamespace(pad_token_id=0)
    dataset = dataset_class([row], tokenizer, 8, strategy=object())

    input_ids, attention, loss_mask, mm_inputs = dataset[0]

    assert input_ids.tolist() == [[1, 2, 3, 4]]
    assert attention.tolist() == [[1, 1, 1, 1]]
    assert loss_mask.tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert mm_inputs == {}


@requires_torch
def test_rendered_dataset_collates_multimodal_tensors(tmp_path, monkeypatch):
    import numpy as np

    config_path = tmp_path / "runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    namespace = {"__name__": "flash_openrlhf_sft_multimodal_test"}
    exec(compile(render_openrlhf_sft_runtime(), "runtime.py", "exec"), namespace)
    dataset_class = namespace["FlashTokenizedSFTDataset"]
    rows = [
        {
            "input_ids": [1, 2, 3],
            "loss_mask": [0, 1, 1],
            "multimodal_inputs": _serialize_multimodal_inputs(
                {
                    "pixel_values": np.array([[1.0, 2.0]], dtype=np.float32),
                    "image_grid_thw": np.array([[1, 2, 3]], dtype=np.int64),
                }
            ),
        },
        {
            "input_ids": [4, 5],
            "loss_mask": [0, 1],
            "multimodal_inputs": _serialize_multimodal_inputs(
                {
                    "pixel_values": np.array([[3.0, 4.0]], dtype=np.float32),
                    "image_grid_thw": np.array([[4, 5, 6]], dtype=np.int64),
                }
            ),
        },
    ]
    tokenizer = SimpleNamespace(pad_token_id=0)
    dataset = dataset_class(rows, tokenizer, 8, strategy=object())

    inputs, attention, loss_mask, mm_inputs = dataset.collate_fn([dataset[0], dataset[1]])

    assert inputs.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert attention.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert loss_mask.tolist() == [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
    assert mm_inputs["pixel_values"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert mm_inputs["image_grid_thw"].tolist() == [[1, 2, 3], [4, 5, 6]]
