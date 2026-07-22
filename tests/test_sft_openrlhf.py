from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import pytest

from flash.engine.worker import sft as sft_mod
from flash.engine.worker.packing import completion_mask_from_ids
from flash.engine.worker.sft_openrlhf import (
    _training_batch_shape,
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


def test_build_sft_openrlhf_args_requires_positive_lora_rank(tmp_path):
    config = _arg_config(tmp_path)
    config["lora_rank"] = 0

    with pytest.raises(ValueError, match="positive LoRA rank"):
        build_sft_openrlhf_args(config)


def test_exact_mask_rows_match_flash_for_multiturn_thinking_case():
    tokenizer = _CharTokenizer()
    prompt = "<system>helpful</system><user>question</user><assistant><think>\n"
    full = (
        "<system>helpful</system><user>question</user>"
        "<assistant><think>\nfirst thought</think>first answer</assistant>"
        "<user>followup</user><assistant><think>\nsecond thought</think>second answer</assistant>"
    )
    rows, dropped = build_text_openrlhf_rows(
        [{"prompt_text": prompt, "text": full}],
        tokenizer,
        4096,
    )
    full_ids = tokenizer([full + tokenizer.eos_token], truncation=True, max_length=4096)[
        "input_ids"
    ][0]
    prompt_ids = tokenizer([prompt], truncation=True, max_length=4096)["input_ids"][0]

    assert dropped == 0
    assert rows[0]["input_ids"] == full_ids
    assert rows[0]["loss_mask"] == completion_mask_from_ids(prompt_ids, full_ids)
    assert rows[0]["loss_mask"][: len(prompt_ids) - 1] == [0] * (len(prompt_ids) - 1)
    assert any(rows[0]["loss_mask"])


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
                    tmp_path
                    / "models--Qwen--Qwen3.5-0.8B"
                    / "snapshots"
                    / revision
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
