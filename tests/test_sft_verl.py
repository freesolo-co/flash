"""CPU unit tests for the sft->verl override builder."""

from __future__ import annotations

import pytest

from flash.engine.worker.sft_verl import (
    build_sft_verl_messages_rows,
    build_sft_verl_overrides,
)


def _cfg(**over):
    base = {
        "train_files": "/w/train.parquet",
        "val_files": "/w/val.parquet",
        "max_length": 32768,
        "micro_batch": 1,
        "max_token_len_per_gpu": 8192,
        "model_path": "Qwen/Qwen3-4B",
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "ulysses_sp_size": 2,
        "lr": 1e-4,
        "local_dir": "/w/ckpt",
        "save_freq": 50,
        "n_gpus_per_node": 2,
        "project_name": "flash-sft",
        "experiment_name": "run-xyz",
        "total_training_steps": 120,
    }
    base.update(over)
    return base


def _as_map(ov):
    return dict(s.split("=", 1) for s in ov)


def test_overrides_cover_the_32k_lora_sp_surface():
    m = _as_map(build_sft_verl_overrides(_cfg()))
    # long context + packing + sequence parallel + liger are the 32k-enabling knobs.
    assert m["data.max_length"] == "32768"
    assert m["model.use_remove_padding"] == "true"
    assert m["engine.ulysses_sequence_parallel_size"] == "2"
    assert m["model.use_liger"] == "true"
    assert m["engine.strategy"] == "fsdp2"
    # sequence-parallel degree and gpu count both come from gpu.count.
    assert m["trainer.n_gpus_per_node"] == "2"
    assert m["trainer.nnodes"] == "1"
    # messages-based dataset (verl computes the role mask); no prompt/response keys.
    assert m["data.messages_key"] == "messages"
    assert m["data.use_dynamic_bsz"] == "true"
    # lora on the immutable base; path key is model.path (not partial_pretrain).
    assert m["model.path"] == "Qwen/Qwen3-4B"
    assert m["model.lora_rank"] == "16"
    # lr renders as a plain decimal hydra parses as a float (1e-4 -> "0.0001", not scientific).
    assert m["optim.lr"] == "0.0001"
    assert "data.train_files=/w/train.parquet" in build_sft_verl_overrides(_cfg())


def test_small_lr_renders_fixed_point_not_scientific():
    # 5e-5 would str() as "5e-05"; hydra should get plain decimal.
    m = _as_map(build_sft_verl_overrides(_cfg(lr=5e-5)))
    assert m["optim.lr"] == "0.00005"


def test_target_modules_list_renders_as_hydra_list():
    m = _as_map(build_sft_verl_overrides(_cfg(target_modules=["q_proj", "v_proj"])))
    assert m["model.target_modules"] == "[q_proj,v_proj]"


def test_epochs_path_when_no_steps():
    m = _as_map(build_sft_verl_overrides(_cfg(total_training_steps=None, total_epochs=3)))
    assert m["trainer.total_epochs"] == "3"
    assert "trainer.total_training_steps" not in m


def test_steps_xor_epochs_is_enforced():
    with pytest.raises(ValueError, match="exactly one"):
        build_sft_verl_overrides(_cfg(total_training_steps=120, total_epochs=3))
    with pytest.raises(ValueError, match="exactly one"):
        build_sft_verl_overrides(_cfg(total_training_steps=None, total_epochs=None))


def test_missing_required_key_raises():
    bad = _cfg()
    del bad["model_path"]
    with pytest.raises(KeyError, match="model_path"):
        build_sft_verl_overrides(bad)


def test_messages_rows_concatenate_prompt_and_completion():
    prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    completion = [{"role": "assistant", "content": "a"}]
    rows = build_sft_verl_messages_rows([(prompt, completion)])
    assert rows == [{"messages": [*prompt, *completion]}]


def test_messages_rows_drop_empty_completion():
    prompt = [{"role": "user", "content": "u"}]
    rows = build_sft_verl_messages_rows(
        [
            (prompt, [{"role": "assistant", "content": "a"}]),
            (prompt, []),
        ]
    )
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["role"] == "assistant"
