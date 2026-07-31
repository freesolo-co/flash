"""verl grpo backend: dispatch, data/config/reward glue, and reward parity (cpu-only, no verl)."""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import shutil
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import flash.engine.worker as W
from flash.engine.worker import rl, rl_verl, verl_common


# ------------------------------- dispatch -------------------------------
def test_run_rl_delegates_to_verl_backend(monkeypatch):
    """FLASH_RL_BACKEND=verl delegates to run_rl_verl without touching the trl body."""
    called = []
    monkeypatch.setattr(rl_verl, "run_rl_verl", lambda: called.append(True))
    monkeypatch.setenv("FLASH_RL_BACKEND", "verl")
    # if the trl body ran instead of delegating, it would fail hard before returning.
    rl.run_rl()
    assert called == [True]


def test_run_rl_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("FLASH_RL_BACKEND", "megatron")
    with pytest.raises(RuntimeError, match="not a known grpo backend"):
        rl.run_rl()


# ------------------------------- data conversion -------------------------------
def test_build_verl_dataset_rows_schema_and_index():
    rows = rl_verl.build_verl_dataset_rows(
        [[{"role": "user", "content": "q0"}], [{"role": "user", "content": "q1"}]],
        [5, 9],
        ["42", ""],
    )
    assert rows[0]["prompt"] == [{"role": "user", "content": "q0"}]
    assert rows[0]["reward_model"] == {"style": "rule", "ground_truth": "42"}
    # the flash rollout index must round-trip through verl's extra_info so the reward maps back.
    assert [r["extra_info"]["index"] for r in rows] == [5, 9]
    assert all(r["data_source"] == rl_verl.DATA_SOURCE for r in rows)


def test_build_verl_dataset_rows_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        rl_verl.build_verl_dataset_rows([[{"role": "user", "content": "q"}]], [1, 2], ["a", "b"])


# ------------------------------- multimodal parquet contract -------------------------------
# verl's RLHFDataset._build_messages re-splits each prompt on "<image>" and then asserts the
# placeholder count equals len(images). the two halves of that invariant are produced in different
# places here (message flattening vs the images column), so they are exactly the kind of pair that
# drifts silently: a row can look well-formed on both sides and still raise inside verl.


def _image_placeholder_count(row) -> int:
    return sum(str(m["content"]).count("<image>") for m in row["prompt"])


def test_multimodal_rows_match_verl_placeholder_assertion():
    rows = rl_verl.build_verl_dataset_rows(
        [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "q0"}, {"type": "image"}],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "q1"}],
                }
            ],
        ],
        [0, 1],
        ["a", "b"],
        image_uris=[["file:///w/0-0.png"], ["file:///w/1-0.png", "file:///w/1-1.png"]],
    )
    # this is verl's own assertion, restated: image_offset == len(images) or the dataset raises.
    for row in rows:
        assert _image_placeholder_count(row) == len(row["images"])
    assert rows[1]["images"] == [
        {"image": "file:///w/1-0.png"},
        {"image": "file:///w/1-1.png"},
    ]
    # content is flattened to a string; verl re-expands it. leaving content blocks in place would
    # make the re-split find zero placeholders while images is non-empty.
    assert all(isinstance(m["content"], str) for row in rows for m in row["prompt"])


def test_text_rows_carry_no_images_column():
    # the control: without image_uris the rows must stay exactly as before. an unconditional images
    # column would make every text job take verl's multimodal dataset path.
    rows = rl_verl.build_verl_dataset_rows([[{"role": "user", "content": "q"}]], [0], ["a"])
    assert "images" not in rows[0]


def test_multimodal_rows_reject_a_mismatched_uri_list():
    with pytest.raises(ValueError, match="image_uris length mismatch"):
        rl_verl.build_verl_dataset_rows(
            [[{"role": "user", "content": "q"}]], [0], ["a"], image_uris=[[], []]
        )


def test_multimodal_rows_reject_a_literal_image_placeholder_in_text():
    # verl splits prompt text on "<image>" and re-expands each hit into a real image block, so a
    # prompt that merely TALKS about the token consumes an image the row does not have. verl would
    # abort dataset loading with a bare offset assertion; catching it here names the example.
    with pytest.raises(ValueError, match="reserved by verl"):
        rl_verl.build_verl_dataset_rows(
            [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "the <image> token marks an image"},
                            {"type": "image"},
                        ],
                    }
                ]
            ],
            [7],
            ["a"],
            image_uris=[["file:///w/0-0.png"]],
        )


def test_text_only_row_of_a_mixed_job_rejects_a_literal_placeholder():
    # the row itself has no images, but verl's split is driven by the row's OWN modality columns and
    # a mixed job writes an images column on every row -- so a text row with a literal "<image>"
    # asserts against its empty list. this is the case a per-job (rather than per-row) check misses.
    with pytest.raises(ValueError, match="reserved by verl"):
        rl_verl.build_verl_dataset_rows(
            [
                [{"role": "user", "content": [{"type": "text", "text": "describe <image> please"}]}],
                [{"role": "user", "content": [{"type": "image"}]}],
            ],
            [0, 1],
            ["a", "b"],
            image_uris=[[], ["file:///w/1-0.png"]],
        )


@pytest.mark.parametrize("reserved", ["<video>", "<audio>"])
def test_multimodal_rows_reject_other_reserved_media_placeholders(reserved):
    # _build_messages splits on all three markers. flash never writes a videos/audios column, so a
    # single literal occurrence asserts against an empty list -- the count check on <image> alone
    # would pass this row straight through.
    with pytest.raises(ValueError, match="reserves as a media placeholder"):
        rl_verl.build_verl_dataset_rows(
            [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"a {reserved} marker"},
                            {"type": "image"},
                        ],
                    }
                ]
            ],
            [3],
            ["a"],
            image_uris=[["file:///w/0-0.png"]],
        )


def test_text_job_does_not_police_reserved_placeholders():
    # the control: without an images column verl never splits, so "<image>" is ordinary text and
    # rejecting it would break text jobs that legitimately discuss the token.
    rows = rl_verl.build_verl_dataset_rows(
        [[{"role": "user", "content": "what does <image> mean?"}]], [0], ["a"]
    )
    assert rows[0]["prompt"] == [{"role": "user", "content": "what does <image> mean?"}]


def test_mixed_job_parquet_round_trips_the_images_column(tmp_path):
    # Dataset.from_list infers ONE type per column across all rows. in a mixed job the text rows
    # have an empty images list, and inference on an all-empty-or-partly-empty column can land on a
    # type verl cannot read back as a struct. this asserts the round trip, not the schema object,
    # because the schema is only interesting insofar as the read-back works.
    rows = rl_verl.build_verl_dataset_rows(
        [
            [{"role": "user", "content": [{"type": "text", "text": "text only"}]}],
            [{"role": "user", "content": [{"type": "image"}]}],
        ],
        [0, 1],
        ["a", "b"],
        image_uris=[[], ["file:///w/1-0.png"]],
    )
    path = str(tmp_path / "train.parquet")
    rl_verl.write_verl_grpo_parquet(rows, path)

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert table.num_rows == 2
    images = table.column("images").to_pylist()
    assert images == [[], [{"image": "file:///w/1-0.png"}]]
    # the empty row must still be a LIST OF STRUCTS, not a null column: verl indexes row["images"]
    # per element, so a null-typed column fails on read rather than on the empty row. asserted
    # structurally because arrow spells the list field "item" or "element" by version.
    import pyarrow as pa

    images_type = table.schema.field("images").type
    assert pa.types.is_list(images_type)
    assert [f.name for f in images_type.value_type] == ["image"]
    assert pa.types.is_string(images_type.value_type.field("image").type)
    assert table.column("extra_info").to_pylist()[1]["index"] == 1


def test_text_only_parquet_does_not_pin_the_multimodal_schema(tmp_path):
    # the control: a text job's rows have no images column at all, so pinning the multimodal schema
    # would fail the write outright.
    rows = rl_verl.build_verl_dataset_rows([[{"role": "user", "content": "q"}]], [0], ["a"])
    path = str(tmp_path / "train.parquet")
    rl_verl.write_verl_grpo_parquet(rows, path)

    import pyarrow.parquet as pq

    assert "images" not in pq.read_table(path).schema.names


# ------------------------------- override generation -------------------------------
def _overrides_cfg(**over):
    cfg = {
        "train_files": "/w/train.parquet", "val_files": "/w/val.parquet",
        "model_id": "Qwen/Qwen3-4B", "lora_rank": 32, "lora_alpha": 64,
        "target_modules": "all-linear", "lr": 1e-5, "group_size": 8,
        "prompts_per_step": 16, "max_prompt_len": 2048,
        "max_model_len": 2368, "max_token_len_per_gpu": 2368,
        # single-turn: the response tensor holds one completion, so it is max_completion wide.
        "max_completion": 320, "max_response_len": 320, "multi_turn": False,
        "temperature": 1.0, "top_p": 0.95, "kl_coef": 0.0,
        "entropy_quantile": None,
        "stop_sequences": (),
        "structured_outputs": None, "thinking": False,
        "loss_agg_mode": "seq-mean-token-sum-norm", "seed": 42, "ppo_epochs": 1,
        "steps": 60, "gpu_mem_util": 0.5, "n_gpus": 1, "loggers": "console", "fp8_kv": False,
        "warmstart_adapter": "", "reward_path": "/w/reward.py", "reward_name": "compute_score",
        "mask_truncated_completions": True,
        "total_epochs": 1, "save_freq": 20, "ckpt_to_keep": 1, "local_dir": "/w/ckpt",
        "project_name": "flash", "experiment_name": "flash-rl-run123",
    }
    cfg.update(over)
    return cfg


def test_build_verl_overrides_carries_dr_grpo_recipe():
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    assert "algorithm.adv_estimator=grpo" in o
    # dr-grpo: no std normalization + constant-length loss aggregation.
    assert "algorithm.norm_adv_by_std_in_grpo=False" in o
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm" in o
    assert "actor_rollout_ref.model.lora_rank=32" in o
    assert "actor_rollout_ref.rollout.n=8" in o
    assert "actor_rollout_ref.rollout.load_format=safetensors" in o
    assert "actor_rollout_ref.rollout.top_p=0.95" in o
    # constant lr, on-policy updates, gradient checkpointing, seed, max-steps horizon, save schedule.
    assert "actor_rollout_ref.actor.optim.weight_decay=0.0" in o
    assert "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0" in o
    assert "actor_rollout_ref.actor.ppo_epochs=1" in o
    assert "actor_rollout_ref.model.enable_gradient_checkpointing=True" in o
    assert "data.seed=42" in o
    # the rollout engine seed rides engine_kwargs, not `rollout.seed`, which verl 0.8.0's
    # RolloutConfig does not declare. see build_verl_overrides for the full reasoning.
    assert "++actor_rollout_ref.rollout.engine_kwargs.vllm.seed=42" in o
    assert "actor_rollout_ref.rollout.seed=42" not in o
    assert "trainer.total_training_steps=60" in o
    assert "trainer.save_freq=20" in o
    assert "trainer.max_actor_ckpt_to_keep=1" in o
    assert "trainer.logger=[console]" in o
    assert "data.train_batch_size=16" in o
    # truncated importance sampling: token-level, cap 2.0 (matches flash's tis recipe).
    assert "algorithm.rollout_correction.rollout_is=token" in o
    assert "algorithm.rollout_correction.rollout_is_threshold=2.0" in o


def test_build_verl_overrides_does_not_emit_inert_drop_last_override():
    # this guards only against flash emitting a misleading no-op; it does not prove verl reads the key.
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    assert not any("drop_last" in override for override in o)


def test_build_verl_overrides_sizes_agent_loop_workers_to_the_rollout_batch():
    # verl chunks prompts_per_step * group_size across agent.num_workers and asserts exact
    # divisibility; its default of 8 aborts before the first step on e.g. 2 x 2 = 4.
    o = rl_verl.build_verl_overrides(_overrides_cfg(prompts_per_step=2, group_size=2))
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in o
    # the common case still gets the full worker pool.
    big = rl_verl.build_verl_overrides(_overrides_cfg(prompts_per_step=64, group_size=8))
    assert "actor_rollout_ref.rollout.agent.num_workers=8" in big


@pytest.mark.parametrize(("count", "expected"), [(None, 1), (1, 1), (2, 2), (8, 8)])
def test_run_rl_verl_sizes_the_run_from_the_spec_gpu_count(count, expected):
    # the wiring, not just the builder: a spec that rents N cards must configure verl for N.
    # gpu_count_of is the same reader the runpod rental path uses, so the rented shape and the
    # trained shape cannot drift apart.
    from flash.spec import GpuSpec, JobSpec, gpu_count_of

    project = "11111111-1111-4111-8111-111111111111"
    spec = (
        JobSpec(project=project)
        if count is None
        else JobSpec(project=project, gpu=GpuSpec(count=count))
    )
    assert gpu_count_of(spec) == expected


def test_build_verl_overrides_single_gpu_is_the_unchanged_default():
    o = rl_verl.build_verl_overrides(_overrides_cfg(n_gpus=1))
    assert "trainer.n_gpus_per_node=1" in o
    assert "trainer.nnodes=1" in o
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in o
    assert "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1" in o


@pytest.mark.parametrize("n_gpus", [2, 4, 8])
def test_build_verl_overrides_shards_every_card_along_the_sequence(n_gpus):
    # verl builds mesh_shape=(dp, sp), so sp == n_gpus pins dp == 1: the optimizer keeps seeing one
    # global batch of prompts_per_step * group_size. sharding the BATCH instead would change the
    # gradient, which is why sp/tp track the card count rather than leaving dp to absorb it.
    o = rl_verl.build_verl_overrides(_overrides_cfg(n_gpus=n_gpus))
    assert f"trainer.n_gpus_per_node={n_gpus}" in o
    assert f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={n_gpus}" in o
    assert f"actor_rollout_ref.rollout.tensor_model_parallel_size={n_gpus}" in o
    # one worker, many cards: nnodes stays 1 so verl's replica_rank offset (and the rollout seed)
    # is unaffected by the card count.
    assert "trainer.nnodes=1" in o
    # sequence parallelism is only legal with padding removed; verl raises otherwise.
    assert "actor_rollout_ref.model.use_remove_padding=True" in o


def test_build_verl_overrides_batch_shape_is_identical_across_gpu_counts():
    # the guard that matters: adding cards must not change what the optimizer sees. anything that
    # would alter the effective batch (or the per-gpu micro batch) is a silent recipe change.
    batch_keys = (
        "data.train_batch_size=",
        "actor_rollout_ref.rollout.n=",
        "actor_rollout_ref.actor.ppo_mini_batch_size=",
        # the per-gpu token budget is part of the shape too: verl scales it by sp_size itself, so
        # emitting a card-dependent value would shrink each micro-batch as cards are added.
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=",
    )
    one = rl_verl.build_verl_overrides(_overrides_cfg(n_gpus=1))
    for n_gpus in (2, 4, 8):
        many = rl_verl.build_verl_overrides(_overrides_cfg(n_gpus=n_gpus))
        for key in batch_keys:
            assert [o for o in one if o.startswith(key)] == [o for o in many if o.startswith(key)]


def test_build_verl_overrides_sets_truncation_mask_when_enabled():
    o = rl_verl.build_verl_overrides(_overrides_cfg(mask_truncated_completions=True))
    # `++` (append-or-override), because the key exists in the fork's rollout.yaml but not stock's.
    assert "++actor_rollout_ref.rollout.mask_truncated_completions=true" in o


def test_build_verl_overrides_omits_truncation_mask_when_disabled():
    # stock verl rejects the unknown key at dataclass conversion, and not masking is already its
    # behavior, so emitting `=false` would break stock runs while changing nothing.
    o = rl_verl.build_verl_overrides(_overrides_cfg(mask_truncated_completions=False))
    assert not any("mask_truncated_completions" in override for override in o)


def test_build_verl_overrides_sizes_engine_to_the_job_not_the_architecture():
    # left unset, verl substitutes the model's full max_position_embeddings and hands it to vllm,
    # so a short job on a long-context model reserves kv cache it can never use. the emitted length
    # must be the job's own engine length.
    o = rl_verl.build_verl_overrides(_overrides_cfg(max_model_len=2368))
    assert "actor_rollout_ref.rollout.max_model_len=2368" in o


def test_engine_len_clamped_to_model_limit():
    # verl raises ValueError when max_model_len exceeds max_position_embeddings, so a job asking
    # for more context than the architecture has must train shorter, not die at rollout startup.
    assert rl_verl.clamp_engine_len(32768, 8192) == 8192
    # under the limit is untouched, and an unknown limit leaves verl's own resolution in charge.
    assert rl_verl.clamp_engine_len(4096, 40960) == 4096
    assert rl_verl.clamp_engine_len(32768, None) == 32768
    assert rl_verl.clamp_engine_len(32768, 0) == 32768


def test_token_budget_admits_a_full_length_sequence():
    # dynamic bsz packs micro-batches up to this budget. below one full sequence, the longest
    # rollout the engine can produce fits in no micro-batch at all.
    cfg = _overrides_cfg(max_prompt_len=31744, max_completion=1024, max_token_len_per_gpu=32768)
    o = rl_verl.build_verl_overrides(cfg)
    assert "actor_rollout_ref.actor.use_dynamic_bsz=true" in o
    assert "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true" in o
    key = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu="
    budget = int(next(x for x in o if x.startswith(key)).split("=")[1])
    assert budget >= cfg["max_prompt_len"] + cfg["max_completion"]
    # the engine asserts the actor and log-prob flags match, so both budgets move together.
    assert f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={budget}" in o


def test_dynamic_bsz_replaces_sequence_count_micro_batches():
    # with use_dynamic_bsz on, verl's actor config validation skips the micro-batch checks entirely
    # and asserts the TOKEN budgets are set instead. a leftover sequence-count key would be dead
    # config claiming to bound memory.
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    assert not any("ppo_micro_batch_size_per_gpu" in x for x in o)
    assert not any("log_prob_micro_batch_size_per_gpu" in x for x in o)


def test_multimodal_overrides_hand_verl_the_images_column():
    # the parquet's images column is inert unless verl is told to read it: without image_key the
    # dataset treats the rows as text, the <image> placeholders never re-expand, and the model
    # trains on the caption alone -- silently, which is the failure this whole port exists to avoid.
    o = rl_verl.build_verl_overrides(_overrides_cfg(multimodal=True))
    assert "data.image_key=images" in o
    # a processor rather than a bare tokenizer, and raw chat so verl owns the expansion.
    assert "actor_rollout_ref.model.trust_remote_code=true" in o
    assert "data.return_raw_chat=true" in o
    # verl RAISES on an over-budget multimodal prompt rather than truncating it, so truncation must
    # be error (never left on a silent trim) and the length filter must be armed.
    assert "data.truncation=error" in o
    assert "data.filter_overlong_prompts=true" in o
    # the processor's image loader is not fork-safe under verl's default dataloader workers.
    assert "data.dataloader_num_workers=0" in o


def test_text_overrides_omit_every_multimodal_key():
    # the control: these keys must be absent, not merely false. data.image_key=images on a text job
    # points verl at a column the parquet does not have.
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    for key in (
        "data.image_key",
        "data.return_raw_chat",
        "data.return_multi_modal_inputs",
        "data.dataloader_num_workers",
    ):
        assert not any(x.startswith(key) for x in o), key


def test_build_verl_training_cfg_carries_the_multimodal_flag():
    # the flag is resolved in _resolve_grpo_inputs but consumed by build_verl_overrides, so
    # a cfg that dropped it would produce a text-shaped override list from a multimodal parquet.
    source = inspect.getsource(rl_verl._build_verl_training_cfg)
    assert '"multimodal": bool(inp.get("multimodal"))' in source


def test_build_verl_training_cfg_derives_engine_len_and_budget():
    inp = {
        "lora_rank": 32, "lora_alpha": 64, "lr": 1e-5, "group_size": 8,
        "prompts_per_step": 16, "mask_truncated_completions": True,
        "max_prompt_len": 3072, "max_completion": 1024, "max_response_len": 1024,
        "multi_turn": False, "engine_len": 4096,
        "temperature": 1.0, "top_p": 0.95, "kl_coef": 0.0, "entropy_quantile": None, "stop_sequences": (), "structured_outputs": None, "seed": 42,
        "ppo_epochs": 1, "steps": 60, "warmstart_adapter": "",
        "verl_total_epochs": 2, "save_freq": 20, "ckpt_to_keep": 1,
    }
    common = {
        "train_files": "/w/t.parquet", "val_files": "/w/v.parquet", "model_id": "Qwen/Qwen3-4B",
        "thinking": False, "loggers": "console", "fp8_kv": False,
        "reward_path": "/w/r.py", "local_dir": "/w/ckpt",
        "project_name": "flash", "experiment_name": "flash-rl-run123",
    }
    cfg = rl_verl._build_verl_training_cfg(inp, **common)
    # the engine gets the full prompt+completion length, not the prompt budget alone, and the token
    # budget matches it. the resolver clamps engine_len, so the builder passes it through unchanged.
    assert cfg["max_model_len"] == 4096
    assert cfg["max_token_len_per_gpu"] == 4096


@pytest.mark.parametrize(
    ("prompt_count", "prompts_per_step", "epochs", "max_steps", "expected_steps", "expected_epochs"),
    [
        pytest.param(33, 16, 2, None, 5, 3, id="partial-batch-derived-horizon"),
        pytest.param(32, 16, 2, 7, 7, 4, id="explicit-horizon-beyond-derived"),
        pytest.param(32, 16, 2, None, 4, 2, id="exactly-divisible"),
    ],
)
def test_verl_epoch_capacity_reaches_update_horizon(
    prompt_count, prompts_per_step, epochs, max_steps, expected_steps, expected_epochs
):
    derived_steps = rl_verl.on_policy_steps(
        epochs=epochs, prompt_count=prompt_count, prompts_per_step=prompts_per_step
    )
    steps = rl_verl.resolve_update_horizon(derived_steps, max_steps)
    resolved_epochs = rl_verl._verl_epochs_for_horizon(
        epochs=epochs,
        prompt_count=prompt_count,
        prompts_per_step=prompts_per_step,
        steps=steps,
    )

    assert steps == expected_steps
    assert resolved_epochs == expected_epochs
    assert (prompt_count // prompts_per_step) * resolved_epochs >= steps


@pytest.mark.parametrize(
    ("prompt_count", "prompts_per_step", "message"),
    [
        pytest.param(0, 16, "prompt_count must be positive", id="no-prompts"),
        pytest.param(5, 0, "prompts_per_step must be positive", id="zero-batch"),
        pytest.param(5, 16, "prompt_count must be at least", id="batch-exceeds-prompts"),
    ],
)
def test_verl_epoch_capacity_rejects_invalid_batch_inputs(
    prompt_count, prompts_per_step, message
):
    with pytest.raises(ValueError, match=message):
        rl_verl._verl_epochs_for_horizon(
            epochs=2,
            prompt_count=prompt_count,
            prompts_per_step=prompts_per_step,
            steps=3,
        )


def test_verl_epoch_capacity_invariant_across_valid_inputs():
    for prompt_count in range(1, 34):
        for prompts_per_step in range(1, prompt_count + 1):
            for epochs in (1, 2, 4):
                for steps in (1, epochs, epochs + 5):
                    resolved_epochs = rl_verl._verl_epochs_for_horizon(
                        epochs=epochs,
                        prompt_count=prompt_count,
                        prompts_per_step=prompts_per_step,
                        steps=steps,
                    )
                    assert (prompt_count // prompts_per_step) * resolved_epochs >= steps


def test_resolver_clamps_prompt_budget_with_the_engine(monkeypatch):
    # regression: clamping only the engine let the prompt filter admit prompts sized against the
    # UNCLAMPED context. those prompts plus the completion allowance overflow the engine vllm was
    # actually given, so they die at rollout instead of training on the shorter context. every
    # length must descend from one clamped value.
    # asks for twice the architecture's context (model_max_position_embeddings is pinned to 32768).
    inp = _capability_resolve(monkeypatch, _capability_env(), train={"max_context_tokens": 65536})
    assert inp["engine_len"] == 32768
    # the prompt filter's budget is carved out of the clamped engine, not the requested 65536.
    assert inp["max_prompt_len"] + inp["max_completion"] == 32768
    # and every length the overrides emit agrees with it.
    cfg = rl_verl._build_verl_training_cfg(
        inp,
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_id=inp["model_id"],
        thinking=False,
        loggers="console",
        fp8_kv=False,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    assert cfg["max_model_len"] == 32768
    assert cfg["max_token_len_per_gpu"] == 32768
    assert cfg["max_prompt_len"] + cfg["max_completion"] == cfg["max_model_len"]


def _save_steps_inputs(monkeypatch, *, save_at_steps=None, save_every=None, max_steps=100):
    """resolve grpo verl inputs for a job with (or without) exact save steps."""
    train: dict = {"max_steps": max_steps}
    if save_at_steps is not None:
        train["save_at_steps"] = list(save_at_steps)
    if save_every is not None:
        train["save_every"] = save_every
    return _capability_resolve(monkeypatch, _capability_env(), train=train)


def test_save_freq_is_the_gcd_so_verl_lands_on_every_required_step(monkeypatch):
    # verl only saves when global_step % save_freq == 0, so it cannot hit an arbitrary set directly.
    # the gcd is the largest interval every required step divides, so verl writes a superset of the
    # checkpoints and the uploader publishes deployables at exactly the requested ones.
    inp = _save_steps_inputs(monkeypatch, save_at_steps=(10, 25, 100))
    assert inp["save_freq"] == 5
    assert inp["save_at_steps"] == (10, 25, 100)
    for step in inp["save_at_steps"]:
        assert step % inp["save_freq"] == 0


def test_save_freq_falls_back_to_save_every_without_exact_steps(monkeypatch):
    # no exact steps: periodic saves are preserved on the customer's own interval.
    inp = _save_steps_inputs(monkeypatch, save_every=15)
    assert inp["save_freq"] == 15
    assert inp["save_at_steps"] == ()


def test_save_steps_reach_the_horizon_they_were_validated_against(monkeypatch):
    # save_at_steps requires max_steps, and the horizon resolves to exactly that, so every required
    # step is reachable by the run. this is the invariant the uploader's completeness check assumes.
    inp = _save_steps_inputs(monkeypatch, save_at_steps=(10, 25, 100), max_steps=100)
    assert inp["steps"] == 100
    assert inp["save_at_steps"][-1] <= inp["steps"]


def test_final_publish_is_suppressed_when_exact_save_steps_are_set():
    # parity with the trl path (rl.py: `if final_save_due(...)`): with save_at_steps set the customer
    # asked for those steps and nothing else, so the final step must not add an unrequested
    # deployable. without them the final checkpoint is still preserved.
    from flash.engine.steps import final_save_due

    assert not final_save_due(100, (10, 25))
    assert final_save_due(100, ())


def test_resume_uploader_publishes_required_steps_and_reports_missing(tmp_path, monkeypatch):
    # the deployable at a required step is the whole point of save_at_steps. a resume-state upload
    # alone leaves the step resumable but not servable, which is the gap this closes.
    published: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(
        rl_verl._w,
        "publish_deployable_checkpoint",
        lambda d, s, **kw: published.append((d, s, kw.get("required", False))),
        raising=False,
    )
    monkeypatch.setattr(rl_verl._w, "upload_resume_checkpoint", lambda *a, **kw: True, raising=False)
    monkeypatch.setattr(rl_verl._w, "write_base_model_provenance", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(rl_verl, "_export_peft_adapter", lambda *a, **kw: None)
    monkeypatch.setattr(rl_verl, "_stamp_adapter_dir_provenance", lambda *a, **kw: None)

    local_dir = tmp_path / "ckpt"
    (local_dir / "global_step_10" / "actor").mkdir(parents=True)
    (local_dir / "global_step_5" / "actor").mkdir(parents=True)
    (local_dir / "latest_checkpointed_iteration.txt").write_text("10")

    class _Tok:
        def save_pretrained(self, path):
            pass

    uploader = rl_verl._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(10, 20),
        export_root=str(tmp_path / "exports"),
        python_bin="python",
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="rev",
        preprocessor=_Tok(),
    )
    uploader.start()
    uploader.stop()

    # step 10 was required and completed, so it published as a REQUIRED deployable. step 5 is a gcd
    # by-product verl wrote on the way there; it is resume state only and must not be published.
    assert [(step, required) for _, step, required in published] == [(10, True)]
    # step 20 never completed, so the run must fail rather than silently ship an incomplete set.
    with pytest.raises(RuntimeError, match="required saves were not durably published: \\[20\\]"):
        uploader.raise_if_incomplete()


def test_resume_credits_required_steps_already_durable_on_hf(tmp_path, monkeypatch):
    # a resumed run never re-saves a step it trained past. without crediting the earlier required
    # steps a retry that resumes at 20 would report step 10 missing and fail a successful run.
    monkeypatch.setattr(rl_verl, "_deployable_adapter_on_hf", lambda step: step == 10)

    class _Tok:
        def save_pretrained(self, path):
            pass

    uploader = rl_verl._VerlResumeUploader(
        str(tmp_path),
        resume_step=20,
        required_steps=(10, 15, 25),
        preprocessor=_Tok(),
    )
    uploader.credit_durable_required_steps(20)

    # step 10 is verified on hf, so it is credited. step 15 is below the resume point but its
    # adapter never landed, so it stays uncredited and completeness still catches it. step 25 is
    # ahead of the resume point and is this run's job to publish.
    assert uploader.published_steps == {10}
    with pytest.raises(RuntimeError, match=r"not durably published: \[15, 25\]"):
        uploader.raise_if_incomplete()


def test_resume_step_is_not_credited_without_a_durable_adapter(tmp_path, monkeypatch):
    # a preempted worker can advance past a required step without its deployable ever reaching hf,
    # so the restored step counter alone must never credit a required save.
    monkeypatch.setattr(rl_verl, "_deployable_adapter_on_hf", lambda step: False)

    uploader = rl_verl._VerlResumeUploader(str(tmp_path), resume_step=10, required_steps=(10,))
    uploader.credit_durable_required_steps(10)

    assert uploader.published_steps == set()


def test_checkpoint_retention_outlives_the_export_when_exact_saves_are_set(monkeypatch):
    # verl prunes a checkpoint once the NEXT save completes, so keeping 1 gives the uploader a
    # single save interval to export before its source is deleted. with a gcd of 1 that interval is
    # one update, which races the export and can lose a required deployable.
    exact = _save_steps_inputs(monkeypatch, save_at_steps=(10, 11))
    assert exact["save_freq"] == 1
    assert exact["ckpt_to_keep"] > 1

    # nothing is exported mid-run without exact saves, so retention stays at its cheapest.
    periodic = _save_steps_inputs(monkeypatch, save_every=15)
    assert periodic["ckpt_to_keep"] == 1


def test_verl_resolver_builds_capacity_overrides_and_configured_metadata(monkeypatch):
    from flash.engine.worker._pkg import W
    from flash.spec import JobSpec

    class _Env:
        multi_turn = False
        is_tool_env = False

        def dataset(self):
            return [{"index": i} for i in range(33)]

        def prompt_messages(self, ex):
            return [{"role": "user", "content": f"question {ex['index']}"}]

    class _Tokenizer:
        pad_token = None
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1])

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
            "train": {"batch_size": 16, "epochs": 2},
        }
    )
    monkeypatch.setattr(W, "JOB_SPEC", spec, raising=False)
    monkeypatch.setattr(W, "SEED", 42, raising=False)
    monkeypatch.setattr(W, "THINKING", False, raising=False)
    monkeypatch.setattr(W, "require_active_env", lambda: _Env(), raising=False)
    monkeypatch.setattr(W, "grpo_overrides", lambda: {}, raising=False)
    monkeypatch.setattr(W, "grpo_mask_truncated_completions", lambda train: False, raising=False)
    monkeypatch.setattr(W, "load_tokenizer", lambda *args, **kwargs: _Tokenizer(), raising=False)
    monkeypatch.setattr(rl_verl, "seed_training_rngs", lambda seed: None)
    # the context-limit probe reads the model config off the hub; keep this unit test offline.
    monkeypatch.setattr(rl_verl, "model_max_position_embeddings", lambda *a, **k: 40960)

    inp = rl_verl._resolve_grpo_inputs()
    cfg = rl_verl._build_verl_training_cfg(
        inp,
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_id=inp["model_id"],
        thinking=False,
        loggers="console",
        fp8_kv=False,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    overrides = rl_verl.build_verl_overrides(cfg)
    notes = rl_verl._build_verl_train_notes(
        inp,
        steps_run=5,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
    )

    assert "trainer.total_training_steps=5" in overrides
    assert "trainer.total_epochs=3" in overrides
    assert notes["epochs"] == 2
    assert notes["grpo_recipe"]["verl_total_epochs"] == 3
    # the builder is called without n_gpus here, exactly as a single-gpu job does: the default must
    # stay 1 so a spec with no gpu.count keeps the historical shape.
    assert cfg["n_gpus"] == 1


def test_build_verl_overrides_wandb_logger_when_enabled():
    o = rl_verl.build_verl_overrides(_overrides_cfg(loggers="console,wandb"))
    assert "trainer.logger=[console,wandb]" in o


def test_build_verl_overrides_warmstart_adapter_path():
    # fresh run: no lora_adapter_path override.
    fresh = rl_verl.build_verl_overrides(_overrides_cfg(warmstart_adapter=""))
    assert not any("lora_adapter_path" in x for x in fresh)
    # warm-start: point verl's lora init at the downloaded source adapter dir.
    warm = rl_verl.build_verl_overrides(_overrides_cfg(warmstart_adapter="/tmp/sft_adapter"))
    assert "actor_rollout_ref.model.lora_adapter_path=/tmp/sft_adapter" in warm


def test_build_verl_overrides_fp8_kv_gated_on_hardware():
    off = rl_verl.build_verl_overrides(_overrides_cfg(fp8_kv=False))
    assert not any("kv_cache_dtype" in x for x in off)
    on = rl_verl.build_verl_overrides(_overrides_cfg(fp8_kv=True))
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8" in on


def test_build_verl_overrides_kl_off_by_default():
    # flash default kl_penalty_coef=0 (dr-grpo, no kl term) -> no reference policy.
    o = rl_verl.build_verl_overrides(_overrides_cfg(kl_coef=0.0))
    assert "actor_rollout_ref.actor.use_kl_loss=False" in o
    assert not any("kl_loss_coef" in x for x in o)
    assert not any("ref.log_prob_micro_batch" in x for x in o)


def test_build_verl_overrides_kl_on_when_requested():
    o = rl_verl.build_verl_overrides(_overrides_cfg(kl_coef=0.02))
    assert "actor_rollout_ref.actor.use_kl_loss=True" in o
    assert "actor_rollout_ref.actor.kl_loss_coef=0.02" in o
    # the ref worker carries no batching keys of its own: ref.yaml resolves
    # log_prob_use_dynamic_bsz / log_prob_max_token_len_per_gpu through oc.select on the actor keys
    # the block above sets, so emitting a sequence-count micro batch here would contradict them.
    assert not any("ref.log_prob_micro_batch" in x for x in o)


def test_verl_uses_canonical_heartbeat_stage_contracts():
    from flash.engine.worker.heartbeat import _HB_THROTTLED_STAGES
    from flash.providers._poll import STEP_GATED_STAGES
    from flash.runner import _TRAINING_STAGES

    src = inspect.getsource(rl_verl.run_rl_verl)
    assert "rl_verl_training" not in src
    assert "rl_verl_finalizing" not in src
    initial_heartbeat = '_w.heartbeat("rl_step", step=0, initial=True)'
    assert initial_heartbeat in src
    # ordering is read off the ast, not off substring offsets: the liveness call spans several lines
    # once it carries keywords, and a text search for the one-line spelling would report "missing"
    # for a call that is present and correctly placed.
    tree = ast.parse(textwrap.dedent(src))
    stage_linenos = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == "rl_step"):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "liveness_heartbeat":
            stage_linenos["liveness"] = node.lineno
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "heartbeat"
            and any(kw.arg == "initial" for kw in node.keywords)
        ):
            stage_linenos["initial"] = node.lineno
    assert "initial" in stage_linenos
    assert "liveness" in stage_linenos
    assert stage_linenos["initial"] < stage_linenos["liveness"]
    assert '"rl_finalizing"' in src
    assert "rl_step" in _HB_THROTTLED_STAGES
    assert "rl_step" in STEP_GATED_STAGES
    assert "rl_step" in _TRAINING_STAGES
    assert "rl_finalizing" in _HB_THROTTLED_STAGES


# ------------------------------- reward module render -------------------------------
def test_render_reward_module_is_valid_and_defines_compute_score():
    src = rl_verl.render_reward_module()
    ns: dict = {}
    exec(compile(src, "<reward>", "exec"), ns)  # compiles + defines, no network call made
    assert callable(ns["compute_score"])
    # no flash import leaks into the verl-side shim.
    assert "import flash" not in src


def test_render_reward_module_missing_index_raises():
    ns: dict = {}
    exec(compile(rl_verl.render_reward_module(), "<reward>", "exec"), ns)
    with pytest.raises(RuntimeError, match="no example index"):
        ns["compute_score"]("flash_env", "answer", "unused", extra_info={})


@pytest.mark.parametrize(
    "index",
    [True, 1.9, np.bool_(True), np.bool_(False), float("nan"), float("inf"), float("-inf")],
    ids=["bool", "fractional", "numpy-true", "numpy-false", "nan", "positive-inf", "negative-inf"],
)
def test_render_reward_module_rejects_invalid_index(monkeypatch, index):
    monkeypatch.setenv("TEST_FLASH_VERL_REWARD_URL", "http://unused")
    ns: dict = {}
    exec(compile(rl_verl.render_reward_module("TEST_FLASH_VERL_REWARD_URL"), "<reward>", "exec"), ns)
    monkeypatch.setattr(
        ns["urllib"].request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("invalid index must not reach the reward server"),
    )

    with pytest.raises(RuntimeError, match="invalid example index"):
        ns["compute_score"]("flash_env", "answer", "unused", extra_info={"index": index})


@pytest.mark.parametrize("index", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_render_reward_module_accepts_exact_integral_index(monkeypatch, index):
    scored = []
    server, url = rl_verl.start_reward_server(
        lambda idx, solution: scored.append((idx, solution)) or 3.0,
        example_count=2,
    )
    try:
        monkeypatch.setenv("TEST_FLASH_VERL_REWARD_URL", url)
        ns: dict = {}
        exec(compile(rl_verl.render_reward_module("TEST_FLASH_VERL_REWARD_URL"), "<reward>", "exec"), ns)
        assert ns["compute_score"](
            "flash_env", "answer", "unused", extra_info={"index": index}
        ) == 3.0
        assert scored == [(1, "answer")]
    finally:
        server.shutdown()


# ------------------------------- reward parity -------------------------------
class _BreakdownEnv:
    def scores_breakdown(self, graded, ex, state):
        return {"total": 1.0 if graded.strip() == ex["gt"] else 0.0}


class _RewardOnlyEnv:
    def reward(self, graded, ex, state):
        return 2.5 if ex["gt"] in graded else 0.0


class _RaisingEnv:
    def scores_breakdown(self, graded, ex, state):
        raise ValueError("boom")


@pytest.fixture
def _identity_graded(monkeypatch):
    monkeypatch.setattr(W, "graded_text", lambda text, prompt_opened_thinking=False: text)
    monkeypatch.setattr(W, "thinking_text", lambda text, prompt_opened_thinking=False: "")
    monkeypatch.setattr(W, "think_token_count", lambda text, tok, prompt_opened_thinking=False: 3)


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_breakdown_and_reward_env():
    s = rl_verl.score_single_turn(
        _BreakdownEnv(), "7", {"gt": "7"}, tok=None, thinking=False,
        prompt_opened_thinking=False, think_penalty=0.0,
    )
    assert s == 1.0
    s2 = rl_verl.score_single_turn(
        _RewardOnlyEnv(), "the answer is 7", {"gt": "7"}, tok=None, thinking=False,
        prompt_opened_thinking=False, think_penalty=0.0,
    )
    assert s2 == 2.5


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_applies_thinking_penalty():
    # base reward 1.0 minus think_penalty(0.1) * think_token_count(3) = 0.7
    s = rl_verl.score_single_turn(
        _BreakdownEnv(), "7", {"gt": "7"}, tok=object(), thinking=True,
        prompt_opened_thinking=True, think_penalty=0.1,
    )
    assert abs(s - 0.7) < 1e-9


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_env_error_is_zero():
    s = rl_verl.score_single_turn(
        _RaisingEnv(), "x", {"gt": "1"}, tok=None, thinking=False,
        prompt_opened_thinking=False, think_penalty=0.0,
    )
    assert s == 0.0


# --------------------- reward_metrics: per-name breakdown collection ---------------------
class _NamedBreakdownEnv:
    def scores_breakdown(self, graded, ex, state):
        hit = 1.0 if graded.strip() == ex["gt"] else 0.0
        return {"success": hit, "quality": 0.5, "total": hit}


class _BadTotalEnv:
    def scores_breakdown(self, graded, ex, state):
        return {"success": 1.0, "total": "not-a-number"}


class _RaisingRewardOnlyEnv:
    def reward(self, graded, ex, state):
        raise ValueError("grader is down")


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_collects_the_named_breakdown_for_reward_metrics():
    breakdowns: list[dict | None] = []
    score = rl_verl.score_single_turn(
        _NamedBreakdownEnv(), "7", {"gt": "7"}, tok=None, thinking=False,
        prompt_opened_thinking=False, think_penalty=0.0, breakdowns=breakdowns,
    )
    assert score == 1.0
    assert breakdowns == [{"success": 1.0, "quality": 0.5, "total": 1.0}]


@pytest.mark.usefixtures("_identity_graded")
def test_a_scalar_reward_env_contributes_no_breakdown_at_all():
    # appending {} for a scores_breakdown-less env would add a denominator under no numerators:
    # _mean_named_reward_metrics divides by every scored completion, so an env mixing the two
    # shapes -- or a run with none at all -- would publish every name shrunk toward 0.
    breakdowns: list[dict | None] = []
    score = rl_verl.score_single_turn(
        _RewardOnlyEnv(), "the answer is 7", {"gt": "7"}, tok=None, thinking=False,
        prompt_opened_thinking=False, think_penalty=0.0, breakdowns=breakdowns,
    )
    assert score == 2.5
    assert breakdowns == []


@pytest.mark.usefixtures("_identity_graded")
def test_a_scalar_reward_env_contributes_nothing_when_its_grading_fails_either():
    # the failure path is where the scores_breakdown gate actually bites: without it, a run whose
    # env has no named components at all would still append None per failed completion, and
    # _latest_named_reward_metrics' outage branch would then republish the LAST run's names as a
    # flat 0 for an env that never reported them.
    breakdowns: list[dict | None] = []
    score = rl_verl.score_single_turn(
        _RaisingRewardOnlyEnv(), "x", {"gt": "1"}, tok=None, thinking=False,
        prompt_opened_thinking=False, think_penalty=0.0, breakdowns=breakdowns,
    )
    assert score == 0.0
    assert breakdowns == []


@pytest.mark.usefixtures("_identity_graded")
def test_a_failed_grading_records_none_so_it_counts_as_a_zero():
    # trl's contract: a completion that failed to grade still scored 0.0, and must pull the mean of
    # every name the OTHER completions reported down with it. dropping it silently would report the
    # surviving completions' average as if the whole generation had earned it.
    breakdowns: list[dict | None] = []
    score = rl_verl.score_single_turn(
        _RaisingEnv(), "x", {"gt": "1"}, tok=None, thinking=False,
        prompt_opened_thinking=False, think_penalty=0.0, breakdowns=breakdowns,
    )
    assert score == 0.0
    assert breakdowns == [None]


@pytest.mark.usefixtures("_identity_graded")
def test_an_unusable_total_records_no_named_components():
    # float(total) raising IS a failed grading -- the completion scores 0.0. crediting its named
    # components would report metrics for a completion that earned nothing.
    breakdowns: list[dict | None] = []
    score = rl_verl.score_single_turn(
        _BadTotalEnv(), "x", {"gt": "1"}, tok=None, thinking=False,
        prompt_opened_thinking=False, think_penalty=0.0, breakdowns=breakdowns,
    )
    assert score == 0.0
    assert breakdowns == [None]


# ------------------------------- reward rpc bridge -------------------------------
def test_reward_server_round_trip():
    server, url = rl_verl.start_reward_server(
        lambda idx, s: float(idx) + len(s), example_count=4
    )
    try:
        body = json.dumps({"index": 3, "solution_str": "abcd"}).encode()
        req = urllib.request.Request(
            url + "/score", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            got = json.loads(r.read().decode())
        assert got["score"] == 7.0  # 3 + len("abcd")
    finally:
        server.shutdown()


@pytest.mark.parametrize("index", [-1, 2])
def test_reward_server_rejects_out_of_range_index_before_lookup(index):
    examples = [{"name": "first"}, {"name": "last"}]
    scored = []

    def scorer(index, solution_str):
        scored.append(examples[index]["name"])
        return 1.0

    server, url = rl_verl.start_reward_server(scorer, example_count=len(examples))
    try:
        body = json.dumps({"index": index, "solution_str": "answer"}).encode()
        req = urllib.request.Request(
            url + "/score", data=body, headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 400
        assert scored == []
    finally:
        server.shutdown()


def test_reward_bridge_lookup_failure_raises(monkeypatch):
    def missing_example(idx, solution_str):
        raise IndexError(idx)

    server, url = rl_verl.start_reward_server(missing_example, example_count=100)
    try:
        monkeypatch.setenv("TEST_FLASH_VERL_REWARD_URL", url)
        ns: dict = {}
        src = rl_verl.render_reward_module("TEST_FLASH_VERL_REWARD_URL")
        exec(compile(src, "<reward>", "exec"), ns)
        with pytest.raises(RuntimeError, match="reward bridge request failed"):
            ns["compute_score"](
                "flash_env",
                "answer",
                "unused",
                extra_info={"index": 99},
            )
    finally:
        server.shutdown()


def test_reward_server_scorer_can_capture_samples():
    # the #607 per-step dump relies on the scoring closure capturing recent completions; verify the
    # reward-server -> scorer -> rolling-buffer path populates in order.
    captured: list = []
    lock = threading.Lock()

    def scorer(idx, sol):
        with lock:
            captured.append((sol, float(len(sol))))
            del captured[:-64]
        return float(len(sol))

    server, url = rl_verl.start_reward_server(scorer, example_count=3)
    try:
        for i in range(3):
            body = json.dumps({"index": i, "solution_str": f"c{i}"}).encode()
            req = urllib.request.Request(
                url + "/score", data=body, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=10).read()
        assert [c[0] for c in captured] == ["c0", "c1", "c2"]
    finally:
        server.shutdown()


def test_entropy_quantile_shim_is_emitted_only_when_masking_is_requested():
    # 1.0 (and unset) means "keep every token", which is verl's own behavior -- emitting a shim then
    # would patch the loss to do nothing. only a real quantile may put a patch on the import path.
    assert rl_verl.render_entropy_quantile_shim(None) == ""
    assert rl_verl.render_entropy_quantile_shim(1.0) == ""
    source = rl_verl.render_entropy_quantile_shim(0.2)
    assert source
    # trl thresholds at 1 - top_entropy_quantile: keeping the top 20% means cutting at the 0.8
    # quantile. carrying the flash value through unconverted would keep the BOTTOM 20% instead.
    assert "_flash_entropy_threshold_q = 0.8" in source
    assert rl_verl._ENTROPY_QUANTILE_MARKER in source


def test_entropy_quantile_shim_refuses_to_wrap_itself_twice():
    # double-wrapping would take the top quantile OF the top quantile: with 0.2 that trains on ~4%
    # of tokens instead of 20%, and nothing in the logs would show it. verified numerically against
    # verl's real ppo_loss -- without this guard the loss drifted from -0.0428 to -0.0251.
    source = rl_verl.render_entropy_quantile_shim(0.2)
    assert '_flash_entropy_masked", False)' in source
    assert "_flash_entropy_masked_ppo_loss._flash_entropy_masked = True" in source


def test_entropy_quantile_shim_masks_only_the_policy_gradient_term():
    # trl multiplies per_token_loss by the entropy mask and THEN adds the kl term, so kl and the
    # entropy bonus stay on the full response mask. masking inside ppo_loss itself would shrink all
    # three. the shim therefore wraps get_policy_loss_fn, not the aggregation.
    source = rl_verl.render_entropy_quantile_shim(0.2)
    assert "_flash_losses.get_policy_loss_fn = _flash_masked_policy_loss_fn" in source
    assert 'kwargs["response_mask"] = _flash_high_entropy_mask' in source
    # equivalence also needs a mask-independent denominator, which is why flash pins this mode.
    assert _overrides_cfg()["loss_agg_mode"] == "seq-mean-token-sum-norm"


def test_entropy_quantile_overrides_enable_verl_entropy_and_stay_off_by_default():
    # the shim reads model_output["entropy"], which verl only populates when calculate_entropy is
    # set. flash's recipe has entropy_coeff 0, so nothing else would turn it on.
    assert "actor_rollout_ref.actor.calculate_entropy=True" in rl_verl.build_verl_overrides(
        _overrides_cfg(entropy_quantile=0.2)
    )
    assert "actor_rollout_ref.actor.calculate_entropy=True" not in rl_verl.build_verl_overrides(
        _overrides_cfg()
    )


def test_resolve_grpo_inputs_no_longer_rejects_entropy_quantile():
    # the guard this replaces raised on any entropy_quantile < 1.0. the shim implements the masking,
    # so the resolver must pass the value through instead of failing the run.
    source = inspect.getsource(rl_verl._resolve_grpo_inputs)
    assert "is not yet supported" not in source.split("entropy_quantile")[1].split("\n\n")[0]
    assert '"entropy_quantile": entropy_quantile' in source


def test_stop_sequences_shim_is_emitted_only_when_stop_strings_are_requested():
    assert rl_verl.render_stop_sequences_shim(()) == ""
    source = rl_verl.render_stop_sequences_shim(("</answer>", "\n\nQ:"))
    assert source
    # the exact list must survive into the child verbatim, escaping included -- a mangled delimiter
    # would silently never fire and the run would look normal.
    assert "_flash_stop_sequences = ['</answer>', '\\n\\nQ:']" in source
    assert rl_verl._STOP_SEQUENCES_MARKER in source


def test_stop_sequences_shim_patches_the_per_sample_params_not_the_config():
    # _run_agent_loop receives the per-sample dict AFTER verl applies its validate/greedy overrides,
    # so patching there keeps stop strings on eval rollouts too -- matching trl, where the stop list
    # lives in generation_kwargs and is not swapped out for validation.
    source = rl_verl.render_stop_sequences_shim(("</answer>",))
    assert "AgentLoopWorker._run_agent_loop" in source
    assert 'params["stop"] = list(_flash_stop_sequences)' in source
    # the dict is copied before mutation: verl reuses sample_sampling_params across the batch.
    assert "params = dict(sampling_params)" in source


def test_stop_sequences_shim_refuses_to_wrap_itself_twice():
    source = rl_verl.render_stop_sequences_shim(("</answer>",))
    assert '_flash_stop_patched", False)' in source


def test_image_pad_ban_shim_is_emitted_only_on_a_multimodal_job():
    # a text run has no image-pad token to ban, and injecting a logit_bias key into every rollout's
    # sampling params would change sampling on jobs that never asked for it.
    assert rl_verl.render_image_pad_ban_shim(None) == ""
    source = rl_verl.render_image_pad_ban_shim(151655)
    assert source
    assert "151655" in source
    assert rl_verl._IMAGE_PAD_BAN_MARKER in source
    # -100.0 matches the bias trl applies through generation_kwargs (rl.py), so the two backends
    # suppress the token equally hard rather than one of them merely discouraging it.
    assert "logit_bias[_flash_image_pad_token_id] = -100.0" in source
    # same reason as the stop shim: verl reuses the params dict across the batch.
    assert "params = dict(sampling_params)" in source


def test_image_pad_ban_and_stop_shims_both_apply_to_the_same_method():
    # both wrap AgentLoopWorker._run_agent_loop. each guards itself with its own marker attribute,
    # and a shared marker would make whichever ran second silently no-op -- leaving either stop
    # strings or the image-pad ban missing with no error. executing both is the only way to catch
    # that: the sources look correct in isolation either way.
    import asyncio
    import sys
    from types import ModuleType

    seen: dict = {}

    class _AgentLoopWorker:
        async def _run_agent_loop(self, sampling_params, *args, **kwargs):
            seen.update(sampling_params)
            return "ok"

    # both shims import the same verl module; hand them one stub so the two patches stack on the
    # very same function object, exactly as they do in the child.
    agent_loop_module = ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop_module.AgentLoopWorker = _AgentLoopWorker
    package = ModuleType("verl.experimental.agent_loop")
    package.agent_loop = agent_loop_module
    stubs = {
        "verl": ModuleType("verl"),
        "verl.experimental": ModuleType("verl.experimental"),
        "verl.experimental.agent_loop": package,
        "verl.experimental.agent_loop.agent_loop": agent_loop_module,
    }
    source = rl_verl.render_stop_sequences_shim(("</answer>",)) + rl_verl.render_image_pad_ban_shim(
        151655
    )
    for name, module in stubs.items():
        sys.modules[name] = module
    try:
        exec(compile(source, "sitecustomize.py", "exec"), {})
        asyncio.run(_AgentLoopWorker()._run_agent_loop({"temperature": 1.0}))
    finally:
        for name in stubs:
            sys.modules.pop(name, None)

    assert seen["stop"] == ["</answer>"]
    assert seen["logit_bias"] == {151655: -100.0}
    # the untouched key proves each patch COPIED the dict rather than replacing it wholesale.
    assert seen["temperature"] == 1.0


def test_image_pad_ban_shim_is_composed_into_the_sitecustomize(monkeypatch):
    source = inspect.getsource(rl_verl.run_rl_verl)
    assert 'render_image_pad_ban_shim(inp["image_pad_token_id"])' in source
    combined = rl_verl.render_stop_sequences_shim(("</answer>",)) + rl_verl.render_image_pad_ban_shim(
        151655
    )
    compile(combined, "sitecustomize.py", "exec")


def test_stop_sequences_gate_off_truncated_completion_masking():
    # main couples these: stop-string rollouts do not end on EOS, so masking truncated completions
    # would wrongly drop every one of them. the verl resolver must inherit that coupling, not
    # re-derive it.
    source = inspect.getsource(rl_verl._resolve_grpo_inputs)
    assert "_w.grpo_mask_truncated_completions(_t)" in source
    assert not W.grpo_mask_truncated_completions(
        SimpleNamespace(stop_sequences=("</answer>",))
    )
    assert W.grpo_mask_truncated_completions(SimpleNamespace(stop_sequences=()))


def test_all_shims_compose_into_one_sitecustomize():
    # python imports sitecustomize once, so a second file would never load. the renderers must
    # concatenate into a single source rather than each owning a file.
    source = inspect.getsource(rl_verl.run_rl_verl)
    assert 'render_entropy_quantile_shim(inp["entropy_quantile"])' in source
    assert 'render_stop_sequences_shim(inp["stop_sequences"])' in source
    assert 'render_structured_outputs_shim(inp["structured_outputs"])' in source
    assert 'render_exact_save_steps_shim(inp["save_at_steps"], inp["steps"])' in source
    combined = (
        rl_verl.render_entropy_quantile_shim(0.2)
        + rl_verl.render_stop_sequences_shim(("</answer>",))
        + rl_verl.render_structured_outputs_shim({"json": {"type": "object"}})
        + rl_verl.render_exact_save_steps_shim((7, 13), 20)
    )
    compile(combined, "sitecustomize.py", "exec")


def test_exact_save_steps_shim_is_emitted_only_when_exact_saves_are_requested():
    # without exact saves verl's own save_every cadence is already what flash wants, so there is
    # nothing to suppress and the shim must stay out of the child's import path.
    assert rl_verl.render_exact_save_steps_shim((), 20) == ""
    source = rl_verl.render_exact_save_steps_shim((7, 13), 20)
    assert source
    assert rl_verl._EXACT_SAVE_STEPS_MARKER in source


def test_exact_save_steps_shim_keeps_required_steps_and_the_final_step():
    # the gcd of the required steps makes verl save a SUPERSET (gcd(7,13) == 1 is a full-state dump
    # every step). the shim drops the writes flash never asked for -- but losing a required step
    # fails the run, and losing the final step leaves the final publish with no source checkpoint.
    source = rl_verl.render_exact_save_steps_shim((7, 13), 20)
    assert "_flash_required_save_steps = frozenset((7, 13))" in source
    assert "_flash_total_steps = 20" in source
    assert (
        "if step not in _flash_required_save_steps and step != _flash_total_steps:" in source
    )
    # it reads the step off the instance: verl's _save_checkpoint takes no step argument.
    assert "step = int(self.global_steps)" in source


def test_exact_save_steps_shim_refuses_to_wrap_itself_twice():
    source = rl_verl.render_exact_save_steps_shim((7,), 20)
    assert '"_flash_save_patched", False' in source
    assert "_flash_save_patched = True" in source


def test_structured_outputs_shim_is_emitted_only_when_a_constraint_is_requested():
    assert rl_verl.render_structured_outputs_shim(None) == ""
    assert rl_verl.render_structured_outputs_shim({}) == ""
    spec = {"json": {"type": "object", "properties": {"a": {"type": "string"}}}}
    source = rl_verl.render_structured_outputs_shim(spec)
    assert source
    assert repr(spec) in source
    assert rl_verl._STRUCTURED_OUTPUTS_MARKER in source


def test_structured_outputs_shim_wraps_the_spec_rather_than_passing_a_raw_dict():
    # the whole point: vllm ACCEPTS a raw dict, passes _verify_args, and then stores a plain dict
    # with no .json attribute -- constraining nothing, silently. trl wraps it in its colocate path,
    # which is why flash's trl path hands over a plain dict; on verl nothing wraps it, so the shim
    # must, or the run trains unconstrained and looks completely normal.
    source = rl_verl.render_structured_outputs_shim({"json": {"type": "object"}})
    assert "StructuredOutputsParams as _FlashStructuredOutputsParams" in source
    assert (
        'params["structured_outputs"] = _FlashStructuredOutputsParams(**_flash_structured_outputs)'
        in source
    )
    # built per request, not once: vllm resolves the backend on first use and caches it on the
    # instance, so a shared object would leak that resolution across requests.
    assert "params = dict(sampling_params)" in source


def test_structured_outputs_shim_refuses_to_wrap_itself_twice():
    source = rl_verl.render_structured_outputs_shim({"json": {"type": "object"}})
    assert '"_flash_so_patched", False' in source
    assert "_flash_so_patched = True" in source


def _load_kl_ref_engine():
    """exec the kl-reference shim against a stub verl engine and hand back the patched class.

    the shim rebinds FSDPEngine._build_lora_module and .disable_adapter on import, so a stub that
    stands in for verl's real class is enough to exercise both halves without a gpu.
    """
    import sys
    from types import ModuleType

    class _FSDPEngine:
        def __init__(self, module):
            self.module = module

        def _build_lora_module(self, module):
            return module

        def disable_adapter(self):
            raise AssertionError("the shim must replace disable_adapter, not defer to it")

    impl = ModuleType("verl.workers.engine.fsdp.transformer_impl")
    impl.FSDPEngine = _FSDPEngine
    fsdp_pkg = ModuleType("verl.workers.engine.fsdp")
    fsdp_pkg.transformer_impl = impl
    stubs = {
        "verl": ModuleType("verl"),
        "verl.workers": ModuleType("verl.workers"),
        "verl.workers.engine": ModuleType("verl.workers.engine"),
        "verl.workers.engine.fsdp": fsdp_pkg,
        "verl.workers.engine.fsdp.transformer_impl": impl,
    }
    for name, module in stubs.items():
        sys.modules[name] = module
    try:
        source = rl_verl.render_kl_ref_adapter_shim(True)
        exec(compile(source, "sitecustomize.py", "exec"), {})
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
    return impl.FSDPEngine


def test_kl_ref_adapter_shim_is_emitted_only_for_a_warm_start():
    # a fresh-start run has no sft adapter to anchor to, so verl's bare-base reference is already
    # what flash wants and the patch must stay out of the child's import path.
    assert rl_verl.render_kl_ref_adapter_shim(False) == ""
    source = rl_verl.render_kl_ref_adapter_shim(True)
    assert source
    assert rl_verl._KL_REF_ADAPTER_MARKER in source


def test_kl_ref_adapter_shim_is_wired_only_when_warm_start_and_kl_are_both_on():
    # with kl off no reference logprob is ever consumed, so patching disable_adapter would add a
    # failure mode and buy nothing. both conditions have to gate the renderer, not just warm start.
    source = inspect.getsource(rl_verl.run_rl_verl)
    assert "render_kl_ref_adapter_shim(" in source
    assert 'bool(inp["warmstart_adapter"]) and float(inp["kl_coef"]) > 0' in source
    combined = rl_verl.render_exact_save_steps_shim((7, 13), 20) + rl_verl.render_kl_ref_adapter_shim(
        True
    )
    compile(combined, "sitecustomize.py", "exec")


def test_kl_ref_adapter_shim_anchors_the_reference_to_the_warm_start_adapter():
    # the defect this removes: verl sets ref_in_actor whenever lora is active (always, on flash) and
    # marks the reference pass no_lora_adapter=True, which engine_workers turns into
    # engine.disable_adapter() -- the BARE BASE. on a warm-started run the kl term would then pull
    # the policy away from the sft adapter the run was told to continue. asserting on the rendered
    # source cannot catch that; only running the patched engine and comparing the three forwards
    # (sft / base / trained policy) can.
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.q_proj(x)

    torch.manual_seed(0)
    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"])
    model = peft.get_peft_model(_Tiny(), config)
    # lora_B initializes to zeros, which makes a fresh adapter a no-op EQUAL to the base -- a
    # fixture left that way could not tell "anchored to sft" from "anchored to base" at all.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.1)

    x = torch.randn(2, 8)
    with torch.no_grad():
        sft_out = model(x).clone()
        with model.disable_adapter():
            base_out = model(x).clone()
    assert not torch.allclose(sft_out, base_out), "fixture cannot discriminate sft from base"

    params_before = set(dict(model.named_parameters()))
    state_before = set(model.state_dict())
    engine = _load_kl_ref_engine()(model)
    engine._build_lora_module(model)

    with torch.no_grad():
        # a training step moves only the trainable default adapter; the frozen snapshot must not
        # follow it, or the anchor drifts with the policy and constrains nothing.
        for name, param in model.named_parameters():
            if ".default." in name and "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.5)
        trained_out = model(x).clone()
        with engine.disable_adapter():
            ref_out = model(x).clone()
        after_out = model(x).clone()

    assert torch.allclose(ref_out, sft_out), "kl reference is not the warm-start adapter"
    assert not torch.allclose(ref_out, base_out), "kl reference fell back to the bare base"
    assert not torch.allclose(ref_out, trained_out), "kl reference drifted with the policy"
    # the policy forward has to come back bit-exact: the reference pass runs inside training.
    assert torch.equal(after_out, trained_out), "policy forward not restored after the reference"
    # non-persistent buffers, not a second adapter's parameters. new named_parameters would be
    # flattened by fsdp and trained by the optimizer; new state_dict keys would reach verl's merger,
    # which hand-builds the shipped adapter from every "lora_" key and derives target_modules from
    # key.split(".")[-3] -- a second adapter's keys resolve to lora_A/lora_B there.
    assert not set(dict(model.named_parameters())) - params_before
    assert not set(model.state_dict()) - state_before


def test_kl_ref_adapter_shim_refuses_to_run_without_a_snapshot():
    # both guards exist because the alternative is silent: an unpatched or half-applied snapshot
    # would leave the reference on the bare base, and the run would look completely healthy while
    # training against the wrong anchor. they must raise, never fall back.
    pytest.importorskip("torch")
    import torch

    engine_cls = _load_kl_ref_engine()

    class _NoAdapterWeights(torch.nn.Module):
        # peft-shaped, but no ModuleDict holds the snapshot's leaves: nothing gets demoted.
        def __init__(self):
            super().__init__()
            self.peft_config = {"default": SimpleNamespace(r=4)}
            self.active_adapter = "default"

        def add_adapter(self, name, config):
            self.peft_config[name] = config

    module = _NoAdapterWeights()
    with pytest.raises(RuntimeError, match="no adapter weights to freeze"):
        engine_cls(module)._build_lora_module(module)

    class _NoSnapshot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.peft_config = {"default": SimpleNamespace(r=4)}

    with pytest.raises(RuntimeError, match="flash kl reference adapter missing"):
        engine_cls(_NoSnapshot()).disable_adapter()


def test_reasoning_parser_override_needs_both_thinking_and_a_constraint():
    # engine half. verl spreads engine_kwargs.vllm straight into AsyncEngineArgs, where
    # reasoning_parser is a real field, so this needs a plain hydra override and no shim.
    spec = {"json": {"type": "object"}}
    key = "+actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=deepseek_r1"
    assert key in rl_verl.build_verl_overrides(
        _overrides_cfg(thinking=True, structured_outputs=spec)
    )
    # thinking off -> no reasoning phase to protect; no constraint -> the grammar gate never runs.
    for off in (
        {"thinking": False, "structured_outputs": spec},
        {"thinking": True, "structured_outputs": None},
    ):
        assert not [
            o for o in rl_verl.build_verl_overrides(_overrides_cfg(**off)) if "reasoning_parser" in o
        ]


def test_build_verl_overrides_enable_fused_linear_ce():
    # 32k GRPO must not materialize [tokens, vocab] logits; fused torch-backend linear-CE
    # computes logprobs from hidden states in chunks (numerically exact).
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    assert "actor_rollout_ref.model.use_fused_kernels=True" in o
    assert "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch" in o


def test_model_revision_resolves_pinned_snapshot_for_verl():
    # model_revision no longer fails closed: prefetch pins the revision and verl gets the pinned
    # snapshot dir as model.path (a bare repo id would resolve the cached "main" ref offline).
    import inspect

    resolver_src = inspect.getsource(rl_verl._resolve_grpo_inputs)
    assert "model_revision pinning is not yet supported" not in resolver_src
    run_src = inspect.getsource(rl_verl.run_rl_verl)
    assert "local_files_only=True" in run_src
    assert 'revision=inp["model_revision"]' in run_src


def test_resolve_verl_loggers_console_when_no_api_key(monkeypatch):
    # no WANDB_API_KEY -> console only, and no wandb probe of the verl interpreter.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(
        rl_verl.subprocess, "run",
        lambda *a, **k: pytest.fail("must not probe verl env without an api key"),
    )
    assert rl_verl._resolve_verl_loggers("/verl/bin/python") == "console"


def test_resolve_verl_loggers_enables_wandb_only_when_verl_env_has_it(monkeypatch):
    # api key set AND wandb importable in the verl interpreter -> wandb logger enabled.
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setattr(rl_verl.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert rl_verl._resolve_verl_loggers("/verl/bin/python") == "console,wandb"


def test_resolve_verl_loggers_falls_back_to_console_when_verl_env_lacks_wandb(monkeypatch):
    # api key set but wandb missing in the verl interpreter -> console only (never aborts verl).
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setattr(rl_verl.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert rl_verl._resolve_verl_loggers("/verl/bin/python") == "console"


# ------------------------------- resume (VERL-018) -------------------------------
def test_build_verl_overrides_enables_resume_mode():
    # without resume_mode=auto verl ignores a staged checkpoint and silently restarts at step 0.
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    assert "trainer.resume_mode=auto" in o


def test_restore_verl_resume_is_a_noop_without_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_verl._w, "hf_resume_checkpoint", lambda *a, **k: None)
    assert rl_verl._restore_verl_resume(str(tmp_path)) == 0
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


def test_restore_verl_resume_stages_the_checkpoint_where_verl_looks(tmp_path, monkeypatch):
    src = tmp_path / "checkpoint-7"
    (src / "actor").mkdir(parents=True)
    (src / "actor" / "model.safetensors").write_text("weights")
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(rl_verl._w, "hf_resume_checkpoint", lambda *a, **k: str(src))

    assert rl_verl._restore_verl_resume(str(local_dir)) == 7
    # verl discovers the checkpoint through this marker plus the global_step_N layout.
    assert (local_dir / "latest_checkpointed_iteration.txt").read_text().strip() == "7"
    assert (local_dir / "global_step_7" / "actor" / "model.safetensors").read_text() == "weights"


def test_restore_verl_resume_rejects_an_unparseable_checkpoint_path(tmp_path, monkeypatch):
    bad = tmp_path / "not-a-checkpoint"
    bad.mkdir()
    monkeypatch.setattr(rl_verl._w, "hf_resume_checkpoint", lambda *a, **k: str(bad))
    with pytest.raises(RuntimeError, match="invalid GRPO resume checkpoint path"):
        rl_verl._restore_verl_resume(str(tmp_path / "ckpt"))


def _write_step(local_dir, step):
    d = local_dir / f"global_step_{step}"
    (d / "actor").mkdir(parents=True)
    (local_dir / "latest_checkpointed_iteration.txt").write_text(str(step))
    return d


def test_resume_uploader_uploads_each_completed_step(tmp_path):
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    seen = []
    uploader = rl_verl._VerlResumeUploader(str(local_dir), resume_step=0)

    import flash.engine.worker.rl_verl as mod

    original = mod._w.upload_resume_checkpoint
    mod._w.upload_resume_checkpoint = lambda step, path, **k: seen.append(int(step))
    try:
        _write_step(local_dir, 4)
        uploader.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and 4 not in seen:
            time.sleep(0.05)
        _write_step(local_dir, 8)
        while time.monotonic() < deadline and 8 not in seen:
            time.sleep(0.05)
        uploader.stop()
    finally:
        mod._w.upload_resume_checkpoint = original
    assert seen == [4, 8]


def test_resume_uploader_skips_the_step_it_resumed_from(tmp_path):
    # that checkpoint is already durable on hf; re-uploading it wastes the upload slot.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    seen = []
    import flash.engine.worker.rl_verl as mod

    original = mod._w.upload_resume_checkpoint
    mod._w.upload_resume_checkpoint = lambda step, path, **k: seen.append(int(step))
    try:
        _write_step(local_dir, 5)
        uploader = rl_verl._VerlResumeUploader(str(local_dir), resume_step=5)
        uploader.start()
        time.sleep(0.5)
        uploader.stop()
    finally:
        mod._w.upload_resume_checkpoint = original
    assert seen == []


def test_resume_uploader_never_fails_the_run_on_an_upload_error(tmp_path):
    # the policy is still trained and published; a failed resume upload only costs restart distance.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    import flash.engine.worker.rl_verl as mod

    def boom(step, path, **k):
        raise RuntimeError("hf is down")

    original = mod._w.upload_resume_checkpoint
    mod._w.upload_resume_checkpoint = boom
    try:
        _write_step(local_dir, 2)
        uploader = rl_verl._VerlResumeUploader(str(local_dir), resume_step=0)
        uploader.start()
        time.sleep(0.5)
        uploader.stop()  # must not raise
    finally:
        mod._w.upload_resume_checkpoint = original
    assert 2 in uploader.processed_steps


def test_grpo_gradient_check_rejects_a_run_whose_rewards_never_varied():
    # the defect this guards: a run on a constant-reward environment reaches state=done with a
    # written checkpoint and an exported adapter, and its reward history looks perfectly healthy,
    # but every advantage was 0 so the published adapter equals its initialization.
    with pytest.raises(RuntimeError, match="zero advantage spread"):
        rl_verl._check_grpo_had_a_gradient([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])


def test_grpo_gradient_check_admits_a_run_with_spread_on_any_step():
    # zero spread on some steps is legitimate (a converged run, or one unlucky all-equal group), so
    # the guard must key on "no step ever had spread" rather than "some step had none".
    rl_verl._check_grpo_had_a_gradient([0.4, 0.6], [0.0, 1.5])
    rl_verl._check_grpo_had_a_gradient([0.4], [2.0])


def test_grpo_gradient_check_rejects_reward_metrics_without_advantage_metrics():
    # both series are parsed off the same verl log line, so advantages missing while rewards are
    # present means the parse regressed. without this the spread check silently cannot fire.
    with pytest.raises(RuntimeError, match="no advantage metrics"):
        rl_verl._check_grpo_had_a_gradient([1.0], [])


def test_grpo_gradient_check_still_rejects_an_unconsulted_reward_bridge():
    with pytest.raises(RuntimeError, match="never consulted"):
        rl_verl._check_grpo_had_a_gradient([], [])


def test_advantage_spread_is_parsed_from_a_real_verl_step_line():
    # the guard is only as good as this parse: verl namespaces both keys under critic/ even though
    # grpo runs without a critic, and emits them outside its use_critic branch
    # (verl/trainer/ppo/metric_utils.py), so they are present for every grpo step.
    line = (
        "step:1 - critic/rewards/mean:1.0 - critic/rewards/max:1.0 - critic/rewards/min:1.0 - "
        "critic/advantages/mean:0.0 - critic/advantages/max:0.0 - critic/advantages/min:0.0 - "
        "actor/pg_loss:0.0"
    )
    adv_max = verl_common.parse_verl_metric(line, "critic/advantages/max")
    adv_min = verl_common.parse_verl_metric(line, "critic/advantages/min")
    assert adv_max == 0.0
    assert adv_min == 0.0
    # this is the exact shape of the run in ISSUES VERL-064: healthy reward, zero spread.
    with pytest.raises(RuntimeError, match="zero advantage spread"):
        rl_verl._check_grpo_had_a_gradient([1.0], [adv_max - adv_min])

    varied = line.replace("critic/advantages/max:0.0", "critic/advantages/max:0.67").replace(
        "critic/advantages/min:0.0", "critic/advantages/min:-0.33"
    )
    spread = verl_common.parse_verl_metric(varied, "critic/advantages/max") - verl_common.parse_verl_metric(
        varied, "critic/advantages/min"
    )
    assert spread > 0.0
    rl_verl._check_grpo_had_a_gradient([0.5], [spread])


def test_run_rl_verl_wires_the_gradient_check_into_the_publish_path():
    # a helper nothing calls is not a guard. assert the training path actually invokes it, and that
    # it does so before the adapter export rather than after a publish has already happened.
    source = inspect.getsource(rl_verl.run_rl_verl)
    assert (
        "_check_grpo_had_a_gradient(reward_history, adv_spread_history, resumed=bool(resume_step))"
        in source
    )
    assert source.index("_check_grpo_had_a_gradient") < source.index("_export_peft_adapter")
    # and that the spread series it passes is actually collected from the child's output.
    assert 'parse_verl_metric(line, "critic/advantages/max")' in source
    assert 'parse_verl_metric(line, "critic/advantages/min")' in source


def test_grpo_gradient_check_abstains_for_a_resumed_run():
    # a run resuming at step 9 of 10 observes ONE step; if that group ties, the spread history is
    # all-zero even though the restored weights carry nine steps of real updates. rejecting it would
    # throw away a correctly trained policy, so the resumed case abstains from the spread verdict.
    rl_verl._check_grpo_had_a_gradient([1.0], [0.0], resumed=True)
    # abstaining is scoped to the spread verdict only: the parse/wiring checks still apply, because
    # a missing metric stream is a regression no matter where training started.
    with pytest.raises(RuntimeError, match="no advantage metrics"):
        rl_verl._check_grpo_had_a_gradient([1.0], [], resumed=True)
    with pytest.raises(RuntimeError, match="never consulted"):
        rl_verl._check_grpo_had_a_gradient([], [], resumed=True)
    # and a FRESH run with the same all-zero history is still rejected -- the abstention must be
    # about the resume boundary, not a weakening of the guard.
    with pytest.raises(RuntimeError, match="zero advantage spread"):
        rl_verl._check_grpo_had_a_gradient([1.0], [0.0], resumed=False)


def test_resume_uploader_withholds_deployables_until_spread_appears():
    # the uploader publishes servable adapters WHILE training runs, so a degenerate-reward run would
    # make untrained adapters durable minutes before the end-of-run guard fails the run.
    spread: list[float] = []
    uploader = rl_verl._VerlResumeUploader(
        "/nonexistent",
        resume_step=0,
        required_steps=(1,),
        had_gradient=lambda: any(s > 0.0 for s in spread),
    )
    assert uploader._deployable_allowed() is False
    spread.append(0.0)  # a step ran, but its group tied: still no gradient evidence
    assert uploader._deployable_allowed() is False
    spread.append(1.25)
    assert uploader._deployable_allowed() is True


def test_resume_uploader_treats_an_unreadable_gradient_signal_as_closed():
    # this gate decides whether an artifact becomes durable and servable, so a callback that raises
    # must not be read as permission to publish.
    def boom() -> bool:
        raise RuntimeError("signal unavailable")

    uploader = rl_verl._VerlResumeUploader("/nonexistent", resume_step=0, had_gradient=boom)
    assert uploader._deployable_allowed() is False
    # and no callback at all means no gate, which is the resume-only configuration.
    assert rl_verl._VerlResumeUploader("/nonexistent", resume_step=0)._deployable_allowed() is True


def test_run_rl_verl_gates_midtraining_deployables_and_exempts_resumes():
    source = inspect.getsource(rl_verl.run_rl_verl)
    # the gate must be wired into the uploader, not merely available on it.
    assert "had_gradient=(" in source
    # a resumed run publishes as before: its restored weights already carry earlier updates that
    # this worker's spread history cannot speak for.
    assert "if resume_step" in source.split("had_gradient=(")[1].split(")")[0] + ")"
    # the spread series must be declared before the uploader closes over it, or the closure raises
    # NameError the first time the drain thread consults it.
    assert source.index("adv_spread_history: list[float] = []") < source.index("_VerlResumeUploader(")


def _patch_stage_and_publish(monkeypatch, staged: list[int], published: list[int]) -> None:
    """record staging and publication separately, without running model_merger or touching hf.

    they are patched as two seams because the production code separates them: staging is bounded by
    verl's checkpoint retention, publication by the gradient gate.
    """
    monkeypatch.setattr(
        rl_verl._VerlResumeUploader,
        "_stage_deployable",
        lambda self, step, path: (staged.append(int(step)), f"{path}-adapter")[1],
    )
    monkeypatch.setattr(
        rl_verl._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (
            published.append(int(step)),
            self.published_steps.add(step),
        )[0],
    )


def test_withheld_required_step_still_uploads_resume_state_exactly_once(tmp_path, monkeypatch):
    # withholding gates PUBLICATION only. the resume upload is internal retry scaffolding, and with
    # exact save_at_steps these are often the only on-disk checkpoints -- skipping it would leave a run
    # preempted before the first nonzero spread with nothing to resume from. neither the upload nor the
    # staging may repeat on every 0.5s sweep while the step waits for the gate.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    uploaded: list[int] = []
    staged: list[int] = []
    published: list[int] = []
    spread: list[float] = []
    monkeypatch.setattr(
        rl_verl._w, "upload_resume_checkpoint",
        lambda step, path, **k: uploaded.append(int(step)), raising=False,
    )
    _patch_stage_and_publish(monkeypatch, staged, published)
    _write_step(local_dir, 3)
    uploader = rl_verl._VerlResumeUploader(
        str(local_dir), resume_step=0, required_steps=(3,),
        had_gradient=lambda: any(s > 0.0 for s in spread),
    )
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and 3 not in uploaded:
            time.sleep(0.05)
        # the gate is shut, so publication is withheld -- but resume state IS durable, and the
        # adapter is already staged out of verl's reach.
        assert uploaded == [3]
        assert staged == [3]
        assert published == []
        time.sleep(1.5)  # several sweeps: neither the upload nor the export may repeat
        assert uploaded == [3]
        assert staged == [3]
        spread.append(2.0)  # gradient evidence appears; the held-back deployable is released
        while time.monotonic() < deadline and not published:
            time.sleep(0.05)
        assert published == [3]
        assert uploaded == [3]
        assert staged == [3]
    finally:
        uploader.stop()
    uploader.raise_if_incomplete()


def test_a_permanently_withheld_step_fails_the_run_and_does_not_hang_stop(tmp_path, monkeypatch):
    # the gate never opening must not wedge stop() waiting for a step it will never release, and the
    # run must still fail rather than silently ship without the customer's requested deployable.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(
        rl_verl._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
    )
    _patch_stage_and_publish(monkeypatch, [], [])
    _write_step(local_dir, 3)
    uploader = rl_verl._VerlResumeUploader(
        str(local_dir), resume_step=0, required_steps=(3,), had_gradient=lambda: False
    )
    uploader.start()
    time.sleep(0.5)
    uploader.stop()  # must return, not hang
    with pytest.raises(RuntimeError, match="required saves were not durably published"):
        uploader.raise_if_incomplete()


def test_gate_opening_just_before_stop_still_publishes_rather_than_failing_on_timing(
    tmp_path, monkeypatch
):
    # the drain loop samples the gate once per sweep. if the main thread records the run's first
    # positive spread and calls stop() after that sample, publishing nothing would fail a genuinely
    # trained run for no reason but thread scheduling. the sweep that observes stop() must therefore
    # still act on the gate as it stands then.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    published: list[int] = []
    monkeypatch.setattr(
        rl_verl._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
    )
    _patch_stage_and_publish(monkeypatch, [], published)
    _write_step(local_dir, 3)
    gate = [False]
    # flips the gate open on the sweep *after* the first sample, mimicking the main thread recording
    # spread while the drain loop is already past its own read.
    reads = [0]

    def _had_gradient():
        reads[0] += 1
        return gate[0]

    uploader = rl_verl._VerlResumeUploader(
        str(local_dir), resume_step=0, required_steps=(3,), had_gradient=_had_gradient
    )
    uploader.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and reads[0] < 1:
        time.sleep(0.01)
    gate[0] = True  # spread appears, then the run ends immediately
    uploader.stop()
    assert published == [3]
    uploader.raise_if_incomplete()


def test_resumed_required_step_can_still_publish_its_withheld_deployable(tmp_path, monkeypatch):
    # a previous worker resume-uploads a required checkpoint while withholding its adapter behind the
    # gradient gate, so the step is durable as resume state but NOT published. seeding processed_steps
    # with resume_step would hide it from _pending forever, and completeness would then fail a run on
    # the one step this worker is both able and allowed to publish.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    published: list[int] = []
    monkeypatch.setattr(
        rl_verl._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
    )
    _patch_stage_and_publish(monkeypatch, [], published)
    _write_step(local_dir, 4)
    # resumed at exactly the required step, and no adapter on hf for it, so it stays uncredited.
    monkeypatch.setattr(rl_verl, "_deployable_adapter_on_hf", lambda step: False)
    uploader = rl_verl._VerlResumeUploader(str(local_dir), resume_step=4, required_steps=(4,))
    uploader.credit_durable_required_steps(4)
    uploader.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not published:
        time.sleep(0.01)
    uploader.stop()
    assert published == [4]
    uploader.raise_if_incomplete()


def test_checkpoint_appearing_at_stop_is_uploaded_before_the_exit(tmp_path, monkeypatch):
    # verl advances latest_checkpointed_iteration.txt right up to the moment the child exits, so the
    # newest resume checkpoint can appear after the drain's last scan but before stop(). exiting
    # without sweeping that checkpoint would drop durable work a preemption then has to redo. resume
    # upload is not gated, and with the gradient gate shut nothing may be PUBLISHED.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    uploaded: list[int] = []
    staged: list[int] = []
    monkeypatch.setattr(
        rl_verl._w,
        "upload_resume_checkpoint",
        lambda step, path, **k: uploaded.append(int(step)),
        raising=False,
    )
    monkeypatch.setattr(
        rl_verl._VerlResumeUploader,
        "_stage_deployable",
        lambda self, step, path: (staged.append(int(step)), f"{path}-adapter")[1],
    )
    monkeypatch.setattr(
        rl_verl._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (_ for _ in ()).throw(AssertionError("gate is shut")),
    )
    # the checkpoint must become visible AFTER a sweep has already decided what to scan, with stop
    # already set -- writing it between sweeps does not discriminate, because the next top-of-loop
    # scan picks it up either way. the tracker read is that boundary: _pending only accepts steps at
    # or below the value it returns, so a step written right after that read is invisible to the
    # sweep holding it and visible to the next one.
    real_completed = rl_verl._VerlResumeUploader._completed_step
    raced = [False]

    def _completed_then_race(self):
        value = real_completed(self)
        if not raced[0]:
            raced[0] = True
            # verl finishes step 5 and advances its tracker here, then the child exits and the main
            # thread calls stop() -- all after this sweep already read the pre-step-5 tracker.
            _write_step(local_dir, 5)
            self._stop.set()
        return value

    monkeypatch.setattr(
        rl_verl._VerlResumeUploader, "_completed_step", _completed_then_race, raising=True
    )
    uploader = rl_verl._VerlResumeUploader(
        str(local_dir), resume_step=0, required_steps=(5,), had_gradient=lambda: False
    )
    uploader.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and uploader._thread.is_alive():
        time.sleep(0.01)
    uploader.stop()
    assert uploaded == [5]
    # staged out of verl's reach on the same sweep, so the gate opening later can still publish it.
    assert staged == [5]


def test_required_step_publishes_after_verl_prunes_its_checkpoint(tmp_path, monkeypatch):
    # verl keeps only max_actor_ckpt_to_keep=3 actor checkpoints, so with four or more required steps
    # written before the first varying-reward group it deletes the earliest source while its
    # deployable is still withheld. deferring the EXPORT until the gate opens would then leave that
    # step unpublishable and fail an otherwise valid run, so the export is staged under export_root
    # (flash's own workdir, outside verl's retention) and only the upload waits for the gate.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    staged: list[int] = []
    published: list[int] = []
    gate = [False]
    monkeypatch.setattr(
        rl_verl._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
    )
    def _stage_requiring_its_source(self, step, path):
        # the real _stage_deployable runs model_merger over <path>/actor, so it cannot succeed once
        # verl has pruned that directory. asserting it here is what makes this test fail on the
        # actual defect -- an unpublishable required step -- rather than on bookkeeping.
        if not os.path.isdir(path):
            raise AssertionError(f"staged step {step} after verl pruned {path}")
        staged.append(int(step))
        return f"{path}-adapter"

    monkeypatch.setattr(
        rl_verl._VerlResumeUploader, "_stage_deployable", _stage_requiring_its_source
    )
    monkeypatch.setattr(
        rl_verl._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (
            published.append(int(step)),
            self.published_steps.add(step),
        )[0],
    )
    for step in (1, 2, 3, 4):
        (local_dir / f"global_step_{step}").mkdir()
    _write_step(local_dir, 4)
    uploader = rl_verl._VerlResumeUploader(
        str(local_dir), resume_step=0, required_steps=(1, 2, 3, 4), had_gradient=lambda: gate[0]
    )
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and len(staged) < 4:
            time.sleep(0.01)
        # nothing may be servable yet: the gate is still shut.
        assert published == []
        # verl prunes the oldest checkpoints now that step 4 has landed -- exactly what strands a
        # step whose export was deferred until the gate opened.
        for step in (1, 2):
            shutil.rmtree(local_dir / f"global_step_{step}")
        gate[0] = True  # the first varying-reward group finally arrives
        while time.monotonic() < deadline and len(published) < 4:
            time.sleep(0.01)
    finally:
        uploader.stop()
    # every required step publishes, including the two whose verl checkpoints no longer exist. this
    # is asserted before `staged` so a deferred export fails here, on the unpublishable step, rather
    # than on the bookkeeping that led to it.
    assert published == [1, 2, 3, 4]
    uploader.raise_if_incomplete()
    assert staged == [1, 2, 3, 4]


def test_staging_failure_does_not_strand_an_earlier_publishable_step(tmp_path, monkeypatch):
    # a sweep can find several new checkpoints at once, and exporting one of them can fail (a corrupt
    # shard, a full disk, an OOM in model_merger). publishing only after the whole sweep finished let
    # that failure abort the thread with earlier, fully exported adapters still local-only -- and the
    # same window swallows a preemption during the resume upload that runs between the two. each step
    # is therefore made durable as soon as it is staged and permitted.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    published: list[int] = []
    monkeypatch.setattr(
        rl_verl._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
    )

    def _stage_failing_on_step_2(self, step, path):
        if int(step) == 2:
            raise RuntimeError("model_merger ran out of memory")
        return f"{path}-adapter"

    monkeypatch.setattr(rl_verl._VerlResumeUploader, "_stage_deployable", _stage_failing_on_step_2)
    monkeypatch.setattr(
        rl_verl._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (
            published.append(int(step)),
            self.published_steps.add(step),
        )[0],
    )
    for step in (1, 2):
        (local_dir / f"global_step_{step}").mkdir()
    _write_step(local_dir, 2)
    uploader = rl_verl._VerlResumeUploader(
        str(local_dir), resume_step=0, required_steps=(1, 2), had_gradient=lambda: True
    )
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and uploader._error is None:
            time.sleep(0.01)
    finally:
        uploader.stop()
    # step 1 was exported before step 2 failed, so it must already be durable. the run still fails --
    # step 2 was required -- but a retry does not have to redo step 1, and step 1 is servable.
    assert published == [1]
    with pytest.raises(RuntimeError, match="verl resume uploader failed"):
        uploader.raise_if_incomplete()


def test_zero_gradient_is_reported_before_a_withheld_required_save(tmp_path, monkeypatch):
    # a zero-spread run withholds every required deployable by design. checking completeness first
    # would raise on artifacts the gate is deliberately holding, reporting a checkpoint-publication
    # failure -- the symptom -- instead of the constant reward signal that caused it.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    _write_step(local_dir, 6)
    monkeypatch.setattr(
        rl_verl._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
    )
    _patch_stage_and_publish(monkeypatch, [], [])
    uploader = rl_verl._VerlResumeUploader(
        str(local_dir), resume_step=0, required_steps=(6,), had_gradient=lambda: False
    )
    uploader.start()
    uploader.stop()
    # both failures are live: the deployable was withheld, and the run produced no spread. the
    # gradient verdict must be the one that speaks.
    with pytest.raises(RuntimeError, match="zero advantage spread on all"):
        rl_verl._check_grpo_had_a_gradient([0.5, 0.5], [0.0, 0.0], resumed=False)
    with pytest.raises(RuntimeError, match="required saves were not durably published"):
        uploader.raise_if_incomplete()
    # ordering is asserted at the call site: the verdict precedes stop()/raise_if_incomplete().
    source = inspect.getsource(rl_verl.run_rl_verl)
    verdict = source.index("_check_grpo_had_a_gradient(reward_history")
    completeness = source.index("resume_uploader.raise_if_incomplete()")
    assert verdict < completeness


def test_train_notes_report_whether_the_run_resumed():
    # without this a resumed run is indistinguishable from a fresh one in train_meta (trl reports it).
    inp = _notes_inp()
    common = _notes_common()
    fresh = rl_verl._build_verl_train_notes(inp, **common)
    assert fresh["resumed"] is False
    resumed = rl_verl._build_verl_train_notes(inp, **common, resumed=True)
    assert resumed["resumed"] is True


def _notes_inp():
    return {
        "epochs": 2,
        "group_size": 4,
        "kl_coef": 0.0,
        "entropy_quantile": None,
        "stop_sequences": (),
        "structured_outputs": None,
        "temperature": 1.0,
        "top_p": 1.0,
        "ppo_epochs": 1,
        "verl_total_epochs": 3,
        "seed": 7,
        "max_completion": 512,
        "prompts_per_step": 8,
        "engine_len": 4096,
    }


def _notes_common():
    return {"steps_run": 3, "retained_prompts": 8, "reward_history": [0.5], "loss_curve": [0.1]}


def test_train_notes_carry_the_trl_observability_fields():
    # the console is uploaded only on FAILURE, so a successful run's train_meta is the sole record
    # of how it ran. the trl path reports these; without them a verl run cannot be compared to a
    # trl one, and the fp8-kv decision (resolved per-card at runtime) leaves no trace at all.
    notes = rl_verl._build_verl_train_notes(
        _notes_inp(),
        **_notes_common(),
        download_seconds=12.5,
        device_peak_gpu_gb=71.25,
        fp8_kv=True,
        wandb_project="acme",
        wandb_run_name="flash-rl-run123",
    )
    assert notes["download_seconds"] == 12.5
    assert notes["vllm_kv_cache_dtype"] == "fp8"
    assert notes["wandb_project"] == "acme"
    assert notes["wandb_run_name"] == "flash-rl-run123"
    # verl trains out-of-process, so nvidia-smi is the only reading that sees the trainer: both keys
    # carry the same device figure rather than a torch-allocated subset that would read ~0 here.
    assert notes["peak_gpu_gb"] == 71.25
    assert notes["device_peak_gpu_gb"] == 71.25
    # chalk installs against an in-process trainer.model, which verl does not have.
    assert notes["chalk_kernels"] is None
    # trl counts generated tokens from a padded upper bound; verl uses observed response lengths.
    # without the flag the two backends' token counts read as comparable when they are not.
    assert notes["gen_tokens_is_upper_bound"] is False


def test_train_notes_report_bf16_kv_when_fp8_did_not_engage():
    # fp8 is gated on cc>=8.9 AND a non-gdn model, so "requested" and "engaged" are not the same
    # thing. reporting fp8 unconditionally would claim a memory saving the run never got.
    notes = rl_verl._build_verl_train_notes(_notes_inp(), **_notes_common(), fp8_kv=False)
    assert notes["vllm_kv_cache_dtype"] is None


def test_train_notes_omit_wandb_identity_when_wandb_is_off():
    # verl logs from its own interpreter, so flash's in-process wandb.run is empty on this path and
    # the names come from the config. recording them when the logger is off would point a reader at
    # a dashboard run that was never created.
    notes = rl_verl._build_verl_train_notes(_notes_inp(), **_notes_common())
    assert notes["wandb_project"] is None
    assert notes["wandb_run_name"] is None
    # a sampler that never saw a card must not report a fabricated zero-gb peak.
    assert notes["peak_gpu_gb"] is None
    assert notes["device_peak_gpu_gb"] is None


def test_verl_grpo_logs_to_the_runs_own_wandb_project_and_name():
    # a hardcoded project/experiment pair lands every grpo run in one wandb experiment, so
    # concurrent runs overwrite each other's curves and an explicit [wandb] project is ignored. the
    # sft and opd verl backends already resolve both from the spec.
    o = rl_verl.build_verl_overrides(
        _overrides_cfg(project_name="acme", experiment_name="flash-rl-run123")
    )
    assert "trainer.project_name=acme" in o
    assert "trainer.experiment_name=flash-rl-run123" in o
    assert "trainer.project_name=flash_verl" not in o
    assert "trainer.experiment_name=grpo" not in o


def test_verl_grpo_wandb_names_survive_hydra_special_characters():
    # a run name is user-settable via [wandb] run_name; an unquoted '=' or ',' would split the
    # override and hydra would compose a different key entirely.
    o = rl_verl.build_verl_overrides(_overrides_cfg(experiment_name="run=a,b"))
    assert 'trainer.experiment_name="run=a,b"' in o


def test_train_notes_record_the_batch_shape_one_step_consumed():
    # the trl path reports the batch shape, so without it a verl run's reward curve cannot be read
    # against a trl one: the same step count at a different batch size is a different experiment.
    notes = rl_verl._build_verl_train_notes(_notes_inp(), **_notes_common())
    assert notes["max_completion_len"] == 512
    assert notes["prompts_per_step"] == 8
    # ulysses shards along the sequence, so dp stays 1 and one optimizer step sees the whole batch.
    assert notes["generations_per_step"] == 8 * 4


def test_train_notes_report_token_bounded_batching_as_unset_not_fabricated():
    # trl fixes a per-device SEQUENCE count; verl bounds the backward pass by tokens, so a
    # micro-batch holds however many sequences fit and varies step to step. reporting a number here
    # would read as directly comparable to trl's when nothing enforces it.
    notes = rl_verl._build_verl_train_notes(_notes_inp(), **_notes_common())
    assert notes["per_device_train_batch_size"] is None
    assert notes["gradient_accumulation_steps"] is None
    # the bound that IS enforced gets recorded in their place.
    assert notes["ppo_max_token_len_per_gpu"] == 4096
    # trl pins vllm's prefill batch because it hardcodes 4096; this path sets no such override.
    assert notes["vllm_max_num_batched_tokens"] is None


# ------------------- capability guards: the specs verl grpo refuses -------------------
# these raises are the ONLY thing standing between a trl-supported job and a verl run that trains
# on a different contract. they had no regression coverage: every resolver test above drives the
# happy path, so deleting any guard left the suite green. each test below asserts one rejection by
# its own message, because a bare pytest.raises(RuntimeError) passes on any of the ~20 other raises
# in this resolver.


def _capability_env(*, multi_turn=False, is_tool_env=False, image_uri=None):
    """a minimal single-turn text env, optionally flipped to a shape verl grpo handles differently.

    ``image_uri`` must be a source the normalizer accepts offline (a data uri): a remote https url
    is rejected outright unless the trusted-dataset opt-in is set, so it would fail the test for
    the wrong reason.
    """

    class _Env:
        package_root = None

        def __init__(self):
            self.multi_turn = multi_turn
            self.is_tool_env = is_tool_env
            self.max_turns = 3 if multi_turn else 0

        def dataset(self):
            return [{"index": i} for i in range(8)]

        # the four calls the multi-turn bridge drives an env through. defined unconditionally so a
        # test can delete one and assert the capability gate catches it.
        def new_rollout_state(self, ex):
            return {}

        def record_model_turn(self, state, text):
            return None

        def env_reply(self, state):
            return [{"role": "user", "content": "reply"}]

        def rollout_done(self, state):
            return True

        def prompt_messages(self, ex):
            if image_uri:
                # record_has_images matches an image content BLOCK, so build the real shape rather
                # than a sentinel key the resolver would not see.
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"question {ex['index']}"},
                            {"type": "image_url", "image_url": {"url": image_uri}},
                        ],
                    }
                ]
            return [{"role": "user", "content": f"question {ex['index']}"}]

    return _Env()


def _capability_image_uri():
    """a 2x2 png as a data uri, the smallest source normalize_image_source accepts offline."""
    import base64
    import io

    image_module = pytest.importorskip("PIL.Image")
    out = io.BytesIO()
    image_module.new("RGB", (2, 2), (255, 0, 0)).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


class _CapabilityTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        # renders each message's content, the way a real template does. a constant string would
        # make the multi-turn glue probe unfindable and fail the pre-rollout template gate -- which
        # is the gate working, not the code under test being wrong.
        rendered = "".join(f"<|{m['role']}|>{m['content']}<eos>" for m in messages)
        return rendered + ("<|assistant|>" if kwargs.get("add_generation_prompt") else "")

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[1])

    def convert_tokens_to_ids(self, token):
        return 151655 if token == "<|image_pad|>" else 0


class _CapabilityProcessor:
    """the shape rl_verl asks of AutoProcessor on a multimodal job.

    it must expose ``tokenizer`` (the resolver reads pad/eos off it), render a chat template, and
    tokenize text+images together. the returned ids are what the prompt-budget filter measures, so
    a test that wants a row dropped controls it through the length here.
    """

    image_token_id = 151655

    def __init__(self, expanded_len=4):
        self.tokenizer = _CapabilityTokenizer()
        self.expanded_len = expanded_len
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, text=None, images=None, **kwargs):
        self.calls.append({"text": text, "images": images})
        return {"input_ids": [[1] * self.expanded_len]}


def _capability_resolve(monkeypatch, env, train=None, overrides=None, processor=None):
    """run the resolver against one env, with everything else on the supported path."""
    import transformers

    from flash.engine.worker._pkg import W as _PkgW
    from flash.spec import JobSpec

    _Tokenizer = _CapabilityTokenizer

    # a multimodal resolve builds a processor rather than a bare tokenizer. AutoProcessor
    # .from_pretrained would hit the hub, so stub it on the live module: the resolver's
    # `from transformers import AutoProcessor` runs inside the function and reads the attribute
    # at call time. imported unconditionally rather than probed out of sys.modules -- a guard that
    # skips the patch when transformers is not yet imported lets the resolver reach the real loader,
    # which fails on a missing backend instead of testing anything.
    monkeypatch.setattr(
        transformers,
        "AutoProcessor",
        SimpleNamespace(from_pretrained=lambda *a, **k: processor or _CapabilityProcessor()),
        raising=False,
    )

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
            "train": {"batch_size": 4, "epochs": 1, **(train or {})},
        }
    )
    monkeypatch.setattr(_PkgW, "JOB_SPEC", spec, raising=False)
    monkeypatch.setattr(_PkgW, "SEED", 42, raising=False)
    monkeypatch.setattr(_PkgW, "THINKING", False, raising=False)
    monkeypatch.setattr(_PkgW, "require_active_env", lambda: env, raising=False)
    monkeypatch.setattr(_PkgW, "grpo_overrides", lambda: dict(overrides or {}), raising=False)
    monkeypatch.setattr(_PkgW, "grpo_mask_truncated_completions", lambda t: False, raising=False)
    monkeypatch.setattr(_PkgW, "load_tokenizer", lambda *a, **k: _Tokenizer(), raising=False)
    monkeypatch.setattr(rl_verl, "seed_training_rngs", lambda seed: None)
    monkeypatch.setattr(rl_verl, "model_max_position_embeddings", lambda *a, **k: 32768)
    return rl_verl._resolve_grpo_inputs()


def test_capability_guard_rejects_tool_env(monkeypatch):
    # trl grpo hands tool schemas AND callables to the trainer; verl gets neither, so a tool env
    # would train against completions that never call a tool.
    with pytest.raises(RuntimeError, match="function-calling tool environments"):
        _capability_resolve(monkeypatch, _capability_env(is_tool_env=True))


def test_multi_turn_env_resolves_and_selects_the_flash_agent_loop(monkeypatch):
    # the inverse of the guard this replaces: a multi-turn env used to be refused outright and fall
    # back to trl. it must now resolve, and the resolution must reach the ONE override that decides
    # which agent loop verl runs -- on the stock single_turn_agent the episode would end after the
    # first assistant turn and every environment reply would be dropped.
    inp = _capability_resolve(monkeypatch, _capability_env(multi_turn=True))
    assert inp["multi_turn"] is True
    assert inp["max_turns"] == 3
    cfg = rl_verl._build_verl_training_cfg(
        inp,
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_id=inp["model_id"],
        thinking=False,
        loggers="console",
        fp8_kv=False,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    o = rl_verl.build_verl_overrides(cfg)
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=flash_grpo_multi_turn" in o


def test_single_turn_env_leaves_the_agent_loop_on_verl_default(monkeypatch):
    # the override must be GATED: emitting it on a single-turn job would route text rollouts
    # through the multi-turn bridge, which has no episode state for them.
    inp = _capability_resolve(monkeypatch, _capability_env())
    assert inp["multi_turn"] is False
    assert not [
        o for o in rl_verl.build_verl_overrides(_overrides_cfg()) if "default_agent_loop" in o
    ]


def test_multi_turn_env_missing_a_rollout_method_is_refused(monkeypatch):
    # the bridge calls exactly four env methods; a missing one would otherwise surface mid-rollout
    # on the first episode, after the gpu time is already spent.
    env = _capability_env(multi_turn=True)
    del type(env).env_reply
    with pytest.raises(RuntimeError, match="missing required rollout methods"):
        _capability_resolve(monkeypatch, env)


def test_multi_turn_env_without_a_turn_limit_is_refused(monkeypatch):
    # an unbounded episode cannot be budgeted: the response tensor is sized from the turn limit and
    # a runaway env would loop until the engine context is exhausted every single rollout.
    env = _capability_env(multi_turn=True)
    env.max_turns = 0
    with pytest.raises(RuntimeError, match="bounded turn limit"):
        _capability_resolve(monkeypatch, env)


def test_resolver_admits_image_prompts_and_carries_the_processor(monkeypatch):
    # the inverse of the guard this replaces: an image env used to be refused outright and fall
    # back to trl. it must now resolve, and it must resolve through a PROCESSOR -- a bare tokenizer
    # would under-count the prompt by the whole placeholder expansion. asserting the processor is
    # carried out (not merely that resolve returned) is what pins the multimodal path: a resolver
    # that quietly took the text branch would still return a valid dict.
    processor = _CapabilityProcessor()
    inp = _capability_resolve(
        monkeypatch,
        _capability_env(image_uri=_capability_image_uri()),
        processor=processor,
    )
    assert inp["multimodal"] is True
    assert inp["processor"] is processor
    assert inp["image_pad_token_id"] == 151655
    # every prompt was measured through the processor, with its decoded image attached: a call
    # carrying images=None would mean the pixels never reached the token count.
    assert len(processor.calls) == len(inp["prompts"])
    assert all(len(call["images"] or []) == 1 for call in processor.calls)


def test_multimodal_prompts_carry_descriptors_and_rendered_text(monkeypatch):
    # the parquet writer needs each prompt's image DESCRIPTORS and the thinking probe needs the
    # RENDERED text. both are produced only on the multimodal branch, so a branch that returned
    # bare messages would break the writer downstream rather than here.
    inp = _capability_resolve(
        monkeypatch, _capability_env(image_uri=_capability_image_uri())
    )
    first = inp["prompts"][0]
    assert len(first["images"]) == 1
    assert first["rendered"] == "prompt"
    # the image block is normalized to a bare {"type": "image"} marker: the source moved into the
    # descriptor list, which is what _materialize_verl_images later writes to disk.
    blocks = first["prompt"][0]["content"]
    assert {"type": "image"} in blocks


def test_multimodal_budget_filter_measures_the_expanded_prompt(monkeypatch):
    # verl RAISES on an over-budget multimodal prompt instead of truncating, so this filter is the
    # only thing between a long image prompt and a dead run. the tokenizer says 1 token; the
    # processor says the prompt is huge. the filter must believe the processor -- if it measured
    # with the tokenizer every row would be admitted and the run would die mid-rollout.
    processor = _CapabilityProcessor(expanded_len=10**6)
    with pytest.raises(ValueError, match="every training prompt exceeds"):
        _capability_resolve(
            monkeypatch,
            _capability_env(image_uri=_capability_image_uri()),
            processor=processor,
        )


def test_text_env_resolves_without_building_a_processor(monkeypatch):
    # the control for the three tests above: a text-only job must NOT pay for a processor, and must
    # not carry an image-pad ban into its rollouts. without this a resolver hardcoded to the
    # multimodal branch would pass every multimodal test above.
    inp = _capability_resolve(monkeypatch, _capability_env())
    assert inp["multimodal"] is False
    assert inp["processor"] is None
    assert inp["image_pad_token_id"] is None


def test_kl_anchored_warm_start_is_accepted(monkeypatch, tmp_path):
    # verl's kl reference is the bare base whenever lora is active, so warm-start + kl used to be
    # refused: the penalty would drag the policy away from the sft adapter the run was told to
    # continue. render_kl_ref_adapter_shim anchors the reference to that adapter instead, so the
    # combination now resolves. the kl coefficient arrives through grpo_overrides, so it must go
    # through the helper rather than being patched separately.
    import flash.engine.worker.adapter as _adapter_mod

    adapter_dir = tmp_path / "warmstart"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    monkeypatch.setattr(_adapter_mod, "_download_adapter", lambda ref: str(adapter_dir))

    inp = _capability_resolve(
        monkeypatch,
        _capability_env(),
        train={"init_from_adapter": "org/some-sft-adapter"},
        overrides={"kl_penalty_coef": 0.1},
    )
    assert inp["warmstart_adapter"]
    assert inp["kl_coef"] == pytest.approx(0.1)


def test_per_turn_credit_assignment_is_accepted_on_single_turn_envs(monkeypatch, capsys):
    # per_turn only diverges from per_episode when there is more than one assistant turn to credit:
    # trl reaches GRPOPerTurnTrainer solely through use_rollout_func, which requires is_multi_turn.
    # the multi-turn/tool guard above already rejects every env that could get there, so anything
    # reaching here is single-turn and the two modes are the same objective -- trl accepts the key on
    # exactly these envs. rejecting it would break configs that run correctly on the trl backend.
    inp = _capability_resolve(
        monkeypatch,
        _capability_env(),
        train={"credit_assignment": "per_turn"},
    )
    assert inp["max_prompt_len"] > 0
    assert "equivalent to per_episode" in capsys.readouterr().out


def test_default_credit_assignment_does_not_log_an_equivalence_note(monkeypatch, capsys):
    # the control: the note above must be tied to an explicitly non-default value, not printed on
    # every run. without this a hardcoded print would still satisfy the test above.
    _capability_resolve(
        monkeypatch,
        _capability_env(),
        train={"credit_assignment": "per_episode"},
    )
    assert "equivalent to per_episode" not in capsys.readouterr().out


def test_capability_guards_admit_the_supported_single_turn_text_env(monkeypatch):
    # the control: with none of the four shapes present the resolver must run to completion, so a
    # guard that fires on every env would fail here instead of passing the four tests above.
    inp = _capability_resolve(monkeypatch, _capability_env())
    assert inp["max_prompt_len"] > 0


# ---------------------- multi-turn child wiring ----------------------
# the parent resolves the episode contract; the child enforces it. everything between the two
# crosses one of exactly three channels -- env vars, copied-in modules, and the bridge's http
# routes -- and none of them is type-checked at either end. a typo'd key or a dropped copy does not
# fail here: it fails on the first rollout of a paid run, after the engine is already up.


def _multi_turn_inp(**over):
    """the keys multi_turn_child_env reads, at their resolved types."""
    return {
        "max_turns": 4,
        "engine_len": 8192,
        "stop_sequences": ("</answer>",),
        "eos_token_ids": frozenset({151645, 151643}),
        **over,
    }


def test_multi_turn_child_env_carries_every_variable_the_loop_reads():
    # the child reads these by name out of os.environ and has no defaults for the first three: a
    # missing FLASH_VERL_MULTITURN_URL/MAX_TURNS/MAX_MODEL_LEN raises KeyError inside the rollout.
    # asserted against the loop's OWN source rather than a hardcoded list, so renaming a key on one
    # side and not the other fails here instead of on the first episode.
    from flash.engine.worker import grpo_multiturn

    emitted = rl_verl.multi_turn_child_env(
        _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=False
    )
    read_by_child = set(
        re.findall(r"os\.environ(?:\.get)?[\[(]\"(FLASH_VERL_[A-Z_]+)\"", inspect.getsource(grpo_multiturn))
    )
    assert read_by_child, "the loop reads no FLASH_VERL_* variable; this test found nothing to pin"
    assert read_by_child <= set(emitted), (
        f"the child reads variables the parent never sets: {sorted(read_by_child - set(emitted))}"
    )


def test_multi_turn_child_env_registers_the_plugin_with_verl():
    # the agent-loop override names `flash_grpo_multi_turn`, but the name only exists once verl
    # imports the plugin -- which happens ONLY through this variable (import_external_libs). without
    # it the child dies at rollout build with an unregistered-loop error, having already paid for
    # engine startup.
    emitted = rl_verl.multi_turn_child_env(
        _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=False
    )
    assert emitted["VERL_USE_EXTERNAL_MODULES"] == "flash_grpo_plugin"
    # the module name must match the file actually copied in, or the import fails at child startup.
    assert ("grpo_plugin.py", "flash_grpo_plugin.py") in rl_verl.MULTI_TURN_CHILD_MODULES


def test_multi_turn_child_env_serializes_values_the_child_can_parse_back():
    # every value crosses as a string. the child json-loads two of them and int()s two others, so a
    # repr() or a str(frozenset) here would raise mid-rollout rather than at launch.
    emitted = rl_verl.multi_turn_child_env(
        _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=True
    )
    assert all(isinstance(value, str) for value in emitted.values())
    assert int(emitted["FLASH_VERL_MAX_TURNS"]) == 4
    assert int(emitted["FLASH_VERL_MAX_MODEL_LEN"]) == 8192
    assert json.loads(emitted["FLASH_VERL_STOP_SEQUENCES"]) == ["</answer>"]
    # sorted, not set-ordered: the child compares against this list every turn and an unstable order
    # would make halting depend on hash seed.
    assert json.loads(emitted["FLASH_VERL_EOS_TOKEN_IDS"]) == [151643, 151645]
    assert emitted["FLASH_VERL_THINKING"] == "1"
    assert (
        rl_verl.multi_turn_child_env(
            _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=False
        )["FLASH_VERL_THINKING"]
        == "0"
    )


def test_multi_turn_child_modules_are_copied_under_the_names_they_import_each_other_by(tmp_path):
    # each module falls back to a flat `flash_`-prefixed import of the next one. copying a file
    # under the wrong name leaves that fallback unresolvable, and the child's ImportError arrives
    # inside verl's plugin loader where it reads as a verl problem.
    written = rl_verl.copy_multi_turn_child_modules(str(tmp_path))
    names = {os.path.basename(path) for path in written}
    assert names == {name for _, name in rl_verl.MULTI_TURN_CHILD_MODULES}
    imported = set()
    for path in written:
        source = Path(path).read_text()
        assert source
        # every copy must parse standalone in the child interpreter.
        ast.parse(source)
        imported |= set(re.findall(r"from (flash_[a-z_]+) import", source))
    # the fallback import targets must be exactly the names copied in.
    assert imported <= {name.removesuffix(".py") for name in names}


def test_multi_turn_child_modules_do_not_import_flash(tmp_path):
    # flash is NOT importable in the verl interpreter (incompatible torch/vllm pins). each module
    # keeps an in-tree `from flash...` fallback for the parent's own lint and tests, so the rule is
    # that no flash import may be reachable without the flat one failing first -- i.e. every one of
    # them sits in an except ImportError handler.
    for path in rl_verl.copy_multi_turn_child_modules(str(tmp_path)):
        tree = ast.parse(Path(path).read_text())
        guarded = {
            id(node)
            for tries in ast.walk(tree)
            if isinstance(tries, ast.Try)
            for handler in tries.handlers
            for node in ast.walk(handler)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("flash."):
                assert id(node) in guarded, (
                    f"{os.path.basename(path)} imports {node.module} outside an ImportError "
                    "fallback; the child interpreter cannot import flash"
                )


def test_the_run_body_puts_the_shim_dir_on_the_child_path_for_multi_turn():
    # the copies above are useless unless shim_dir is importable. the condition used to be
    # `if shim_source:` -- true only when some OTHER feature wanted a sitecustomize patch, so a
    # plain multi-turn job copied three modules the child could never import. source-level because
    # the assignment sits inside run_rl_verl, past the subprocess launch.
    src = inspect.getsource(rl_verl.run_rl_verl)
    assert 'if shim_source or inp["multi_turn"]:' in src, (
        "PYTHONPATH is not extended for a multi-turn job with no other shim"
    )
    assert 'if inp["multi_turn"]:\n        copy_multi_turn_child_modules(shim_dir)' in src


# ---------------------- multi-turn bridge routes ----------------------
class _BridgeEnv:
    """the four calls MultiTurnBridge drives, recording what it was asked."""

    def __init__(self, *, replies=None, done_after=1, episode=1.0, max_episode_turns=None):
        self.replies = replies if replies is not None else [{"role": "user", "content": "next"}]
        self.done_after = done_after
        self.episode = episode
        self.max_episode_turns = max_episode_turns
        self.recorded: list[str] = []
        self.scored: list[dict] = []

    def new_rollout_state(self, example):
        state: dict = {"messages": [], "example": example}
        if self.max_episode_turns is not None:
            state["max_episode_turns"] = self.max_episode_turns
        return state

    def record_model_turn(self, state, text):
        self.recorded.append(text)
        state["messages"].append({"role": "assistant", "content": text})

    def env_reply(self, messages, state):
        state["messages"].extend(self.replies)
        return self.replies

    def rollout_done(self, state, max_turns):
        return len(self.recorded) >= self.done_after

    def rollout_rewards_many(self, items):
        from flash.envs.base import RolloutReward

        self.scored.extend(state for _, state in items)
        return [RolloutReward(episode=self.episode, turns=None) for _ in items]


def _bridge(env, *, max_turns=4, examples=None):
    return rl_verl.MultiTurnBridge(
        env, examples if examples is not None else [{"index": 0}, {"index": 1}], max_turns=max_turns
    )


def test_bridge_exposes_exactly_the_routes_the_child_posts_to():
    # the child posts to four literal paths and the server 404s anything else, with the failure
    # surfacing as a transport error mid-episode. pinned against the child's own source so a rename
    # on either side fails here.
    from flash.engine.worker import grpo_multiturn

    routes = set(_bridge(_BridgeEnv()).routes())
    posted = set(re.findall(r"\"(/multiturn/[a-z]+)\"", inspect.getsource(grpo_multiturn)))
    assert posted, "the child posts to no /multiturn path; this test found nothing to pin"
    assert posted <= routes, f"the child posts to unrouted paths: {sorted(posted - routes)}"


def test_bridge_start_mints_a_session_and_returns_the_turn_budget():
    env = _BridgeEnv()
    bridge = _bridge(env, max_turns=4)
    assert bridge.start({"index": 1, "session_id": "a"}) == {"max_turns": 4}
    assert bridge.open_sessions() == 1


def test_bridge_start_lets_a_per_example_budget_lower_the_cap_but_never_raise_it():
    # a per-example limit is the env asking for a SHORTER episode; honoring one that is longer would
    # let a single row overrun the response tensor the batch was sized for.
    assert _bridge(_BridgeEnv(max_episode_turns=2), max_turns=4).start(
        {"index": 0, "session_id": "a"}
    ) == {"max_turns": 2}
    assert _bridge(_BridgeEnv(max_episode_turns=99), max_turns=4).start(
        {"index": 0, "session_id": "a"}
    ) == {"max_turns": 4}
    # zero would make the loop skip generation entirely and score an empty transcript.
    assert _bridge(_BridgeEnv(max_episode_turns=0), max_turns=4).start(
        {"index": 0, "session_id": "a"}
    ) == {"max_turns": 1}


def test_bridge_start_rejects_an_out_of_range_index_before_touching_the_env():
    # the index selects which dataset row the episode scores against. python's negative indexing
    # would silently score the wrong row, so the check has to run before the lookup.
    env = _BridgeEnv()
    bridge = _bridge(env, examples=[{"index": 0}])
    for index in (-1, 1, 99):
        with pytest.raises(IndexError, match="outside"):
            bridge.start({"index": index, "session_id": "a"})
    assert bridge.open_sessions() == 0


def test_bridge_start_refuses_to_reuse_a_live_session_id():
    # session ids key the episode state. silently replacing one would strand the first episode's
    # transcript and score the second one twice.
    bridge = _bridge(_BridgeEnv())
    bridge.start({"index": 0, "session_id": "a"})
    with pytest.raises(KeyError, match="duplicate"):
        bridge.start({"index": 1, "session_id": "a"})


def test_bridge_step_records_the_turn_and_returns_the_env_reply():
    env = _BridgeEnv(done_after=2, replies=[{"role": "user", "content": "again"}])
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    out = bridge.step({"session_id": "a", "completion_text": "first"})
    assert env.recorded == ["first"]
    assert out == {"terminal": False, "messages": [{"role": "user", "content": "again"}]}


def test_bridge_step_does_not_show_the_env_an_unusable_turn():
    # a truncated or skipped turn is terminal on the child side too. recording it would append a
    # cut-off assistant message to the transcript that then gets SCORED as if the model produced it.
    for payload in (
        {"session_id": "a", "completion_text": "cut", "truncated": True},
        {"session_id": "a", "completion_text": "cut", "skip_reason": "no room for a turn"},
    ):
        env = _BridgeEnv()
        bridge = _bridge(env)
        bridge.start({"index": 0, "session_id": "a"})
        assert bridge.step(payload) == {"terminal": True, "messages": []}
        assert env.recorded == [], "an unusable turn reached the env"


def test_bridge_step_stops_before_asking_a_finished_env_for_a_reply():
    # rollout_done is checked between recording and replying: an env that ended the episode must not
    # be asked to produce another user message, which its contract does not define past terminal.
    env = _BridgeEnv(done_after=1)
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    assert bridge.step({"session_id": "a", "completion_text": "done"}) == {
        "terminal": True,
        "messages": [],
    }


def test_bridge_step_on_an_unknown_session_raises_rather_than_scoring_a_blank_episode():
    with pytest.raises(KeyError, match="unknown multi-turn session"):
        _bridge(_BridgeEnv()).step({"session_id": "ghost", "completion_text": "x"})


def test_bridge_score_returns_the_episode_reward_for_that_session():
    env = _BridgeEnv(episode=0.75)
    bridge = _bridge(env)
    bridge.start({"index": 1, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    assert bridge.score({"session_id": "a", "turn_count": 1}) == {"score": 0.75}
    # scored against the state the turns accumulated into, not a fresh one.
    assert env.scored[0]["messages"][0]["content"] == "answer"


def test_bridge_score_converts_an_unscorable_episode_to_zero(capsys):
    # nan is score_rollouts' unscorable marker. verl has no equivalent: a nan advantage propagates
    # through the group baseline and poisons every OTHER rollout in the group, so one ungradable
    # episode would corrupt the whole step.
    env = _BridgeEnv(episode=float("nan"))
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    assert bridge.score({"session_id": "a", "turn_count": 1}) == {"score": 0.0}
    assert "unscorable" in capsys.readouterr().out


def test_bridge_score_converts_a_non_finite_episode_to_zero():
    # inf reaches the group baseline the same way nan does, and is NOT caught by an isnan check.
    for episode in (float("inf"), float("-inf")):
        env = _BridgeEnv(episode=episode)
        bridge = _bridge(env)
        bridge.start({"index": 0, "session_id": "a"})
        bridge.step({"session_id": "a", "completion_text": "answer"})
        assert bridge.score({"session_id": "a", "turn_count": 1}) == {"score": 0.0}


def test_bridge_hands_each_scored_episode_to_the_sample_recorder():
    # multi-turn has no per-completion breakdown -- the env scores a whole episode to a scalar --
    # so the transcript IS the only thing this path can publish for `flash runs log`.
    recorded: list[tuple] = []
    env = _BridgeEnv(episode=0.75)
    bridge = rl_verl.MultiTurnBridge(
        env, [{"index": 0}], max_turns=4, on_episode_scored=lambda *row: recorded.append(row)
    )
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})

    assert len(recorded) == 1
    prompt, transcript, reward = recorded[0]
    assert reward == 0.75
    # the whole accumulated transcript, not just the last turn.
    assert [m["content"] for m in transcript] == ["answer"]
    assert prompt == []  # _BridgeEnv seeds no prompt; the shape is what matters here


def test_the_recorded_episode_is_the_zeroed_reward_not_the_raw_nan():
    # the sample carries the reward the rollout actually trained on. publishing nan here would show
    # a reward in the log that no advantage was ever computed from.
    recorded: list[tuple] = []
    bridge = rl_verl.MultiTurnBridge(
        _BridgeEnv(episode=float("nan")), [{"index": 0}], max_turns=4,
        on_episode_scored=lambda *row: recorded.append(row),
    )
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})

    assert [row[2] for row in recorded] == [0.0]


def test_the_recorded_transcript_is_a_snapshot_that_later_turns_cannot_mutate():
    # `step` appends to state["messages"] IN PLACE. handing the live list to the recorder would let
    # a concurrent episode's turn appear inside an already-published sample.
    recorded: list[tuple] = []
    env = _BridgeEnv(done_after=99)
    bridge = rl_verl.MultiTurnBridge(
        env, [{"index": 0}], max_turns=4, on_episode_scored=lambda *row: recorded.append(row)
    )
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "first"})
    bridge.score({"session_id": "a", "turn_count": 1})
    snapshot = list(recorded[0][1])
    bridge.step({"session_id": "a", "completion_text": "second"})

    assert list(recorded[0][1]) == snapshot
    assert "second" not in [m.get("content") for m in recorded[0][1]]


def test_the_episode_recorder_runs_outside_the_session_lock():
    # the recorder is the caller's sample buffer, which has its own lock. taking it while holding
    # the session lock inverts the single-turn path's order (buffer lock only, never nested) and
    # any grading that touches both deadlocks.
    observed: list[bool] = []
    env = _BridgeEnv()
    bridge = rl_verl.MultiTurnBridge(
        env, [{"index": 0}], max_turns=4,
        on_episode_scored=lambda *_: observed.append(bridge._lock.acquire(blocking=False)),
    )
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})

    assert observed == [True], "the session lock was still held when the recorder ran"
    bridge._lock.release()


def test_a_bridge_without_a_recorder_still_scores():
    # single-turn jobs build no bridge, but the recorder stays optional so the bridge is usable
    # (and testable) without a sample buffer behind it.
    bridge = _bridge(_BridgeEnv(episode=0.5))
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    assert bridge.score({"session_id": "a", "turn_count": 1}) == {"score": 0.5}


def test_bridge_close_releases_the_session():
    # every in-flight episode holds env state. a leak here grows for the whole run, and the child
    # closes in a finally precisely so a failed episode still frees it -- so close must tolerate a
    # session that was never started.
    bridge = _bridge(_BridgeEnv())
    bridge.start({"index": 0, "session_id": "a"})
    assert bridge.close({"session_id": "a"}) == {"closed": True}
    assert bridge.open_sessions() == 0
    assert bridge.close({"session_id": "a"}) == {"closed": True}


def test_bridge_routes_are_served_alongside_single_turn_scoring():
    # one server, one port: the child gets a single url and posts both /score and /multiturn/* to
    # it. mounting the bridge on its own server would leave the child's reward path pointing at a
    # port that only answers episodes.
    env = _BridgeEnv()
    bridge = _bridge(env)
    server, url = rl_verl.start_reward_server(
        lambda index, text: 1.0, example_count=2, multi_turn_bridge=bridge
    )
    try:
        def _post(path, payload):
            request = urllib.request.Request(
                url + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode())

        assert _post("/score", {"index": 0, "solution_str": "x"}) == {"score": 1.0}
        assert _post("/multiturn/start", {"index": 0, "session_id": "a"}) == {"max_turns": 4}
        _post("/multiturn/step", {"session_id": "a", "completion_text": "answer"})
        assert _post("/multiturn/score", {"session_id": "a", "turn_count": 1}) == {"score": 1.0}
        assert _post("/multiturn/close", {"session_id": "a"}) == {"closed": True}
    finally:
        server.shutdown()


def test_the_bridge_is_built_only_for_multi_turn_jobs():
    # a bridge on a single-turn job would expose episode routes with no episode state behind them,
    # and mounting it costs a lock the single-turn scoring path already has.
    # whitespace-normalized: the construction spans several lines, and what is under test is the
    # guard around it, not how the formatter wrapped the call.
    src = " ".join(inspect.getsource(rl_verl.run_rl_verl).split())
    assert src.count("MultiTurnBridge(") == 1
    assert 'MultiTurnBridge( env, rollout_examples, max_turns=int(inp["max_turns"]),' in src
    assert 'if inp["multi_turn"] else None' in src


# ---------------------- multi-turn response tensor width ----------------------
def test_multi_turn_widens_the_response_tensor_to_hold_a_whole_episode(monkeypatch):
    # verl right-pads response_ids to data.max_response_length and DROPS the overflow
    # (_pad_token_ids). on multi-turn the response is the whole transcript -- every assistant turn
    # plus every glued env reply -- so a max_completion-wide tensor would cut episodes mid-turn and
    # train on the fragment, silently. the width has to cover the longest episode the engine can
    # produce: the child stops generating at max_model_len, so engine_len minus the SHORTEST
    # admitted prompt bounds it.
    inp = _capability_resolve(monkeypatch, _capability_env(multi_turn=True))
    assert inp["max_response_len"] > inp["max_completion"], "the episode tensor was not widened"
    assert inp["max_response_len"] == inp["engine_len"] - min(
        int(p["prompt_len"]) for p in inp["prompts"]
    )
    # max_completion stays the PER-TURN cap, exactly as the trl driver uses it.
    assert inp["max_completion"] < inp["engine_len"]


def test_single_turn_leaves_the_response_tensor_at_the_completion_width(monkeypatch):
    # the control: one completion IS the response, so widening it would inflate every rollout's
    # padded tensor and the token budget derived from it for no reason.
    inp = _capability_resolve(monkeypatch, _capability_env())
    assert inp["max_response_len"] == inp["max_completion"]


def test_the_response_width_reaches_verls_config_rather_than_max_completion(monkeypatch):
    # the derivation is worthless if the override still emits max_completion. this is the one line
    # that decides how wide the tensor verl allocates actually is.
    inp = _capability_resolve(monkeypatch, _capability_env(multi_turn=True))
    cfg = rl_verl._build_verl_training_cfg(
        inp,
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_id=inp["model_id"],
        thinking=False,
        loggers="console",
        fp8_kv=False,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    assert f"data.max_response_length={inp['max_response_len']}" in rl_verl.build_verl_overrides(cfg)


# ---------------------- measured reward latency in train_meta ----------------------
def _resolved_inputs_for_notes(monkeypatch):
    """A resolved single-turn input dict, offline.

    Mirrors the resolver fixture above: _resolve_grpo_inputs needs a loaded env, a spec and
    a tokenizer, none of which exist in a unit test.
    """
    from flash.engine.worker._pkg import W
    from flash.spec import JobSpec

    class _Env:
        multi_turn = False
        is_tool_env = False

        def dataset(self):
            return [{"index": i} for i in range(33)]

        def prompt_messages(self, ex):
            return [{"role": "user", "content": f"question {ex['index']}"}]

    class _Tokenizer:
        pad_token = None
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1])

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
            "train": {"batch_size": 16, "epochs": 2},
        }
    )
    monkeypatch.setattr(W, "JOB_SPEC", spec, raising=False)
    monkeypatch.setattr(W, "SEED", 42, raising=False)
    monkeypatch.setattr(W, "THINKING", False, raising=False)
    monkeypatch.setattr(W, "require_active_env", lambda: _Env(), raising=False)
    monkeypatch.setattr(W, "grpo_overrides", lambda: {}, raising=False)
    monkeypatch.setattr(W, "grpo_mask_truncated_completions", lambda train: False, raising=False)
    monkeypatch.setattr(W, "load_tokenizer", lambda *args, **kwargs: _Tokenizer(), raising=False)
    monkeypatch.setattr(rl_verl, "seed_training_rngs", lambda seed: None)
    monkeypatch.setattr(rl_verl, "model_max_position_embeddings", lambda *a, **k: 40960)
    return rl_verl._resolve_grpo_inputs()


def _profile(seconds: float, *, trustworthy: bool = True):
    """A RewardProfile shaped like the profiler's real output."""
    from flash.engine.reward_profile import RewardProfile

    return RewardProfile(
        seconds_per_completion=seconds,
        samples=0 if not trustworthy else 3,
        degenerate=False,
        failures=0,
    )


def test_train_notes_record_the_measured_grading_latency(monkeypatch):
    """The measurement has to outlive the log line to be usable by anything downstream.

    A latency that only reaches stdout cannot price a run or place one: the cost model would keep
    using its single 1.0s average for an env just measured at 0.02s.
    """
    inp = _resolved_inputs_for_notes(monkeypatch)
    notes = rl_verl._build_verl_train_notes(
        inp,
        steps_run=5,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
        reward_profile=_profile(0.25),
        step_intervals=[10.0],
    )
    assert notes["reward_seconds_per_completion"] == 0.25


def test_train_notes_omit_an_untrustworthy_profile(monkeypatch):
    """A profile that measured nothing must record None, not a number.

    RewardProfile.trustworthy is False when no sample graded successfully; writing its 0.0 into
    train_meta would read downstream as a genuinely instant grader.
    """
    inp = _resolved_inputs_for_notes(monkeypatch)
    notes = rl_verl._build_verl_train_notes(
        inp,
        steps_run=5,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
        reward_profile=_profile(0.0, trustworthy=False),
        step_intervals=[10.0],
    )
    assert notes["reward_seconds_per_completion"] is None
    assert notes["reward_gpu_idle_fraction"] is None


def test_train_notes_report_idle_fraction_from_the_runs_own_step_wall(monkeypatch):
    """The idle share is computed from measured wall time, not from the cost model's estimate.

    Deriving it from the modelled step would report the estimator's own opinion back to it, which
    is exactly the number an operator would want to CHECK against reality.
    """
    inp = _resolved_inputs_for_notes(monkeypatch)
    completions = inp["prompts_per_step"] * inp["group_size"]
    # a step wall of 10s, of which grading accounts for 8s -> 80% idle.
    latency = 8.0 / completions
    notes = rl_verl._build_verl_train_notes(
        inp,
        steps_run=10,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
        reward_profile=_profile(latency),
        step_intervals=[10.0] * 10,
    )
    assert notes["reward_gpu_idle_fraction"] == pytest.approx(0.8, abs=0.01)


def test_idle_fraction_is_none_when_grading_exceeds_the_measured_step(monkeypatch):
    """Grading that fills the whole step leaves no gpu-bound remainder to divide.

    The profile and the observed wall disagree here (a warm-up latency that no longer holds, or a
    step wall dominated by something else), and neither can arbitrate, so the honest record is no
    reading rather than a fabricated 100%.
    """
    inp = _resolved_inputs_for_notes(monkeypatch)
    completions = inp["prompts_per_step"] * inp["group_size"]
    notes = rl_verl._build_verl_train_notes(
        inp,
        steps_run=10,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
        # 20s of grading inside a 10s measured step
        reward_profile=_profile(20.0 / completions),
        step_intervals=[10.0] * 10,
    )
    assert notes["reward_gpu_idle_fraction"] is None


def test_idle_fraction_ignores_steps_this_worker_did_not_run(monkeypatch):
    """A resumed run must divide by ITS steps, not by the checkpoint's absolute step number.

    steps_run comes from the checkpoint directory and counts every step the run has ever taken;
    the wall clock only ever covers this session. A worker resuming at 90 and training to 100
    walled ten steps, and charging those seconds against a hundred understates each step by 10x --
    which inflates the idle share toward 100% and would route the run to a co-location tier it
    cannot sustain. Feeding the measured intervals removes the term entirely.
    """
    inp = _resolved_inputs_for_notes(monkeypatch)
    completions = inp["prompts_per_step"] * inp["group_size"]
    latency = 8.0 / completions
    notes = rl_verl._build_verl_train_notes(
        inp,
        # the checkpoint says 100 steps; this worker observed ten, each a real 10s step.
        steps_run=100,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
        reward_profile=_profile(latency),
        step_intervals=[10.0] * 10,
    )
    # unchanged from the fresh-run case above: resume does not move the reading.
    assert notes["reward_gpu_idle_fraction"] == pytest.approx(0.8, abs=0.01)


def test_idle_fraction_is_unmoved_by_one_slow_step(monkeypatch):
    """A checkpoint save lands inside a single step interval and must not set the verdict.

    Every save-step gap carries an upload that the other steps never pay. Averaging lets one such
    step drag the whole reading -- here a 200s outlier among 10s steps would report ~28s/step and
    flip an 80% idle run to 29%. The median needs MOST steps to be slow before it moves.
    """
    inp = _resolved_inputs_for_notes(monkeypatch)
    completions = inp["prompts_per_step"] * inp["group_size"]
    latency = 8.0 / completions
    notes = rl_verl._build_verl_train_notes(
        inp,
        steps_run=10,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
        reward_profile=_profile(latency),
        step_intervals=[10.0] * 9 + [200.0],
    )
    assert notes["reward_gpu_idle_fraction"] == pytest.approx(0.8, abs=0.01)


def test_idle_fraction_is_none_when_no_step_was_timed(monkeypatch):
    """A run that never emitted two step lines has no measured step to divide by.

    One step line yields zero intervals by design (the span before it is engine startup), and a
    run that died in its first step yields none at all. Both must record no reading rather than
    fall back to a modelled step, which is the number this metric exists to check.
    """
    inp = _resolved_inputs_for_notes(monkeypatch)
    completions = inp["prompts_per_step"] * inp["group_size"]
    notes = rl_verl._build_verl_train_notes(
        inp,
        steps_run=1,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
        reward_profile=_profile(8.0 / completions),
        step_intervals=[],
    )
    assert notes["reward_gpu_idle_fraction"] is None


def test_step_intervals_exclude_the_span_before_the_first_step():
    """Engine init, weight load and cudagraph capture precede the first step line.

    That span is setup, paid once, and often larger than a step. Counting it as a step would
    inflate the step wall and understate the idle share on exactly the short runs where the
    startup cost dominates. N step lines bound N-1 steps, never N.
    """
    assert rl_verl._step_intervals([100.0, 110.0, 122.0]) == [10.0, 12.0]
    # a single step line bounds no whole step: nothing is known about what came before it or after.
    assert rl_verl._step_intervals([100.0]) == []
    assert rl_verl._step_intervals([]) == []


def test_the_profile_hook_returns_its_reading_to_the_caller():
    """_log_reward_profile must RETURN the profile, not only print it.

    Asserted against the real hook rather than a fixture: the wiring under test is that the run
    body can capture what the profiler measured.
    """

    class Env:
        def sft_completion(self, example):
            return [{"role": "assistant", "content": "an answer worth grading"}]

    profile = rl_verl._log_reward_profile(
        Env(), lambda index, completion: 1.0, [{"id": i} for i in range(4)], 32
    )
    assert profile is not None
    assert profile.trustworthy


def test_the_run_body_passes_the_measured_profile_into_train_meta():
    """The captured profile must actually reach the notes builder.

    Source-level, because reaching this line in a live run needs a gpu. Without it the hook could
    return a profile that the run body drops on the floor, and every other test here would still
    pass while train_meta always recorded None.
    """
    src = inspect.getsource(rl_verl.run_rl_verl)
    assert "_log_reward_profile(" in src, "the hook is never called"
    assert "reward_profile = " in src, "the hook's reading is discarded"
    assert "reward_profile=reward_profile" in src, "the reading never reaches train_meta"


def test_the_reward_profiler_is_skipped_on_multi_turn():
    """The profiler times the SINGLE-TURN grading path, which a multi-turn env does not have.

    Source-level for the same reason as above. Running it on a multi-turn env would call
    env.reward/scores_breakdown on one completion -- a call that env's contract does not define --
    and record the resulting number as if it described the episode reward path.
    """
    src = inspect.getsource(rl_verl.run_rl_verl)
    profile_call = src[src.index("reward_profile = ") : src.index("multi_turn_bridge = ")]
    assert 'if inp["multi_turn"]' in profile_call, "the profiler is not gated off multi-turn"
    assert "None" in profile_call, "multi-turn must record no profile rather than a wrong one"


# ---------------- reward observability: the buffers and the heartbeat drain ----------------
def _closure_namespace(name: str, namespace: dict):
    """Compile one closure out of run_rl_verl and return it bound to `namespace`.

    `_score` and `_reward_observability` are locals of a body that needs a model, a dataset and a
    verl interpreter to reach, so they are lifted out and run against fakes -- the same technique
    the trl `reward_fn` tests use. What executes is the shipped source, not a restatement of it.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(rl_verl.run_rl_verl)))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, f"<{name}>", "exec"), namespace)
    return namespace[name]


def _score_namespace(env, *, prompts=None, examples=None):
    namespace = {
        "env": env,
        "rollout_examples": examples if examples is not None else [{"gt": "7"}],
        "message_prompts": prompts if prompts is not None else ["prompt-0"],
        "recent_samples": [],
        "pending_named_breakdowns": [],
        "_samples_lock": threading.Lock(),
        "_MAX_PENDING_BREAKDOWNS": rl_verl._MAX_PENDING_BREAKDOWNS,
        "score_single_turn": rl_verl.score_single_turn,
        "tok": None,
        "inp": {"prompt_opened_thinking": False, "think_penalty": 0.0},
        "_w": SimpleNamespace(THINKING=False),
    }
    return _closure_namespace("_score", namespace), namespace


@pytest.mark.usefixtures("_identity_graded")
def test_score_buffers_the_rollout_sample_and_its_named_breakdown():
    score, ns = _score_namespace(_NamedBreakdownEnv())

    assert score(0, "7") == 1.0
    assert ns["recent_samples"] == [("prompt-0", "7", 1.0)]
    assert ns["pending_named_breakdowns"] == [{"success": 1.0, "quality": 0.5, "total": 1.0}]


@pytest.mark.usefixtures("_identity_graded")
def test_the_breakdown_buffer_is_bounded_when_the_heartbeat_stops_draining():
    # one generation is bounded, but nothing bounds how many generations pass between drains. an
    # unbounded list here grows for the whole run in a process that is already memory-tight.
    score, ns = _score_namespace(_NamedBreakdownEnv())
    for _ in range(rl_verl._MAX_PENDING_BREAKDOWNS + 50):
        score(0, "7")

    assert len(ns["pending_named_breakdowns"]) == rl_verl._MAX_PENDING_BREAKDOWNS
    assert len(ns["recent_samples"]) == 64  # the sample buffer keeps its own rolling bound


@pytest.mark.usefixtures("_identity_graded")
def test_score_grades_outside_the_buffer_lock():
    # grading calls user code and can block on i/o for seconds while verl scores many rollouts at
    # once. holding the buffer lock across it serializes every grading in the run behind the
    # slowest one, which on a slow grader is the whole reward wall.
    held: list[bool] = []

    class _Env:
        def reward(self, graded, ex, state):
            held.append(ns["_samples_lock"].acquire(blocking=False))
            if held[-1]:
                ns["_samples_lock"].release()
            return 1.0

    score, ns = _score_namespace(_Env())
    score(0, "x")

    assert held == [True], "the buffer lock was held across env grading"


@pytest.mark.usefixtures("_identity_graded")
def test_a_scalar_reward_run_publishes_no_named_metrics_at_all():
    # end to end for the empty case: a scores_breakdown-less env must reach the wire with the key
    # ABSENT, not with every name flattened to 0 by an empty-dict denominator.
    score, ns = _score_namespace(_RewardOnlyEnv(), examples=[{"gt": "7"}])
    score(0, "the answer is 7")

    assert ns["pending_named_breakdowns"] == []
    observability = _observability_namespace(ns)()
    assert "reward_metrics" not in observability
    assert len(observability["sampled_completions"]) == 1


def _observability_namespace(score_ns: dict, *, step: int = 5):
    from flash.engine.worker.heartbeat import (
        _latest_named_reward_metrics,
        reward_observability_fields,
    )
    from flash.engine.worker.rollout_samples import select_rollout_samples

    namespace = dict(score_ns)
    namespace.update(
        {
            "latest_named_metrics": score_ns.get("latest_named_metrics", {}),
            "step_box": [step],
            "_latest_named_reward_metrics": _latest_named_reward_metrics,
            "select_rollout_samples": select_rollout_samples,
            "reward_observability_fields": reward_observability_fields,
        }
    )
    return _closure_namespace("_reward_observability", namespace)


@pytest.mark.usefixtures("_identity_graded")
def test_the_heartbeat_publishes_averaged_metrics_and_bounded_samples():
    score, ns = _score_namespace(
        _NamedBreakdownEnv(),
        prompts=["p0", "p1", "p2", "p3"],
        examples=[{"gt": "7"}, {"gt": "7"}, {"gt": "9"}, {"gt": "9"}],
    )
    for index, completion in enumerate(["7", "7", "7", "7"]):
        score(index, completion)

    fields = _observability_namespace(ns)()

    # two of four completions matched their gt, so success averages 0.5 across the generation.
    assert fields["reward_metrics"] == {"success": 0.5, "quality": 0.5}
    assert len(fields["sampled_completions"]) == 3  # hard cap, four rollouts buffered
    assert {s["generated_at_step"] for s in fields["sampled_completions"]} == {5}
    assert [s["reward"] for s in fields["sampled_completions"]] == [1.0, 1.0, 0.0]


@pytest.mark.usefixtures("_identity_graded")
def test_the_drain_clears_pending_breakdowns_and_then_repeats_the_last_reading():
    # _latest_named_reward_metrics CLEARS the pending list. between generations there is nothing new
    # to average, and reporting {} there would blank the metric on every heartbeat that lands
    # mid-generation rather than holding the last real reading.
    score, ns = _score_namespace(_NamedBreakdownEnv())
    score(0, "7")
    observability = _observability_namespace(ns)

    assert observability()["reward_metrics"] == {"success": 1.0, "quality": 0.5}
    assert ns["pending_named_breakdowns"] == []
    assert observability()["reward_metrics"] == {"success": 1.0, "quality": 0.5}

    score(0, "wrong")
    assert observability()["reward_metrics"] == {"success": 0.0, "quality": 0.5}


@pytest.mark.usefixtures("_identity_graded")
def test_the_drain_and_the_sample_read_share_one_lock_acquisition():
    """The drain CLEARS the pending list, so it and the sample read must be one atomic section.

    Asserted on the closure's own source: reproducing the interleave needs a grading to land
    between the clear and the read, and a test that merely calls the closure passes with the lock
    split in two.
    """
    body = textwrap.dedent(inspect.getsource(rl_verl.run_rl_verl))
    body = body[body.index("def _reward_observability") :]
    body = body[: body.index("return reward_observability_fields")]

    assert body.count("with _samples_lock:") == 1
    assert body.index("with _samples_lock:") < body.index("_latest_named_reward_metrics(")
    assert body.index("_latest_named_reward_metrics(") < body.index("select_rollout_samples(")


def test_the_first_sample_bearing_heartbeat_is_forced():
    # the liveness daemon can claim a step before the stdout loop reaches it, and a step-gated stage
    # drops a second payload at an already-committed step. without force, the first heartbeat
    # carrying samples is exactly the one most likely to be suppressed.
    src = inspect.getsource(rl_verl.run_rl_verl)
    forced = src[src.index("if not sent_first_metrics:") :]
    forced = forced[: forced.index("gpu=gpu_diagnostics")]
    assert "force=True" in forced
    assert "**_reward_observability()" in forced


def test_the_liveness_fields_hook_carries_reward_observability():
    # the rl_step liveness wrap is what publishes between stdout lines. without the fields hook
    # merging it, samples would only ever reach the wire on the one forced first-metrics heartbeat.
    src = " ".join(inspect.getsource(rl_verl.run_rl_verl).split())
    assert 'fields=lambda: {"metrics_last": list(metrics_last), **_reward_observability()}' in src
