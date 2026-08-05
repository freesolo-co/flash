"""verl grpo backend: dispatch, data/config/reward glue, and reward parity (cpu-only, no verl)."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import flash.engine.worker as W
from flash.engine.worker import backend_common, rl, rl_train, sft_train
from flash.engine.worker.heartbeat import RewardObservabilityBuffer


# ------------------------------- dispatch -------------------------------
@pytest.mark.parametrize("stale", [None, "verl", "trl", "megatron"])
def test_run_rl_always_delegates_to_verl(monkeypatch, stale):
    """run_rl delegates to run_rl_train unconditionally -- no env key selects a backend.

    verl is the only trainer, so a stale FLASH_RL_BACKEND left in a config must be inert rather than
    routing anywhere else or raising.
    """
    called = []
    monkeypatch.setattr(rl_train, "run_rl_train", lambda: called.append(True))
    if stale is None:
        monkeypatch.delenv("FLASH_RL_BACKEND", raising=False)
    else:
        monkeypatch.setenv("FLASH_RL_BACKEND", stale)
    rl.run_rl()
    assert called == [True]


class _FakeGrpoProcess:
    def __init__(self, lines, *, wait_code, stale_return_code):
        self.stdout = iter(lines)
        self.pid = 424242
        self.returncode = stale_return_code
        self._wait_code = wait_code
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1
        return self._wait_code


def test_grpo_subprocess_stream_classifies_the_recorded_nonzero_exit(monkeypatch):
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    terminated = []
    monkeypatch.setattr(
        rl_train,
        "kill_process_group",
        lambda proc, *, process_group_id: terminated.append((proc, process_group_id)),
    )
    signature = "cudaErrorDevicesUnavailable\n"
    lines = [signature, *(f"filler-{i}\n" for i in range(150))]
    proc = _FakeGrpoProcess(lines, wait_code=17, stale_return_code=0)
    stream = rl_train._GrpoSubprocessStream(proc)

    assert list(stream) == lines
    with pytest.raises(RetriableInfraError) as exc_info:
        stream.wait_and_classify()

    assert "cudaErrorDevicesUnavailable" in str(exc_info.value)
    assert proc.wait_calls == 1
    assert terminated == [(proc, proc.pid)]


def test_grpo_subprocess_stream_does_not_classify_a_zero_exit():
    lines = ["cudaErrorDevicesUnavailable\n"]
    proc = _FakeGrpoProcess(lines, wait_code=0, stale_return_code=17)
    stream = rl_train._GrpoSubprocessStream(proc)

    assert list(stream) == lines
    assert stream.wait_and_classify() == 0
    assert proc.wait_calls == 1


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="requires linux process groups")
def test_grpo_classified_exit_drains_group_after_leader_is_reaped(tmp_path, monkeypatch):
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    if not backend_common.adopt_orphaned_descendants():
        pytest.skip("child subreaper unavailable")
    monkeypatch.setattr(backend_common, "_TEARDOWN_GRACE_S", 0.5)
    marker = tmp_path / "grpo-classified-grandchild.pid"
    grandchild = (
        "import signal,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(300)\n"
    )
    leader = (
        "import pathlib,subprocess,sys\n"
        f"g = subprocess.Popen([sys.executable, '-c', {grandchild!r}], "
        "stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)\n"
        "assert g.stdout.readline().strip() == 'ready'\n"
        "g.stdout.close()\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(g.pid))\n"
        "print('cudaErrorDevicesUnavailable', flush=True)\n"
        "raise SystemExit(1)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", leader],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    grandchild_pid = None
    try:
        stream = rl_train._GrpoSubprocessStream(proc)
        assert "cudaErrorDevicesUnavailable\n" in list(stream)
        with pytest.raises(RetriableInfraError, match="cudaErrorDevicesUnavailable"):
            stream.wait_and_classify()

        assert proc.poll() == 1, "test did not exercise classification after leader reaping"
        assert marker.exists(), "leader exited before recording its surviving grandchild"
        grandchild_pid = int(marker.read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and os.path.exists(f"/proc/{grandchild_pid}"):
            time.sleep(0.05)
        assert not os.path.exists(f"/proc/{grandchild_pid}"), (
            f"classified grpo exit left grandchild {grandchild_pid} alive or unreaped"
        )
    finally:
        if grandchild_pid is not None:  # pragma: no cover - only on an unexpected failure
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(grandchild_pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(grandchild_pid, 0)
        if proc.poll() is None:  # pragma: no cover - only on an unexpected failure
            backend_common.kill_process_group(proc)


def test_run_rl_train_reaches_the_executable_grpo_subprocess_stream():
    source = inspect.getsource(rl_train.run_rl_train)

    assert "child_stream = _GrpoSubprocessStream(proc)" in source
    assert "for line in child_stream" in source
    assert "rc = child_stream.wait_and_classify()" in source


# ------------------------------- data conversion -------------------------------
def test_build_verl_dataset_rows_schema_and_index():
    rows = rl_train.build_verl_dataset_rows(
        [[{"role": "user", "content": "q0"}], [{"role": "user", "content": "q1"}]],
        [5, 9],
        ["42", ""],
    )
    assert rows[0]["prompt"] == [{"role": "user", "content": "q0"}]
    assert rows[0]["reward_model"] == {"style": "rule", "ground_truth": "42"}
    # the flash rollout index must round-trip through verl's extra_info so the reward maps back.
    assert [r["extra_info"]["index"] for r in rows] == [5, 9]
    assert all(r["data_source"] == rl_train.DATA_SOURCE for r in rows)


def test_build_verl_dataset_rows_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        rl_train.build_verl_dataset_rows([[{"role": "user", "content": "q"}]], [1, 2], ["a", "b"])


# ------------------------------- multimodal parquet contract -------------------------------
# verl's RLHFDataset._build_messages re-splits each prompt on "<image>" and then asserts the
# placeholder count equals len(images). the two halves of that invariant are produced in different
# places here (message flattening vs the images column), so they are exactly the kind of pair that
# drifts silently: a row can look well-formed on both sides and still raise inside verl.


def _image_placeholder_count(row) -> int:
    return sum(str(m["content"]).count("<image>") for m in row["prompt"])


def test_multimodal_rows_match_verl_placeholder_assertion():
    rows = rl_train.build_verl_dataset_rows(
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
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "q1"},
                    ],
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
    rows = rl_train.build_verl_dataset_rows([[{"role": "user", "content": "q"}]], [0], ["a"])
    assert "images" not in rows[0]


def test_multimodal_rows_reject_a_mismatched_uri_list():
    with pytest.raises(ValueError, match="image_uris length mismatch"):
        rl_train.build_verl_dataset_rows(
            [[{"role": "user", "content": "q"}]], [0], ["a"], image_uris=[[], []]
        )


def test_multimodal_rows_reject_a_literal_image_placeholder_in_text():
    # verl splits prompt text on "<image>" and re-expands each hit into a real image block, so a
    # prompt that merely TALKS about the token consumes an image the row does not have. verl would
    # abort dataset loading with a bare offset assertion; catching it here names the example.
    with pytest.raises(ValueError, match="reserved by verl"):
        rl_train.build_verl_dataset_rows(
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
        rl_train.build_verl_dataset_rows(
            [
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "describe <image> please"}],
                    }
                ],
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
        rl_train.build_verl_dataset_rows(
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
    rows = rl_train.build_verl_dataset_rows(
        [[{"role": "user", "content": "what does <image> mean?"}]], [0], ["a"]
    )
    assert rows[0]["prompt"] == [{"role": "user", "content": "what does <image> mean?"}]


def test_mixed_job_parquet_round_trips_the_images_column(tmp_path):
    # Dataset.from_list infers ONE type per column across all rows. in a mixed job the text rows
    # have an empty images list, and inference on an all-empty-or-partly-empty column can land on a
    # type verl cannot read back as a struct. this asserts the round trip, not the schema object,
    # because the schema is only interesting insofar as the read-back works.
    rows = rl_train.build_verl_dataset_rows(
        [
            [{"role": "user", "content": [{"type": "text", "text": "text only"}]}],
            [{"role": "user", "content": [{"type": "image"}]}],
        ],
        [0, 1],
        ["a", "b"],
        image_uris=[[], ["file:///w/1-0.png"]],
    )
    path = str(tmp_path / "train.parquet")
    rl_train.write_verl_grpo_parquet(rows, path)

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
    rows = rl_train.build_verl_dataset_rows([[{"role": "user", "content": "q"}]], [0], ["a"])
    path = str(tmp_path / "train.parquet")
    rl_train.write_verl_grpo_parquet(rows, path)

    import pyarrow.parquet as pq

    assert "images" not in pq.read_table(path).schema.names


# ------------------------------- override generation -------------------------------
def _overrides_cfg(**over):
    cfg = {
        "train_files": "/w/train.parquet",
        "val_files": "/w/val.parquet",
        "model_id": "Qwen/Qwen3-4B",
        "lora_rank": 32,
        "lora_alpha": 64,
        "target_modules": "all-linear",
        "lr": 1e-5,
        "group_size": 8,
        "prompts_per_step": 16,
        "max_prompt_len": 2048,
        "max_model_len": 2368,
        "max_token_len_per_gpu": 2368,
        # single-turn: the response tensor holds one completion, so it is max_completion wide.
        "max_completion": 320,
        "max_response_len": 320,
        "multi_turn": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "kl_coef": 0.0,
        "entropy_quantile": None,
        "stop_sequences": (),
        "structured_outputs": None,
        "thinking": False,
        "loss_agg_mode": "seq-mean-token-sum-norm",
        "seed": 42,
        "ppo_epochs": 1,
        "steps": 60,
        "gpu_mem_util": 0.5,
        "n_gpus": 1,
        "loggers": ["console"],
        "fp8_kv": False,
        "enforce_eager": False,
        "warmstart_adapter": "",
        "reward_path": "/w/reward.py",
        "reward_name": "compute_score",
        "mask_truncated_completions": True,
        "total_epochs": 1,
        "save_freq": 20,
        "ckpt_to_keep": 1,
        "local_dir": "/w/ckpt",
        "project_name": "flash",
        "experiment_name": "flash-rl-run123",
    }
    cfg.update(over)
    return cfg


def test_build_verl_overrides_carries_dr_grpo_recipe():
    o = rl_train.build_verl_overrides(_overrides_cfg())
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


def test_build_verl_overrides_carries_fused_expert_target_parameters():
    o = rl_train.build_verl_overrides(
        _overrides_cfg(
            target_parameters=[
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj",
            ]
        )
    )

    assert (
        "++actor_rollout_ref.model.target_parameters="
        "[mlp.experts.gate_up_proj,mlp.experts.down_proj]"
    ) in o


def test_build_verl_overrides_does_not_emit_inert_drop_last_override():
    # this guards only against flash emitting a misleading no-op; it does not prove verl reads the key.
    o = rl_train.build_verl_overrides(_overrides_cfg())
    assert not any("drop_last" in override for override in o)


def test_build_verl_overrides_sizes_agent_loop_workers_to_the_rollout_batch():
    # verl chunks prompts_per_step * group_size across agent.num_workers and asserts exact
    # divisibility; its default of 8 aborts before the first step on e.g. 2 x 2 = 4.
    o = rl_train.build_verl_overrides(_overrides_cfg(prompts_per_step=2, group_size=2))
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in o
    # the common case still gets the full worker pool.
    big = rl_train.build_verl_overrides(_overrides_cfg(prompts_per_step=64, group_size=8))
    assert "actor_rollout_ref.rollout.agent.num_workers=8" in big


@pytest.mark.parametrize(("count", "expected"), [(None, 1), (1, 1), (2, 2), (8, 8)])
def test_run_rl_train_sizes_the_run_from_the_spec_gpu_count(count, expected):
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
    o = rl_train.build_verl_overrides(_overrides_cfg(n_gpus=1))
    assert "+ray_kwargs.ray_init.num_gpus=1" in o
    assert "trainer.n_gpus_per_node=1" in o
    assert "trainer.nnodes=1" in o
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in o
    assert "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1" in o


@pytest.mark.parametrize("n_gpus", [2, 4, 8])
def test_build_verl_overrides_shards_every_card_along_the_sequence(n_gpus):
    # verl builds mesh_shape=(dp, sp), so sp == n_gpus pins dp == 1: the optimizer keeps seeing one
    # global batch of prompts_per_step * group_size. sharding the BATCH instead would change the
    # gradient, which is why sp/tp track the card count rather than leaving dp to absorb it.
    o = rl_train.build_verl_overrides(_overrides_cfg(n_gpus=n_gpus))
    assert f"+ray_kwargs.ray_init.num_gpus={n_gpus}" in o
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
    one = rl_train.build_verl_overrides(_overrides_cfg(n_gpus=1))
    for n_gpus in (2, 4, 8):
        many = rl_train.build_verl_overrides(_overrides_cfg(n_gpus=n_gpus))
        for key in batch_keys:
            assert [o for o in one if o.startswith(key)] == [o for o in many if o.startswith(key)]


def test_one_optimizer_step_consumes_exactly_the_requested_unique_prompts():
    # the invariant the old trl batching helper existed to protect: an optimizer step must optimize
    # prompts_per_step UNIQUE PROMPTS, not prompts_per_step COMPLETIONS. trl sized batches in
    # completions, so it needed group_size folded into grad-accum or a step silently optimized
    # prompts_per_step/group_size prompts. verl sizes in prompts and expands group_size itself via
    # rollout.n, so the guard here is that flash never pre-divides or pre-multiplies by the group:
    # train_batch_size (prompts drawn) and ppo_mini_batch_size (prompts per update) both stay the
    # raw request, and rollout.n carries the group. off-by-a-group here is silent -- the run trains,
    # just on 1/group_size of the intended data.
    for prompts, group in ((64, 8), (5, 8), (2, 2), (1, 6)):
        o = rl_train.build_verl_overrides(
            _overrides_cfg(prompts_per_step=prompts, group_size=group)
        )
        assert f"data.train_batch_size={prompts}" in o
        assert f"actor_rollout_ref.actor.ppo_mini_batch_size={prompts}" in o
        assert f"actor_rollout_ref.rollout.n={group}" in o


def test_build_verl_overrides_sets_truncation_mask_when_enabled():
    o = rl_train.build_verl_overrides(_overrides_cfg(mask_truncated_completions=True))
    # `++` (append-or-override), because the key exists in the fork's rollout.yaml but not stock's.
    assert "++actor_rollout_ref.rollout.mask_truncated_completions=true" in o


def test_build_verl_overrides_omits_truncation_mask_when_disabled():
    # stock verl rejects the unknown key at dataclass conversion, and not masking is already its
    # behavior, so emitting `=false` would break stock runs while changing nothing.
    o = rl_train.build_verl_overrides(_overrides_cfg(mask_truncated_completions=False))
    assert not any("mask_truncated_completions" in override for override in o)


def test_build_verl_overrides_pins_both_blackwell_attention_backends():
    o = rl_train.build_verl_overrides(
        _overrides_cfg(attention_backend="FLASHINFER", mm_encoder_attn_backend="TORCH_SDPA")
    )
    # verl spreads engine_kwargs.vllm straight into AsyncEngineArgs, where both are real fields in
    # the pinned vllm 0.19.1. `+` appends under the existing struct, as kv_cache_dtype does.
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASHINFER" in o
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend=TORCH_SDPA" in o


def test_build_verl_overrides_leaves_attention_backends_to_vllm_off_blackwell():
    # off blackwell vllm's own capability-ordered defaults are correct, and pinning a backend there
    # would override a working choice. the resolver returns None/None, so nothing may be emitted.
    o = rl_train.build_verl_overrides(
        _overrides_cfg(attention_backend=None, mm_encoder_attn_backend=None)
    )
    assert not any("attention_backend" in override for override in o)


def test_build_verl_overrides_sizes_engine_to_the_job_not_the_architecture():
    # left unset, verl substitutes the model's full max_position_embeddings and hands it to vllm,
    # so a short job on a long-context model reserves kv cache it can never use. the emitted length
    # must be the job's own engine length.
    o = rl_train.build_verl_overrides(_overrides_cfg(max_model_len=2368))
    assert "actor_rollout_ref.rollout.max_model_len=2368" in o


def test_engine_len_clamped_to_model_limit():
    # verl raises ValueError when max_model_len exceeds max_position_embeddings, so a job asking
    # for more context than the architecture has must train shorter, not die at rollout startup.
    assert rl_train.clamp_engine_len(32768, 8192) == 8192
    # under the limit is untouched, and an unknown limit leaves verl's own resolution in charge.
    assert rl_train.clamp_engine_len(4096, 40960) == 4096
    assert rl_train.clamp_engine_len(32768, None) == 32768
    assert rl_train.clamp_engine_len(32768, 0) == 32768


def test_token_budget_admits_a_full_length_sequence():
    # dynamic bsz packs micro-batches up to this budget. below one full sequence, the longest
    # rollout the engine can produce fits in no micro-batch at all.
    cfg = _overrides_cfg(max_prompt_len=31744, max_completion=1024, max_token_len_per_gpu=32768)
    o = rl_train.build_verl_overrides(cfg)
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
    o = rl_train.build_verl_overrides(_overrides_cfg())
    assert not any("ppo_micro_batch_size_per_gpu" in x for x in o)
    assert not any("log_prob_micro_batch_size_per_gpu" in x for x in o)


def test_multimodal_overrides_hand_verl_the_images_column():
    # the parquet's images column is inert unless verl is told to read it: without image_key the
    # dataset treats the rows as text, the <image> placeholders never re-expand, and the model
    # trains on the caption alone -- silently, which is the failure this whole port exists to avoid.
    o = rl_train.build_verl_overrides(_overrides_cfg(multimodal=True))
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
    o = rl_train.build_verl_overrides(_overrides_cfg())
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
    source = inspect.getsource(rl_train._build_verl_training_cfg)
    assert '"multimodal": bool(inp.get("multimodal"))' in source


def test_sleep_unsupported_models_keep_the_rollout_engine_resident():
    """VERL-091: a model whose vLLM wake HANGS must never be offloaded between steps.

    verl defaults free_cache_engine and enable_sleep_mode BOTH True and sleeps the rollout engine at
    every step boundary, so without an explicit override a catalog model flagged sleep_unsupported
    wedges on the first wake instead of failing fast. The flag comes from the catalog, so assert
    against a real flagged entry rather than a fabricated one.
    """
    from flash.catalog import MODELS

    flagged = [m for m, i in MODELS.items() if getattr(i, "sleep_unsupported", False)]
    assert flagged, "no catalog model is sleep_unsupported; this guard now has no subject"

    def _argv(model_id):
        inp = {
            "lora_rank": 32,
            "lora_alpha": 64,
            "lr": 1e-5,
            "group_size": 8,
            "prompts_per_step": 16,
            "mask_truncated_completions": True,
            "max_prompt_len": 3072,
            "max_completion": 1024,
            "max_response_len": 1024,
            "multi_turn": False,
            "engine_len": 4096,
            "temperature": 1.0,
            "top_p": 0.95,
            "kl_coef": 0.0,
            "entropy_quantile": None,
            "stop_sequences": (),
            "structured_outputs": None,
            "seed": 42,
            "ppo_epochs": 1,
            "steps": 60,
            "warmstart_adapter": "",
            "model_id": model_id,
            "verl_total_epochs": 2,
            "save_freq": 20,
            "ckpt_to_keep": 1,
        }
        # go through the real builder so the flag cannot drift out of the cfg it emits.
        cfg = rl_train._build_verl_training_cfg(
            inp,
            train_files="/w/t.parquet",
            val_files="/w/v.parquet",
            model_id="/w/model",
            thinking=False,
            loggers=["console"],
            fp8_kv=False,
            enforce_eager=False,
            attention_backend=None,
            mm_encoder_attn_backend=None,
            reward_path="/w/r.py",
            local_dir="/w/ckpt",
            project_name="flash",
            experiment_name="flash-rl-run123",
        )
        return rl_train.build_verl_overrides(cfg)

    # the two knobs need DIFFERENT hydra prefixes -- rollout_resident_overrides' docstring has the
    # why. asserted EXACTLY rather than as a substring, because "x=false" is a substring of
    # "+x=false": the obvious assertion passes against the spelling that kills the run at parse,
    # which is how this shipped. see ISSUES.md VERL-148.
    for override in (
        "actor_rollout_ref.rollout.free_cache_engine=false",
        "+actor_rollout_ref.rollout.enable_sleep_mode=false",
    ):
        assert override in _argv(flagged[0])
    # the bare spelling must be ABSENT, not merely accompanied by the prefixed one.
    assert "actor_rollout_ref.rollout.enable_sleep_mode=false" not in _argv(flagged[0])
    # and the override is scoped: an ordinary model keeps verl's own sleep/wake offload, which is
    # what lets a large rollout fit alongside the training weights.
    for key in ("free_cache_engine", "enable_sleep_mode"):
        assert not [a for a in _argv("Qwen/Qwen3-4B") if key in a]


def test_build_verl_training_cfg_derives_engine_len_and_budget():
    inp = {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lr": 1e-5,
        "group_size": 8,
        "prompts_per_step": 16,
        "mask_truncated_completions": True,
        "max_prompt_len": 3072,
        "max_completion": 1024,
        "max_response_len": 1024,
        "multi_turn": False,
        "engine_len": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "kl_coef": 0.0,
        "entropy_quantile": None,
        "stop_sequences": (),
        "structured_outputs": None,
        "seed": 42,
        "ppo_epochs": 1,
        "steps": 60,
        "warmstart_adapter": "",
        "model_id": "Qwen/Qwen3-4B",
        "verl_total_epochs": 2,
        "save_freq": 20,
        "ckpt_to_keep": 1,
    }
    common = {
        "train_files": "/w/t.parquet",
        "val_files": "/w/v.parquet",
        "model_id": "Qwen/Qwen3-4B",
        "thinking": False,
        "loggers": ["console"],
        "fp8_kv": False,
        "enforce_eager": False,
        "attention_backend": None,
        "mm_encoder_attn_backend": None,
        "reward_path": "/w/r.py",
        "local_dir": "/w/ckpt",
        "project_name": "flash",
        "experiment_name": "flash-rl-run123",
    }
    cfg = rl_train._build_verl_training_cfg(inp, **common)
    # the engine gets the full prompt+completion length, not the prompt budget alone, and the token
    # budget matches it. the resolver clamps engine_len, so the builder passes it through unchanged.
    assert cfg["max_model_len"] == 4096
    assert cfg["max_token_len_per_gpu"] == 4096


@pytest.mark.parametrize(
    (
        "prompt_count",
        "prompts_per_step",
        "epochs",
        "max_steps",
        "expected_steps",
        "expected_epochs",
    ),
    [
        pytest.param(33, 16, 2, None, 5, 3, id="partial-batch-derived-horizon"),
        pytest.param(32, 16, 2, 7, 7, 4, id="explicit-horizon-beyond-derived"),
        pytest.param(32, 16, 2, None, 4, 2, id="exactly-divisible"),
    ],
)
def test_verl_epoch_capacity_reaches_update_horizon(
    prompt_count, prompts_per_step, epochs, max_steps, expected_steps, expected_epochs
):
    derived_steps = rl_train.on_policy_steps(
        epochs=epochs, prompt_count=prompt_count, prompts_per_step=prompts_per_step
    )
    steps = rl_train.resolve_update_horizon(derived_steps, max_steps)
    resolved_epochs = rl_train._verl_epochs_for_horizon(
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
def test_verl_epoch_capacity_rejects_invalid_batch_inputs(prompt_count, prompts_per_step, message):
    with pytest.raises(ValueError, match=message):
        rl_train._verl_epochs_for_horizon(
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
                    resolved_epochs = rl_train._verl_epochs_for_horizon(
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
    cfg = rl_train._build_verl_training_cfg(
        inp,
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_id=inp["model_id"],
        thinking=False,
        loggers=["console"],
        fp8_kv=False,
        enforce_eager=False,
        attention_backend=None,
        mm_encoder_attn_backend=None,
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
    # parity with the retired trl path: with save_at_steps set the customer
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
        rl_train._w,
        "publish_deployable_checkpoint",
        lambda d, s, **kw: published.append((d, s, kw.get("required", False))),
        raising=False,
    )
    monkeypatch.setattr(
        rl_train._w, "upload_resume_checkpoint", lambda *a, **kw: True, raising=False
    )
    monkeypatch.setattr(
        rl_train._w, "write_base_model_provenance", lambda *a, **kw: None, raising=False
    )
    monkeypatch.setattr(rl_train, "export_peft_adapter", lambda *a, **kw: None)
    monkeypatch.setattr(rl_train, "stamp_adapter_dir_provenance", lambda *a, **kw: None)

    local_dir = tmp_path / "ckpt"
    (local_dir / "global_step_10" / "actor").mkdir(parents=True)
    (local_dir / "global_step_5" / "actor").mkdir(parents=True)
    (local_dir / "latest_checkpointed_iteration.txt").write_text("10")

    class _Tok:
        def save_pretrained(self, path):
            pass

    uploader = rl_train._VerlResumeUploader(
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
    monkeypatch.setattr(rl_train, "_deployable_adapter_on_hf", lambda step: step == 10)

    class _Tok:
        def save_pretrained(self, path):
            pass

    uploader = rl_train._VerlResumeUploader(
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
    monkeypatch.setattr(rl_train, "_deployable_adapter_on_hf", lambda step: False)

    uploader = rl_train._VerlResumeUploader(str(tmp_path), resume_step=10, required_steps=(10,))
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
    monkeypatch.setattr(W, "grpo_overrides", dict, raising=False)
    monkeypatch.setattr(W, "grpo_mask_truncated_completions", lambda train: False, raising=False)
    monkeypatch.setattr(W, "load_tokenizer", lambda *args, **kwargs: _Tokenizer(), raising=False)
    monkeypatch.setattr(rl_train, "seed_training_rngs", lambda seed: None)
    # the context-limit probe reads the model config off the hub; keep this unit test offline.
    monkeypatch.setattr(rl_train, "model_max_position_embeddings", lambda *a, **k: 40960)

    inp = rl_train._resolve_grpo_inputs()
    cfg = rl_train._build_verl_training_cfg(
        inp,
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_id=inp["model_id"],
        thinking=False,
        loggers=["console"],
        fp8_kv=False,
        enforce_eager=False,
        attention_backend=None,
        mm_encoder_attn_backend=None,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    overrides = rl_train.build_verl_overrides(cfg)
    notes = rl_train._build_verl_train_notes(
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
    o = rl_train.build_verl_overrides(_overrides_cfg(loggers=["console", "wandb"]))
    assert "trainer.logger=[console,wandb]" in o


def test_build_verl_overrides_warmstart_adapter_path():
    # fresh run: no lora_adapter_path override.
    fresh = rl_train.build_verl_overrides(_overrides_cfg(warmstart_adapter=""))
    assert not any("lora_adapter_path" in x for x in fresh)
    # warm-start: point verl's lora init at the downloaded source adapter dir.
    warm = rl_train.build_verl_overrides(_overrides_cfg(warmstart_adapter="/tmp/sft_adapter"))
    assert "actor_rollout_ref.model.lora_adapter_path=/tmp/sft_adapter" in warm


def test_build_verl_overrides_fp8_kv_gated_on_hardware():
    off = rl_train.build_verl_overrides(_overrides_cfg(fp8_kv=False))
    assert not any("kv_cache_dtype" in x for x in off)
    on = rl_train.build_verl_overrides(_overrides_cfg(fp8_kv=True))
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8" in on


def test_build_verl_overrides_enforce_eager_gated_on_hardware():
    off = rl_train.build_verl_overrides(_overrides_cfg(enforce_eager=False))
    assert not any("enforce_eager" in x for x in off)
    on = rl_train.build_verl_overrides(_overrides_cfg(enforce_eager=True))
    # plain override, not '+': enforce_eager is a declared verl RolloutConfig field
    # (workers/config/rollout.py:195), so appending it would be a duplicate-key error.
    assert "actor_rollout_ref.rollout.enforce_eager=True" in on


def test_the_resolved_eager_flag_reaches_the_verl_config():
    # the string assertions above pass against a resolver whose answer is never carried into the
    # config, which is exactly how the retired trl workaround got dropped. pin the wiring.
    built = inspect.getsource(rl_train.run_rl_train)
    assert "enforce_eager = resolve_rollout_enforce_eager(verl_cc)" in built
    assert "enforce_eager=enforce_eager," in built
    # and the capability it decides from is the one probe both rollout decisions share.
    assert "verl_cc = resolve_verl_device_capability(python_bin)" in built
    assert (
        "resolve_blackwell_attention_backends(\n            python_bin, verl_cc\n        )" in built
    )
    cfg = inspect.getsource(rl_train._build_verl_training_cfg)
    assert '"enforce_eager": enforce_eager,' in cfg


def test_build_verl_overrides_kl_off_by_default():
    # flash default kl_penalty_coef=0 (dr-grpo, no kl term) -> no reference policy.
    o = rl_train.build_verl_overrides(_overrides_cfg(kl_coef=0.0))
    assert "actor_rollout_ref.actor.use_kl_loss=False" in o
    assert not any("kl_loss_coef" in x for x in o)
    assert not any("ref.log_prob_micro_batch" in x for x in o)


def test_build_verl_overrides_kl_on_when_requested():
    o = rl_train.build_verl_overrides(_overrides_cfg(kl_coef=0.02))
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

    src = inspect.getsource(rl_train.run_rl_train)
    # the stage names are a cross-process contract: the poller, the throttle table and the runner
    # all key off "rl_step"/"rl_finalizing". the tempting mistake is to coin a module-prefixed
    # variant, which reports a stage none of those three recognise. assert against the spelling the
    # CURRENT module name would produce -- pinning the pre-rename "rl_verl_*" spelling here would be
    # asserting the absence of a string nothing can emit any more, which no regression can fail.
    assert "rl_train_training" not in src
    assert "rl_train_finalizing" not in src
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
    src = rl_train.render_reward_module()
    ns: dict = {}
    exec(compile(src, "<reward>", "exec"), ns)  # compiles + defines, no network call made
    assert callable(ns["compute_score"])
    # no flash import leaks into the verl-side shim.
    assert "import flash" not in src


def test_render_reward_module_missing_index_raises():
    ns: dict = {}
    exec(compile(rl_train.render_reward_module(), "<reward>", "exec"), ns)
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
    exec(
        compile(rl_train.render_reward_module("TEST_FLASH_VERL_REWARD_URL"), "<reward>", "exec"), ns
    )
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
    server, url = rl_train.start_reward_server(
        lambda idx, solution: scored.append((idx, solution)) or 3.0,
        example_count=2,
    )
    try:
        monkeypatch.setenv("TEST_FLASH_VERL_REWARD_URL", url)
        ns: dict = {}
        exec(
            compile(
                rl_train.render_reward_module("TEST_FLASH_VERL_REWARD_URL"), "<reward>", "exec"
            ),
            ns,
        )
        assert (
            ns["compute_score"]("flash_env", "answer", "unused", extra_info={"index": index}) == 3.0
        )
        assert scored == [(1, "answer")]
    finally:
        server.shutdown()


def test_a_slow_env_call_is_not_cut_off_by_a_client_deadline(monkeypatch):
    # verl fans reward scoring out hard -- RewardLoopManager spawns reward.num_workers ray workers
    # and each asyncio.gathers its whole chunk -- and start_reward_server serializes them behind one
    # lock. a per-request deadline therefore bounds QUEUE WAIT, not the env call, so the Nth caller
    # in line fails for arriving Nth. a wedged env is the stall watchdog's job, not this client's.
    waited = []
    server, url = rl_train.start_reward_server(
        lambda idx, solution: waited.append(idx) or 7.0, example_count=2
    )
    try:
        ns: dict = {}
        exec(compile(rl_train.render_reward_module("TEST_URL"), "<reward>", "exec"), ns)
        real_urlopen = ns["urllib"].request.urlopen
        seen = []

        def urlopen_recording_deadline(req, *args, **kwargs):
            seen.append((args, kwargs))
            return real_urlopen(req)

        monkeypatch.setattr(ns["urllib"].request, "urlopen", urlopen_recording_deadline)
        ns["_URL"] = url
        assert ns["compute_score"]("env", "answer", "unused", extra_info={"index": 0}) == 7.0
        assert waited == [0]
        assert seen == [((), {})], f"reward client still carries a deadline: {seen!r}"
    finally:
        server.shutdown()


def test_concurrent_scorers_are_serialized_for_the_env():
    # a flash env is a plain python object with no concurrency contract; the retired trl path only
    # ever called it from one thread. verl's reward workers do not, so the server must impose it.
    import concurrent.futures

    live = []
    peak = []
    lock = threading.Lock()

    def score(idx, solution):
        with lock:
            live.append(idx)
            peak.append(len(live))
        time.sleep(0.02)
        with lock:
            live.remove(idx)
        return float(idx)

    server, url = rl_train.start_reward_server(score, example_count=8)
    try:
        ns: dict = {}
        exec(compile(rl_train.render_reward_module("TEST_URL"), "<reward>", "exec"), ns)
        ns["_URL"] = url
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda i: ns["compute_score"]("env", "a", "u", extra_info={"index": i}),
                    range(8),
                )
            )
        assert sorted(results) == [float(i) for i in range(8)]
        assert max(peak) == 1, f"env saw {max(peak)} concurrent calls"
    finally:
        server.shutdown()


def test_reward_server_accept_queue_holds_a_whole_rollout_batch(monkeypatch):
    # verl opens one connection per episode and starts a whole step at once, so the accept queue
    # sees prompts_per_step * group_size connects in a burst. socketserver's default backlog of 5
    # overflows there and the kernel RESETS the excess, which reaches the client as
    # ConnectionResetError at getresponse(); bridge_post does not retry, so that kills the run.
    rollout_batch = 512  # the default 64x8 recipe

    # record the argument the server actually hands to listen(). asserting on request_queue_size
    # alone cannot tell a working fix from a no-op: server_activate() reads that attribute once,
    # so a value set after server_bind() would never reach the socket at all.
    backlogs = []
    real_listen = socket.socket.listen

    def spy_listen(self, *args):
        backlogs.append(args[0] if args else None)
        return real_listen(self, *args)

    monkeypatch.setattr(socket.socket, "listen", spy_listen)

    server, _url = rl_train.start_reward_server(
        lambda idx, solution: 1.0, example_count=8, rollout_batch=rollout_batch
    )
    try:
        assert server.request_queue_size >= rollout_batch
        assert backlogs, "server never called listen()"
        assert backlogs[-1] is not None, "listen() was called with no backlog argument"
        assert backlogs[-1] >= rollout_batch, f"listen() backlog is {backlogs[-1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("rollout_batch", [0, 32, 64 * 8, 2048])
def test_reward_bridge_backlog_never_falls_below_the_burst(rollout_batch):
    # a fixed constant would only move the cliff, so the queue is sized from the caller's burst.
    # an unspecified batch still keeps a floor well clear of socketserver's default of 5.
    server, _url = rl_train.start_reward_server(
        lambda idx, solution: 1.0, example_count=1, rollout_batch=rollout_batch
    )
    try:
        assert server.request_queue_size >= max(128, rollout_batch)
    finally:
        server.shutdown()
        server.server_close()


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
    s = rl_train.score_single_turn(
        _BreakdownEnv(),
        "7",
        {"gt": "7"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )
    assert s == 1.0
    s2 = rl_train.score_single_turn(
        _RewardOnlyEnv(),
        "the answer is 7",
        {"gt": "7"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )
    assert s2 == 2.5


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_applies_thinking_penalty():
    # base reward 1.0 minus think_penalty(0.1) * think_token_count(3) = 0.7
    s = rl_train.score_single_turn(
        _BreakdownEnv(),
        "7",
        {"gt": "7"},
        tok=object(),
        thinking=True,
        prompt_opened_thinking=True,
        think_penalty=0.1,
    )
    assert abs(s - 0.7) < 1e-9


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_env_error_is_zero():
    s = rl_train.score_single_turn(
        _RaisingEnv(),
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )
    assert s == 0.0


class _UnscorableRewardEnv:
    def reward(self, graded, ex, state):
        return float("nan")


class _UnscorableBreakdownEnv:
    def scores_breakdown(self, graded, ex, state):
        return {"total": float("nan"), "judge": 1.0}


@pytest.mark.usefixtures("_identity_graded")
@pytest.mark.parametrize(
    "env", [_UnscorableRewardEnv(), _UnscorableBreakdownEnv()], ids=["reward", "scores_breakdown"]
)
def test_an_unscorable_reward_is_masked_before_it_reaches_verl(env):
    """A non-finite reward must not be forwarded, from EITHER env hook.

    verl's grpo baseline is a plain torch.mean/torch.std over the group (core_algos.py:320-326)
    with no nan-aware variant on its path, so one nan row makes the mean, the std, and all
    `group_size` advantages nan -- the whole group, not just the unscorable row. The retired trl
    path could forward it because it masked nan rows out of the baseline and zeroed their
    advantage (grpo_trainer.py:2171, :2222); nothing downstream of here does that now (codex[bot]).
    """
    breakdowns: list[dict[str, float] | None] = []
    s = rl_train.score_single_turn(
        env,
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert s == 0.0
    assert math.isfinite(s)


@pytest.mark.usefixtures("_identity_graded")
def test_an_infinite_reward_is_masked_too_even_with_a_penalty_applied():
    # inf, not just nan: nan is score_rollouts' canonical unscorable marker, but an env returning
    # inf poisons verl's baseline exactly as thoroughly and is not covered by testing nan alone.
    # the penalty is live here because that arithmetic runs between the env's return and the mask;
    # inf minus a finite penalty is still inf, so the mask is what has to catch it.
    class _InfEnv:
        def reward(self, graded, ex, state):
            return float("inf")

    s = rl_train.score_single_turn(
        _InfEnv(),
        "x",
        {"gt": "1"},
        tok=object(),
        thinking=True,
        prompt_opened_thinking=True,
        think_penalty=0.1,
    )
    assert s == 0.0


@pytest.mark.usefixtures("_identity_graded")
def test_an_unscorable_reward_still_re_raises_for_the_latency_profiler():
    # same contract as a raising env: the profiler must tell a real 0.0 apart from a grader that
    # is not returning a usable number, or it reports a broken grader as fast and confident.
    with pytest.raises(ValueError, match="non-finite"):
        rl_train.score_single_turn(
            _UnscorableRewardEnv(),
            "x",
            {"gt": "1"},
            tok=None,
            thinking=False,
            prompt_opened_thinking=False,
            think_penalty=0.0,
            raise_on_error=True,
        )


class _RaisingProbeEnv:
    """An env whose `scores_breakdown` ATTRIBUTE LOOKUP raises, not its call.

    Real shapes that do this: a `@property` that touches a closed handle, or a lazy proxy that
    dials a sidecar on first access. `hasattr` only swallows AttributeError, so anything else
    propagates out of the probe itself.
    """

    def __getattr__(self, name):
        if name == "scores_breakdown":
            raise RuntimeError("scoring sidecar is gone")
        raise AttributeError(name)


@pytest.mark.usefixtures("_identity_graded")
def test_a_capability_probe_that_raises_scores_zero_and_counts_as_a_failed_grading():
    """The probe is env code too, so it has to sit inside the guard that turns env faults into 0.0.

    Outside it, the exception escapes into the reward http handler, the verl child reads a bridge
    failure and aborts the whole run -- over one env's attribute access (codex[bot]).

    The `None` matters just as much as the 0.0: a raising probe is a completion that FAILED to
    score, and the mean counts None as a zero for every name the other completions reported.
    Dropping it instead would shrink the denominator and bias every named metric high.
    """
    breakdowns: list[dict[str, float] | None] = []
    s = rl_train.score_single_turn(
        _RaisingProbeEnv(),
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )

    assert s == 0.0
    assert breakdowns == [None]


@pytest.mark.usefixtures("_identity_graded")
def test_a_raising_probe_still_re_raises_for_the_latency_profiler():
    # raise_on_error is the profiler's way of telling a real 0.0 apart from a broken grader. the
    # guard must not swallow the probe's fault for that caller either.
    with pytest.raises(RuntimeError, match="scoring sidecar is gone"):
        rl_train.score_single_turn(
            _RaisingProbeEnv(),
            "x",
            {"gt": "1"},
            tok=None,
            thinking=False,
            prompt_opened_thinking=False,
            think_penalty=0.0,
            raise_on_error=True,
        )


# --------------------- reward_metrics: per-name breakdown collection ---------------------
class _NamedBreakdownEnv:
    def scores_breakdown(self, graded, ex, state):
        hit = 1.0 if graded.strip() == ex["gt"] else 0.0
        return {"success": hit, "quality": 0.5, "total": hit}


class _CountingBreakdownEnv:
    """Scores each grading with its own ordinal, so a retained buffer window is identifiable."""

    def __init__(self):
        self.n = 0

    def scores_breakdown(self, graded, ex, state):
        n = float(self.n)
        self.n += 1
        return {"n": n, "total": n}


class _OverflowingBreakdownEnv:
    """A named component too large to be a float, beside a usable one and a usable total.

    `float()` raises OverflowError on this, NOT ValueError -- a distinction the observability pass
    has to make because it runs outside the grading error guard."""

    def scores_breakdown(self, graded, ex, state):
        hit = 1.0 if graded.strip() == ex["gt"] else 0.0
        return {"success": hit, "enormous": 10**400, "total": hit}


class _UnusableComponentEnv:
    """Emits a component whose value never coerces to a finite float, alongside a usable one."""

    def scores_breakdown(self, graded, ex, state):
        return {"broken": None, "diverged": float("nan"), "quality": 0.5, "total": 1.0}


class _BadTotalEnv:
    def scores_breakdown(self, graded, ex, state):
        return {"success": 1.0, "total": "not-a-number"}


class _RaisingRewardOnlyEnv:
    def reward(self, graded, ex, state):
        raise ValueError("grader is down")


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_collects_the_named_breakdown_for_reward_metrics():
    breakdowns: list[dict | None] = []
    score = rl_train.score_single_turn(
        _NamedBreakdownEnv(),
        "7",
        {"gt": "7"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert score == 1.0
    assert breakdowns == [{"success": 1.0, "quality": 0.5, "total": 1.0}]


@pytest.mark.usefixtures("_identity_graded")
def test_a_scalar_reward_env_contributes_no_breakdown_at_all():
    # appending {} for a scores_breakdown-less env would add a denominator under no numerators:
    # _mean_named_reward_metrics divides by every scored completion, so an env mixing the two
    # shapes -- or a run with none at all -- would publish every name shrunk toward 0.
    breakdowns: list[dict | None] = []
    score = rl_train.score_single_turn(
        _RewardOnlyEnv(),
        "the answer is 7",
        {"gt": "7"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
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
    score = rl_train.score_single_turn(
        _RaisingRewardOnlyEnv(),
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert score == 0.0
    assert breakdowns == []


@pytest.mark.usefixtures("_identity_graded")
def test_a_failed_grading_records_none_so_it_counts_as_a_zero():
    # trl's contract: a completion that failed to grade still scored 0.0, and must pull the mean of
    # every name the OTHER completions reported down with it. dropping it silently would report the
    # surviving completions' average as if the whole generation had earned it.
    breakdowns: list[dict | None] = []
    score = rl_train.score_single_turn(
        _RaisingEnv(),
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert score == 0.0
    assert breakdowns == [None]


@pytest.mark.usefixtures("_identity_graded")
def test_an_unusable_total_records_no_named_components():
    # float(total) raising IS a failed grading -- the completion scores 0.0. crediting its named
    # components would report metrics for a completion that earned nothing.
    breakdowns: list[dict | None] = []
    score = rl_train.score_single_turn(
        _BadTotalEnv(),
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert score == 0.0
    assert breakdowns == [None]


# ------------------------------- reward rpc bridge -------------------------------
def test_reward_server_round_trip():
    server, url = rl_train.start_reward_server(lambda idx, s: float(idx) + len(s), example_count=4)
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

    server, url = rl_train.start_reward_server(scorer, example_count=len(examples))
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

    server, url = rl_train.start_reward_server(missing_example, example_count=100)
    try:
        monkeypatch.setenv("TEST_FLASH_VERL_REWARD_URL", url)
        ns: dict = {}
        src = rl_train.render_reward_module("TEST_FLASH_VERL_REWARD_URL")
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

    server, url = rl_train.start_reward_server(scorer, example_count=3)
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


def test_reentrant_checkpointing_shim_is_emitted_only_for_models_that_need_it():
    # verl hardcodes use_reentrant=False and offers no knob. a dense non-GDN model is fine on that
    # default and must not get a patch on its import path; a GDN/MoE model dies on the FIRST
    # backward without one, so for those the shim is what makes the run possible at all.
    assert rl_train.render_reentrant_checkpointing_shim(False) == ""
    source = rl_train.render_reentrant_checkpointing_shim(True)
    assert source
    assert '{"use_reentrant": True}' in source
    # must patch the class GRPO's actor actually builds through: FSDPEngineWithLMHead inherits
    # _build_module from FSDPEngine, so patching the base covers the actor.
    assert "FSDPEngine as _FlashReentrantEngine" in source
    assert "_FlashReentrantEngine._build_module = _flash_reentrant_build_module" in source


def test_the_reentrant_shim_is_wired_for_gdn_and_moe_models_and_not_for_dense_ones():
    # the flag has to be resolved from the model id, not left to verl. grpo_use_reentrant is the
    # same helper the sft verl path and the retired trl path both keyed on.
    resolved = inspect.getsource(rl_train._resolve_grpo_inputs)
    assert '"reentrant_checkpointing": bool(_w.grpo_use_reentrant(model_id))' in resolved
    written = inspect.getsource(rl_train.run_rl_train)
    assert (
        "render_reentrant_checkpointing_shim( "
        'inp["reentrant_checkpointing"], multimodal=bool(inp["multimodal"]) )'
        in " ".join(written.split())
    )
    # the curated GDN hybrids and the MoE need it; an uncataloged dense model does not.
    assert W.grpo_use_reentrant("Qwen/Qwen3.5-4B") is True
    assert W.grpo_use_reentrant("Qwen/Qwen3.6-35B-A3B") is True
    assert W.grpo_use_reentrant("meta-llama/Llama-3.1-8B") is False


def test_the_reentrant_shim_installs_vision_input_grads_only_for_multimodal_runs():
    # reentrant recompute DROPS the backward for a checkpointed block when none of that block's
    # inputs require grad. the vision tower's patch embeddings are exactly that case -- the pixels
    # are frozen inputs -- so without a forward hook marking the patch-embed output as requiring
    # grad the visual modules silently train on nothing while the language side trains normally and
    # the run reports success. the retired trl path installed the same hook via a trainer callback;
    # verl has no callback surface, so it rides this shim.
    text_only = rl_train.render_reentrant_checkpointing_shim(True)
    assert "_flash_install_vision_input_grads" not in text_only, (
        "a text-only run pays for a vision hook that can never match"
    )
    multimodal = rl_train.render_reentrant_checkpointing_shim(True, multimodal=True)
    # compile FIRST. an earlier version asserted only that the call text appeared, which the
    # helper's own `def _flash_install_vision_input_grads(module):` line satisfies on its own --
    # so the assertion held while the call site was emitted at the wrong indent and the rendered
    # sitecustomize was a SyntaxError. this shim is exec'd in verl's child, where a syntax error
    # is a silent no-op rather than a test failure, so compiling here is the only real gate.
    compile(multimodal, "sitecustomize.py", "exec")
    assert "visual.patch_embed" in multimodal
    # the call has to land INSIDE the checkpointing branch: dedented one level it would run for
    # every module verl builds, including engines whose checkpointing verl deliberately left off.
    assert "\n        _flash_install_vision_input_grads(module)\n" in multimodal
    # the flag is independent of the shim: a multimodal run on a model that does not need reentrant
    # checkpointing gets no shim at all, and therefore no hook.
    assert rl_train.render_reentrant_checkpointing_shim(False, multimodal=True) == ""


def test_the_reentrant_shim_enables_language_side_input_grads_before_checkpointing():
    """GRAD-001: every rl run is lora, so lora freezes the embeddings and nothing entering the
    first checkpointed decoder layer requires grad. reentrant recompute then drops the backward
    for the whole segment -- where every lora parameter lives -- and the run reports success while
    training nothing. the vision hook only ever covered the patch embeddings on multimodal runs;
    text-only runs had no hook at all."""
    calls = []

    class FakeModule:
        def enable_input_require_grads(self):
            calls.append("require_grads")

        def gradient_checkpointing_enable(self, **kwargs):
            calls.append(("gc_enable", kwargs))

    class FakeEngine:
        def __init__(self, checkpointing):
            self.model_config = SimpleNamespace(enable_gradient_checkpointing=checkpointing)

        def _build_module(self):
            return FakeModule()

    source = rl_train.render_reentrant_checkpointing_shim(True)
    start = source.index("def _flash_reentrant_build_module")
    end = source.index("_FlashReentrantEngine._build_module = _flash_reentrant_build_module")
    namespace = {"_flash_reentrant_original_build_module": FakeEngine._build_module}
    exec(compile(source[start:end], "sitecustomize.py", "exec"), namespace)
    build = namespace["_flash_reentrant_build_module"]

    build(FakeEngine(checkpointing=True))
    # order matters: enabling checkpointing first captures the graph before any input requires
    # grad, so asserting mere presence would pass on a broken shim.
    assert calls[0] == "require_grads"
    assert calls[1] == ("gc_enable", {"gradient_checkpointing_kwargs": {"use_reentrant": True}})

    # the guard is load-bearing: when verl left checkpointing OFF, touching the module at all
    # would turn on a feature verl deliberately declined and change the memory profile.
    calls.clear()
    build(FakeEngine(checkpointing=False))
    assert calls == []


@contextlib.contextmanager
def _vision_hook_installer(torch_module):
    """yield the rendered vision helper with a stand-in torch visible to it.

    the helper imports torch INSIDE the function body, so the stand-in has to stay in sys.modules
    for the duration of the call rather than just the exec -- which is also the property that lets
    the real shim run inside verl's child, where torch is imported long after sitecustomize.
    """
    source = rl_train.render_reentrant_checkpointing_shim(True, multimodal=True)
    helper_start = source.index("def _flash_install_vision_input_grads")
    helper_end = source.index("def _flash_reentrant_build_module")
    namespace: dict = {}
    exec(source[helper_start:helper_end], namespace)
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch_module
    try:
        yield namespace["_flash_install_vision_input_grads"]
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


class _FakeTensor:
    """the two attributes the hook interrogates, plus the mutation it performs."""

    def __init__(self, *, floating=True):
        self.requires_grad = False
        self._floating = floating

    def is_floating_point(self):
        return self._floating

    def requires_grad_(self, value=True):
        self.requires_grad = value
        return self


class _FakeTorch:
    Tensor = _FakeTensor


class _FakeSubmodule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)

    def forward(self, output):
        for hook in self.hooks:
            hook(self, (), output)
        return output


class _FakeVisionModel:
    """named_modules() the way the hook walks it, at the paths a real vlm exposes."""

    def __init__(self, *, patch_embed_path="visual.patch_embed"):
        self.patch_embed = _FakeSubmodule()
        self.other = _FakeSubmodule()
        self._paths = {patch_embed_path: self.patch_embed, "model.layers.0": self.other}

    def named_modules(self):
        return list(self._paths.items())


def test_the_vision_hook_marks_patch_embed_output_as_requiring_grad():
    # runs the rendered helper for real rather than asserting on its text: a hook registered on the
    # wrong submodule, or one that inspects the output incorrectly, still renders a plausible
    # string. torch is absent from the unit env, so the stand-in reproduces exactly the contract
    # the hook depends on -- `is_floating_point()` and `requires_grad_()` on a `torch.Tensor`.
    model = _FakeVisionModel()
    with _vision_hook_installer(_FakeTorch) as install:
        install(model)
        assert len(model.patch_embed.hooks) == 1
        assert model.other.hooks == [], "the hook was installed on a non-vision submodule"

        output = _FakeTensor()
        assert output.requires_grad is False, "the fixture cannot demonstrate the bug"
        assert model.patch_embed.forward(output).requires_grad is True

        # a tuple output is what a real patch-embed returns; the hook must reach inside it.
        tupled = _FakeTensor()
        model.patch_embed.forward((tupled, None))
        assert tupled.requires_grad is True

        # an integer output (ids, not activations) must be left alone: requires_grad_ on a
        # non-float tensor raises in real torch, taking the run down at the first forward.
        integral = _FakeTensor(floating=False)
        model.patch_embed.forward(integral)
        assert integral.requires_grad is False


def test_the_vision_hook_reports_when_it_finds_no_patch_embed(capsys):
    # a silent no-op is the failure mode being fixed: visual modules training on nothing while the
    # run reports success. if the path ever moves, the log line is what makes that visible.
    with _vision_hook_installer(_FakeTorch) as install:
        install(_FakeVisionModel(patch_embed_path="vision_tower.embeddings"))
    assert "no visual.patch_embed found" in capsys.readouterr().out


def test_the_vision_hook_unwraps_a_peft_model_to_find_the_vision_tower():
    # grpo trains through a peft wrapper, so named_modules() on the wrapper is prefixed. the hook
    # unwraps to the base model first; without that it would find nothing and silently no-op.
    class _PeftWrapped:
        def __init__(self, base):
            self._base = base

        def get_base_model(self):
            return self._base

        def named_modules(self):
            raise AssertionError("the wrapper's own module list must not be walked")

    base = _FakeVisionModel()
    with _vision_hook_installer(_FakeTorch) as install:
        install(_PeftWrapped(base))
    assert len(base.patch_embed.hooks) == 1


def test_the_reentrant_shim_flips_the_flag_and_leaves_uncheckpointed_models_alone():
    # execute the rendered source against a stand-in engine: asserting on the string alone would not
    # catch a patch that never runs, or one that turns checkpointing ON for a model verl left off.
    class _Cfg:
        def __init__(self, on):
            self.enable_gradient_checkpointing = on

    class _Module:
        def __init__(self):
            self.kwargs = None
            self.input_grads = False

        def enable_input_require_grads(self):
            self.input_grads = True

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.kwargs = gradient_checkpointing_kwargs

    class _Engine:
        def __init__(self, on):
            self.model_config = _Cfg(on)
            self.module = _Module()

        def _build_module(self):
            return self.module

    module_stub = types.ModuleType("verl.workers.engine.fsdp.transformer_impl")
    module_stub.FSDPEngine = _Engine
    parents = [
        "verl",
        "verl.workers",
        "verl.workers.engine",
        "verl.workers.engine.fsdp",
    ]
    saved = {name: sys.modules.get(name) for name in [*parents, module_stub.__name__]}
    try:
        for name in parents:
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules[module_stub.__name__] = module_stub
        exec(rl_train.render_reentrant_checkpointing_shim(True), {})
        # checkpointing on -> the flag is put back to reentrant, and the lora-frozen embeddings
        # get input grads so the checkpointed segment actually produces a backward (GRAD-001)
        engine = _Engine(True)
        built = engine._build_module()
        assert built.kwargs == {"use_reentrant": True}
        assert built.input_grads is True
        # checkpointing off -> untouched, so the shim cannot silently raise the memory profile
        off = _Engine(False)
        off_built = off._build_module()
        assert off_built.kwargs is None
        assert off_built.input_grads is False
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_entropy_quantile_shim_is_emitted_only_when_masking_is_requested():
    # 1.0 (and unset) means "keep every token", which is verl's own behavior -- emitting a shim then
    # would patch the loss to do nothing. only a real quantile may put a patch on the import path.
    assert rl_train.render_entropy_quantile_shim(None) == ""
    assert rl_train.render_entropy_quantile_shim(1.0) == ""
    source = rl_train.render_entropy_quantile_shim(0.2)
    assert source
    # trl thresholds at 1 - top_entropy_quantile: keeping the top 20% means cutting at the 0.8
    # quantile. carrying the flash value through unconverted would keep the BOTTOM 20% instead.
    assert "_flash_entropy_threshold_q = 0.8" in source
    assert rl_train._ENTROPY_QUANTILE_MARKER in source


def test_entropy_quantile_shim_refuses_to_wrap_itself_twice():
    # double-wrapping would take the top quantile OF the top quantile: with 0.2 that trains on ~4%
    # of tokens instead of 20%, and nothing in the logs would show it. verified numerically against
    # verl's real ppo_loss -- without this guard the loss drifted from -0.0428 to -0.0251.
    source = rl_train.render_entropy_quantile_shim(0.2)
    assert '_flash_entropy_masked", False)' in source
    assert "_flash_entropy_masked_ppo_loss._flash_entropy_masked = True" in source


def test_entropy_quantile_shim_masks_only_the_policy_gradient_term():
    # trl multiplies per_token_loss by the entropy mask and THEN adds the kl term, so kl and the
    # entropy bonus stay on the full response mask. masking inside ppo_loss itself would shrink all
    # three. the shim therefore wraps get_policy_loss_fn, not the aggregation.
    source = rl_train.render_entropy_quantile_shim(0.2)
    assert "_flash_losses.get_policy_loss_fn = _flash_masked_policy_loss_fn" in source
    assert 'kwargs["response_mask"] = _flash_high_entropy_mask' in source
    # equivalence also needs a mask-independent denominator, which is why flash pins this mode.
    assert _overrides_cfg()["loss_agg_mode"] == "seq-mean-token-sum-norm"


def test_entropy_quantile_overrides_enable_verl_entropy_and_stay_off_by_default():
    # the shim reads model_output["entropy"], which verl only populates when calculate_entropy is
    # set. flash's recipe has entropy_coeff 0, so nothing else would turn it on.
    assert "actor_rollout_ref.actor.calculate_entropy=True" in rl_train.build_verl_overrides(
        _overrides_cfg(entropy_quantile=0.2)
    )
    assert "actor_rollout_ref.actor.calculate_entropy=True" not in rl_train.build_verl_overrides(
        _overrides_cfg()
    )


def test_resolve_grpo_inputs_no_longer_rejects_entropy_quantile():
    # the guard this replaces raised on any entropy_quantile < 1.0. the shim implements the masking,
    # so the resolver must pass the value through instead of failing the run.
    source = inspect.getsource(rl_train._resolve_grpo_inputs)
    assert "is not yet supported" not in source.split("entropy_quantile")[1].split("\n\n")[0]
    assert '"entropy_quantile": entropy_quantile' in source


def test_stop_sequences_shim_is_emitted_only_when_stop_strings_are_requested():
    assert rl_train.render_stop_sequences_shim(()) == ""
    source = rl_train.render_stop_sequences_shim(("</answer>", "\n\nQ:"))
    assert source
    # the exact list must survive into the child verbatim, escaping included -- a mangled delimiter
    # would silently never fire and the run would look normal.
    assert "_flash_stop_sequences = ['</answer>', '\\n\\nQ:']" in source
    assert rl_train._STOP_SEQUENCES_MARKER in source


def test_stop_sequences_shim_patches_the_per_sample_params_not_the_config():
    # _run_agent_loop receives the per-sample dict AFTER verl applies its validate/greedy overrides,
    # so patching there keeps stop strings on eval rollouts too -- matching trl, where the stop list
    # lives in generation_kwargs and is not swapped out for validation.
    source = rl_train.render_stop_sequences_shim(("</answer>",))
    assert "AgentLoopWorker._run_agent_loop" in source
    assert 'params["stop"] = list(_flash_stop_sequences)' in source
    # the dict is copied before mutation: verl reuses sample_sampling_params across the batch.
    assert "params = dict(sampling_params)" in source


def test_stop_sequences_shim_refuses_to_wrap_itself_twice():
    source = rl_train.render_stop_sequences_shim(("</answer>",))
    assert '_flash_stop_patched", False)' in source


def test_image_pad_ban_shim_is_emitted_only_on_a_multimodal_job():
    # a text run has no image-pad token to ban, and injecting a logit_bias key into every rollout's
    # sampling params would change sampling on jobs that never asked for it.
    assert rl_train.render_image_pad_ban_shim(None) == ""
    source = rl_train.render_image_pad_ban_shim(151655)
    assert source
    assert "151655" in source
    assert rl_train._IMAGE_PAD_BAN_MARKER in source
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
    source = rl_train.render_stop_sequences_shim(
        ("</answer>",)
    ) + rl_train.render_image_pad_ban_shim(151655)
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
    source = inspect.getsource(rl_train.run_rl_train)
    assert 'render_image_pad_ban_shim(inp["image_pad_token_id"])' in source
    combined = rl_train.render_stop_sequences_shim(
        ("</answer>",)
    ) + rl_train.render_image_pad_ban_shim(151655)
    compile(combined, "sitecustomize.py", "exec")


def test_stop_sequences_gate_off_truncated_completion_masking():
    # main couples these: stop-string rollouts do not end on EOS, so masking truncated completions
    # would wrongly drop every one of them. the verl resolver must inherit that coupling, not
    # re-derive it.
    source = inspect.getsource(rl_train._resolve_grpo_inputs)
    assert "_w.grpo_mask_truncated_completions(_t)" in source
    assert not W.grpo_mask_truncated_completions(SimpleNamespace(stop_sequences=("</answer>",)))
    assert W.grpo_mask_truncated_completions(SimpleNamespace(stop_sequences=()))


def test_all_shims_compose_into_one_sitecustomize():
    # python imports sitecustomize once, so a second file would never load. the renderers must
    # concatenate into a single source rather than each owning a file.
    source = inspect.getsource(rl_train.run_rl_train)
    assert 'render_entropy_quantile_shim(inp["entropy_quantile"])' in source
    assert 'render_stop_sequences_shim(inp["stop_sequences"])' in source
    assert 'render_structured_outputs_shim(inp["structured_outputs"])' in source
    assert 'render_exact_save_steps_shim(inp["save_at_steps"], inp["steps"])' in source
    combined = (
        rl_train.render_entropy_quantile_shim(0.2)
        + rl_train.render_stop_sequences_shim(("</answer>",))
        + rl_train.render_structured_outputs_shim({"json": {"type": "object"}})
        + rl_train.render_exact_save_steps_shim((7, 13), 20)
    )
    compile(combined, "sitecustomize.py", "exec")


def test_exact_save_steps_shim_is_emitted_only_when_exact_saves_are_requested():
    # without exact saves verl's own save_every cadence is already what flash wants, so there is
    # nothing to suppress and the shim must stay out of the child's import path.
    assert rl_train.render_exact_save_steps_shim((), 20) == ""
    source = rl_train.render_exact_save_steps_shim((7, 13), 20)
    assert source
    assert rl_train._EXACT_SAVE_STEPS_MARKER in source


def test_exact_save_steps_shim_keeps_required_steps_and_the_final_step():
    # the gcd of the required steps makes verl save a SUPERSET (gcd(7,13) == 1 is a full-state dump
    # every step). the shim drops the writes flash never asked for -- but losing a required step
    # fails the run, and losing the final step leaves the final publish with no source checkpoint.
    source = rl_train.render_exact_save_steps_shim((7, 13), 20)
    assert "_flash_required_save_steps = frozenset((7, 13))" in source
    assert "_flash_total_steps = 20" in source
    assert "if step not in _flash_required_save_steps and step != _flash_total_steps:" in source
    # it reads the step off the instance: verl's _save_checkpoint takes no step argument.
    assert "step = int(self.global_steps)" in source


def test_exact_save_steps_shim_refuses_to_wrap_itself_twice():
    source = rl_train.render_exact_save_steps_shim((7,), 20)
    assert '"_flash_save_patched", False' in source
    assert "_flash_save_patched = True" in source


def test_structured_outputs_shim_is_emitted_only_when_a_constraint_is_requested():
    assert rl_train.render_structured_outputs_shim(None) == ""
    assert rl_train.render_structured_outputs_shim({}) == ""
    spec = {"json": {"type": "object", "properties": {"a": {"type": "string"}}}}
    source = rl_train.render_structured_outputs_shim(spec)
    assert source
    assert repr(spec) in source
    assert rl_train._STRUCTURED_OUTPUTS_MARKER in source


def test_structured_outputs_shim_wraps_the_spec_rather_than_passing_a_raw_dict():
    # the whole point: vllm ACCEPTS a raw dict, passes _verify_args, and then stores a plain dict
    # with no .json attribute -- constraining nothing, silently. trl wraps it in its colocate path,
    # which is why flash's trl path hands over a plain dict; on verl nothing wraps it, so the shim
    # must, or the run trains unconstrained and looks completely normal.
    source = rl_train.render_structured_outputs_shim({"json": {"type": "object"}})
    assert "StructuredOutputsParams as _FlashStructuredOutputsParams" in source
    assert (
        'params["structured_outputs"] = _FlashStructuredOutputsParams(**_flash_structured_outputs)'
        in source
    )
    # built per request, not once: vllm resolves the backend on first use and caches it on the
    # instance, so a shared object would leak that resolution across requests.
    assert "params = dict(sampling_params)" in source


def test_structured_outputs_shim_refuses_to_wrap_itself_twice():
    source = rl_train.render_structured_outputs_shim({"json": {"type": "object"}})
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
        source = rl_train.render_kl_ref_adapter_shim(True)
        exec(compile(source, "sitecustomize.py", "exec"), {})
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
    return impl.FSDPEngine


def test_kl_ref_adapter_shim_is_emitted_only_for_a_warm_start():
    # a fresh-start run has no sft adapter to anchor to, so verl's bare-base reference is already
    # what flash wants and the patch must stay out of the child's import path.
    assert rl_train.render_kl_ref_adapter_shim(False) == ""
    source = rl_train.render_kl_ref_adapter_shim(True)
    assert source
    assert rl_train._KL_REF_ADAPTER_MARKER in source


def test_kl_ref_adapter_shim_is_wired_only_when_warm_start_and_kl_are_both_on():
    # with kl off no reference logprob is ever consumed, so patching disable_adapter would add a
    # failure mode and buy nothing. both conditions have to gate the renderer, not just warm start.
    source = inspect.getsource(rl_train.run_rl_train)
    assert "render_kl_ref_adapter_shim(" in source
    assert 'bool(inp["warmstart_adapter"]) and float(inp["kl_coef"]) > 0' in source
    combined = rl_train.render_exact_save_steps_shim(
        (7, 13), 20
    ) + rl_train.render_kl_ref_adapter_shim(True)
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
    assert key in rl_train.build_verl_overrides(
        _overrides_cfg(thinking=True, structured_outputs=spec)
    )
    # thinking off -> no reasoning phase to protect; no constraint -> the grammar gate never runs.
    for off in (
        {"thinking": False, "structured_outputs": spec},
        {"thinking": True, "structured_outputs": None},
    ):
        assert not [
            o
            for o in rl_train.build_verl_overrides(_overrides_cfg(**off))
            if "reasoning_parser" in o
        ]


def test_build_verl_overrides_enable_fused_linear_ce():
    # 32k GRPO must not materialize [tokens, vocab] logits; fused torch-backend linear-CE
    # computes logprobs from hidden states in chunks (numerically exact).
    o = rl_train.build_verl_overrides(_overrides_cfg())
    assert "actor_rollout_ref.model.use_fused_kernels=True" in o
    assert "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch" in o


def test_model_revision_resolves_pinned_snapshot_for_verl():
    # model_revision no longer fails closed: prefetch pins the revision and verl gets the pinned
    # snapshot dir as model.path (a bare repo id would resolve the cached "main" ref offline).
    import inspect

    resolver_src = inspect.getsource(rl_train._resolve_grpo_inputs)
    assert "model_revision pinning is not yet supported" not in resolver_src
    # assert on the resolver being CALLED, not on snapshot_download's keywords appearing inline:
    # the resolution moved into _cached_model_path (shared with sft/opd), so pinning the argument
    # spelling here would only re-assert where the code happens to live today.
    run_src = inspect.getsource(rl_train.run_rl_train)
    assert '_cached_model_path(inp["model_id"], inp["model_revision"])' in run_src
    helper_src = inspect.getsource(sft_train._cached_model_path)
    assert "local_files_only=True" in helper_src


def test_unpinned_model_also_resolves_a_snapshot_dir_for_verl():
    """An EMPTY model_revision must resolve a real snapshot dir too, not pass the bare repo id.

    verl runs with HF_HUB_OFFLINE=1, so a bare repo id resolves only through cache symlinks that
    are best-effort on this worker. When they do not land, verl raises "does not appear to have a
    file named pytorch_model.bin or model.safetensors" -- a PERMANENT OSError, thrown after the GPU
    is already rented, so the user is billed for a pod that never trains. The pinned branch was
    always resolved; the unpinned one is the path most runs take, and it was the one left bare.

    _cached_model_path raises RetriableInfraError instead, so an unresolvable cache relands the run
    on a healthy worker. Assert it is called UNCONDITIONALLY -- outside any `if model_revision`
    branch -- because a resolver reachable on only one branch is exactly the defect.
    """
    import ast
    import inspect
    import textwrap

    fn = ast.parse(textwrap.dedent(inspect.getsource(rl_train.run_rl_train))).body[0]

    def _calls(node):
        return [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_cached_model_path"
        ]

    assert _calls(fn), "run_rl_train must resolve the model path through _cached_model_path"
    # every call sits at statement level in the function body, so none is guarded by a revision test
    guarded = [c for stmt in fn.body if isinstance(stmt, ast.If) for c in _calls(stmt)]
    assert not guarded, "the resolver must run for an unpinned revision too, not only a pinned one"


def test_pinned_snapshot_dir_is_what_reaches_verl_model_path():
    # resolving the pinned snapshot is only half the invariant: the RESOLVED path has to be the value
    # verl gets as actor_rollout_ref.model.path. passing inp["model_id"] here would resolve the cached
    # "main" ref offline and silently train the wrong commit, with every other assertion still green.
    # read the call with ast, not a substring: the argument list is multi-line and reformats.
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(rl_train.run_rl_train)))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_build_verl_training_cfg"
    ]
    assert len(calls) == 1
    model_id = next(k for k in calls[0].keywords if k.arg == "model_id")
    assert isinstance(model_id.value, ast.Name)
    assert model_id.value.id == "model_path_for_verl"


# ------------------------------- resume (VERL-018) -------------------------------
def test_build_verl_overrides_enables_resume_mode():
    # without resume_mode=auto verl ignores a staged checkpoint and silently restarts at step 0.
    o = rl_train.build_verl_overrides(_overrides_cfg())
    assert "trainer.resume_mode=auto" in o


def test_restore_verl_resume_is_a_noop_without_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_train._w, "hf_resume_checkpoint", lambda *a, **k: None)
    assert rl_train._restore_verl_resume(str(tmp_path)) == 0
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


def test_restore_verl_resume_stages_the_checkpoint_where_verl_looks(tmp_path, monkeypatch):
    src = tmp_path / "checkpoint-7"
    (src / "actor").mkdir(parents=True)
    (src / "actor" / "model.safetensors").write_text("weights")
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(rl_train._w, "hf_resume_checkpoint", lambda *a, **k: str(src))

    assert rl_train._restore_verl_resume(str(local_dir)) == 7
    # verl discovers the checkpoint through this marker plus the global_step_N layout.
    assert (local_dir / "latest_checkpointed_iteration.txt").read_text().strip() == "7"
    assert (local_dir / "global_step_7" / "actor" / "model.safetensors").read_text() == "weights"


def test_restore_verl_resume_rejects_an_unparseable_checkpoint_path(tmp_path, monkeypatch):
    bad = tmp_path / "not-a-checkpoint"
    bad.mkdir()
    monkeypatch.setattr(rl_train._w, "hf_resume_checkpoint", lambda *a, **k: str(bad))
    with pytest.raises(RuntimeError, match="invalid GRPO resume checkpoint path"):
        rl_train._restore_verl_resume(str(tmp_path / "ckpt"))


def _write_step(local_dir, step):
    d = local_dir / f"global_step_{step}"
    (d / "actor").mkdir(parents=True)
    (local_dir / "latest_checkpointed_iteration.txt").write_text(str(step))
    return d


def test_resume_uploader_uploads_each_completed_step(tmp_path):
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    seen = []
    uploader = rl_train._VerlResumeUploader(str(local_dir), resume_step=0)

    import flash.engine.worker.rl_train as mod

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
    import flash.engine.worker.rl_train as mod

    original = mod._w.upload_resume_checkpoint
    mod._w.upload_resume_checkpoint = lambda step, path, **k: seen.append(int(step))
    try:
        _write_step(local_dir, 5)
        uploader = rl_train._VerlResumeUploader(str(local_dir), resume_step=5)
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
    import flash.engine.worker.rl_train as mod

    def boom(step, path, **k):
        raise RuntimeError("hf is down")

    original = mod._w.upload_resume_checkpoint
    mod._w.upload_resume_checkpoint = boom
    try:
        _write_step(local_dir, 2)
        uploader = rl_train._VerlResumeUploader(str(local_dir), resume_step=0)
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
        rl_train._check_grpo_had_a_gradient([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])


def test_grpo_gradient_check_admits_a_run_with_spread_on_any_step():
    # zero spread on some steps is legitimate (a converged run, or one unlucky all-equal group), so
    # the guard must key on "no step ever had spread" rather than "some step had none".
    rl_train._check_grpo_had_a_gradient([0.4, 0.6], [0.0, 1.5])
    rl_train._check_grpo_had_a_gradient([0.4], [2.0])


def test_grpo_gradient_check_rejects_reward_metrics_without_advantage_metrics():
    # both series are parsed off the same verl log line, so advantages missing while rewards are
    # present means the parse regressed. without this the spread check silently cannot fire.
    with pytest.raises(RuntimeError, match="no advantage metrics"):
        rl_train._check_grpo_had_a_gradient([1.0], [])


def test_grpo_gradient_check_still_rejects_an_unconsulted_reward_bridge():
    with pytest.raises(RuntimeError, match="never consulted"):
        rl_train._check_grpo_had_a_gradient([], [])


def test_advantage_spread_is_parsed_from_a_real_verl_step_line():
    # the guard is only as good as this parse: verl namespaces both keys under critic/ even though
    # grpo runs without a critic, and emits them outside its use_critic branch
    # (verl/trainer/ppo/metric_utils.py), so they are present for every grpo step.
    line = (
        "step:1 - critic/rewards/mean:1.0 - critic/rewards/max:1.0 - critic/rewards/min:1.0 - "
        "critic/advantages/mean:0.0 - critic/advantages/max:0.0 - critic/advantages/min:0.0 - "
        "actor/pg_loss:0.0"
    )
    adv_max = backend_common.parse_verl_metric(line, "critic/advantages/max")
    adv_min = backend_common.parse_verl_metric(line, "critic/advantages/min")
    assert adv_max == 0.0
    assert adv_min == 0.0
    # this is the exact shape of the run in ISSUES VERL-064: healthy reward, zero spread.
    with pytest.raises(RuntimeError, match="zero advantage spread"):
        rl_train._check_grpo_had_a_gradient([1.0], [adv_max - adv_min])

    varied = line.replace("critic/advantages/max:0.0", "critic/advantages/max:0.67").replace(
        "critic/advantages/min:0.0", "critic/advantages/min:-0.33"
    )
    spread = backend_common.parse_verl_metric(
        varied, "critic/advantages/max"
    ) - backend_common.parse_verl_metric(varied, "critic/advantages/min")
    assert spread > 0.0
    rl_train._check_grpo_had_a_gradient([0.5], [spread])


def test_run_rl_train_wires_the_gradient_check_into_the_publish_path():
    # a helper nothing calls is not a guard. assert the training path actually invokes it, and that
    # it does so before the adapter export rather than after a publish has already happened.
    source = inspect.getsource(rl_train.run_rl_train)
    assert "_check_grpo_had_a_gradient(" in source
    assert "resumed=bool(resume_step)," in source
    assert "already_complete=bool(resume_step) and resume_step >= expected_steps," in source
    assert source.index("_check_grpo_had_a_gradient") < source.index("export_peft_adapter")
    # and that the spread series it passes is actually collected from the child's output.
    assert 'parse_verl_metric(line, "critic/advantages/max")' in source
    assert 'parse_verl_metric(line, "critic/advantages/min")' in source


def test_grpo_gradient_check_abstains_for_a_resumed_run():
    # a run resuming at step 9 of 10 observes ONE step; if that group ties, the spread history is
    # all-zero even though the restored weights carry nine steps of real updates. rejecting it would
    # throw away a correctly trained policy, so the resumed case abstains from the spread verdict.
    rl_train._check_grpo_had_a_gradient([1.0], [0.0], resumed=True)
    # abstaining is scoped to the spread verdict only: the parse/wiring checks still apply, because
    # a missing metric stream is a regression no matter where training started.
    with pytest.raises(RuntimeError, match="no advantage metrics"):
        rl_train._check_grpo_had_a_gradient([1.0], [], resumed=True)
    with pytest.raises(RuntimeError, match="never consulted"):
        rl_train._check_grpo_had_a_gradient([], [], resumed=True)
    # and a FRESH run with the same all-zero history is still rejected -- the abstention must be
    # about the resume boundary, not a weakening of the guard.
    with pytest.raises(RuntimeError, match="zero advantage spread"):
        rl_train._check_grpo_had_a_gradient([1.0], [0.0], resumed=False)


def test_grpo_gradient_check_accepts_a_resume_that_is_already_complete():
    # a resume whose checkpoint ALREADY sits at the target runs zero steps: verl computes
    # current_epoch = global_steps // len(dataloader), the epoch range comes out empty, and the child
    # exits 0 having emitted no metric lines. both histories are therefore empty for a policy that is
    # fully trained -- which the empty-history branch would report as a reward-bridge wiring
    # regression, failing a complete run. (trl exempted exactly this via
    # _grpo_resume_already_complete; the port dropped the exemption.)
    rl_train._check_grpo_had_a_gradient([], [], resumed=True, already_complete=True)

    # the exemption is ONLY for the zero-step case. a resume that ran steps and produced no metrics
    # is still a wiring regression, and a fresh run can never claim it.
    with pytest.raises(RuntimeError, match="never consulted"):
        rl_train._check_grpo_had_a_gradient([], [], resumed=True, already_complete=False)
    with pytest.raises(RuntimeError, match="never consulted"):
        rl_train._check_grpo_had_a_gradient([], [], resumed=False)


def test_resume_uploader_withholds_deployables_until_spread_appears():
    # the uploader publishes servable adapters WHILE training runs, so a degenerate-reward run would
    # make untrained adapters durable minutes before the end-of-run guard fails the run.
    spread: list[float] = []
    uploader = rl_train._VerlResumeUploader(
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

    uploader = rl_train._VerlResumeUploader("/nonexistent", resume_step=0, had_gradient=boom)
    assert uploader._deployable_allowed() is False
    # and no callback at all means no gate, which is the resume-only configuration.
    assert rl_train._VerlResumeUploader("/nonexistent", resume_step=0)._deployable_allowed() is True


def test_run_rl_train_gates_midtraining_deployables_and_exempts_resumes():
    source = inspect.getsource(rl_train.run_rl_train)
    # the gate must be wired into the uploader, not merely available on it.
    assert "had_gradient=(" in source
    # a resumed run publishes as before: its restored weights already carry earlier updates that
    # this worker's spread history cannot speak for.
    assert "if resume_step" in source.split("had_gradient=(")[1].split(")")[0] + ")"
    # the spread series must be declared before the uploader closes over it, or the closure raises
    # NameError the first time the drain thread consults it.
    assert source.index("adv_spread_history: list[float] = []") < source.index(
        "_VerlResumeUploader("
    )


def _patch_stage_and_publish(monkeypatch, staged: list[int], published: list[int]) -> None:
    """record staging and publication separately, without running model_merger or touching hf.

    they are patched as two seams because the production code separates them: staging is bounded by
    verl's checkpoint retention, publication by the gradient gate.
    """
    monkeypatch.setattr(
        rl_train._VerlResumeUploader,
        "_stage_deployable",
        lambda self, step, path: (staged.append(int(step)), f"{path}-adapter")[1],
    )
    monkeypatch.setattr(
        rl_train._VerlResumeUploader,
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
        rl_train._w,
        "upload_resume_checkpoint",
        lambda step, path, **k: uploaded.append(int(step)),
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, staged, published)
    _write_step(local_dir, 3)
    uploader = rl_train._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(3,),
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
        rl_train._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
    )
    _patch_stage_and_publish(monkeypatch, [], [])
    _write_step(local_dir, 3)
    uploader = rl_train._VerlResumeUploader(
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
        rl_train._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
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

    uploader = rl_train._VerlResumeUploader(
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
        rl_train._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
    )
    _patch_stage_and_publish(monkeypatch, [], published)
    _write_step(local_dir, 4)
    # resumed at exactly the required step, and no adapter on hf for it, so it stays uncredited.
    monkeypatch.setattr(rl_train, "_deployable_adapter_on_hf", lambda step: False)
    uploader = rl_train._VerlResumeUploader(str(local_dir), resume_step=4, required_steps=(4,))
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
        rl_train._w,
        "upload_resume_checkpoint",
        lambda step, path, **k: uploaded.append(int(step)),
        raising=False,
    )
    monkeypatch.setattr(
        rl_train._VerlResumeUploader,
        "_stage_deployable",
        lambda self, step, path: (staged.append(int(step)), f"{path}-adapter")[1],
    )
    monkeypatch.setattr(
        rl_train._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (_ for _ in ()).throw(AssertionError("gate is shut")),
    )
    # the checkpoint must become visible AFTER a sweep has already decided what to scan, with stop
    # already set -- writing it between sweeps does not discriminate, because the next top-of-loop
    # scan picks it up either way. the tracker read is that boundary: _pending only accepts steps at
    # or below the value it returns, so a step written right after that read is invisible to the
    # sweep holding it and visible to the next one.
    real_completed = rl_train._VerlResumeUploader._completed_step
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
        rl_train._VerlResumeUploader, "_completed_step", _completed_then_race, raising=True
    )
    uploader = rl_train._VerlResumeUploader(
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
        rl_train._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
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
        rl_train._VerlResumeUploader, "_stage_deployable", _stage_requiring_its_source
    )
    monkeypatch.setattr(
        rl_train._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (
            published.append(int(step)),
            self.published_steps.add(step),
        )[0],
    )
    for step in (1, 2, 3, 4):
        (local_dir / f"global_step_{step}").mkdir()
    _write_step(local_dir, 4)
    uploader = rl_train._VerlResumeUploader(
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
        rl_train._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
    )

    def _stage_failing_on_step_2(self, step, path):
        if int(step) == 2:
            raise RuntimeError("model_merger ran out of memory")
        return f"{path}-adapter"

    monkeypatch.setattr(rl_train._VerlResumeUploader, "_stage_deployable", _stage_failing_on_step_2)
    monkeypatch.setattr(
        rl_train._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (
            published.append(int(step)),
            self.published_steps.add(step),
        )[0],
    )
    for step in (1, 2):
        (local_dir / f"global_step_{step}").mkdir()
    _write_step(local_dir, 2)
    uploader = rl_train._VerlResumeUploader(
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
        rl_train._w, "upload_resume_checkpoint", lambda step, path, **k: True, raising=False
    )
    _patch_stage_and_publish(monkeypatch, [], [])
    uploader = rl_train._VerlResumeUploader(
        str(local_dir), resume_step=0, required_steps=(6,), had_gradient=lambda: False
    )
    uploader.start()
    uploader.stop()
    # both failures are live: the deployable was withheld, and the run produced no spread. the
    # gradient verdict must be the one that speaks.
    with pytest.raises(RuntimeError, match="zero advantage spread on all"):
        rl_train._check_grpo_had_a_gradient([0.5, 0.5], [0.0, 0.0], resumed=False)
    with pytest.raises(RuntimeError, match="required saves were not durably published"):
        uploader.raise_if_incomplete()
    # ordering is asserted at the call site: the verdict precedes stop()/raise_if_incomplete().
    # match on the call name alone -- the argument list spans several lines, so pinning an argument
    # would make this fail on a reformat rather than on a reordering, which is the real invariant.
    source = inspect.getsource(rl_train.run_rl_train)
    assert source.count("_check_grpo_had_a_gradient(") == 1
    assert source.count("resume_uploader.raise_if_incomplete()") == 1
    verdict = source.index("_check_grpo_had_a_gradient(")
    completeness = source.index("resume_uploader.raise_if_incomplete()")
    assert verdict < completeness


def test_train_notes_report_whether_the_run_resumed():
    # without this a resumed run is indistinguishable from a fresh one in train_meta (trl reports it).
    inp = _notes_inp()
    common = _notes_common()
    fresh = rl_train._build_verl_train_notes(inp, **common)
    assert fresh["resumed"] is False
    resumed = rl_train._build_verl_train_notes(inp, **common, resumed=True)
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
        "model_id": "Qwen/Qwen3-4B",
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
    # of how it ran. the retired trl path reported these; without them a verl run cannot be compared to a
    # trl one, and the fp8-kv decision (resolved per-card at runtime) leaves no trace at all.
    notes = rl_train._build_verl_train_notes(
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
    notes = rl_train._build_verl_train_notes(_notes_inp(), **_notes_common(), fp8_kv=False)
    assert notes["vllm_kv_cache_dtype"] is None


def test_train_notes_omit_wandb_identity_when_wandb_is_off():
    # verl logs from its own interpreter, so flash's in-process wandb.run is empty on this path and
    # the names come from the config. recording them when the logger is off would point a reader at
    # a dashboard run that was never created.
    notes = rl_train._build_verl_train_notes(_notes_inp(), **_notes_common())
    assert notes["wandb_project"] is None
    assert notes["wandb_run_name"] is None
    # a sampler that never saw a card must not report a fabricated zero-gb peak.
    assert notes["peak_gpu_gb"] is None
    assert notes["device_peak_gpu_gb"] is None


def test_verl_grpo_logs_to_the_runs_own_wandb_project_and_name():
    # a hardcoded project/experiment pair lands every grpo run in one wandb experiment, so
    # concurrent runs overwrite each other's curves and an explicit [wandb] project is ignored. the
    # sft and opd verl backends already resolve both from the spec.
    o = rl_train.build_verl_overrides(
        _overrides_cfg(project_name="acme", experiment_name="flash-rl-run123")
    )
    assert "trainer.project_name=acme" in o
    assert "trainer.experiment_name=flash-rl-run123" in o
    assert "trainer.project_name=flash_verl" not in o
    assert "trainer.experiment_name=grpo" not in o


def test_verl_grpo_wandb_names_survive_hydra_special_characters():
    # a run name is user-settable via [wandb] run_name; an unquoted '=' or ',' would split the
    # override and hydra would compose a different key entirely.
    o = rl_train.build_verl_overrides(_overrides_cfg(experiment_name="run=a,b"))
    assert 'trainer.experiment_name="run=a,b"' in o


def test_train_notes_record_the_batch_shape_one_step_consumed():
    # the retired trl path reported the batch shape, so without it a verl run's reward curve cannot be read
    # against a trl one: the same step count at a different batch size is a different experiment.
    notes = rl_train._build_verl_train_notes(_notes_inp(), **_notes_common())
    assert notes["max_completion_len"] == 512
    assert notes["prompts_per_step"] == 8
    # ulysses shards along the sequence, so dp stays 1 and one optimizer step sees the whole batch.
    assert notes["generations_per_step"] == 8 * 4


def test_train_notes_report_token_bounded_batching_as_unset_not_fabricated():
    # trl fixes a per-device SEQUENCE count; verl bounds the backward pass by tokens, so a
    # micro-batch holds however many sequences fit and varies step to step. reporting a number here
    # would read as directly comparable to trl's when nothing enforces it.
    notes = rl_train._build_verl_train_notes(_notes_inp(), **_notes_common())
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
    """the shape rl_train asks of AutoProcessor on a multimodal job.

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


def _capability_resolve(
    monkeypatch,
    env,
    train=None,
    overrides=None,
    processor=None,
    model="Qwen/Qwen3.5-0.8B",
    gpu_count=1,
):
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
            "model": model,
            "algorithm": "grpo",
            "train": {"batch_size": 4, "epochs": 1, **(train or {})},
            "gpu": {"count": gpu_count},
        }
    )
    monkeypatch.setattr(_PkgW, "JOB_SPEC", spec, raising=False)
    monkeypatch.setattr(_PkgW, "SEED", 42, raising=False)
    monkeypatch.setattr(_PkgW, "THINKING", False, raising=False)
    monkeypatch.setattr(_PkgW, "require_active_env", lambda: env, raising=False)
    monkeypatch.setattr(_PkgW, "grpo_overrides", lambda: dict(overrides or {}), raising=False)
    monkeypatch.setattr(_PkgW, "grpo_mask_truncated_completions", lambda t: False, raising=False)
    monkeypatch.setattr(_PkgW, "load_tokenizer", lambda *a, **k: _Tokenizer(), raising=False)
    monkeypatch.setattr(rl_train, "seed_training_rngs", lambda seed: None)
    monkeypatch.setattr(rl_train, "model_max_position_embeddings", lambda *a, **k: 32768)
    return rl_train._resolve_grpo_inputs()


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
    cfg = rl_train._build_verl_training_cfg(
        inp,
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_id=inp["model_id"],
        thinking=False,
        loggers=["console"],
        fp8_kv=False,
        enforce_eager=False,
        attention_backend=None,
        mm_encoder_attn_backend=None,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    o = rl_train.build_verl_overrides(cfg)
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=flash_grpo_multi_turn" in o


def test_single_turn_env_leaves_the_agent_loop_on_verl_default(monkeypatch):
    # the override must be GATED: emitting it on a single-turn job would route text rollouts
    # through the multi-turn bridge, which has no episode state for them.
    inp = _capability_resolve(monkeypatch, _capability_env())
    assert inp["multi_turn"] is False
    assert not [
        o for o in rl_train.build_verl_overrides(_overrides_cfg()) if "default_agent_loop" in o
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
    inp = _capability_resolve(monkeypatch, _capability_env(image_uri=_capability_image_uri()))
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


def test_35b_grpo_warm_start_requires_fused_expert_targets(monkeypatch, tmp_path):
    import flash.engine.worker.adapter as adapter_mod

    adapter_dir = tmp_path / "warmstart"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 32, "lora_alpha": 64}), encoding="utf-8"
    )
    monkeypatch.setattr(adapter_mod, "_download_adapter", lambda ref: str(adapter_dir))

    with pytest.raises(ValueError, match="omits required expert targets"):
        _capability_resolve(
            monkeypatch,
            _capability_env(),
            train={"init_from_adapter": "org/pre-expert-adapter"},
            model="Qwen/Qwen3.6-35B-A3B",
            gpu_count=2,
        )


def test_per_turn_credit_assignment_is_accepted_on_single_turn_envs(monkeypatch, capsys):
    # per_turn only diverges from per_episode when there is more than one assistant turn to credit.
    # the multi-turn/tool guard above already rejects every env that could get there, so anything
    # reaching here is single-turn and the two modes are the same objective. rejecting a key that
    # is merely redundant would break configs that are asking for nothing wrong.
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
        "max_completion": 512,
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

    emitted = rl_train.multi_turn_child_env(
        _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=False
    )
    read_by_child = set(
        re.findall(
            r"os\.environ(?:\.get)?[\[(]\"(FLASH_VERL_[A-Z_]+)\"", inspect.getsource(grpo_multiturn)
        )
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
    emitted = rl_train.multi_turn_child_env(
        _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=False
    )
    assert emitted["VERL_USE_EXTERNAL_MODULES"] == "flash_grpo_plugin"
    # the module name must match the file actually copied in, or the import fails at child startup.
    assert ("grpo_plugin.py", "flash_grpo_plugin.py") in rl_train.MULTI_TURN_CHILD_MODULES


def test_multi_turn_child_env_serializes_values_the_child_can_parse_back():
    # every value crosses as a string. the child json-loads two of them and int()s two others, so a
    # repr() or a str(frozenset) here would raise mid-rollout rather than at launch.
    emitted = rl_train.multi_turn_child_env(
        _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=True
    )
    assert all(isinstance(value, str) for value in emitted.values())
    assert int(emitted["FLASH_VERL_MAX_TURNS"]) == 4
    assert int(emitted["FLASH_VERL_MAX_MODEL_LEN"]) == 8192
    assert int(emitted["FLASH_VERL_MAX_COMPLETION_TOKENS"]) == 512
    assert json.loads(emitted["FLASH_VERL_STOP_SEQUENCES"]) == ["</answer>"]
    # sorted, not set-ordered: the child compares against this list every turn and an unstable order
    # would make halting depend on hash seed.
    assert json.loads(emitted["FLASH_VERL_EOS_TOKEN_IDS"]) == [151643, 151645]
    assert emitted["FLASH_VERL_THINKING"] == "1"
    assert (
        rl_train.multi_turn_child_env(
            _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=False
        )["FLASH_VERL_THINKING"]
        == "0"
    )


def test_multi_turn_child_modules_are_copied_under_the_names_they_import_each_other_by(tmp_path):
    # each module falls back to a flat `flash_`-prefixed import of the next one. copying a file
    # under the wrong name leaves that fallback unresolvable, and the child's ImportError arrives
    # inside verl's plugin loader where it reads as a verl problem.
    written = rl_train.copy_multi_turn_child_modules(str(tmp_path))
    names = {os.path.basename(path) for path in written}
    assert names == {name for _, name in rl_train.MULTI_TURN_CHILD_MODULES}
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
    for path in rl_train.copy_multi_turn_child_modules(str(tmp_path)):
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
    # the assignment sits inside run_rl_train, past the subprocess launch.
    src = inspect.getsource(rl_train.run_rl_train)
    assert 'if shim_source or inp["multi_turn"]:' in src, (
        "PYTHONPATH is not extended for a multi-turn job with no other shim"
    )
    assert 'if inp["multi_turn"]:\n        copy_multi_turn_child_modules(shim_dir)' in src


# ---------------------- multi-turn per-turn generation cap ----------------------
def test_the_child_caps_each_turn_at_max_completion_tokens_not_the_whole_episode():
    # parity with the retired trl driver, which passed per_turn_max_tokens=max_completion into
    # _turn_budget. without the cap the first turn may spend the ENTIRE episode budget: the other
    # two bounds are transcript-wide, so a 4096-token engine window lets turn one generate 4096
    # tokens and leaves nothing for the rest of the episode. asserted on the child's own source
    # because the alternative is a full engine rollout to observe one min().
    from flash.engine.worker import grpo_multiturn

    body = " ".join(inspect.getsource(grpo_multiturn).split())
    assert 'max_completion_tokens = int(os.environ["FLASH_VERL_MAX_COMPLETION_TOKENS"])' in body
    assert (
        "max_tokens = min( max_completion_tokens, max_model_len - len(prefix_ids), "
        "response_capacity - len(response_ids), )" in body
    ), "the per-turn cap is not one of the three budgets bounding a turn"


def test_the_child_puts_no_deadline_on_a_bridge_call():
    # MultiTurnBridge serializes every env touch behind one lock, so with a whole generation in
    # flight a request spends most of its life QUEUED rather than being served slowly. a client
    # timeout there fails healthy episodes for arriving Nth -- a function of batch size, not of
    # the environment. a genuinely wedged env is caught by the stall watchdog instead, which
    # measures training progress rather than one request.
    from flash.engine.worker import grpo_multiturn

    body = " ".join(inspect.getsource(grpo_multiturn.post_json).split())
    assert "urllib.request.urlopen(request) as response" in body
    assert "timeout=" not in body.split('"""')[-1], (
        "a client-side deadline is back on the bridge call"
    )


def test_the_parent_sends_the_per_turn_cap_from_the_configured_completion_budget():
    # the cap is only real if the parent actually exports it; the child KeyErrors mid-rollout
    # otherwise, after the engine is already up and paid for.
    emitted = rl_train.multi_turn_child_env(
        _multi_turn_inp(max_completion=321), reward_url="http://127.0.0.1:9/", thinking=False
    )
    assert emitted["FLASH_VERL_MAX_COMPLETION_TOKENS"] == "321"


# ---------------------- multi-turn bridge routes ----------------------
class _BridgeEnv:
    """the four calls MultiTurnBridge drives, recording what it was asked."""

    def __init__(
        self, *, replies=None, done_after=1, episode=1.0, max_episode_turns=None, prompt=None
    ):
        self.replies = replies if replies is not None else [{"role": "user", "content": "next"}]
        self.done_after = done_after
        self.episode = episode
        self.max_episode_turns = max_episode_turns
        self.prompt = list(prompt or ())
        self.recorded: list[str] = []
        self.scored: list[dict] = []

    def new_rollout_state(self, example):
        # `messages` starts as a COPY of `prompt` and turns are appended onto it, matching
        # flash.envs.adapter.new_rollout_state. anything reading the transcript has to account
        # for that seeding rather than treating `messages` as turns-only.
        state: dict = {
            "example": example,
            "prompt": list(self.prompt),
            "messages": [dict(message) for message in self.prompt],
        }
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


def _bridge(env, *, max_turns=4, examples=None, env_prompts=None, **kwargs):
    examples = examples if examples is not None else [{"index": 0}, {"index": 1}]
    if env_prompts is None:
        # what dataset preparation would have produced for these examples: the same opening the
        # env's own start_episode returns. tests that care about the two DISAGREEING pass their own.
        env_prompts = [[dict(message) for message in getattr(env, "prompt", ())] for _ in examples]
    return rl_train.MultiTurnBridge(
        env, examples, env_prompts=env_prompts, max_turns=max_turns, **kwargs
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


def test_bridge_start_adopts_the_datasets_prompt_over_a_second_start_episode():
    # `new_rollout_state` calls `start_episode` a SECOND time; dataset preparation already called
    # it to build the prompt the child generates against. an env that randomizes per episode hands
    # back a DIFFERENT opening here, and the run would then score a response generated for prompt A
    # against a reward computed for prompt B. the env below returns a fresh secret every call, the
    # way a randomized env does.
    class _RandomizingEnv(_BridgeEnv):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def new_rollout_state(self, example):
            self.calls += 1
            return {
                "example": example,
                "prompt": [{"role": "user", "content": f"secret-{self.calls}"}],
                "messages": [{"role": "user", "content": f"secret-{self.calls}"}],
            }

    env = _RandomizingEnv()
    dataset_prompt = [{"role": "user", "content": "secret-0"}]
    bridge = _bridge(env, examples=[{"index": 0}], env_prompts=[dataset_prompt])
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})

    scored = env.scored[0]
    assert scored["prompt"] == dataset_prompt, (
        "the episode was scored against a prompt the model never saw"
    )
    assert scored["messages"][0] == dataset_prompt[0]


def test_bridge_rejects_prompts_that_do_not_align_with_its_examples():
    # the two are indexed by the SAME integer the child sends. a length mismatch means some index
    # reads the wrong row's prompt, or IndexErrors mid-rollout; both are worth failing at
    # construction, before the engine is paid for.
    with pytest.raises(ValueError, match="one-to-one"):
        rl_train.MultiTurnBridge(
            _BridgeEnv(), [{"index": 0}, {"index": 1}], env_prompts=[[]], max_turns=4
        )


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
    bridge = _bridge(
        env, examples=[{"index": 0}], on_episode_scored=lambda *row: recorded.append(row)
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


def test_the_recorded_transcript_excludes_the_prompt_it_was_seeded_from():
    """`messages` starts as a copy of `prompt`, so it is not the transcript -- it CONTAINS it.

    Publishing the whole list repeats the prompt inside `completion` when it already rides the
    sample as `prompt_tail`: the reader sees it twice, and the doubled text eats the payload budget
    a long episode needs for its actual turns (codex[bot]).
    """
    recorded: list[tuple] = []
    prompt = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is 3+4"},
    ]
    env = _BridgeEnv(episode=0.75, prompt=prompt)
    bridge = _bridge(
        env, examples=[{"index": 0}], on_episode_scored=lambda *row: recorded.append(row)
    )
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "7"})
    bridge.score({"session_id": "a", "turn_count": 1})

    published_prompt, transcript, _ = recorded[0]
    assert published_prompt == prompt
    assert [m["content"] for m in transcript] == ["7"]


def test_the_transcript_slice_keeps_a_turn_that_merely_looks_like_a_prompt_message():
    """Slice by LENGTH, not by equality.

    An env may legitimately produce a turn identical to a prompt message -- an echo env, a
    re-issued instruction, a two-token action space. Dropping matches instead of the seeded prefix
    would silently truncate exactly those episodes, and the sample would understate the turn count
    that the reward was computed over.
    """
    recorded: list[tuple] = []
    prompt = [{"role": "user", "content": "repeat after me: go"}]
    env = _BridgeEnv(episode=1.0, done_after=99, prompt=prompt)
    bridge = _bridge(
        env, examples=[{"index": 0}], on_episode_scored=lambda *row: recorded.append(row)
    )
    bridge.start({"index": 0, "session_id": "a"})
    # the env replies with the prompt message verbatim, then the model answers.
    env.replies = [dict(prompt[0])]
    bridge.step({"session_id": "a", "completion_text": "go"})
    bridge.score({"session_id": "a", "turn_count": 1})

    transcript = recorded[0][1]
    assert [m["content"] for m in transcript] == ["go", "repeat after me: go"]


def test_the_recorded_episode_is_the_zeroed_reward_not_the_raw_nan():
    # the sample carries the reward the rollout actually trained on. publishing nan here would show
    # a reward in the log that no advantage was ever computed from.
    recorded: list[tuple] = []
    bridge = _bridge(
        _BridgeEnv(episode=float("nan")),
        examples=[{"index": 0}],
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
    bridge = _bridge(
        env, examples=[{"index": 0}], on_episode_scored=lambda *row: recorded.append(row)
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
    bridge = _bridge(
        env,
        examples=[{"index": 0}],
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


@pytest.mark.parametrize(
    "abort", [{"truncated": True}, {"skip_reason": "length"}], ids=["truncated", "skipped"]
)
def test_a_first_turn_abort_is_still_shown_in_the_sample(abort):
    """The turn the episode DIED on is the one worth reading, and it is trained on either way.

    `step` keeps an unusable turn out of `messages` so the env never scores it. Building the sample
    from that state alone therefore publishes an empty completion for a first-turn truncation --
    a model that generated right up to its token limit reads as a model that generated nothing
    (codex[bot])."""
    recorded: list[tuple] = []
    env = _BridgeEnv(done_after=99)
    bridge = _bridge(
        env, examples=[{"index": 0}], on_episode_scored=lambda *row: recorded.append(row)
    )
    bridge.start({"index": 0, "session_id": "a"})
    assert bridge.step({"session_id": "a", "completion_text": "ran out of ro", **abort}) == {
        "terminal": True,
        "messages": [],
    }
    bridge.score({"session_id": "a", "turn_count": 1})

    assert env.recorded == [], "the env was shown a turn it must never score"
    assert [m.get("content") for m in recorded[0][1]] == ["ran out of ro"]


def test_the_env_never_scores_the_aborted_turn_it_is_shown_in_the_sample():
    # the two sides are separate on purpose: the sample gains the turn, the scored state does not.
    # asserting only on the sample would pass an implementation that also appended it to `messages`,
    # which is the truncated-text-gets-graded bug the abort branch exists to prevent.
    env = _BridgeEnv(done_after=99)
    bridge = _bridge(env, examples=[{"index": 0}])
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "good turn"})
    bridge.step({"session_id": "a", "completion_text": "cut off", "truncated": True})
    bridge.score({"session_id": "a", "turn_count": 2})

    scored = [m.get("content") for m in env.scored[0]["messages"]]
    assert "cut off" not in scored, "the truncated turn reached the env's scoring state"
    assert "good turn" in scored


# ---------------------- multi-turn batched episode scoring ----------------------
def test_concurrently_finished_episodes_are_scored_in_one_env_call():
    # a generation is prompts_per_step * group_size episodes. scoring them one at a time turns one
    # judge round into hundreds of serial round-trips with the gpu idle. the env below records the
    # SIZE of every batch it is handed, so a regression to per-episode scoring shows up as many
    # calls of one rather than one call of many.
    class _BatchRecordingEnv(_BridgeEnv):
        def __init__(self):
            super().__init__()
            self.batch_sizes: list[int] = []

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            self.batch_sizes.append(len(items))
            return [RolloutReward(episode=1.0, turns=None) for _ in items]

    env = _BatchRecordingEnv()
    examples = [{"index": i} for i in range(8)]
    bridge = _bridge(env, examples=examples)
    for i in range(8):
        bridge.start({"index": i, "session_id": f"s{i}"})
        bridge.step({"session_id": f"s{i}", "completion_text": "answer"})

    scores: dict[int, float] = {}
    threads = [
        threading.Thread(
            target=lambda i=i: scores.__setitem__(
                i, bridge.score({"session_id": f"s{i}", "turn_count": 1})["score"]
            )
        )
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not [t for t in threads if t.is_alive()], "a scoring episode never completed"

    assert scores == dict.fromkeys(range(8), 1.0)
    assert max(env.batch_sizes) > 1, (
        f"every episode was scored alone: batch sizes {env.batch_sizes}"
    )
    assert sum(env.batch_sizes) == 8, "an episode was scored twice or not at all"


def test_a_batched_score_reaches_the_env_under_the_same_lock_every_other_call_takes():
    # scoring is batched to shorten how long the lock is held, NOT to drop it. `reward_thread_safe`
    # licenses racing the scorer against ITSELF; it says nothing about racing it against a
    # concurrent episode's `env_reply`, and no env contract permits that.
    class _LockObservingEnv(_BridgeEnv):
        def __init__(self):
            super().__init__()
            self.held_during_scoring: list[bool] = []

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            acquired = bridge._lock.acquire(blocking=False)
            self.held_during_scoring.append(not acquired)
            if acquired:
                bridge._lock.release()
            return [RolloutReward(episode=1.0, turns=None) for _ in items]

    env = _LockObservingEnv()
    bridge = _bridge(env, examples=[{"index": 0}])
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})
    assert env.held_during_scoring == [True], "the env was scored without the session lock held"


def test_a_failing_batch_fails_every_episode_in_it_rather_than_hanging_them():
    # the scoring thread scatters results back to waiters. if a raise completed only the waiter
    # that provoked it, every OTHER episode in that batch would block on its event forever and the
    # run would wedge with no error -- the worst shape of failure, since the stall watchdog only
    # fires 25 minutes later.
    #
    # the batch has to actually CONTAIN several episodes for this to test anything. an earlier
    # version just started four threads and hoped; each arrived alone, `_take_batch` returned it by
    # itself, and `batch[0]` WAS the whole batch -- so it passed against code that completed only
    # the first waiter (VERL-100). the gate below parks the scorer inside the env call until the
    # rest have queued, which is the interleave the assertion is actually about.
    entered = threading.Event()
    release = threading.Event()

    batch_sizes: list[int] = []

    class _FailingEnv(_BridgeEnv):
        def rollout_rewards_many(self, items):
            batch_sizes.append(len(items))
            entered.set()
            release.wait(timeout=30)
            raise RuntimeError("judge is down")

    bridge = _bridge(_FailingEnv(), examples=[{"index": i} for i in range(5)])
    for i in range(5):
        bridge.start({"index": i, "session_id": f"s{i}"})
        bridge.step({"session_id": f"s{i}", "completion_text": "answer"})

    outcomes: list[str] = []
    outcome_lock = threading.Lock()

    def _record(session_id):
        result = _score_capturing(bridge, session_id)
        with outcome_lock:
            outcomes.append(result)

    # s0 goes first and parks the scorer inside the env call. the other four queue behind it and
    # are taken as ONE batch on the next pass -- that is the batch whose failure must scatter.
    # daemon: a waiter this test proves is WEDGED never returns, and a non-daemon thread would
    # then block interpreter shutdown after the assertion had already fired -- turning a named
    # failure into a hang that reports nothing and burns the whole ci timeout.
    first = threading.Thread(target=lambda: _record("s0"), daemon=True)
    first.start()
    assert entered.wait(timeout=30), "the scorer never reached the env"

    rest = [threading.Thread(target=lambda i=i: _record(f"s{i}"), daemon=True) for i in range(1, 5)]
    for thread in rest:
        thread.start()

    # wait for all four to be QUEUED before releasing the in-flight call. `_pending` is the wrong
    # field to watch: the batcher drains it into `_in_flight` the moment it takes a batch, so
    # sampling it races to zero and the assertion fails against working code. count the waiters
    # that exist in either place instead.
    # do NOT gate on `_scorer._pending`: `bridge.score()` takes the bridge lock to look up its
    # session, and `_score_batch` holds that same lock for the whole parked env call -- so the four
    # sit inside `score()` and never reach the batcher's queue while s0 is in flight. `_pending`
    # stays 0 the entire time and any wait on it times out against perfectly good code. they
    # coalesce the moment the lock is released, which is what `batch_sizes` below actually proves.
    time.sleep(0.5)
    release.set()

    threads = [first, *rest]
    for thread in threads:
        thread.join(timeout=30)
    alive = [t for t in threads if t.is_alive()]
    assert not alive, f"{len(alive)} episode(s) left hanging by a failed batch"
    assert sorted(outcomes) == ["judge is down"] * 5
    # the point of the test: a batch of MORE THAN ONE actually failed together. without this the
    # queue-depth wait above would still pass if each episode were scored on its own, which is the
    # shape under which the buggy `batch[0]`-only completion is indistinguishable from correct.
    assert max(batch_sizes) > 1, f"no shared batch ever formed; env saw batches {batch_sizes}"


def _score_capturing(bridge, session_id):
    """score one episode, returning the failure text instead of raising (for use off-thread)."""
    try:
        bridge.score({"session_id": session_id, "turn_count": 1})
    except Exception as error:
        return str(error)
    return "scored"


def test_the_scoring_thread_starts_on_first_use_rather_than_an_explicit_call():
    # the batcher's consumer thread is what drains the queue. constructed-but-never-started is a
    # silent wedge: `score` blocks on an event nothing will ever set, and it is invisible until a
    # multi-turn run hangs on real gpu. binding the start to first use makes that state
    # unreachable, so no caller can forget.
    #
    # the score call is bounded on its own thread rather than made inline: if the start is ever
    # dropped, an inline call would HANG here, and a hanging test is only marginally better than one
    # that cannot fail -- it burns the whole ci timeout and reports no assertion. off-thread with a
    # join deadline turns that same regression into a named failure.
    bridge = _bridge(_BridgeEnv(), examples=[{"index": 0}])
    assert bridge._scorer._thread is None, "the thread was started before any episode needed it"
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})

    scored: list = []
    worker = threading.Thread(
        target=lambda: scored.append(bridge.score({"session_id": "a", "turn_count": 1})),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "score blocked forever: nothing is draining the queue"
    assert scored == [{"score": 1.0}]
    assert bridge._scorer._thread is not None


def test_bridge_shutdown_stops_the_scoring_thread():
    # the run's finally calls this before the server goes down. a thread left running would keep
    # the worker process alive past the point flash considers the run finished.
    bridge = _bridge(_BridgeEnv(), examples=[{"index": 0}])
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})
    thread = bridge._scorer._thread
    assert thread is not None
    assert thread.is_alive()
    bridge.shutdown()
    thread.join(timeout=10)
    assert not thread.is_alive(), "the scoring thread outlived the bridge"


def test_the_run_shuts_the_bridge_down_before_the_server_it_is_mounted_on():
    # ordering matters: the server's routes block on the scoring thread, so stopping the server
    # first would strand a scoring episode on an event nothing will ever set.
    src = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    assert "multi_turn_bridge.shutdown()" in src
    assert src.index("multi_turn_bridge.shutdown()") < src.index("server.shutdown()")


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
    server, url = rl_train.start_reward_server(
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
    src = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    assert src.count("MultiTurnBridge(") == 1
    assert (
        "MultiTurnBridge( env, rollout_examples, "
        "# index-aligned with rollout_examples: build_grpo_prompt_dataset preserves order. "
        'env_prompts=[p["env_prompt"] for p in prompts], max_turns=int(inp["max_turns"]), '
        'per_turn_credit=bool(inp["per_turn_credit"]), '
        "on_episode_scored=observability.record, )" in src
    )
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
    cfg = rl_train._build_verl_training_cfg(
        inp,
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_id=inp["model_id"],
        thinking=False,
        loggers=["console"],
        fp8_kv=False,
        enforce_eager=False,
        attention_backend=None,
        mm_encoder_attn_backend=None,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    assert f"data.max_response_length={inp['max_response_len']}" in rl_train.build_verl_overrides(
        cfg
    )


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
    monkeypatch.setattr(W, "grpo_overrides", dict, raising=False)
    monkeypatch.setattr(W, "grpo_mask_truncated_completions", lambda train: False, raising=False)
    monkeypatch.setattr(W, "load_tokenizer", lambda *args, **kwargs: _Tokenizer(), raising=False)
    monkeypatch.setattr(rl_train, "seed_training_rngs", lambda seed: None)
    monkeypatch.setattr(rl_train, "model_max_position_embeddings", lambda *a, **k: 40960)
    return rl_train._resolve_grpo_inputs()


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
    notes = rl_train._build_verl_train_notes(
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
    notes = rl_train._build_verl_train_notes(
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
    notes = rl_train._build_verl_train_notes(
        inp,
        steps_run=10,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
        reward_profile=_profile(latency),
        step_intervals=[10.0] * 10,
    )
    assert notes["reward_gpu_idle_fraction"] == pytest.approx(0.8, abs=0.01)


def test_train_notes_do_not_publish_the_serial_idle_projection_when_batching(monkeypatch):
    inp = _resolved_inputs_for_notes(monkeypatch)
    completions = inp["prompts_per_step"] * inp["group_size"]
    notes = rl_train._build_verl_train_notes(
        inp,
        steps_run=10,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
        reward_profile=_profile(8.0 / completions),
        step_intervals=[10.0] * 10,
        reward_bridge_batching=True,
    )

    assert notes["reward_bridge_batching"] is True
    assert notes["reward_seconds_per_completion"] == 8.0 / completions
    assert notes["reward_gpu_idle_fraction"] is None


def test_idle_fraction_is_none_when_grading_exceeds_the_measured_step(monkeypatch):
    """Grading that fills the whole step leaves no gpu-bound remainder to divide.

    The profile and the observed wall disagree here (a warm-up latency that no longer holds, or a
    step wall dominated by something else), and neither can arbitrate, so the honest record is no
    reading rather than a fabricated 100%.
    """
    inp = _resolved_inputs_for_notes(monkeypatch)
    completions = inp["prompts_per_step"] * inp["group_size"]
    notes = rl_train._build_verl_train_notes(
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
    notes = rl_train._build_verl_train_notes(
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
    notes = rl_train._build_verl_train_notes(
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
    notes = rl_train._build_verl_train_notes(
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
    assert rl_train._step_intervals([100.0, 110.0, 122.0]) == [10.0, 12.0]
    # a single step line bounds no whole step: nothing is known about what came before it or after.
    assert rl_train._step_intervals([100.0]) == []
    assert rl_train._step_intervals([]) == []


def test_the_profile_hook_returns_its_reading_to_the_caller():
    """_log_reward_profile must RETURN the profile, not only print it.

    Asserted against the real hook rather than a fixture: the wiring under test is that the run
    body can capture what the profiler measured.
    """

    class Env:
        def sft_completion(self, example):
            return [{"role": "assistant", "content": "an answer worth grading"}]

    profile = rl_train._log_reward_profile(
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
    src = inspect.getsource(rl_train.run_rl_train)
    assert "_log_reward_profile(" in src, "the hook is never called"
    assert "reward_profile = " in src, "the hook's reading is discarded"
    assert "reward_profile=reward_profile" in src, "the reading never reaches train_meta"
    assert 'reward_bridge_batching=not inp["multi_turn"]' in src


def test_the_reward_profiler_is_skipped_on_multi_turn():
    """The profiler times the SINGLE-TURN grading path, which a multi-turn env does not have.

    Source-level for the same reason as above. Running it on a multi-turn env would call
    env.reward/scores_breakdown on one completion -- a call that env's contract does not define --
    and record the resulting number as if it described the episode reward path.
    """
    src = inspect.getsource(rl_train.run_rl_train)
    profile_call = src[src.index("reward_profile = ") : src.index("multi_turn_bridge = ")]
    assert 'if inp["multi_turn"]' in profile_call, "the profiler is not gated off multi-turn"
    assert "None" in profile_call, "multi-turn must record no profile rather than a wrong one"


# ---------------- reward observability: the buffer and the heartbeat drain ----------------
def _score_buffer(env, *, prompts=None, examples=None, generation_size=0):
    """`_score`'s grade-then-record pair, against a real buffer and fake env.

    `_score` itself is a local of a body that needs a model, a dataset and a verl interpreter to
    reach, so the two calls it makes are made here directly; the wiring that they ARE what `_score`
    does is asserted separately below, on its source.

    `generation_size` defaults to 0 -- the boundary stays caller-driven -- so a test that only cares
    about grading never trips the counted seal.
    """
    buffer = RewardObservabilityBuffer(generation_size=generation_size)
    rollout_examples = examples if examples is not None else [{"gt": "7"}]
    message_prompts = prompts if prompts is not None else ["prompt-0"]

    def score(index: int, solution_str: str) -> float:
        breakdowns: list[dict[str, float] | None] = []
        value = rl_train.score_single_turn(
            env,
            solution_str,
            rollout_examples[int(index)],
            tok=None,
            thinking=False,
            prompt_opened_thinking=False,
            think_penalty=0.0,
            breakdowns=breakdowns,
        )
        buffer.record(message_prompts[int(index)], solution_str, value, breakdowns)
        return value

    return score, buffer


def test_score_batch_grades_before_it_records():
    """User grading must finish before the observability lock is taken per result."""
    src = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    body = src[src.index("def _score_batch(requests:") :]
    body = body[: body.index("def _score_for_profile")]

    assert body.count("observability.record(") == 1
    assert body.index("scored = score_single_turn_batch(") < body.index("observability.record(")


def test_the_recorded_prompt_is_the_one_the_batched_completion_was_graded_against():
    """Each scattered sample must use the same request index for its example and prompt."""
    src = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    body = src[src.index("def _score_batch(requests:") :]
    body = body[: body.index("def _score_for_profile")]

    assert (
        "(solution_str, rollout_examples[int(index)]) for index, solution_str in requests" in body
    )
    assert (
        "for (index, solution_str), (score, breakdowns) in zip(requests, scored, strict=True):"
        in body
    )
    assert (
        "observability.record(message_prompts[int(index)], solution_str, score, breakdowns)" in body
    )


@pytest.mark.usefixtures("_identity_graded")
def test_the_buffer_keeps_the_rollout_sample_and_its_named_breakdown():
    score, buffer = _score_buffer(_NamedBreakdownEnv())

    assert score(0, "7") == 1.0
    assert buffer.latest() == ("prompt-0", "7", 1.0)
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0, "quality": 0.5}


@pytest.mark.usefixtures("_identity_graded")
def test_the_buffers_are_bounded_when_the_generation_never_closes():
    # nothing bounds how many completions arrive before a boundary. the sample buffer evicts; the
    # metric accumulator must not GROW instead -- this process is already memory-tight.
    score, buffer = _score_buffer(_NamedBreakdownEnv())
    for _ in range(5000):
        score(0, "7")

    assert len(buffer._samples) == RewardObservabilityBuffer._SAMPLE_BUFFER_LIMIT
    # one float per NAME, whatever the completion count: the env reports two.
    assert set(buffer._pending_totals) == {"success", "quality"}
    assert buffer._pending_count == 5000


@pytest.mark.usefixtures("_identity_graded")
def test_every_completion_counts_toward_the_mean_however_large_the_generation():
    """A generation is ``batch_size * group_size`` completions, both unbounded, so no retention cap
    can hold one. Dropping the overflow biases the published mean toward whichever completions were
    graded last -- here, a run that succeeded early and failed late would report a flat 0
    (codex[bot])."""
    score, buffer = _score_buffer(
        _CountingBreakdownEnv(),
        prompts=["prompt-0"],
        examples=[{"gt": "7"}],
    )
    total = 5000
    for _ in range(total):
        score(0, "7")
    buffer.close_generation(1)

    # the env numbers each grading, so the true mean of `n` is the mean of 0..total-1.
    assert buffer.heartbeat_fields()["reward_metrics"]["n"] == (total - 1) / 2


@pytest.mark.usefixtures("_identity_graded")
def test_eviction_drops_the_oldest_rollouts_and_keeps_the_newest():
    """Over the limit, what SURVIVES has to be the recent end: samples answer "what is the model
    doing now", so evicting the newest would pin a stalled run's diagnostics to its oldest
    gradings. A length-only assertion passes either way -- this reads the retained values."""
    score, buffer = _score_buffer(
        _CountingBreakdownEnv(),
        prompts=["prompt-0"],
        examples=[{"gt": "7"}],
    )
    total = RewardObservabilityBuffer._SAMPLE_BUFFER_LIMIT + 50
    for _ in range(total):
        score(0, "7")

    assert buffer.latest()[2] == float(total - 1)
    assert [row[2] for row in buffer._samples] == [
        float(n) for n in range(total - RewardObservabilityBuffer._SAMPLE_BUFFER_LIMIT, total)
    ]


@pytest.mark.usefixtures("_identity_graded")
def test_a_run_with_no_rollouts_yet_omits_sampled_completions():
    # symmetric with the reward_metrics case below: an empty list on the wire reads as "this step
    # produced no rollouts", which is a different claim from "none have been scored yet".
    buffer = RewardObservabilityBuffer()

    assert "sampled_completions" not in buffer.heartbeat_fields()


@pytest.mark.usefixtures("_identity_graded")
def test_both_signals_pass_their_wire_bounds_before_publication():
    """The bounds are what make a payload safe to commit, and both are applied HERE -- the caller
    publishes whatever this returns. Asserted through the buffer rather than on the helpers (which
    have their own tests) so dropping either call from the publisher fails.

    The sample side asserts on neutralization and the cap together, because on this path both come
    from `select_rollout_samples` itself rather than from a second bounding pass.
    """
    buffer = RewardObservabilityBuffer()
    buffer.record("prompt-0", "done\x1b[2Jcleared", 1.0)
    for i in range(1, 5):  # distinct prompts, so dedup can't stand in for the cap
        buffer.record(f"prompt-{i}", f"completion-{i}", float(i))
    buffer.close_generation(1)
    with buffer._lock:  # 13 names, over the 12-metric cap
        buffer._latest_metrics.update({f"m{i}": 1.0 for i in range(13)})

    fields = buffer.heartbeat_fields()

    assert len(fields["reward_metrics"]) == 12
    assert len(fields["sampled_completions"]) == 3
    # a raw escape would let a rollout repaint the terminal of whoever runs `flash runs log`.
    assert "\x1b" not in fields["sampled_completions"][0]["completion"]


@pytest.mark.usefixtures("_identity_graded")
def test_a_reward_that_is_not_a_float_is_coerced_at_the_boundary():
    # rewards arrive from user grading code and go out as json. coercing on the way IN keeps a
    # numpy scalar or a bool from reaching the serializer a heartbeat away from the call site.
    buffer = RewardObservabilityBuffer()
    buffer.record("prompt-0", "completion-0", np.float32(0.25))

    reward = buffer.latest()[2]
    assert type(reward) is float
    assert reward == 0.25


@pytest.mark.usefixtures("_identity_graded")
def test_a_scalar_reward_run_publishes_no_named_metrics_at_all():
    # end to end for the empty case: a scores_breakdown-less env must reach the wire with the key
    # ABSENT, not with every name flattened to 0 by an empty-dict denominator.
    score, buffer = _score_buffer(_RewardOnlyEnv())
    score(0, "the answer is 7")
    buffer.close_generation(1)

    fields = buffer.heartbeat_fields()
    assert "reward_metrics" not in fields
    assert len(fields["sampled_completions"]) == 1


@pytest.mark.usefixtures("_identity_graded")
def test_the_heartbeat_publishes_averaged_metrics_and_bounded_samples():
    score, buffer = _score_buffer(
        _NamedBreakdownEnv(),
        prompts=["p0", "p1", "p2", "p3"],
        examples=[{"gt": "7"}, {"gt": "7"}, {"gt": "9"}, {"gt": "9"}],
    )
    for index, completion in enumerate(["7", "7", "7", "7"]):
        score(index, completion)

    buffer.close_generation(5)
    fields = buffer.heartbeat_fields()

    # two of four completions matched their gt, so success averages 0.5 across the generation.
    assert fields["reward_metrics"] == {"success": 0.5, "quality": 0.5}
    assert len(fields["sampled_completions"]) == 3  # hard cap, four rollouts buffered
    assert {s["generated_at_step"] for s in fields["sampled_completions"]} == {5}
    assert [s["reward"] for s in fields["sampled_completions"]] == [1.0, 1.0, 0.0]


@pytest.mark.usefixtures("_identity_graded")
def test_a_heartbeat_landing_mid_generation_republishes_the_last_complete_one():
    """A 30s liveness tick is not a generation boundary, and publishing on it is latency-biased.

    The completions that finish first are the fast ones -- short outputs, cache hits, envs that
    grade without i/o. A drain on the heartbeat cadence therefore reports THAT subset's mean as the
    step's reward, systematically over-representing whatever is cheap to produce. The reading has to
    stay pinned to the last whole generation until the next boundary seals a new one (codex[bot]).
    """
    score, buffer = _score_buffer(
        _NamedBreakdownEnv(),
        prompts=["p0", "p1"],
        examples=[{"gt": "7"}, {"gt": "7"}],
    )
    score(0, "7")  # the fast completion: success 1.0
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"]["success"] == 1.0

    # generation 2 is under way and only its fast half has been graded.
    score(1, "wrong")  # success 0.0

    mid = buffer.heartbeat_fields()
    assert mid["reward_metrics"]["success"] == 1.0, "published a partial generation's mean"
    assert [s["reward"] for s in mid["sampled_completions"]] == [1.0]

    # the slow half lands, then the boundary: now the whole generation publishes at once.
    score(0, "7")
    buffer.close_generation(2)
    assert buffer.heartbeat_fields()["reward_metrics"]["success"] == 0.5


@pytest.mark.usefixtures("_identity_graded")
def test_the_next_generation_cannot_be_sealed_into_the_step_line_that_is_still_in_flight():
    """The child's stdout is delivered asynchronously, so `step:N` can reach the parent AFTER
    generation N+1 has started scoring. A boundary taken at that moment seals both generations
    under step N and leaves N+1 with nothing of its own to publish (codex[bot]).

    Counting closes the generation on the scoring thread that finishes it, so the in-flight line
    only names what was already sealed."""
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    score(0, "7")
    score(0, "7")  # generation 1 is complete here, whatever stdout is doing
    score(0, "wrong")  # generation 2 begins while `step:1` is still in the pipe

    buffer.close_generation(1)
    first = buffer.heartbeat_fields()
    assert first["reward_metrics"]["success"] == 1.0, "next generation leaked into this mean"
    assert [s["completion"] for s in first["sampled_completions"]] == ["7", "7"]
    assert {s["generated_at_step"] for s in first["sampled_completions"]} == {1}

    score(0, "wrong")
    buffer.close_generation(2)
    second = buffer.heartbeat_fields()
    assert second["reward_metrics"]["success"] == 0.0, "step 2 republished step 1"
    assert {s["generated_at_step"] for s in second["sampled_completions"]} == {2}


@pytest.mark.usefixtures("_identity_graded")
def test_a_generation_that_completes_before_the_previous_step_line_is_not_lost():
    """Stdout can fall a WHOLE generation behind, not just part of one.

    A single "already sealed" flag only remembers one unacknowledged generation, so the second seal
    overwrites the first: generation 1 is dropped and generation 2 publishes under step 1, leaving
    every later step misaligned. Small generations make that window ordinary (cursor, codex[bot])."""
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    score(0, "7")
    score(0, "7")  # generation 1 complete: success 1.0
    score(0, "wrong")
    score(0, "wrong")  # generation 2 complete too, and NEITHER step line has arrived

    buffer.close_generation(1)
    first = buffer.heartbeat_fields()
    assert first["reward_metrics"]["success"] == 1.0, (
        "generation 1 was overwritten before it published"
    )
    assert [s["completion"] for s in first["sampled_completions"]] == ["7", "7"]
    assert {s["generated_at_step"] for s in first["sampled_completions"]} == {1}

    buffer.close_generation(2)
    second = buffer.heartbeat_fields()
    assert second["reward_metrics"]["success"] == 0.0
    assert [s["completion"] for s in second["sampled_completions"]] == ["wrong", "wrong"]
    assert {s["generated_at_step"] for s in second["sampled_completions"]} == {2}


@pytest.mark.usefixtures("_identity_graded")
def test_the_queue_of_unnamed_generations_is_bounded():
    # the queue holds whole generations, each retaining up to _SAMPLE_BUFFER_LIMIT completions. a
    # child that stops printing step lines never drains it, and this process is already memory-tight.
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    for _ in range(4 * RewardObservabilityBuffer._SEALED_QUEUE_LIMIT):
        score(0, "7")

    assert len(buffer._sealed_by_count) == RewardObservabilityBuffer._SEALED_QUEUE_LIMIT


@pytest.mark.usefixtures("_identity_graded")
def test_an_eviction_does_not_shift_every_later_step_onto_the_wrong_generation():
    """Dropping the oldest queued generation is not enough: its `step:N` line still arrives.

    Handing that line to the oldest SURVIVOR consumes a generation whose own line is still coming,
    so the offset never closes -- every step for the rest of the run publishes the next generation's
    output under the previous step's number. One eviction, permanently wrong diagnostics (cursor,
    codex[bot]).
    """
    limit = RewardObservabilityBuffer._SEALED_QUEUE_LIMIT
    buffer = RewardObservabilityBuffer(generation_size=1)
    for i in range(limit + 1):  # one more than the queue holds: generation 0 is evicted
        buffer.record(f"prompt-{i}", f"gen-{i}", float(i))

    assert len(buffer._sealed_by_count) == limit

    # stdout catches up and names them in order. the line for the evicted generation is spent on it.
    published = []
    for step in range(1, limit + 2):
        buffer.close_generation(step)
        published.append(buffer._published[-1][1] if buffer._published else None)

    # step 1's generation is genuinely gone, so its reading is stale rather than another
    # generation's. every step after it is matched to the generation that actually produced it.
    assert published[0] is None
    assert published[1:] == [f"gen-{i}" for i in range(1, limit + 1)]


@pytest.mark.usefixtures("_identity_graded")
def test_the_step_preview_reads_the_generation_that_step_published():
    """The caller closes the generation and then previews it under that same step number.

    A late `step:N` line arrives with generation N+1 already scoring, so the newest recorded sample
    belongs to N+1. Previewing that labels N+1's completion as step N -- the mislabelling the queue
    exists to prevent, reintroduced one line later, and disagreeing with the heartbeat about the
    very same step (codex[bot]).
    """
    buffer = RewardObservabilityBuffer(generation_size=2)
    buffer.record("p", "gen1-a", 1.0)
    buffer.record("p", "gen1-b", 1.0)  # generation 1 sealed by count
    buffer.record("p", "gen2-a", 9.0)  # generation 2 already scoring; `step:1` still in the pipe

    buffer.close_generation(1)

    assert buffer.latest()[1] == "gen1-b", "the preview labelled the next generation as this step"
    # and the heartbeat agrees with it, which is the point of reading the published generation.
    fields = buffer.heartbeat_fields()
    assert [s["completion"] for s in fields["sampled_completions"]] == ["gen1-a", "gen1-b"]


def test_a_preview_before_the_first_boundary_still_shows_a_rollout():
    # the fallback direction: with nothing published yet, the open generation is all there is, and
    # blanking the preview would read as "no rollouts" rather than "no boundary yet".
    buffer = RewardObservabilityBuffer()
    buffer.record("p", "first", 0.5)

    assert buffer.latest() == ("p", "first", 0.5)


@pytest.mark.usefixtures("_identity_graded")
def test_a_component_too_large_to_be_a_float_does_not_fail_the_reward_request():
    """`record` runs OUTSIDE `score_single_turn`'s error guard, so anything it raises 400s the
    reward request and aborts the run. An int larger than a float can hold raises OverflowError,
    which is neither TypeError nor ValueError (codex[bot])."""
    score, buffer = _score_buffer(_OverflowingBreakdownEnv())

    assert score(0, "7") == 1.0  # the total graded fine; only the diagnostic component is unusable
    buffer.close_generation(1)
    metrics = buffer.heartbeat_fields()["reward_metrics"]
    assert metrics["success"] == 1.0, "a usable component was dropped with the unusable one"
    assert metrics["enormous"] == 0.0


def test_the_published_metric_bound_survives_a_value_too_large_to_be_a_float():
    # the same coercion runs again on the publish side, on the heartbeat thread, over a dict the
    # trl callback takes from its caller. escaping there kills liveness reporting for the whole run.
    from flash.engine.worker.heartbeat import _bounded_reward_metrics

    assert _bounded_reward_metrics({"huge": 10**400, "fine": 0.25}) == {"fine": 0.25}


@pytest.mark.usefixtures("_identity_graded")
def test_the_step_line_names_the_generation_the_count_already_sealed():
    """A counted seal publishes under the buffer's OWN ordinal, which is only a guess at what verl
    logged -- it counts from 1 and assumes no skipped or resumed steps. The arriving line carries
    the real number, so dropping the relabel would stamp every sample with a step the run never
    logged, and a reader correlating samples against the loss curve would line them up wrong."""
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    score(0, "7")
    score(0, "7")

    # verl resumed from a checkpoint: its first logged step is 41, not the buffer's internal 1.
    buffer.close_generation(41)

    assert {s["generated_at_step"] for s in buffer.heartbeat_fields()["sampled_completions"]} == {
        41
    }


@pytest.mark.usefixtures("_identity_graded")
def test_a_counted_seal_does_not_disarm_the_boundary_for_later_generations():
    """`_sealed_by_count` holds the generations the count sealed, waiting to be named. A step line
    that names one without taking it off the queue turns every later `close_generation` into a
    relabel of the same entry, so from the second generation on the buffer publishes nothing new --
    metrics frozen at generation 1 while the run continues."""
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    for _ in range(2):
        score(0, "7")
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"]["success"] == 1.0

    for _ in range(2):
        score(0, "wrong")
    buffer.close_generation(2)

    assert buffer.heartbeat_fields()["reward_metrics"]["success"] == 0.0, "boundary went dead"


@pytest.mark.usefixtures("_identity_graded")
def test_a_generation_short_of_its_count_is_still_sealed_by_the_step_line():
    """The count seals a FULL generation; the step line is what seals a short one. If a named
    generation stays on `_sealed_by_count`, the next `close_generation` relabels it instead of
    sealing what is open -- so a generation that lost a completion (a `_score` that raised before
    reaching the bridge) republishes the PREVIOUS generation's numbers under the new step, and
    carries its samples forward. The run reads as healthy at exactly the step where grading broke."""
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    for _ in range(2):
        score(0, "7")
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"]["success"] == 1.0

    # one of the two completions never reached the bridge, so the count cannot fire
    score(0, "wrong")
    buffer.close_generation(2)

    fields = buffer.heartbeat_fields()
    assert fields["reward_metrics"]["success"] == 0.0, "stale generation republished as step 2"
    assert len(fields["sampled_completions"]) == 1, "generation 1's samples leaked into 2"


@pytest.mark.usefixtures("_identity_graded")
def test_a_completion_that_failed_scoring_still_counts_toward_the_denominator():
    """A completion the env could not grade contributes no breakdown but is still part of the
    generation. Counting only the ones that produced a dict would divide by the scored subset --
    biasing every named metric HIGH exactly when scoring is degraded, so a half-broken env reports
    the same number as a healthy one."""
    buffer = RewardObservabilityBuffer()
    buffer.record("prompt-0", "a", 1.0, [{"success": 1.0, "total": 1.0}])
    buffer.record("prompt-0", "b", 0.0, [None])
    buffer.close_generation(1)

    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 0.5}


@pytest.mark.usefixtures("_identity_graded")
def test_a_component_whose_values_are_all_unusable_is_reported_as_zero_not_dropped():
    """Registering a name only once one of its values coerces would delete the component from the
    payload entirely -- and an env whose component is broken for the WHOLE generation is exactly
    when someone needs to see it. A flat 0 reads as "this scored nothing"; absence is
    indistinguishable from "this env has no such component"."""
    score, buffer = _score_buffer(_UnusableComponentEnv())
    score(0, "7")
    buffer.close_generation(1)

    assert buffer.heartbeat_fields()["reward_metrics"] == {
        "broken": 0.0,
        "diverged": 0.0,
        "quality": 0.5,
    }


@pytest.mark.usefixtures("_identity_graded")
def test_a_generation_that_reports_no_components_at_all_leaves_the_metrics_standing():
    """ "No breakdowns" and "breakdowns that all failed" look the same at the seal but mean opposite
    things. A multi-turn episode grades to a scalar and never reports components, so zeroing the
    known metrics for it would publish a scoring outage the env never had -- and the two record
    paths share one buffer, so a run that scores some rows per-completion and some per-episode
    would flip its metrics to 0 on every episode generation."""
    buffer = RewardObservabilityBuffer()
    buffer.record("p", "a", 1.0, [{"success": 1.0, "total": 1.0}])
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0}

    buffer.record("p", "b", 1.0)  # a scored episode, no per-completion breakdown to report
    buffer.close_generation(2)

    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0}, "read as an outage"


@pytest.mark.usefixtures("_identity_graded")
def test_one_non_finite_score_cannot_poison_a_whole_components_mean():
    """Summing a NaN in makes the running total NaN forever: every later completion adds to it and
    the name publishes NaN for the rest of the generation. One diverged grading would take out a
    component that scored fine on every other completion -- and `json.dumps` writes bare `NaN`,
    which is not JSON, so a strict reader rejects the whole heartbeat over it."""
    buffer = RewardObservabilityBuffer()
    buffer.record("p", "a", 1.0, [{"quality": 1.0, "total": 1.0}])
    buffer.record("p", "b", 1.0, [{"quality": float("nan"), "total": 1.0}])
    buffer.record("p", "c", 1.0, [{"quality": 1.0, "total": 1.0}])
    buffer.close_generation(1)

    assert buffer.heartbeat_fields()["reward_metrics"] == {"quality": 2.0 / 3.0}


@pytest.mark.usefixtures("_identity_graded")
def test_the_published_payload_is_read_under_one_acquisition():
    """Both fields describe the same generation, so reading them under separate acquisitions lets a
    seal land between the two and the payload tears: metrics from generation N+1 shipped beside
    samples from N. Asserted on the source because reproducing the interleave needs a scoring
    thread to win a specific race -- a test that passes when it loses proves nothing."""
    body = inspect.getsource(RewardObservabilityBuffer.heartbeat_fields)
    snapshot = body[body.index("with self._lock:") : body.index("fields: dict = {}")]

    assert snapshot.count("with self._lock:") == 1, "the two reads can straddle a seal"
    assert "self._latest_metrics" in snapshot
    assert "self._published" in snapshot


def test_the_generation_size_is_the_configured_rollout_count():
    # the counted boundary is only correct if it counts a whole generation. verl runs with
    # test_freq=-1 and val_before_train=False, so every completion reaching the bridge is one of
    # these -- a validation pass would desynchronize the count from the step lines.
    src = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    construction = src[src.index("RewardObservabilityBuffer(") :]
    construction = construction[: construction.index("wandb_link")]
    assert 'generation_size=int(inp["prompts_per_step"]) * int(inp["group_size"])' in construction
    overrides = rl_train.build_verl_overrides(_overrides_cfg())
    assert "trainer.test_freq=-1" in overrides
    assert "trainer.val_before_train=False" in overrides


@pytest.mark.usefixtures("_identity_graded")
def test_samples_carry_the_step_they_were_generated_at_not_the_current_one():
    """`generated_at_step` names the generation that PRODUCED the completion.

    The buffer is rolling, so a drain that stamps everything in it with the current step
    re-publishes older rollouts as if the model had just produced them -- a reader watching for
    behaviour change sees old text under a new step number (codex[bot]).
    """
    score, buffer = _score_buffer(_NamedBreakdownEnv())
    score(0, "7")
    buffer.close_generation(4)

    # steps 5 and 6 generate nothing (the run is stalled, or verl logged without new rollouts).
    buffer.close_generation(5)
    buffer.close_generation(6)

    fields = buffer.heartbeat_fields()
    assert {s["generated_at_step"] for s in fields["sampled_completions"]} == {4}


@pytest.mark.usefixtures("_identity_graded")
def test_the_drain_clears_pending_breakdowns_and_then_repeats_the_last_reading():
    # the drain CLEARS the pending list. between generations there is nothing new to average, and
    # reporting {} there would blank the metric on every heartbeat that lands mid-generation rather
    # than holding the last real reading.
    score, buffer = _score_buffer(_NamedBreakdownEnv())
    score(0, "7")
    buffer.close_generation(1)

    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0, "quality": 0.5}
    assert buffer._pending_totals == {}
    assert buffer._pending_count == 0
    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0, "quality": 0.5}

    score(0, "wrong")
    buffer.close_generation(2)
    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 0.0, "quality": 0.5}


class _ReleaseHookedLock:
    """The buffer's lock, with a callback fired the instant it is released.

    Landing a grading at that exact moment is what makes the atomicity test deterministic: a real
    thread race can't be, because lock handoff is barging-prone and a sleep long enough to make the
    interleave reliable is a sleep the correct code also passes.
    """

    def __init__(self, lock, on_release):
        self._lock = lock
        self._on_release = on_release

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *exc):
        released = self._lock.__exit__(*exc)
        self._on_release()
        return released


@pytest.mark.usefixtures("_identity_graded")
def test_metrics_and_samples_in_one_payload_describe_the_same_gradings():
    """The boundary's drain and sample seal must be ONE acquisition, or a payload tears.

    A grading landing between them is not lost -- it just belongs to the NEXT generation. What
    breaks is agreement: its sample would ride this generation's publication while its reward
    doesn't reach this generation's metrics, so `sampled_completions` and `reward_metrics` describe
    different gradings and a reader diagnosing a reward drop sees a sample the numbers next to it
    never scored.
    """
    score, buffer = _score_buffer(
        _NamedBreakdownEnv(),
        prompts=["prompt-0", "prompt-1"],  # distinct: both survive the per-prompt dedup
        examples=[{"gt": "7"}, {"gt": "7"}],
    )
    score(0, "7")  # success 1.0

    landed = []

    def _land_a_grading_the_instant_the_lock_drops():
        if landed:
            return  # record() takes the same lock; don't recurse into it
        landed.append(True)
        score(1, "wrong")  # success 0.0

    buffer._lock = _ReleaseHookedLock(buffer._lock, _land_a_grading_the_instant_the_lock_drops)
    buffer.close_generation(1)
    fields = buffer.heartbeat_fields()

    assert landed, "the hook never fired, so this asserts nothing about atomicity"
    # the second grading landed after the whole section, so NEITHER signal carries it.
    assert fields["reward_metrics"] == {"success": 1.0, "quality": 0.5}
    assert [sample["reward"] for sample in fields["sampled_completions"]] == [1.0]


def test_the_first_sample_bearing_heartbeat_is_forced():
    # the liveness daemon can claim a step before the stdout loop reaches it, and a step-gated stage
    # drops a second payload at an already-committed step. without force, the first heartbeat
    # carrying samples is exactly the one most likely to be suppressed.
    src = inspect.getsource(rl_train.run_rl_train)
    forced = src[src.index("if not sent_first_metrics:") :]
    forced = forced[: forced.index("gpu=gpu_diagnostics")]
    assert "force=True" in forced
    assert "**_reward_observability()" in forced


def test_the_liveness_fields_hook_carries_reward_observability():
    # the rl_step liveness wrap is what publishes between stdout lines. without the fields hook
    # merging it, samples would only ever reach the wire on the one forced first-metrics heartbeat.
    src = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    assert 'fields=lambda: {"metrics_last": list(metrics_last), **_reward_observability()}' in src


def test_the_generation_boundary_is_the_step_line_and_the_heartbeat_never_drains():
    """The boundary is verl's step line, and it is the ONLY drain.

    Both halves are load-bearing and neither is reachable from a unit test -- `_reward_observability`
    is a local of a body that needs a model, a dataset and a verl interpreter. If the heartbeat
    drained as well, a 30s tick landing mid-generation would publish the subset of completions that
    had finished by then. If the step line did not close the generation, nothing ever would, and the
    buffer would report its first generation for the whole run.
    """
    src = " ".join(inspect.getsource(rl_train.run_rl_train).split())

    hook = src[src.index("def _reward_observability()") :]
    hook = hook[: hook.index("with liveness_heartbeat(")]
    assert "observability.heartbeat_fields()" in hook
    assert "close_generation" not in hook, "the heartbeat drains; the boundary would be bypassed"

    # sealed on the new-step branch, and BEFORE the preview reads the published rows so the logged
    # sample and the heartbeat describe the same generation.
    stdout_loop = src[src.index("step_box[0] = int(m.group(1))") :]
    assert "observability.close_generation(step_box[0])" in stdout_loop
    assert stdout_loop.index("observability.close_generation(") < stdout_loop.index(
        "samp = observability.latest_for_step(step_box[0])"
    )
    # and the preview asks for THIS step's rows. the unchecked accessor answers with whatever was
    # published last, which on the drop-spend path belongs to an earlier step (cursor).
    assert "observability.latest()" not in stdout_loop, (
        "the preview would print older rows under this step number"
    )


def test_a_step_whose_generation_was_dropped_previews_nothing_rather_than_older_text():
    """The drop-spend path publishes nothing, so the newest rows belong to an EARLIER step.

    `close_generation` spends this step's line on a generation the queue already dropped and
    publishes nothing. `latest` keeps answering with the previous generation's rows, and the caller
    -- which cannot see that no publish happened -- prints them under the new step number, so the
    log claims this step generated text that a different step produced (cursor).
    """
    buffer = RewardObservabilityBuffer(generation_size=2)
    buffer.record("pA", "gen-A-a", 1.0)
    buffer.record("pA", "gen-A-b", 1.0)
    buffer.close_generation(10)  # generation A is published under step 10

    # overflow the sealed queue so a later generation is dropped before it was ever named.
    for gen in range(RewardObservabilityBuffer._SEALED_QUEUE_LIMIT + 2):
        buffer.record("pB", f"gen-B{gen}-a", 2.0)
        buffer.record("pB", f"gen-B{gen}-b", 2.0)
    assert buffer._dropped_unnamed, "the queue never dropped a generation, so the path is untested"

    buffer.close_generation(11)  # spent on the drop: nothing is published for step 11

    assert buffer.latest_for_step(11) is None, (
        "step 11 previewed rows that belong to an earlier generation"
    )
    # the stale reading is still reachable, deliberately: the heartbeat reports it as step 10's.
    assert buffer.latest()[1] == "gen-A-b"


def test_a_step_that_did_publish_still_previews_its_own_rows():
    # the control: the ordinary path must keep previewing, or the fix above silences every preview.
    buffer = RewardObservabilityBuffer(generation_size=2)
    buffer.record("p", "gen1-a", 1.0)
    buffer.record("p", "gen1-b", 1.0)

    buffer.close_generation(7)

    assert buffer.latest_for_step(7) == ("p", "gen1-b", 1.0)


def test_a_late_step_line_previews_the_generation_that_step_named():
    # the queue's whole purpose, asserted through the step-checked accessor: a late line names the
    # OLDEST unnamed generation, so that is the one this step may preview.
    buffer = RewardObservabilityBuffer(generation_size=2)
    buffer.record("p", "gen1-a", 1.0)
    buffer.record("p", "gen1-b", 1.0)
    buffer.record("p", "gen2-a", 9.0)  # generation 2 already scoring

    buffer.close_generation(1)

    assert buffer.latest_for_step(1)[1] == "gen1-b"
    assert buffer.latest_for_step(2) is None, "step 2 has not been named yet"


def test_nothing_is_previewed_for_a_step_before_the_first_boundary():
    # `latest` falls back to the open generation so an early preview is not blank. that fallback
    # must NOT leak into the step-checked accessor: those rows have not been named, so no step
    # number is correct for them.
    buffer = RewardObservabilityBuffer()
    buffer.record("p", "first", 0.5)

    assert buffer.latest() == ("p", "first", 0.5)
    assert buffer.latest_for_step(0) is None
    assert buffer.latest_for_step(1) is None


def _run_per_turn_shim(rows, uids, episode_advantages, response_mask=None):
    """execute the rendered shim against a stub verl and return the advantages it writes.

    executing the source is the only test that can fail for the right reason: the shim's whole job
    is to replace a module-global that verl calls by name, and a string assertion would pass just
    as happily on a shim that never installed itself.
    """
    import sys
    from types import ModuleType, SimpleNamespace

    batch_size = len(uids)
    width = episode_advantages.shape[1]
    spans = np.empty(batch_size, dtype=object)
    turns = np.empty(batch_size, dtype=object)
    for row_index, row in enumerate(rows):
        spans[row_index] = None if row is None else list(row[0])
        turns[row_index] = None if row is None else list(row[1])

    batch = {"advantages": episode_advantages}
    if response_mask is not None:
        batch["response_mask"] = response_mask
    data = SimpleNamespace(
        batch=batch,
        non_tensor_batch={
            "uid": np.array(uids, dtype=object),
            "flash_turn_spans": spans,
            "flash_turn_rewards": turns,
        },
    )

    ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")
    # stock grpo's contribution: the shim must call through to it and build on what it returns.
    ray_trainer.compute_advantage = lambda payload, *args, **kwargs: payload
    ppo = ModuleType("verl.trainer.ppo")
    ppo.ray_trainer = ray_trainer
    stubs = {
        "verl": ModuleType("verl"),
        "verl.trainer": ModuleType("verl.trainer"),
        "verl.trainer.ppo": ppo,
        "verl.trainer.ppo.ray_trainer": ray_trainer,
    }
    for name, module in stubs.items():
        sys.modules[name] = module
    try:
        exec(compile(rl_train.render_per_turn_credit_shim(True), "sitecustomize.py", "exec"), {})
        # call the module global by name, exactly as ray_trainer.fit does at its call site.
        out = ray_trainer.compute_advantage(data, adv_estimator="grpo")
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
    return out.batch["advantages"]


def test_per_turn_credit_shim_is_emitted_only_when_per_turn_credit_is_requested():
    # the default path must put nothing on the child's import path: this shim replaces the
    # advantage computation itself, so emitting it unconditionally would put every episode-credit
    # run through a rewrite it never asked for.
    assert rl_train.render_per_turn_credit_shim(False) == ""
    source = rl_train.render_per_turn_credit_shim(True)
    assert source
    assert "_flash_pt_ray_trainer.compute_advantage = _flash_pt_compute_advantage" in source


def test_per_turn_credit_shim_centres_each_turn_against_its_group_sibling():
    pytest.importorskip("torch")
    import torch

    # two rollouts of the same prompt, two turns each, spans [0,2) and [2,4).
    # turn 0: 1.0 vs 0.0 -> baseline 0.5 -> +0.5 / -0.5
    # turn 1: 0.0 vs 1.0 -> baseline 0.5 -> -0.5 / +0.5
    # the second rollout is the WORSE episode overall on turn 0 yet must still earn positive credit
    # on turn 1; that inversion is the entire point of per-turn credit and episode credit cannot
    # produce it.
    rows = [
        (((0, 2), (2, 4)), (1.0, 0.0)),
        (((0, 2), (2, 4)), (0.0, 1.0)),
    ]
    advantages = _run_per_turn_shim(rows, ["p0", "p0"], torch.zeros((2, 4), dtype=torch.float32))
    assert advantages[0].tolist() == [0.5, 0.5, -0.5, -0.5]
    assert advantages[1].tolist() == [-0.5, -0.5, 0.5, 0.5]


def test_per_turn_credit_shim_reproduces_the_reference_advantages():
    # the port's defining property: for the same rollouts it must produce the SAME advantages the
    # original per-turn builder did, which is what makes it a port rather than a second
    # implementation.
    #
    # the reference values below were computed from that builder before it was deleted with the trl
    # backend. they are pinned as literals deliberately: an oracle that no longer ships cannot be
    # imported, and re-deriving them from the shim under test would make this assert on itself.
    #
    # by hand, for spans [(0,3),(3,5)] / [(0,2),(2,6)] and turn rewards [.25,.75] / [1.0,.5]:
    # turn 0 group mean is (0.25+1.0)/2 = 0.625, so its centred credits are -0.375 and +0.375;
    # turn 1 group mean is (0.75+0.5)/2 = 0.625, giving +0.125 and -0.125. each credit is broadcast
    # across its own token span, and index 5 of row 0 lies past its last span, so it stays 0.
    pytest.importorskip("torch")
    import torch

    spans = [[(0, 3), (3, 5)], [(0, 2), (2, 6)]]
    turns = [[0.25, 0.75], [1.0, 0.5]]
    expected = torch.tensor(
        [
            [-0.375, -0.375, -0.375, 0.125, 0.125, 0.0],
            [0.375, 0.375, -0.125, -0.125, -0.125, -0.125],
        ],
        dtype=torch.float32,
    )
    got = _run_per_turn_shim(
        [
            (tuple(map(tuple, spans[0])), tuple(turns[0])),
            (tuple(map(tuple, spans[1])), tuple(turns[1])),
        ],
        ["p0", "p0"],
        # the episode tensor is what per-turn credit REPLACES, so its value cannot affect the
        # result; zeros make an accidental passthrough visible as a row of zeros.
        torch.zeros((2, 6), dtype=torch.float32),
    )
    assert torch.allclose(got, expected)


def test_per_turn_credit_shim_drops_a_whole_group_when_one_row_is_unusable():
    pytest.importorskip("torch")
    import torch

    # an unscorable row (bridge returned turns=None) must not leave its group centred on a smaller
    # sample: the surviving rows would be compared against a baseline built from a different
    # population than grpo's own. the whole group keeps stock grpo's tensor untouched.
    episode = torch.tensor([[0.3, 0.3, 0.3, 0.3], [-0.3, -0.3, -0.3, -0.3]], dtype=torch.float32)
    advantages = _run_per_turn_shim(
        [(((0, 2), (2, 4)), (1.0, 0.0)), None], ["p0", "p0"], episode.clone()
    )
    assert torch.equal(advantages, episode)


def test_per_turn_credit_shim_leaves_other_groups_on_episode_credit():
    pytest.importorskip("torch")
    import torch

    # the fallback is per group, not per batch: one broken group must not cost every other prompt
    # in the step its per-turn credit.
    episode = torch.zeros((4, 4), dtype=torch.float32)
    episode[2] = 0.7
    episode[3] = -0.7
    rows = [
        (((0, 2), (2, 4)), (1.0, 0.0)),
        (((0, 2), (2, 4)), (0.0, 1.0)),
        (((0, 2), (2, 4)), (1.0, 0.0)),
        None,
    ]
    advantages = _run_per_turn_shim(rows, ["p0", "p0", "p1", "p1"], episode.clone())
    assert advantages[0].tolist() == [0.5, 0.5, -0.5, -0.5]
    # the broken group kept exactly what stock grpo produced. compared against the input tensor
    # rather than a literal: 0.7 has no exact float32 representation, so a literal would compare
    # the shim's output against a value grpo never actually held.
    assert torch.equal(advantages[2], episode[2])
    assert torch.equal(advantages[3], episode[3])


def test_per_turn_credit_shim_ignores_a_turn_no_group_member_emitted():
    pytest.importorskip("torch")
    import torch

    # a zero-width span is a turn the model never produced. it must be excluded from the BASELINE,
    # not merely written nowhere: one sibling emits turn 1 and the other does not, so the emitting
    # row is the only member and its advantage is 0.0. counting the absent member would centre it
    # against a reward for tokens that do not exist -- here (2.0 - (2.0 + 8.0) / 2) = -3.0, a large
    # negative signal on a turn the model actually produced.
    rows = [
        (((0, 2), (2, 4)), (1.0, 2.0)),
        (((0, 2), (2, 2)), (0.0, 8.0)),
    ]
    advantages = _run_per_turn_shim(rows, ["p0", "p0"], torch.zeros((2, 4), dtype=torch.float32))
    assert advantages[0].tolist() == [0.5, 0.5, 0.0, 0.0]
    assert advantages[1].tolist() == [-0.5, -0.5, 0.0, 0.0]


def test_per_turn_credit_shim_keeps_glue_tokens_out_of_the_gradient():
    pytest.importorskip("torch")
    import torch

    # environment replies sit inside the transcript with response_mask 0. a turn span that reaches
    # over one must not hand it advantage -- the model did not generate those tokens.
    mask = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]], dtype=torch.float32)
    rows = [
        (((0, 2), (2, 4)), (1.0, 0.0)),
        (((0, 2), (2, 4)), (0.0, 1.0)),
    ]
    advantages = _run_per_turn_shim(
        rows, ["p0", "p0"], torch.zeros((2, 4), dtype=torch.float32), response_mask=mask
    )
    assert advantages[0].tolist() == [0.5, 0.5, 0.0, -0.5]
    assert advantages[1].tolist() == [-0.5, -0.5, 0.0, 0.5]


def test_per_turn_credit_shim_rejects_a_span_past_the_response_width():
    pytest.importorskip("torch")
    import torch

    # a span beyond the tensor would silently write nothing (python slicing clamps), training on
    # episode credit while the logs claim per-turn. fail loudly instead.
    rows = [(((0, 2), (2, 99)), (1.0, 0.0)), (((0, 2), (2, 4)), (0.0, 1.0))]
    with pytest.raises(ValueError, match="exceeds the response width"):
        _run_per_turn_shim(rows, ["p0", "p0"], torch.zeros((2, 4), dtype=torch.float32))


def test_per_turn_credit_shim_passes_through_a_batch_without_per_turn_metadata():
    # validation batches and any single-turn rollout carry no spans. the shim must return stock
    # grpo's tensor untouched rather than zeroing a batch it cannot credit.
    pytest.importorskip("torch")
    import sys
    from types import ModuleType, SimpleNamespace

    import torch

    episode = torch.full((2, 3), 0.4, dtype=torch.float32)
    data = SimpleNamespace(
        batch={"advantages": episode.clone()},
        non_tensor_batch={"uid": np.array(["p0", "p0"], dtype=object)},
    )
    ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")
    ray_trainer.compute_advantage = lambda payload, *args, **kwargs: payload
    ppo = ModuleType("verl.trainer.ppo")
    ppo.ray_trainer = ray_trainer
    stubs = {
        "verl": ModuleType("verl"),
        "verl.trainer": ModuleType("verl.trainer"),
        "verl.trainer.ppo": ppo,
        "verl.trainer.ppo.ray_trainer": ray_trainer,
    }
    for name, module in stubs.items():
        sys.modules[name] = module
    try:
        exec(compile(rl_train.render_per_turn_credit_shim(True), "sitecustomize.py", "exec"), {})
        out = ray_trainer.compute_advantage(data, adv_estimator="grpo")
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
    assert torch.equal(out.batch["advantages"], episode)


def test_per_turn_credit_is_resolved_only_for_multi_turn_and_reaches_the_bridge():
    # single-turn envs cannot express per-turn credit (there is one turn), and trl says so while
    # accepting the key. the verl resolver must match that, and must no longer REJECT multi-turn.
    source = inspect.getsource(rl_train._resolve_grpo_inputs)
    assert "not supported for multi-turn environments" not in source
    assert '"per_turn_credit": per_turn_credit' in source
    run_source = inspect.getsource(rl_train.run_rl_train)
    assert 'render_per_turn_credit_shim(inp["per_turn_credit"])' in run_source
    assert 'per_turn_credit=bool(inp["per_turn_credit"])' in run_source


def test_multi_turn_bridge_returns_turns_only_under_per_turn_credit():
    # the loop keys off the presence of `turns`, so an episode-credit run must not send the key at
    # all: sending it would put every ordinary multi-turn run through the per-turn rewrite.
    class _Env:
        max_turns = 2

        def new_rollout_state(self, example):
            return {"prompt": [], "messages": []}

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            return [RolloutReward(episode=1.0, turns=(0.25, 0.75)) for _ in items]

    examples = [{"question": "q"}]
    episode_only = _bridge(_Env(), examples=examples, max_turns=2)
    per_turn = _bridge(_Env(), examples=examples, max_turns=2, per_turn_credit=True)
    for bridge in (episode_only, per_turn):
        bridge.start({"index": 0, "session_id": "s", "prompt_ids": []})
    assert episode_only.score({"session_id": "s", "turn_count": 2}) == {"score": 1.0}
    assert per_turn.score({"session_id": "s", "turn_count": 2}) == {
        "score": 1.0,
        "turns": [0.25, 0.75],
    }


def test_multi_turn_bridge_sends_no_turns_when_the_env_vector_is_unusable():
    # score_rollouts canonicalises a bad vector to None. the bridge must forward that None rather
    # than a partial list, so the group falls back cleanly.
    class _Env:
        max_turns = 2

        def new_rollout_state(self, example):
            return {"prompt": [], "messages": []}

        def rollout_rewards_many(self, items):
            from flash.envs.base import RolloutReward

            # one reward for two turns: the validator rejects the count and drops to None.
            return [RolloutReward(episode=1.0, turns=(0.5,)) for _ in items]

    bridge = _bridge(_Env(), examples=[{"question": "q"}], max_turns=2, per_turn_credit=True)
    bridge.start({"index": 0, "session_id": "s", "prompt_ids": []})
    assert bridge.score({"session_id": "s", "turn_count": 2}) == {"score": 1.0, "turns": None}


def _drive_multi_turn_episode(
    *,
    stop_reasons,
    env,
    per_turn_credit=True,
    max_turns=4,
    monkeypatch=None,
    multi_modal_data=None,
    return_instance=False,
):
    """run the real child loop end to end against a real bridge, returning its agent loop output.

    the loop is the thing under test here: it is what appends turn_spans and what tells the bridge
    how many turns to score. a hand-built bridge conversation would restate that bookkeeping
    instead of exercising it.
    """
    from flash.engine.worker import grpo_multiturn

    monkeypatch.setenv("FLASH_VERL_MULTITURN_URL", "http://bridge.invalid")
    monkeypatch.setenv("FLASH_VERL_MAX_TURNS", str(max_turns))
    monkeypatch.setenv("FLASH_VERL_MAX_MODEL_LEN", "4096")
    # generous enough that the per-turn cap never binds here: these tests exercise termination and
    # span accounting, and a cap that clipped a turn would change what they are measuring.
    monkeypatch.setenv("FLASH_VERL_MAX_COMPLETION_TOKENS", "4096")

    bridge = _bridge(
        env, examples=[{"question": "q"}], max_turns=max_turns, per_turn_credit=per_turn_credit
    )
    routes = bridge.routes()

    def bridge_post(url, path, payload):
        return routes[path](payload)

    class _Tokenizer:
        """one codepoint per token, so spans are readable straight off response_ids."""

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(int(i)) for i in ids)

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def apply_chat_template(self, messages, **kwargs):
            return "".join(str(m.get("content") or "") for m in messages)

    class _Base:
        """mirrors the parts of verl's AgentLoopBase the loop actually calls."""

        def __init__(self):
            self.tokenizer = _Tokenizer()
            self.rollout_config = SimpleNamespace(response_length=256)
            self.server_manager = self
            self._sent = list(stop_reasons)
            # every generate call's media, so a test can assert the pixels ride along on turn 2+.
            self.generate_media = []

        def _get_mm_processor_kwargs(self, audio_data=None):
            return {}

        async def process_multi_modal_info(self, messages):
            return dict(multi_modal_data or {})

        async def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3]

        async def generate(
            self,
            *,
            request_id,
            prompt_ids,
            sampling_params,
            image_data=None,
            video_data=None,
            audio_data=None,
            mm_processor_kwargs=None,
        ):
            self.generate_media.append(image_data)
            text, stop_reason = self._sent.pop(0)
            return SimpleNamespace(
                token_ids=[ord(c) for c in text],
                log_probs=[0.0] * len(text),
                num_preempted=0,
                stop_reason=stop_reason,
            )

    captured = {}

    def agent_loop_output(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    loop_class = grpo_multiturn.build_flash_grpo_multi_turn_agent_loop(
        register=lambda name: lambda cls: cls,
        agent_loop_base=_Base,
        agent_loop_output=agent_loop_output,
        bridge_post=bridge_post,
    )

    driven = {}

    async def _go():
        instance = loop_class()
        # the loop offloads the bridge's blocking posts onto this executor, so it has to be the
        # one actually running the coroutine.
        instance.loop = asyncio.get_running_loop()
        driven["instance"] = instance
        await instance.run({}, raw_prompt=[{"role": "user", "content": "go"}], index=0)

    asyncio.run(_go())
    if return_instance:
        return captured, driven["instance"]
    return captured


class _SpanEnv:
    """an env that returns one reward per turn it was actually told about."""

    max_turns = 4

    def __init__(self):
        self.recorded: list[str] = []

    def new_rollout_state(self, example):
        return {"prompt": [], "messages": []}

    def record_model_turn(self, state, text):
        self.recorded.append(text)
        state["messages"].append({"role": "assistant", "content": text})

    def rollout_done(self, state, max_turns=None):
        return False

    def env_reply(self, messages, state):
        return [{"role": "user", "content": "next"}]

    def rollout_rewards_many(self, items):
        from flash.envs.base import RolloutReward

        return [RolloutReward(episode=1.0, turns=tuple(0.5 for _ in self.recorded)) for _ in items]


def test_a_truncated_final_turn_still_earns_per_turn_credit_for_the_turns_before_it(monkeypatch):
    # the bridge does not record an aborted turn into env state (MultiTurnBridge.step returns before
    # record_model_turn), so the env returns no reward for it. the loop must not span it either, or
    # the vector is one short of the spans, score_rollouts rejects the count, and the row -- and via
    # the shim its whole group -- silently drops to episode credit (cursor).
    env = _SpanEnv()
    out = _drive_multi_turn_episode(
        stop_reasons=[("ab", "completed"), ("cd", "aborted")], env=env, monkeypatch=monkeypatch
    )
    assert env.recorded == ["ab"], "the aborted turn was recorded into env state"
    assert len(out["extra_fields"]["flash_turn_spans"]) == 1, "the aborted turn was spanned"
    assert out["extra_fields"]["flash_turn_rewards"] == [0.5], (
        "per-turn credit was dropped: the span count disagreed with the env's reward count"
    )
    # the truncated tokens are still trained on, they just carry no turn coordinate: the transcript
    # is turn "ab", the env's "next" glue, then the aborted "cd".
    assert out["num_turns"] == 2
    assert out["response_ids"] == [ord(c) for c in "abnextcd"]
    assert out["response_mask"] == [1, 1, 0, 0, 0, 0, 1, 1]


def test_an_unspanned_truncated_turns_tokens_are_still_generated_and_masked_as_model_output(
    monkeypatch,
):
    # the control for the fix above: dropping the SPAN must not drop the TOKENS. if the fix had
    # skipped the turn entirely, the child would train on a transcript that never contained it.
    env = _SpanEnv()
    out = _drive_multi_turn_episode(
        stop_reasons=[("ab", "completed"), ("cd", "aborted")], env=env, monkeypatch=monkeypatch
    )
    assert out["response_ids"][-2:] == [ord("c"), ord("d")], "the truncated turn's tokens were lost"
    assert out["response_mask"][-2:] == [1, 1], "the truncated turn's tokens were not model-masked"


def test_multi_turn_rollout_carries_the_prompts_images_into_every_turn(monkeypatch):
    # an image-bearing prompt tokenizes to placeholder tokens that carry no pixels. the engine needs
    # the decoded media alongside them on EVERY generate call, because each turn re-sends the whole
    # prefix -- turn 2 conditioning on placeholders alone is the same failure as turn 1 doing it.
    # the training pass re-tokenizes the episode through the processor, so the output has to carry
    # the media too.
    env = _SpanEnv()
    sentinel = ["<pil-image>"]
    out, instance = _drive_multi_turn_episode(
        stop_reasons=[("ab", "completed")] * 4,
        env=env,
        monkeypatch=monkeypatch,
        multi_modal_data={"images": sentinel},
        return_instance=True,
    )
    assert instance.generate_media == [sentinel] * 4, (
        "the prompt's images did not reach every turn's generate call"
    )
    assert out["multi_modal_data"] == {"images": sentinel}, (
        "the episode was emitted without the media the training pass re-tokenizes against"
    )


def test_a_text_only_multi_turn_rollout_sends_no_media(monkeypatch):
    # the control: a text-only prompt must not start shipping an empty media payload. verl treats an
    # empty multi_modal_data dict as a multimodal row, so passing {} rather than None would push a
    # text-only episode down the processor path it has no pixels for.
    env = _SpanEnv()
    out, instance = _drive_multi_turn_episode(
        stop_reasons=[("ab", "completed")] * 4,
        env=env,
        monkeypatch=monkeypatch,
        return_instance=True,
    )
    assert instance.generate_media == [None] * 4
    assert out["multi_modal_data"] is None


def test_every_turn_is_spanned_when_none_of_them_abort(monkeypatch):
    # the negative control: without an abort, spans and rewards are one per turn and per-turn credit
    # is live. an over-eager skip would show up here as a missing span.
    env = _SpanEnv()
    out = _drive_multi_turn_episode(
        stop_reasons=[
            ("ab", "completed"),
            ("cd", "completed"),
            ("ef", "completed"),
            ("gh", "completed"),
        ],
        env=env,
        monkeypatch=monkeypatch,
    )
    assert env.recorded == ["ab", "cd", "ef", "gh"]
    assert len(out["extra_fields"]["flash_turn_spans"]) == 4
    assert out["extra_fields"]["flash_turn_rewards"] == [0.5, 0.5, 0.5, 0.5]


def test_the_rl_trainer_stores_the_frozen_base_in_bf16():
    """VERL-150: verl's fsdp.yaml default is fp32, which doubles the trainer's resident base.

    the fp32 copy is storage-only -- FSDP already wraps the module MixedPrecision(param_dtype=bf16),
    so params are cast to bf16 for compute either way -- and the base is FROZEN, since verl's
    ref_in_actor (lora_rank > 0 or lora_adapter_path is not None) is always true here. what is
    actually optimized stays fp32: peft's autocast_adapter_dtype casts lora_* weights UP to fp32.
    so this frees ~51 GB at 27B and changes nothing about the gradient.

    the opd driver's half of this lives in test_opd_train. asserted in both because the sft driver
    has set a dtype since it was written and these two never did, which is exactly why g5's sft leg
    succeeded and its grpo and opd legs failed on the same model and the same card.
    """
    overrides = rl_train.build_verl_overrides(_overrides_cfg())
    want = "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
    # exact, not substring: "x=bfloat16" is a substring of "+x=bfloat16", so the obvious `in`
    # assertion passes against a spelling hydra REJECTS here ("Could not append to config. An item
    # is already at ..."). the key is declared in the yaml and takes a BARE override -- the inverse
    # of enable_sleep_mode, which is dataclass-only and requires `+`. neighbouring keys, opposite
    # prefixes. see ISSUES.md VERL-150.
    assert want in overrides
    assert f"+{want}" not in overrides, "must not be + prefixed"
    # ref is deliberately NOT set. it reads like a second resident copy and is not one:
    # ray_trainer.py:897 aliases ref_policy_wg to the actor worker whenever ref_in_actor holds, and
    # flash parses lora_rank with minimum=1 (schema/__init__.py:485) so it always holds. setting it
    # would free nothing. this asserts the absence so a future reader has to re-derive the above
    # rather than pattern-match it back in.
    assert not [o for o in overrides if "ref.fsdp_config.model_dtype" in o]
