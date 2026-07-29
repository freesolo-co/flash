"""verl grpo backend: dispatch, data/config/reward glue, and reward parity (cpu-only, no verl)."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

import flash.engine.worker as W
from flash.engine.worker import rl, rl_verl


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


# ------------------------------- override generation -------------------------------
def _overrides_cfg(**over):
    cfg = {
        "train_files": "/w/train.parquet", "val_files": "/w/val.parquet",
        "model_id": "Qwen/Qwen3-4B", "lora_rank": 32, "lora_alpha": 64,
        "target_modules": "all-linear", "lr": 1e-5, "group_size": 8,
        "prompts_per_step": 16, "max_prompt_len": 2048,
        "max_model_len": 2368, "max_token_len_per_gpu": 2368,
        "max_completion": 320, "temperature": 1.0, "top_p": 0.95, "kl_coef": 0.0,
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


def test_build_verl_training_cfg_derives_engine_len_and_budget():
    inp = {
        "lora_rank": 32, "lora_alpha": 64, "lr": 1e-5, "group_size": 8,
        "prompts_per_step": 16, "mask_truncated_completions": True,
        "max_prompt_len": 3072, "max_completion": 1024, "engine_len": 4096,
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
        tokenizer=_Tok(),
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
        tokenizer=_Tok(),
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

    inp = rl_verl._resolve_single_turn_inputs()
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
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "heartbeat":
            if any(kw.arg == "initial" for kw in node.keywords):
                stage_linenos["initial"] = node.lineno
    assert "initial" in stage_linenos and "liveness" in stage_linenos
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


# ------------------------------- reward rpc bridge -------------------------------
def test_reward_server_round_trip():
    server, url = rl_verl.start_reward_server(
        lambda idx, s: float(idx) + len(s), example_count=4
    )
    try:
        body = json.dumps({"index": 3, "solution_str": "abcd"}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
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
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
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
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
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


def test_resolve_single_turn_inputs_no_longer_rejects_entropy_quantile():
    # the guard this replaces raised on any entropy_quantile < 1.0. the shim implements the masking,
    # so the resolver must pass the value through instead of failing the run.
    source = inspect.getsource(rl_verl._resolve_single_turn_inputs)
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


def test_stop_sequences_gate_off_truncated_completion_masking():
    # main couples these: stop-string rollouts do not end on EOS, so masking truncated completions
    # would wrongly drop every one of them. the verl resolver must inherit that coupling, not
    # re-derive it.
    source = inspect.getsource(rl_verl._resolve_single_turn_inputs)
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

    resolver_src = inspect.getsource(rl_verl._resolve_single_turn_inputs)
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


# ------------------- capability guards: the four specs verl grpo refuses -------------------
# these four raises are the ONLY thing standing between a trl-supported job and a verl run that
# trains on a different contract. they had no regression coverage: every resolver test above drives
# the happy path, so deleting any guard left the suite green. each test below asserts one rejection
# by its own message, because a bare pytest.raises(RuntimeError) passes on any of the ~20 other
# raises in this resolver.


def _capability_env(*, multi_turn=False, is_tool_env=False, images=False):
    """a minimal single-turn text env, optionally flipped to a shape verl grpo rejects."""

    class _Env:
        def __init__(self):
            self.multi_turn = multi_turn
            self.is_tool_env = is_tool_env

        def dataset(self):
            return [{"index": i} for i in range(8)]

        def prompt_messages(self, ex):
            if images:
                # record_has_images matches an image content BLOCK, so build the real shape rather
                # than a sentinel key the guard would not see.
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"question {ex['index']}"},
                            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                        ],
                    }
                ]
            return [{"role": "user", "content": f"question {ex['index']}"}]

    return _Env()


def _capability_resolve(monkeypatch, env, train=None, overrides=None):
    """run the resolver against one env, with everything else on the supported path."""
    from flash.engine.worker._pkg import W as _PkgW
    from flash.spec import JobSpec

    class _Tokenizer:
        pad_token = None
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1])

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
    return rl_verl._resolve_single_turn_inputs()


def test_capability_guard_rejects_multi_turn_env(monkeypatch):
    # trl grpo drives multi-turn through a rollout func (rl.py select_grpo_trainer); the verl path
    # has no equivalent, so it must refuse rather than train only the first turn.
    with pytest.raises(RuntimeError, match="single-turn, non-tool"):
        _capability_resolve(monkeypatch, _capability_env(multi_turn=True))


def test_capability_guard_rejects_tool_env(monkeypatch):
    # trl grpo hands tool schemas AND callables to the trainer; verl gets neither, so a tool env
    # would train against completions that never call a tool.
    with pytest.raises(RuntimeError, match="single-turn, non-tool"):
        _capability_resolve(monkeypatch, _capability_env(is_tool_env=True))


def test_capability_guard_rejects_image_prompts(monkeypatch):
    # this guard runs AFTER prompt_messages is built, so it needs a real image content block, not a
    # flag. trl selects a multimodal trainer here; verl grpo is text-only and would silently drop
    # the image, training the model on the caption alone.
    with pytest.raises(RuntimeError, match="non-multimodal grpo only"):
        _capability_resolve(monkeypatch, _capability_env(images=True))


def test_capability_guard_rejects_kl_anchored_warm_start(monkeypatch):
    # verl computes its kl reference with adapters DISABLED, so under init_from_adapter the
    # reference is the bare base model and the penalty drags the policy away from the sft start -
    # the opposite of what the knob asks for. must raise until verl can hold a reference adapter.
    # the kl coefficient arrives through grpo_overrides, so it must go through the helper: patching
    # it separately gets clobbered by the helper's own patch and the test then fails on the adapter
    # download instead, passing a bare raises() while proving nothing about this guard.
    with pytest.raises(RuntimeError, match="kl_penalty_coef"):
        _capability_resolve(
            monkeypatch,
            _capability_env(),
            train={"init_from_adapter": "org/some-sft-adapter"},
            overrides={"kl_penalty_coef": 0.1},
        )


def test_capability_guards_admit_the_supported_single_turn_text_env(monkeypatch):
    # the control: with none of the four shapes present the resolver must run to completion, so a
    # guard that fires on every env would fail here instead of passing the four tests above.
    inp = _capability_resolve(monkeypatch, _capability_env())
    assert inp["max_prompt_len"] > 0
