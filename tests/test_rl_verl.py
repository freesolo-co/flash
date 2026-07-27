"""verl grpo backend: dispatch, data/config/reward glue, and reward parity (cpu-only, no verl)."""

from __future__ import annotations

import json
import threading
import urllib.request
from types import SimpleNamespace

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
        "prompts_per_step": 16, "micro_batch": 2, "max_prompt_len": 2048,
        "max_completion": 320, "temperature": 1.0, "top_p": 0.95, "kl_coef": 0.0,
        "loss_agg_mode": "seq-mean-token-sum-norm", "seed": 42, "num_iterations": 2,
        "steps": 60, "gpu_mem_util": 0.5, "tp_size": 1, "loggers": "console", "fp8_kv": False,
        "warmstart_adapter": "", "reward_path": "/w/reward.py", "reward_name": "compute_score",
        "total_epochs": 1, "save_freq": 20, "local_dir": "/w/ckpt",
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
    # constant lr, num_iterations, gradient checkpointing, seed, max-steps horizon, save schedule.
    assert "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0" in o
    assert "actor_rollout_ref.actor.ppo_epochs=2" in o
    assert "actor_rollout_ref.model.enable_gradient_checkpointing=True" in o
    assert "data.seed=42" in o
    assert "trainer.total_training_steps=60" in o
    assert "trainer.save_freq=20" in o
    assert "trainer.max_actor_ckpt_to_keep=1" in o
    assert "trainer.logger=[console]" in o
    assert "data.train_batch_size=16" in o
    # truncated importance sampling: token-level, cap 2.0 (matches flash's tis recipe).
    assert "algorithm.rollout_correction.rollout_is=token" in o
    assert "algorithm.rollout_correction.rollout_is_threshold=2.0" in o


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
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2" in o


# ------------------------------- reward module render -------------------------------
def test_render_reward_module_is_valid_and_defines_compute_score():
    src = rl_verl.render_reward_module()
    ns: dict = {}
    exec(compile(src, "<reward>", "exec"), ns)  # compiles + defines, no network call made
    assert callable(ns["compute_score"])
    # no flash import leaks into the verl-side shim.
    assert "import flash" not in src


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
    server, url = rl_verl.start_reward_server(lambda idx, s: float(idx) + len(s))
    try:
        body = json.dumps({"index": 3, "solution_str": "abcd"}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            got = json.loads(r.read().decode())
        assert got["score"] == 7.0  # 3 + len("abcd")
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

    server, url = rl_verl.start_reward_server(scorer)
    try:
        for i in range(3):
            body = json.dumps({"index": i, "solution_str": f"c{i}"}).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        assert [c[0] for c in captured] == ["c0", "c1", "c2"]
    finally:
        server.shutdown()


def test_resolve_single_turn_inputs_guards_entropy_quantile(monkeypatch):
    # entropy_quantile<1.0 has no verl equivalent (trl top-entropy masking); the single-turn resolver
    # must fail loud rather than silently train without the requested masking.
    import pytest

    import flash.engine.worker.rl_verl as rlv
    from flash.engine.worker._pkg import W
    from flash.spec import JobSpec

    spec = JobSpec.from_dict(
        {"model": "Qwen/Qwen3.5-0.8B", "algorithm": "grpo", "train": {"entropy_quantile": 0.2}}
    )

    class _Env:
        is_tool_env = False
        multi_turn = False

    monkeypatch.setattr(W, "JOB_SPEC", spec, raising=False)
    monkeypatch.setattr(W, "SEED", 42, raising=False)
    monkeypatch.setattr(W, "require_active_env", lambda: _Env(), raising=False)
    # the resolver seeds rngs (torch) before validating; stub it so this stays cpu/offline-runnable.
    monkeypatch.setattr(rlv, "seed_training_rngs", lambda seed: None)

    with pytest.raises(RuntimeError, match="entropy_quantile"):
        rlv._resolve_single_turn_inputs()


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
