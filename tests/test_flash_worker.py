"""Regression tests for the Flash worker plumbing fixed in this PR.

Covers:
- build_worker_env forwards the documented RL/vLLM tuning knobs to the GPU worker
  (they were silently dropped) and sets a fragmentation-safe allocator default;
- the runpod_flash backoff OverflowError that aborted long runs is patched;
- the serve cold-start execution timeout is generous + env-overridable;
- per-phase error artifact names don't collide (train error survives a later eval error).
"""

from __future__ import annotations

import pytest


def _spec():
    from flash.spec import JobSpec, TrainSpec

    return JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(steps=10, seeds=(0,), hf_repo="owner/runs"),
    )


def test_build_worker_env_forwards_tuning_knobs(monkeypatch):
    """Genuine runtime tuning knobs (not in the spec) reach the worker via the allowlist."""
    from flash.providers.runpod.train import build_worker_env

    knobs = {
        "RL_VLLM_GPU_UTIL": "0.40",
        "RL_VLLM_SLEEP": "1",
        "VLLM_USE_V1": "0",
        "SFT_PER_DEVICE_BS": "4",
    }
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("RL_PER_DEVICE_PROMPTS", "2")

    env = build_worker_env(_spec(), 0)
    for k, v in knobs.items():
        assert env.get(k) == v, f"{k} not forwarded to worker"
    assert "RL_PER_DEVICE_PROMPTS" not in env  # default + auto-caps only
    # fragmentation-safe allocator default is always set
    assert "PYTORCH_CUDA_ALLOC_CONF" in env


def test_build_worker_env_respects_alloc_conf_override(monkeypatch):
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    env = build_worker_env(_spec(), 0)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:256"


def test_build_worker_env_does_not_forward_midrun_eval_knobs(monkeypatch):
    """The mid-run eval cadence comes from the run spec ([train] eval_every_steps), not env, so the
    old FLASH_EVAL_* operator overrides are NOT forwarded to the worker."""
    from flash.providers.runpod.train import build_worker_env

    for k in ("FLASH_EVAL_EVERY_STEPS", "FLASH_EVAL_NUM"):
        monkeypatch.setenv(k, "20")
    env = build_worker_env(_spec(), 0)
    for k in ("FLASH_EVAL_EVERY_STEPS", "FLASH_EVAL_NUM"):
        assert k not in env, f"{k} should no longer be forwarded (cadence comes from the spec)"


def test_build_worker_env_forwards_heartbeat_throttle(monkeypatch):
    """FLASH_HEARTBEAT_MIN_S is read by engine.worker on the GPU side to throttle rl_step
    heartbeat commits under HuggingFace's 128/hr-per-repo cap; operators raise it when several
    concurrent GRPO runs share one HF_REPO, so it MUST be on the forward allowlist."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("FLASH_HEARTBEAT_MIN_S", "180")
    assert build_worker_env(_spec(), 0).get("FLASH_HEARTBEAT_MIN_S") == "180"
    monkeypatch.delenv("FLASH_HEARTBEAT_MIN_S", raising=False)
    assert "FLASH_HEARTBEAT_MIN_S" not in build_worker_env(_spec(), 0)


def test_build_worker_env_forwards_judge_model(monkeypatch):
    """The optimizer-authored verifiers env reads FLASH_JUDGE_MODEL on the worker to pick its
    JudgeRubric client model (SFT-eval / GRPO-reward / rejection-sampling); the control-plane
    override must be forwarded, else the env silently falls back to its generated default."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("FLASH_JUDGE_MODEL", "openai/gpt-oss-120b")
    assert build_worker_env(_spec(), 0).get("FLASH_JUDGE_MODEL") == "openai/gpt-oss-120b"
    monkeypatch.delenv("FLASH_JUDGE_MODEL", raising=False)
    assert "FLASH_JUDGE_MODEL" not in build_worker_env(_spec(), 0)


def test_build_worker_env_forwards_prime_api_key(monkeypatch):
    """The worker needs PRIME_API_KEY to `prime env install` the run's Hub env(s)."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("PRIME_API_KEY", "pit-secret")
    assert build_worker_env(_spec(), 0).get("PRIME_API_KEY") == "pit-secret"
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    assert "PRIME_API_KEY" not in build_worker_env(_spec(), 0)


def test_build_worker_env_hf_repo_is_per_run(monkeypatch):
    """The worker env's HF_REPO is seeded from the run's [train] hf_repo, NOT the operator's
    HF_REPO env var (which no longer exists). An operator HF_REPO in the process env is
    ignored — the worker reads its own seeded value, sourced from the spec."""
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    # an operator HF_REPO in the env must NOT leak into the worker env
    monkeypatch.setenv("HF_REPO", "operator/default")
    per_run = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(steps=10, seeds=(0,), hf_repo="myorg/runs"),
    )
    assert build_worker_env(per_run, 0)["HF_REPO"] == "myorg/runs"
    # still the per-run value even with no operator HF_REPO at all
    monkeypatch.delenv("HF_REPO", raising=False)
    assert build_worker_env(per_run, 0)["HF_REPO"] == "myorg/runs"


def test_alloc_conf_default_avoids_expandable_under_grpo_sleep(monkeypatch):
    # vLLM sleep-mode CuMemAllocator is incompatible with expandable_segments; GRPO with sleep
    # ON (the default) must NOT default to expandable_segments or the run crashes at engine init.
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("RL_VLLM_SLEEP", raising=False)  # default = sleep on
    env = build_worker_env(_spec(), 0)
    assert "expandable_segments" not in env["PYTORCH_ALLOC_CONF"]
    assert env["PYTORCH_ALLOC_CONF"] == env["PYTORCH_CUDA_ALLOC_CONF"]


def test_alloc_conf_default_expandable_when_sleep_off(monkeypatch):
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setenv("RL_VLLM_SLEEP", "0")  # sleep off -> expandable is safe + preferred
    env = build_worker_env(_spec(), 0)
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_alloc_conf_default_expandable_for_sft(monkeypatch):
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    spec = JobSpec(model="Qwen/Qwen3.5-0.8B", algorithm="sft", train=TrainSpec(steps=2, seeds=(0,)))
    env = build_worker_env(spec, 0)
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_runpod_backoff_no_overflow_on_long_runs():
    """DEFECT: runpod_flash computed base*(2**attempt) then clamped, so a long poll loop
    overflowed (~80 min in) and killed a healthy job. The patch caps the exponent first."""
    pytest.importorskip("runpod_flash")
    from flash.providers.runpod.train import _patch_runpod_backoff

    _patch_runpod_backoff()
    from runpod_flash.core.utils import backoff

    # Pre-patch this raised OverflowError; now it must return a clamped, finite delay.
    delay = backoff.get_backoff_delay(5000, max_seconds=5)
    assert delay <= 5 * 1.2 + 1e-9
    # the serverless module's imported reference is patched too (that's the real call site)
    from runpod_flash.core.resources import serverless

    assert serverless.get_backoff_delay(100000, max_seconds=5) <= 5 * 1.2 + 1e-9


def test_require_vllm_for_rollout_func_rejects_vllm_off_multiturn():
    """Multi-turn GRPO with vLLM disabled (the 35B tier's grpo_use_vllm=False, or RL_USE_VLLM=0)
    must fail fast — the rollout closure reads trainer.vllm_generation.llm, which only exists
    when use_vllm=True, so otherwise it would AttributeError deep in the first rollout turn."""
    from flash.engine.worker import require_vllm_for_rollout_func

    with pytest.raises(RuntimeError, match="needs colocated vLLM"):
        require_vllm_for_rollout_func(True, False, "Qwen/Qwen3.6-35B-A3B")
    # every supported combination is a no-op (single-turn, or vLLM enabled)
    require_vllm_for_rollout_func(True, True, "m")
    require_vllm_for_rollout_func(False, False, "m")
    require_vllm_for_rollout_func(False, True, "m")


def test_error_artifact_name_is_per_phase():
    """Per-phase error files keep the root-cause train traceback under a stable name."""
    from flash.engine.worker import error_artifact_name

    names = {error_artifact_name(m) for m in ("sft", "rl")}
    assert len(names) == 2  # distinct per phase -> no clobber
    assert error_artifact_name("rl") == "error_rl.txt"


def test_train_body_imports_every_name_it_uses():
    """Flash ships only _train_body's source to the worker, where module-level
    imports are out of scope, so every stdlib/3p name it references must be
    imported inside the function body (else NameError before training)."""
    import ast
    import inspect

    from flash.providers.runpod import train

    tree = ast.parse(inspect.getsource(train._train_body))
    fn = tree.body[0]
    imported = {
        alias.asname or alias.name.split(".")[0]
        for node in ast.walk(fn)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    # Names that must be locally imported (regression: contextlib was missing). shutil is used
    # to gate the conditional `prime` install, so it must also be imported locally.
    for name in ("contextlib", "json", "os", "shutil", "subprocess", "sys"):
        assert name in imported, f"_train_body uses {name!r} without a local import"


def test_train_body_installs_prime_only_when_absent():
    """`prime` is often baked into the worker image; an unconditional `pip install prime`
    every run adds latency + a per-run PyPI failure point. The handler must guard the install
    behind `shutil.which("prime") is None`."""
    import inspect

    from flash.providers.runpod import train

    src = inspect.getsource(train._train_body)
    assert 'shutil.which("prime")' in src
    # The pip install of `prime` must be conditional, not at module/handler top level.
    install_idx = src.index('"install", "prime"')
    guard_idx = src.index('shutil.which("prime") is None')
    assert guard_idx < install_idx, "the prime install must be gated by the which() check"
