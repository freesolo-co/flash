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


def test_build_worker_env_does_not_forward_tuning_knobs(monkeypatch):
    """Run tuning is fully managed: operator process-env knobs are NOT forwarded to the worker —
    the worker uses optimal defaults, and per-run config comes from the spec / [worker_env] TOML."""
    from flash.providers.runpod.train import build_worker_env

    managed = (
        "RL_VLLM_GPU_UTIL", "RL_VLLM_SLEEP", "VLLM_USE_V1", "VLLM_ATTENTION_BACKEND",
        "SFT_PER_DEVICE_BS", "SFT_PACKING", "FLASH_QUANT", "LORA_TARGETS",
        "FLASH_HEARTBEAT_MIN_S", "RL_PER_DEVICE_PROMPTS",
    )
    for k in managed:
        monkeypatch.setenv(k, "1")
    env = build_worker_env(_spec(), 0)
    for k in managed:
        assert k not in env, f"{k} should not be forwarded (tuning is managed)"
    # W&B credential/routing is still forwarded; fragmentation-safe allocator default still set
    monkeypatch.setenv("WANDB_API_KEY", "wb")
    assert build_worker_env(_spec(), 0)["WANDB_API_KEY"] == "wb"
    assert "PYTORCH_CUDA_ALLOC_CONF" in env


def test_build_worker_env_alloc_conf_pinned_via_worker_env(monkeypatch):
    """A per-run [worker_env] alloc-conf pin wins over the computed default; an operator process-env
    PYTORCH_*_ALLOC_CONF no longer does."""
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:999")  # process env: ignored
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(steps=10, seeds=(0,), hf_repo="owner/runs"),
        worker_env={"PYTORCH_ALLOC_CONF": "max_split_size_mb:256"},
    )
    env = build_worker_env(spec, 0)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:256"  # [worker_env] wins, not the env


def test_build_worker_env_does_not_forward_midrun_eval_knobs(monkeypatch):
    """The mid-run eval cadence comes from the run spec ([train] eval_every_steps), not env, so the
    old FLASH_EVAL_* operator overrides are NOT forwarded to the worker."""
    from flash.providers.runpod.train import build_worker_env

    for k in ("FLASH_EVAL_EVERY_STEPS", "FLASH_EVAL_NUM"):
        monkeypatch.setenv(k, "20")
    env = build_worker_env(_spec(), 0)
    for k in ("FLASH_EVAL_EVERY_STEPS", "FLASH_EVAL_NUM"):
        assert k not in env, f"{k} should no longer be forwarded (cadence comes from the spec)"


def test_build_worker_env_heartbeat_throttle_via_worker_env(monkeypatch):
    """The rl_step heartbeat throttle is not an operator process-env knob; a run pins it via the
    [worker_env] TOML table (process env is ignored)."""
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    monkeypatch.setenv("FLASH_HEARTBEAT_MIN_S", "999")  # process env: ignored
    assert "FLASH_HEARTBEAT_MIN_S" not in build_worker_env(_spec(), 0)
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(steps=10, seeds=(0,), hf_repo="owner/runs"),
        worker_env={"FLASH_HEARTBEAT_MIN_S": "180"},
    )
    assert build_worker_env(spec, 0).get("FLASH_HEARTBEAT_MIN_S") == "180"


def test_build_worker_env_does_not_set_judge_model(monkeypatch):
    """The judge model is no longer an env knob in flash — it's fixed in the env-authoring side
    (gpt-oss-120b). flash neither forwards an operator value nor sets one on the worker."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("FLASH_JUDGE_MODEL", "some/other-model")  # operator value: ignored
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
    from flash.spec import JobSpec, TrainSpec

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    # sleep off -> expandable is safe + preferred; pinned via [worker_env] (not process env)
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(steps=10, seeds=(0,), hf_repo="owner/runs"),
        worker_env={"RL_VLLM_SLEEP": "0"},
    )
    env = build_worker_env(spec, 0)
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
