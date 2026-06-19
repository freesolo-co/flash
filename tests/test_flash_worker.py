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
    """DEFECT: RL_VLLM_GPU_UTIL / RL_PER_DEVICE_PROMPTS (and the others) were never added to
    the forward list, so the docs' OOM-fix advice couldn't reach the worker."""
    from flash.providers.runpod.train import build_worker_env

    knobs = {
        "RL_VLLM_GPU_UTIL": "0.40",
        "RL_VLLM_SLEEP": "1",
        "RL_PER_DEVICE_PROMPTS": "2",
        "VLLM_USE_V1": "0",
        "SFT_PER_DEVICE_BS": "4",
    }
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)

    env = build_worker_env(_spec(), 0)
    for k, v in knobs.items():
        assert env.get(k) == v, f"{k} not forwarded to worker"
    # fragmentation-safe allocator default is always set
    assert "PYTORCH_CUDA_ALLOC_CONF" in env


def test_build_worker_env_respects_alloc_conf_override(monkeypatch):
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    env = build_worker_env(_spec(), 0)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:256"


def test_build_worker_env_forwards_midrun_eval_knobs(monkeypatch):
    """The periodic mid-run eval knobs are read via os.environ on the worker, so they MUST be
    on the forward allowlist or the feature silently no-ops on every remote run (RunPod + Vast,
    which reuses this same build_worker_env)."""
    from flash.providers.runpod.train import build_worker_env

    knobs = {
        "FLASH_EVAL_EVERY_STEPS": "20",
        "FLASH_EVAL_NUM": "16",
    }
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)
    env = build_worker_env(_spec(), 0)
    for k, v in knobs.items():
        assert env.get(k) == v, f"{k} not forwarded to worker"


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


def test_build_worker_env_forwards_chalk_kernel_flags(monkeypatch):
    """The opt-in chalk-kernel install hook (engine.chalk_kernels) runs inside the worker
    subprocess and selects which install-on-call chalk installers to run from FLASH_* flags read
    from its OWN process env, so an operator-set FLASH_* selection (and FLASH_CHALK_SPEC) MUST be
    forwarded by the allowlist or every chalk kernel silently no-ops on every remote run."""
    from flash.providers.runpod.train import build_worker_env

    flags = {
        "FLASH_MLP_KERNEL": "1",
        "FLASH_MLP_FP8": "1",
        "FLASH_MLP_FP8_DOWN": "0",
        "FLASH_FP8_BASE": "1",
        "FLASH_FP8_BASE_ATTN": "0",
        "FLASH_FP8_BASE_MLP": "1",
        "FLASH_FP8_BASE_MIN_K": "512",
        "FLASH_TRITON_LORA": "1",
        "FLASH_EMBED_KERNEL": "1",
        "FLASH_QKV_KERNEL": "1",
        "FLASH_ROPE_KERNEL": "1",
        "FLASH_CHALK_SPEC": "git+https://github.com/freesolo-co/chalk@main",
    }
    for k, v in flags.items():
        monkeypatch.setenv(k, v)
    env = build_worker_env(_spec(), 0)
    for k, v in flags.items():
        assert env.get(k) == v, f"{k} not forwarded to worker"
    # unset chalk flags are not invented
    for k in flags:
        monkeypatch.delenv(k, raising=False)
    env2 = build_worker_env(_spec(), 0)
    assert not any(k in env2 for k in flags)


def _clear_chalk_flags(monkeypatch):
    for k in (
        "FLASH_MLP_KERNEL",
        "FLASH_MLP_FP8",
        "FLASH_FP8_BASE",
        "FLASH_TRITON_LORA",
        "FLASH_EMBED_KERNEL",
        "FLASH_QKV_KERNEL",
        "FLASH_ROPE_KERNEL",
        "FLASH_CHALK_SPEC",
    ):
        monkeypatch.delenv(k, raising=False)


def test_chalk_extra_pip_empty_without_flags(monkeypatch):
    """No FLASH_* kernel flag -> nothing added to the worker's extra_pip (default = no chalk)."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_CHALK_SPEC", "freesolo-chalk")  # spec alone must not opt in
    assert chalk_extra_pip() == []


def test_chalk_extra_pip_empty_without_spec(monkeypatch):
    """A kernel flag is set but FLASH_CHALK_SPEC is unset -> nothing added (chalk is unpublished,
    can't auto-install) — install_chalk_kernels then safely no-ops on the worker."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_MLP_KERNEL", "1")
    assert chalk_extra_pip() == []


def test_chalk_extra_pip_adds_spec_when_selected(monkeypatch):
    """Kernel flag + FLASH_CHALK_SPEC -> the chalk spec is appended to extra_pip, which the worker
    installs for EVERY job (the durable baked-image path that bypasses resolve_worker_deps)."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_FP8_BASE", "1")
    monkeypatch.setenv("FLASH_CHALK_SPEC", "git+https://github.com/freesolo-co/chalk@main")
    assert chalk_extra_pip() == ["git+https://github.com/freesolo-co/chalk@main"]


def test_chalk_extra_pip_falsey_flag_does_not_opt_in(monkeypatch):
    """FLASH_*=0 is inert: a leftover 0 must not pull chalk in even with a spec set."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_MLP_KERNEL", "0")
    monkeypatch.setenv("FLASH_CHALK_SPEC", "freesolo-chalk")
    assert chalk_extra_pip() == []


def _spec_worker_env(worker_env: dict):
    """A grpo JobSpec carrying a per-run [worker_env] block (the TOML override map)."""
    from flash.spec import JobSpec, TrainSpec

    return JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(steps=10, seeds=(0,), hf_repo="owner/runs"),
        worker_env=dict(worker_env),
    )


def test_chalk_extra_pip_detects_per_run_worker_env_optin(monkeypatch):
    """DEFECT: a run that enables chalk only via its [worker_env] block (not the control-plane
    process env) was NOT detected — chalk_extra_pip read bare os.environ, so the spec was never
    appended and the kernels never installed for that run. The effective worker env (worker_env
    merged over os.environ) must be what decides selection + the FLASH_CHALK_SPEC lookup."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)  # nothing in os.environ
    spec = _spec_worker_env(
        {
            "FLASH_FP8_BASE": "1",
            "FLASH_CHALK_SPEC": "git+https://github.com/freesolo-co/chalk@main",
        }
    )
    # bare process env -> nothing; the opt-in lives only in [worker_env]
    assert chalk_extra_pip() == []
    assert chalk_extra_pip(spec) == ["git+https://github.com/freesolo-co/chalk@main"]


def test_chalk_extra_pip_worker_env_spec_with_os_env_flag(monkeypatch):
    """Mixed source: the kernel flag is in the control-plane env but FLASH_CHALK_SPEC is pinned
    per-run in [worker_env]. The merged effective env resolves both, so the spec is added."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_TRITON_LORA", "1")  # selection from the process env
    spec = _spec_worker_env({"FLASH_CHALK_SPEC": "git+https://github.com/freesolo-co/chalk@dev"})
    assert chalk_extra_pip(spec) == ["git+https://github.com/freesolo-co/chalk@dev"]


def test_chalk_extra_pip_worker_env_overrides_os_env_flag(monkeypatch):
    """[worker_env] wins over os.environ (same precedence build_worker_env applies): a per-run
    FLASH_*=0 must DISABLE a chalk flag set globally, so chalk is not installed for that run."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_MLP_KERNEL", "1")  # global opt-in
    monkeypatch.setenv("FLASH_CHALK_SPEC", "git+https://github.com/freesolo-co/chalk@main")
    # without an override the global flag opts in
    assert chalk_extra_pip() == ["git+https://github.com/freesolo-co/chalk@main"]
    # a per-run [worker_env] FLASH_MLP_KERNEL=0 turns it OFF for this run
    spec = _spec_worker_env({"FLASH_MLP_KERNEL": "0"})
    assert chalk_extra_pip(spec) == []


def test_chalk_selection_matches_what_worker_env_forwards(monkeypatch):
    """Consistency: the SAME effective env that decides chalk install must be what the worker
    process sees. A [worker_env] chalk opt-in both (a) triggers chalk_extra_pip and (b) reaches
    the worker via build_worker_env, so install_chalk_kernels sees the flag on the worker."""
    from flash.providers.runpod.train import build_worker_env, chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    spec = _spec_worker_env(
        {
            "FLASH_QKV_KERNEL": "1",
            "FLASH_CHALK_SPEC": "git+https://github.com/freesolo-co/chalk@main",
        }
    )
    # (a) submit path appends the chalk spec to extra_pip
    assert chalk_extra_pip(spec) == ["git+https://github.com/freesolo-co/chalk@main"]
    # (b) the same flag is forwarded into the worker's env, so install_chalk_kernels (which reads
    #     the worker's own process env) selects the kernel on the worker
    env = build_worker_env(spec, 0)
    assert env.get("FLASH_QKV_KERNEL") == "1"
    assert env.get("FLASH_CHALK_SPEC") == "git+https://github.com/freesolo-co/chalk@main"


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


def test_serve_execution_timeout_is_fixed():
    """DEFECT: serve execution cap (10 min) was shorter than a cold serving worker's startup, so
    the first slm chat/deploy failed with 'executionTimeout exceeded'. It is now a fixed, generous
    constant (no env override)."""
    from flash.serve import deploy

    assert deploy.serve_execution_timeout_ms() >= 20 * 60 * 1000


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
