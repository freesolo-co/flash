"""Regression tests for the Flash worker plumbing fixed in this PR.

Covers:
- build_worker_env sets managed worker defaults without forwarding removed tuning knobs;
- the runpod_flash backoff OverflowError that aborted long runs is patched;
- per-phase error artifact names don't collide (train error survives a later eval error).
"""

from __future__ import annotations

import time

import pytest


def _spec():
    from flash.spec import JobSpec, TrainSpec

    return JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )


def _run_deadline_fields() -> dict[str, float | int]:
    run_created_at = time.time()
    run_max_wall_seconds = 3600
    return {
        "run_created_at": run_created_at,
        "run_max_wall_seconds": run_max_wall_seconds,
        "deadline_at": run_created_at + run_max_wall_seconds,
    }


def test_build_worker_env_does_not_forward_removed_tuning_knobs(monkeypatch):
    """Flash is managed: process-env tuning toggles do not change worker behavior."""
    from flash.providers.runpod.train import build_worker_env

    knobs = {
        "VLLM_USE_V1": "0",
        "SFT_PER_DEVICE_BS": "4",
    }
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)

    env = build_worker_env(_spec(), 0)
    for k in knobs:
        assert k not in env, f"{k} should not be forwarded to worker"
    # fragmentation-safe allocator default is always set
    assert "PYTORCH_CUDA_ALLOC_CONF" in env


def test_build_worker_env_ignores_alloc_conf_override(monkeypatch):
    """flash is fully managed: an operator PYTORCH_CUDA_ALLOC_CONF in the process env does NOT
    override flash's computed allocator conf (RL is non-expandable, sleep-safe)."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:999")
    env = build_worker_env(_spec(), 0)  # grpo -> sleep-safe non-expandable
    assert env["PYTORCH_CUDA_ALLOC_CONF"] != "max_split_size_mb:999"
    assert "expandable_segments" not in env["PYTORCH_CUDA_ALLOC_CONF"]


def test_build_worker_env_opd_uses_sleep_safe_allocator(monkeypatch):
    """OPD must get the sleep-safe NON-expandable allocator conf, like GRPO.

    This inverts an earlier regression test, and the inversion is the point. Under trl, OPD drove its
    own HF generate loop with no sleep engine, so the anti-fragmentation expandable allocator was
    right. run_opd now delegates to verl unconditionally, and verl leaves rollout.enable_sleep_mode
    defaulted True -- so the engine always builds a CuMemAllocator, which asserts outright on
    expandable_segments (vllm cumem.py, pytorch#147851). An OPD run with the old conf dies before
    step 1. SFT keeps expandable: its verl trainer is pure FSDP and builds no rollout at all.
    """
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    opd_spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )
    env = build_worker_env(
        opd_spec,
        0,
        runtime_secrets={
            "FLASH_CONTROL_PANEL_URL": "https://broker.example",
            "FLASH_TEACHER_CAPABILITY": "capability-test-value",
        },
    )
    assert "expandable_segments" not in env["PYTORCH_CUDA_ALLOC_CONF"]
    assert "expandable_segments" not in env["PYTORCH_ALLOC_CONF"]
    # GRPO still ships the sleep-safe non-expandable conf.
    grpo_env = build_worker_env(_spec(), 0)
    assert "expandable_segments" not in grpo_env["PYTORCH_CUDA_ALLOC_CONF"]


def test_build_worker_env_does_not_forward_judge_creds(monkeypatch):
    """flash is fully managed: reward-judge creds and the judge-model id are NOT hardcoded
    control-plane forwards. An env that needs a judge provider key declares it as an
    [environment].secrets entry (forwarded via runtime_secrets); the env's own default judge model
    otherwise applies. A stray control-plane OPENROUTER_API_KEY / OPENAI_API_KEY / FLASH_JUDGE_MODEL
    must NOT leak into every worker."""
    from flash.providers.runpod.train import build_worker_env

    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "FLASH_JUDGE_MODEL"):
        monkeypatch.setenv(key, "control-plane-should-not-forward")
    env = build_worker_env(_spec(), 0)
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "FLASH_JUDGE_MODEL"):
        assert key not in env


def test_build_worker_env_forwards_github_env_source_token(monkeypatch):
    """The worker receives the control-plane token used for managed Freesolo environments."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("GITHUB_TOKEN", "ghp-secret")
    assert build_worker_env(_spec(), 0).get("GITHUB_TOKEN") == "ghp-secret"
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


def test_build_worker_env_forwards_only_managed_teacher_capability_for_opd(monkeypatch):
    """opd receives bounded broker transport while provider credentials remain control-plane-only."""
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    monkeypatch.setenv("PARASAIL_API_KEY", "platform-managed-parasail")
    opd_spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )
    env = build_worker_env(
        opd_spec,
        0,
        runtime_secrets={
            "FLASH_CONTROL_PANEL_URL": "https://broker.example",
            "FLASH_TEACHER_CAPABILITY": "capability-test-value",
        },
    )
    assert env["FLASH_CONTROL_PANEL_URL"] == "https://broker.example"
    assert env["FLASH_TEACHER_CAPABILITY"] == "capability-test-value"
    assert "PARASAIL_API_KEY" not in env
    grpo = build_worker_env(_spec(), 0)
    assert "FLASH_CONTROL_PANEL_URL" not in grpo
    assert "FLASH_TEACHER_CAPABILITY" not in grpo


def test_build_worker_env_does_not_accept_legacy_teacher_broker_url():
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    opd_spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )

    with pytest.raises(RuntimeError, match="control-panel teacher transport is missing"):
        build_worker_env(
            opd_spec,
            0,
            runtime_secrets={
                "FLASH_TEACHER_BROKER_URL": "https://broker.example",
                "FLASH_TEACHER_CAPABILITY": "capability-test-value",
            },
        )


def test_build_worker_env_rejects_managed_teacher_byo_names():
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import EnvironmentSpec, JobSpec, TrainSpec

    opd_spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        environment=EnvironmentSpec(id="org/env", secrets=("PARASAIL_API_KEY",)),
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )
    with pytest.raises(ValueError, match="managed teacher credential names"):
        build_worker_env(
            opd_spec,
            0,
            runtime_secrets={
                "PARASAIL_API_KEY": "byo-parasail-key",
                "FLASH_CONTROL_PANEL_URL": "https://broker.example",
                "FLASH_TEACHER_CAPABILITY": "capability-test-value",
            },
        )


def test_build_worker_env_wandb_is_user_runtime_secret_not_control_plane_env(monkeypatch):
    """Provider/platform creds are supplied by the control plane, but W&B belongs to the user.

    WANDB_API_KEY must therefore come from the per-submit runtime secret path, not from the
    control-plane process env.
    """
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("WANDB_API_KEY", "platform-should-not-forward")
    env = build_worker_env(_spec(), 0)
    assert "WANDB_API_KEY" not in env

    env = build_worker_env(_spec(), 0, runtime_secrets={"WANDB_API_KEY": "user-wb"})
    assert env["WANDB_API_KEY"] == "user-wb"


def test_build_worker_env_forwards_declared_environment_runtime_secrets():
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import EnvironmentSpec, JobSpec, TrainSpec

    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        environment=EnvironmentSpec(id="owner/env", secrets=("SERPAPI_API_KEY",)),
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )

    env = build_worker_env(
        spec,
        0,
        runtime_secrets={
            "SERPAPI_API_KEY": "serp-user",
            "UNDECLARED_API_KEY": "must-not-forward",
        },
    )
    assert env["SERPAPI_API_KEY"] == "serp-user"
    assert "UNDECLARED_API_KEY" not in env


def test_worker_console_always_uploaded_and_no_flag(monkeypatch):
    """The worker console is ALWAYS uploaded — live (periodic) while the worker runs and once more
    when it exits — so every print reaches `flash runs log`, not just a post-mortem tail on
    crash. There is no FLASH_UPLOAD_CONSOLE flag to forget: it is NOT forwarded to the worker (even
    if an operator sets it), and neither worker run_mode path gates the upload."""
    import inspect

    from flash.providers import _instance_bootstrap
    from flash.providers.runpod.train import build_worker_env, endpoints

    # the flag is gone — setting it in the control-plane env does not reach the worker
    monkeypatch.setenv("FLASH_UPLOAD_CONSOLE", "1")
    assert "FLASH_UPLOAD_CONSOLE" not in build_worker_env(_spec(), 0)

    # both worker run_mode paths upload unconditionally (no flag, no gating var)
    for src in (
        inspect.getsource(_instance_bootstrap.run_mode),
        inspect.getsource(endpoints._train_body),
    ):
        assert "FLASH_UPLOAD_CONSOLE" not in src
        assert "upload_enabled" not in src
        assert "_force_console" not in src


def _spec_worker_env(worker_env: dict):
    """A grpo JobSpec carrying a per-run [worker_env] block (the TOML override map)."""
    from flash.spec import JobSpec, TrainSpec

    return JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
        worker_env=dict(worker_env),
    )


def test_build_worker_env_filters_removed_optimization_toggles(monkeypatch):
    """A per-run [worker_env] block can NOT re-inject the optimization toggles removed in PR #175
    (flash is deterministic + fully managed). The dangerous case: a recipe pinning
    PYTORCH_ALLOC_CONF=expandable_segments:True would crash GRPO vLLM sleep mode — it must be
    dropped, and flash's computed sleep-safe RL conf must survive. Non-removed keys still merge."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    spec = _spec_worker_env(
        {
            # Removed optimization toggles — must all be stripped.
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "VLLM_FLASH_ATTN_VERSION": "2",
            "VLLM_USE_V1": "0",
            "SFT_PER_DEVICE_BS": "1",
            "TORCHDYNAMO_DISABLE": "1",
            "FLASH_DISABLE_FA2": "1",
            "RL_VLLM_SLEEP": "0",
            "FLASH_ROPE_KERNEL": "0",
            "FLASH_WORKER_DEPS": "evil==9",
            # Case-insensitive match: a lower-cased re-injection is also stripped.
            "pytorch_alloc_conf": "expandable_segments:True",
            # A legitimate, non-removed per-run override still wins.
            "MY_ENV_FLAG": "keep-me",
        }
    )
    env = build_worker_env(spec, 0)  # grpo -> sleep-safe non-expandable alloc conf
    # The unsafe alloc conf was NOT injected; flash's computed RL conf stands.
    assert "expandable_segments" not in env["PYTORCH_ALLOC_CONF"]
    assert "expandable_segments" not in env["PYTORCH_CUDA_ALLOC_CONF"]
    for stripped in (
        "VLLM_ATTENTION_BACKEND",
        "VLLM_FLASH_ATTN_VERSION",
        "VLLM_USE_V1",
        "SFT_PER_DEVICE_BS",
        "TORCHDYNAMO_DISABLE",
        "FLASH_DISABLE_FA2",
        "RL_VLLM_SLEEP",
        "FLASH_ROPE_KERNEL",
        "FLASH_WORKER_DEPS",
    ):
        assert stripped not in env, f"{stripped} should have been filtered from worker_env"
    # A non-removed per-run key is honored — the filter is targeted, not a blanket block.
    assert env["MY_ENV_FLAG"] == "keep-me"


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
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="myorg/runs"),
        seed=0,
    )
    assert build_worker_env(per_run, 0)["HF_REPO"] == "myorg/runs"
    # still the per-run value even with no operator HF_REPO at all
    monkeypatch.delenv("HF_REPO", raising=False)
    assert build_worker_env(per_run, 0)["HF_REPO"] == "myorg/runs"


def test_alloc_conf_rl_is_non_expandable(monkeypatch):
    # vLLM sleep-mode CuMemAllocator is incompatible with expandable_segments, so RL ships the
    # sleep-SAFE non-expandable conf; the worker upgrades to expandable_segments at boot once it
    # resolves sleep OFF for the model/context (engine.worker.finalize_alloc_conf_for_sleep). The
    # conf is deterministic — there is no launcher sleep/alloc knob.
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    env = build_worker_env(_spec(), 0)  # grpo
    assert "expandable_segments" not in env["PYTORCH_ALLOC_CONF"]
    assert env["PYTORCH_ALLOC_CONF"] == env["PYTORCH_CUDA_ALLOC_CONF"]
    # no launcher->worker FLASH_ALLOC_AUTO signal anymore (the worker gates on PHASE == "rl")
    assert "FLASH_ALLOC_AUTO" not in env


def test_alloc_conf_default_expandable_for_sft(monkeypatch):
    from flash.providers.runpod.train import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    spec = JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        algorithm="sft",
        train=TrainSpec(epochs=1, max_examples=2),
        seed=0,
    )
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


def test_error_artifact_name_is_per_phase_and_attempt():
    """Error files are scoped per-phase and per-attempt so a stale prior-attempt
    traceback can't be mistaken for the current attempt's crash on a retry."""
    from flash.engine.worker import error_artifact_name

    names = {error_artifact_name(m) for m in ("sft", "rl")}
    assert len(names) == 2  # distinct per phase -> no clobber
    assert error_artifact_name("rl") == "error_rl_attempt0.txt"
    assert error_artifact_name("rl", 0) != error_artifact_name("rl", 1)
    assert error_artifact_name("sft", 2) == "error_sft_attempt2.txt"
    for invalid in ("3", "", True, 1.5, -1, 1 << 63):
        with pytest.raises(ValueError, match="attempt must be"):
            error_artifact_name("sft", invalid)


def test_ray_log_artifact_name_is_scoped_exactly_like_the_traceback_beside_it():
    """Ray's failure logs upload to the same per-RUN hf_prefix() as the traceback, so they need the
    same per-attempt scoping. A raylet failure is precisely the case that gets retried, and an
    unscoped name would let the retry overwrite the attempt that actually reproduced it."""
    from flash.engine.worker import error_artifact_name, ray_log_artifact_name

    assert ray_log_artifact_name("rl") == "raylogs_rl_attempt0.txt"
    assert ray_log_artifact_name("rl", 0) != ray_log_artifact_name("rl", 1)
    assert ray_log_artifact_name("rl", 1) != error_artifact_name("rl", 1)
    for invalid in ("3", "", True, 1.5, -1, 1 << 63):
        with pytest.raises(ValueError, match="attempt must be"):
            ray_log_artifact_name("sft", invalid)


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
    # Names that must be locally imported (regression: contextlib was missing; threading is used by
    # the always-on console uploader).
    for name in ("contextlib", "json", "os", "subprocess", "sys", "threading"):
        assert name in imported, f"_train_body uses {name!r} without a local import"
    assert "_CONSOLE_UPLOAD_INTERVAL_S" not in inspect.getsource(train._train_body)


def test_train_body_has_no_prime_install_path():
    import inspect

    from flash.providers.runpod import train

    src = inspect.getsource(train._train_body)
    assert '"install", "prime"' not in src
    assert 'shutil.which("prime")' not in src


def test_train_body_extra_pip_uses_worker_env_credentials(monkeypatch):
    import os
    from pathlib import Path

    from flash.providers.runpod.train import endpoints

    calls = []
    askpass_paths = []

    def fake_run(cmd, *, check, env=None):
        askpass = Path(env["GIT_ASKPASS"])
        assert askpass.exists()
        assert os.access(askpass, os.X_OK)
        assert "ghp-secret" not in askpass.read_text()
        askpass_paths.append(askpass)
        calls.append({"cmd": cmd, "check": check, "env": env})

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(ValueError, match="invalid code_prefix"):
        endpoints._train_body(
            {
                "phase": "sft",
                "seed": 0,
                "hf_repo": "owner/runs",
                "job_spec_json": '{"algorithm": "sft", "run_id": "flash-test-run"}',
                "env": {"GITHUB_TOKEN": "ghp-secret", "PYTHONPATH": ""},
                "extra_pip": ["git+https://github.com/example/env-pkg.git@abc123"],
                "code_prefix": "../code/flash",
                **_run_deadline_fields(),
            }
        )

    assert len(calls) == 1
    env = calls[0]["env"]
    assert env["GITHUB_TOKEN"] == "ghp-secret"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert askpass_paths
    assert all(not p.exists() for p in askpass_paths)


def test_train_body_extra_pip_ignores_askpass_cleanup_errors(monkeypatch):
    import os
    from pathlib import Path

    from flash.providers.runpod.train import endpoints

    askpass_paths = []

    def fake_run(cmd, *, check, env=None):
        askpass_paths.append(Path(env["GIT_ASKPASS"]))

    original_remove = os.remove

    def fake_remove(path):
        if Path(path) in askpass_paths:
            raise PermissionError("locked askpass helper")
        return original_remove(path)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(os, "remove", fake_remove)

    try:
        with pytest.raises(ValueError, match="invalid code_prefix"):
            endpoints._train_body(
                {
                    "phase": "sft",
                    "seed": 0,
                    "hf_repo": "owner/runs",
                    "job_spec_json": '{"algorithm": "sft", "run_id": "flash-test-run"}',
                    "env": {"GITHUB_TOKEN": "ghp-secret", "PYTHONPATH": ""},
                    "extra_pip": ["git+https://github.com/example/env-pkg.git@abc123"],
                    "code_prefix": "../code/flash",
                    **_run_deadline_fields(),
                }
            )
    finally:
        for askpass in askpass_paths:
            if askpass.exists():
                original_remove(askpass)

    assert askpass_paths


def test_sft_train_keeps_the_optimizations_that_survived_the_trl_deletion():
    """The verl SFT path must still carry the sizing/memory optimizations, not just train.

    The previous version of this test read run_sft's source. That body was trl's and is deleted:
    run_sft now delegates to run_sft_train. Rather than drop the coverage, assert against the module
    that really runs. Two of the old assertions are intentionally NOT reproduced -- LoRA+ B-matrix
    ratio plumbing and the chunked_nll loss_type -- because they were properties of trl's
    SFTTrainer call, and verl owns its own loss and kernel path.
    """
    import inspect

    from flash.engine.worker import sft, sft_train

    # run_sft is now a pure delegation: no backend selector, no trainer of its own.
    assert "run_sft_train()" in inspect.getsource(sft.run_sft)

    src = inspect.getsource(sft_train)
    # completion-only supervision survives, as verl's loss_mask rather than trl's completion_mask.
    assert "_pretokenize_completion_only(" in src
    assert "completion_mask_from_ids(" in src
    assert '"loss_mask": tokenized["completion_mask"]' in src
    # revision-aware vocab resolution: the worker must size the realized batch through the SAME
    # resolver the cost quote priced with, else a revision-pinned run drifts from its quote.
    assert "resolve_vocab_size(" in src
    assert "vocab_size_for(model_id)" not in src
    # per-device micro-batch / grad-accum sizing for the large-vocab logits cap.
    assert "sft_grad_accum(" in src
    # gradient checkpointing, with the MoE/GDN reentrant rule shared with grpo.
    assert "grad_checkpointing_on(" in src
    assert "grpo_use_reentrant(" in src
    # LoRA+ survives (verl builds the optimizer itself, but flash still supplies the grouping).
    assert "create_loraplus_optimizer" in src


def test_train_body_uploads_console_on_missing_metrics(monkeypatch, tmp_path):
    """The 'crashed before finishing' path (no /tmp/metrics.json) MUST upload the captured console
    even when the worker exited 0 — run_mode only uploads on a non-zero exit, so an OOM/segfault or
    silent early-exit otherwise leaves the failure undebuggable (no metrics, often no error_<phase>,
    and the message points at a console that was never uploaded)."""
    import contextlib
    import os
    import subprocess
    import types

    import huggingface_hub

    from flash.providers.runpod.train import endpoints

    code_prefix = "code/0123456789abcdef0123456789abcdef/flash"
    list_calls = []
    download_calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **k: pytest.fail("code download should not use snapshot_download"),
    )

    uploads = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_tree(self, **kw):
            list_calls.append(kw)
            if len(list_calls) == 1:
                raise _RateLimited("slow down")
            return [
                types.SimpleNamespace(path=f"{code_prefix}/__init__.py", size=0),
                types.SimpleNamespace(path=f"{code_prefix}/engine/worker.py", size=10),
                types.SimpleNamespace(path=f"{code_prefix}/engine", tree_id="folder"),
            ]

        def upload_file(self, **kw):
            uploads.append(kw)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    class _Response:
        status_code = 429

        def __init__(self) -> None:
            self.headers = {"Retry-After": "0"}

    class _RateLimited(Exception):
        response = _Response()

    def fake_hf_hub_download(*, filename, local_dir, **kw):
        download_calls.append({"filename": filename, "local_dir": local_dir, **kw})
        return os.path.join(local_dir, filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    class _FakeProc:
        # Worker boots, logs an OOM, then the kernel/clean-exit leaves NO metrics.json.
        def __init__(self, *a, **k):
            assert k["cwd"] == "/runcode/code/0123456789abcdef0123456789abcdef"
            self.stdout = iter(
                [
                    "worker booting\n",
                    ("x" * 70_000) + "\n",
                    "torch.cuda.OutOfMemoryError: CUDA OOM\n",
                ]
            )
            self.returncode = 0  # the bug case: exits 0, so run_mode skips the console upload

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakeProc)

    job_spec = '{"algorithm": "sft", "run_id": "flash-test-run"}'
    input_data = {
        "phase": "sft",
        "seed": 0,
        "hf_repo": "owner/runs",
        "job_spec_json": job_spec,
        "env": {"HF_TOKEN": "tok", "PYTHONPATH": ""},
        "code_prefix": code_prefix,
        **_run_deadline_fields(),
    }

    try:
        with pytest.raises(RuntimeError, match=r"produced no /tmp/metrics\.json"):
            endpoints._train_body(input_data)

        # The fix: the console for the crashed phase is uploaded so the failure is root-causable.
        console_uploads = [
            u for u in uploads if str(u.get("path_in_repo", "")).endswith("console_sft.txt")
        ]
        assert console_uploads, (
            f"console_sft.txt was not uploaded on the no-metrics crash path: {uploads}"
        )
        assert console_uploads[0]["path_in_repo"] == "sft/flash-test-run/console_sft.txt"
        with open(console_uploads[0]["path_or_fileobj"], encoding="utf-8") as f:
            uploaded_console = f.read()
        assert not uploaded_console.startswith("worker booting\n")
        assert uploaded_console.endswith("torch.cuda.OutOfMemoryError: CUDA OOM\n")
        assert len(uploaded_console) == 64_000
        assert [call["path_in_repo"] for call in list_calls] == [code_prefix, code_prefix]
        assert [call["filename"] for call in download_calls] == [
            f"{code_prefix}/__init__.py",
            f"{code_prefix}/engine/worker.py",
        ]
    finally:
        # _train_body writes the hardcoded /tmp/console_sft.txt(.tail); remove them so this test
        # doesn't leak state across tests (flaky under isolated/parallel runners).
        for _p in ("/tmp/console_sft.txt", "/tmp/console_sft.txt.tail"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(_p)


def test_train_body_rejects_unsafe_code_prefix(monkeypatch):
    import huggingface_hub

    from flash.providers.runpod.train import endpoints

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **k: pytest.fail("snapshot_download should not run with an invalid code prefix"),
    )
    with pytest.raises(ValueError, match="invalid code_prefix"):
        endpoints._train_body(
            {
                "phase": "sft",
                "seed": 0,
                "hf_repo": "owner/runs",
                "job_spec_json": '{"algorithm": "sft", "run_id": "flash-test-run"}',
                "env": {"HF_TOKEN": "tok"},
                "code_prefix": "../code/flash",
                **_run_deadline_fields(),
            }
        )


def test_live_console_uploads_are_throttled_for_shared_artifact_repos():
    import flash.engine.worker as worker
    from flash.providers import _instance_bootstrap
    from flash.providers.runpod.train import endpoints

    assert endpoints._CONSOLE_UPLOAD_INTERVAL_S == 3600.0
    assert _instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S == 3600.0
    steady_state_commits_per_hour = (
        3600.0 / worker._HB_MIN_INTERVAL_S + 3600.0 / endpoints._CONSOLE_UPLOAD_INTERVAL_S
    )
    assert steady_state_commits_per_hour <= 5.0


def test_worker_image_override_carries_its_registry_credential(monkeypatch):
    # a private override image needs its provider-side pull credential; that cannot be derived
    # from the image ref, so it rides alongside it.
    from flash.providers import _worker

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/example/private-worker:cu13")
    monkeypatch.setenv("FLASH_WORKER_IMAGE_REGISTRY_AUTH", "auth-123")
    o = _worker.worker_image_override()
    assert o.image == "ghcr.io/example/private-worker:cu13"
    assert o.registry_auth_id == "auth-123"
    assert _worker.worker_image_for_gpu("H200") == o.image


def test_worker_image_override_absent_is_none(monkeypatch):
    from flash.providers import _worker

    monkeypatch.delenv("FLASH_WORKER_IMAGE", raising=False)
    assert _worker.worker_image_override() is None


def test_min_cuda_for_uses_the_gpu_class_floor(monkeypatch):
    # the CUDA floor is a property of the GPU class, not of an operator-supplied image tag
    from flash.providers.runpod.train.endpoints import min_cuda_for

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/example/w:cu13")
    assert min_cuda_for("B200") == "13.0"  # blackwell needs cu13 drivers
    assert min_cuda_for("H200") == "12.8"


def test_apply_disk_raises_to_the_requested_floor(monkeypatch):
    from types import SimpleNamespace

    from flash.providers.runpod.jobs import apply_disk_gb, apply_image_override_constraints

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/example/w:big")
    monkeypatch.setenv("FLASH_WORKER_IMAGE_REGISTRY_AUTH", "auth-xyz")
    tpl = SimpleNamespace(containerDiskInGb=64, containerRegistryAuthId=None)
    cfg = SimpleNamespace(template=tpl)
    apply_disk_gb(cfg, 80)
    assert tpl.containerDiskInGb == 80  # raise-only: the request wins over the smaller default
    apply_disk_gb(cfg, 32)
    assert tpl.containerDiskInGb == 80  # never lowers an already-larger disk
    apply_image_override_constraints(cfg)
    assert tpl.containerRegistryAuthId == "auth-xyz"


def test_snapshot_weight_validation(tmp_path):
    from flash.engine.worker.hf import _snapshot_has_weights

    d = tmp_path / "snap"
    d.mkdir()
    (d / "config.json").write_text("{}")
    assert not _snapshot_has_weights(str(d))  # configs only = stale partial snapshot
    (d / "model.safetensors-00001-of-00001.safetensors").write_text("x")
    assert _snapshot_has_weights(str(d))
