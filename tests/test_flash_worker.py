"""Regression tests for the Flash worker plumbing fixed in this PR.

Covers:
- build_worker_env forwards the documented RL/vLLM tuning knobs to the GPU worker
  (they were silently dropped) and sets a fragmentation-safe allocator default;
- the runpod_flash backoff OverflowError that aborted long runs is patched;
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
    """The worker-side knobs the worker / vLLM actually read are forwarded to the GPU worker."""
    from flash.providers.runpod.train import build_worker_env

    knobs = {
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


def test_build_worker_env_ignores_alloc_conf_override(monkeypatch):
    """flash is fully managed: an operator PYTORCH_CUDA_ALLOC_CONF in the process env does NOT
    override flash's computed allocator conf (RL is non-expandable, sleep-safe)."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:999")
    env = build_worker_env(_spec(), 0)  # grpo -> sleep-safe non-expandable
    assert env["PYTORCH_CUDA_ALLOC_CONF"] != "max_split_size_mb:999"
    assert "expandable_segments" not in env["PYTORCH_CUDA_ALLOC_CONF"]


def test_build_worker_env_forwards_judge_model(monkeypatch):
    """The optimizer-authored verifiers env reads FLASH_JUDGE_MODEL on the worker to pick its
    JudgeRubric client model (SFT-eval / GRPO-reward / rejection-sampling); the control-plane
    override must be forwarded, else the env silently falls back to its generated default."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("FLASH_JUDGE_MODEL", "openai/gpt-oss-120b")
    assert build_worker_env(_spec(), 0).get("FLASH_JUDGE_MODEL") == "openai/gpt-oss-120b"
    monkeypatch.delenv("FLASH_JUDGE_MODEL", raising=False)
    assert "FLASH_JUDGE_MODEL" not in build_worker_env(_spec(), 0)


def test_build_worker_env_forwards_github_env_source_token(monkeypatch):
    """The worker receives the control-plane token used for managed Freesolo environments."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("GITHUB_TOKEN", "ghp-secret")
    assert build_worker_env(_spec(), 0).get("GITHUB_TOKEN") == "ghp-secret"
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert "GITHUB_TOKEN" not in build_worker_env(_spec(), 0)


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
        train=TrainSpec(steps=10, seeds=(0,), hf_repo="owner/runs"),
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


def test_build_worker_env_forwards_upload_console(monkeypatch):
    """FLASH_UPLOAD_CONSOLE (upload the worker console on SUCCESS, not just on crash) is read on the
    worker by run_mode() from the forwarded env dict. It MUST be on the allowlist or the
    success-console upload silently no-ops on every remote run."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("FLASH_UPLOAD_CONSOLE", "1")
    assert build_worker_env(_spec(), 0).get("FLASH_UPLOAD_CONSOLE") == "1"
    monkeypatch.delenv("FLASH_UPLOAD_CONSOLE", raising=False)
    assert "FLASH_UPLOAD_CONSOLE" not in build_worker_env(_spec(), 0)


def _clear_chalk_flags(monkeypatch):
    monkeypatch.delenv("FLASH_CHALK_SPEC", raising=False)


def test_chalk_extra_pip_default_on_with_spec(monkeypatch):
    """chalk is always selected (fixed gap-fillers), so a set FLASH_CHALK_SPEC IS appended to
    extra_pip (chalk installs + auto-applies, like Liger)."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_CHALK_SPEC", "freesolo-chalk")
    assert chalk_extra_pip() == ["freesolo-chalk"]


def test_chalk_extra_pip_defaults_to_pypi_without_spec(monkeypatch):
    """chalk is published on PyPI, so with FLASH_CHALK_SPEC unset flash auto-installs the
    VERSION-PINNED PyPI package by default (just like Liger) — no operator spec required, and a
    breaking release can't silently land (the pin is bounded)."""
    from flash.providers.runpod.train import DEFAULT_CHALK_SPEC, chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    assert chalk_extra_pip() == [DEFAULT_CHALK_SPEC]
    assert DEFAULT_CHALK_SPEC.startswith("freesolo-chalk")
    assert "<" in DEFAULT_CHALK_SPEC  # bounded range, not an unpinned floating install


def test_chalk_extra_pip_adds_spec_when_set(monkeypatch):
    """FLASH_CHALK_SPEC -> the chalk spec is appended to extra_pip, which the worker installs for
    EVERY job (the durable baked-image path that bypasses resolve_worker_deps)."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_CHALK_SPEC", "git+https://github.com/freesolo-co/chalk@main")
    assert chalk_extra_pip() == ["git+https://github.com/freesolo-co/chalk@main"]


def _spec_worker_env(worker_env: dict):
    """A grpo JobSpec carrying a per-run [worker_env] block (the TOML override map)."""
    from flash.spec import JobSpec, TrainSpec

    return JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(steps=10, seeds=(0,), hf_repo="owner/runs"),
        worker_env=dict(worker_env),
    )


def test_chalk_extra_pip_per_run_worker_env_spec_override(monkeypatch):
    """A per-run [worker_env] FLASH_CHALK_SPEC overrides the PyPI default install SOURCE for THAT
    run — resolved against the effective worker env (worker_env merged over os.environ)."""
    from flash.providers.runpod.train import DEFAULT_CHALK_SPEC, chalk_extra_pip

    _clear_chalk_flags(monkeypatch)  # nothing in os.environ
    spec = _spec_worker_env({"FLASH_CHALK_SPEC": "git+https://github.com/freesolo-co/chalk@main"})
    # bare env: the version-pinned PyPI default installs
    assert chalk_extra_pip() == [DEFAULT_CHALK_SPEC]
    # the per-run [worker_env] spec overrides the source for that run
    assert chalk_extra_pip(spec) == ["git+https://github.com/freesolo-co/chalk@main"]


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
        train=TrainSpec(steps=10, seeds=(0,), hf_repo="myorg/runs"),
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
    # Names that must be locally imported (regression: contextlib was missing).
    for name in ("contextlib", "json", "os", "subprocess", "sys"):
        assert name in imported, f"_train_body uses {name!r} without a local import"


def test_train_body_has_no_prime_install_path():
    import inspect

    from flash.providers.runpod import train

    src = inspect.getsource(train._train_body)
    assert '"install", "prime"' not in src
    assert 'shutil.which("prime")' not in src


def test_train_body_uploads_console_on_missing_metrics(monkeypatch, tmp_path):
    """The 'crashed before finishing' path (no /tmp/metrics.json) MUST upload the captured console
    even when the worker exited 0 — run_mode only uploads on a non-zero exit, so an OOM/segfault or
    silent early-exit otherwise leaves the failure undebuggable (no metrics, often no error_<phase>,
    and the message points at a console that was never uploaded)."""
    import contextlib
    import os
    import subprocess

    import huggingface_hub

    from flash.providers.runpod.train import endpoints

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path))

    uploads = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def upload_file(self, **kw):
            uploads.append(kw)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    class _FakeProc:
        # Worker boots, logs an OOM, then the kernel/clean-exit leaves NO metrics.json.
        def __init__(self, *a, **k):
            self.stdout = iter(["worker booting\n", "torch.cuda.OutOfMemoryError: CUDA OOM\n"])
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
    }

    try:
        with pytest.raises(RuntimeError, match=r"produced no /tmp/metrics\.json"):
            endpoints._train_body(input_data)

        # The fix: the console for the crashed phase is uploaded so the failure is root-causable.
        console_uploads = [u for u in uploads if str(u.get("path_in_repo", "")).endswith("console_sft.txt")]
        assert console_uploads, f"console_sft.txt was not uploaded on the no-metrics crash path: {uploads}"
        assert console_uploads[0]["path_in_repo"] == "sft/flash-test-run/seed0/console_sft.txt"
    finally:
        # _train_body writes the hardcoded /tmp/console_sft.txt(.tail); remove them so this test
        # doesn't leak state across tests (flaky under isolated/parallel runners).
        for _p in ("/tmp/console_sft.txt", "/tmp/console_sft.txt.tail"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(_p)
