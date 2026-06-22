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
        "RL_VLLM_SLEEP": "1",
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


def test_build_worker_env_forwards_upload_console(monkeypatch):
    """FLASH_UPLOAD_CONSOLE (upload the worker console on SUCCESS, not just on crash) is read on the
    worker by run_mode() from the forwarded env dict — RunPod _train_body AND the Vast bootstrap,
    both of which reuse this build_worker_env. It MUST be on the allowlist or the success-console
    upload silently no-ops on every remote run."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.setenv("FLASH_UPLOAD_CONSOLE", "1")
    assert build_worker_env(_spec(), 0).get("FLASH_UPLOAD_CONSOLE") == "1"
    monkeypatch.delenv("FLASH_UPLOAD_CONSOLE", raising=False)
    assert "FLASH_UPLOAD_CONSOLE" not in build_worker_env(_spec(), 0)


def test_build_worker_env_forwards_chalk_kernel_flags(monkeypatch):
    """The chalk-kernel install hook (engine.chalk_kernels) runs inside the worker subprocess and
    resolves which chalk kernels to apply from FLASH_* flags read from its OWN process env (gap-
    fillers default-on; FLASH_* are overrides). So an operator-set FLASH_* override (and
    FLASH_CHALK_SPEC) MUST be forwarded by the allowlist or it silently no-ops on every remote run."""
    from flash.providers.runpod.train import build_worker_env

    # Exactly the per-kernel boolean flags in chalk_kernels._KERNELS (one per
    # apply_chalk_kernel_to_qwen35 keyword) plus the chalk install spec.
    flags = {
        "FLASH_MLP_KERNEL": "1",
        "FLASH_FP8_BASE": "1",
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
        "FLASH_FP8_BASE",
        "FLASH_TRITON_LORA",
        "FLASH_EMBED_KERNEL",
        "FLASH_QKV_KERNEL",
        "FLASH_ROPE_KERNEL",
        "FLASH_CHALK_SPEC",
    ):
        monkeypatch.delenv(k, raising=False)


# The three default-on gap-filler flags (chalk_kernels._KERNELS): disabling exactly these
# deselects chalk. QKV/MLP/FP8 base are opt-in (default-off) and need no flag to stay off.
_DEFAULT_ON_CHALK_FLAGS = ("FLASH_ROPE_KERNEL", "FLASH_TRITON_LORA", "FLASH_EMBED_KERNEL")


def test_chalk_extra_pip_default_on_with_spec(monkeypatch):
    """Default (no FLASH_* flags) -> chalk's gap-fillers are ON, so a set FLASH_CHALK_SPEC IS
    appended to extra_pip (chalk installs + auto-applies, like Liger)."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_CHALK_SPEC", "freesolo-chalk")
    assert chalk_extra_pip() == ["freesolo-chalk"]


def test_chalk_extra_pip_defaults_to_pypi_without_spec(monkeypatch):
    """chalk is published on PyPI, so a SELECTED run with FLASH_CHALK_SPEC unset auto-installs the
    VERSION-PINNED PyPI package by default (just like Liger) — no operator spec required, and a
    breaking release can't silently land (the pin is bounded)."""
    from flash.providers.runpod.train import DEFAULT_CHALK_SPEC, chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_MLP_KERNEL", "1")
    assert chalk_extra_pip() == [DEFAULT_CHALK_SPEC]
    assert DEFAULT_CHALK_SPEC.startswith("freesolo-chalk")
    assert "<" in DEFAULT_CHALK_SPEC  # bounded range, not an unpinned floating install


def test_chalk_extra_pip_adds_spec_when_selected(monkeypatch):
    """Kernel flag + FLASH_CHALK_SPEC -> the chalk spec is appended to extra_pip, which the worker
    installs for EVERY job (the durable baked-image path that bypasses resolve_worker_deps)."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_FP8_BASE", "1")
    monkeypatch.setenv("FLASH_CHALK_SPEC", "git+https://github.com/freesolo-co/chalk@main")
    assert chalk_extra_pip() == ["git+https://github.com/freesolo-co/chalk@main"]


def test_chalk_extra_pip_all_kernels_disabled_adds_nothing(monkeypatch):
    """Disabling every default-on gap-filler (FLASH_<K>=0 on ROPE/TRITON_LORA/EMBED) -> chalk not
    selected -> nothing added, even with a spec set. The opt-in kernels (QKV/MLP/FP8 base) are
    default-off, so they need no flag to keep chalk deselected."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    for k in _DEFAULT_ON_CHALK_FLAGS:
        monkeypatch.setenv(k, "0")
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


def test_chalk_extra_pip_per_run_worker_env_spec_override(monkeypatch):
    """A per-run [worker_env] FLASH_CHALK_SPEC overrides the PyPI default for THAT run — resolved
    against the effective worker env (worker_env merged over os.environ), not bare os.environ, so a
    per-run source pin actually reaches the worker's extra_pip."""
    from flash.providers.runpod.train import DEFAULT_CHALK_SPEC, chalk_extra_pip

    _clear_chalk_flags(monkeypatch)  # nothing in os.environ
    spec = _spec_worker_env({"FLASH_CHALK_SPEC": "git+https://github.com/freesolo-co/chalk@main"})
    # bare env: chalk's gap-fillers are default-on -> the version-pinned PyPI default installs
    assert chalk_extra_pip() == [DEFAULT_CHALK_SPEC]
    # the per-run [worker_env] spec overrides the source for that run
    assert chalk_extra_pip(spec) == ["git+https://github.com/freesolo-co/chalk@main"]


def _fake_smi_compute_cap(value: str):
    """subprocess.run stub mimicking `nvidia-smi --query-gpu=compute_cap` (one line per GPU)."""

    class _Out:
        stdout = value

    def _run(cmd, *a, **k):
        return _Out()

    return _run


def test_fp8_native_no_cuda_context_reads_nvidia_smi(monkeypatch):
    """The DDP-launcher fp8 KV decision must read compute_cap from nvidia-smi (CUDA-free), NOT
    torch.cuda.get_device_capability (which binds a context on the first trainer card and lingers
    while the accelerate child trains on it — VRAM theft / rank-0 OOM on tight ratios)."""
    import subprocess

    from flash.engine.worker import _fp8_native_no_cuda_context

    monkeypatch.setattr(subprocess, "run", _fake_smi_compute_cap("9.0\n"))  # Hopper sm90
    assert _fp8_native_no_cuda_context() is True
    monkeypatch.setattr(subprocess, "run", _fake_smi_compute_cap("8.9\n"))  # Ada sm89
    assert _fp8_native_no_cuda_context() is True
    monkeypatch.setattr(subprocess, "run", _fake_smi_compute_cap("8.0\n"))  # Ampere sm80
    assert _fp8_native_no_cuda_context() is False

    def _boom(*a, **k):  # nvidia-smi unavailable -> None (caller defaults to no fp8)
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert _fp8_native_no_cuda_context() is None


def test_chalk_extra_pip_worker_env_spec_with_os_env_flag(monkeypatch):
    """Mixed source: the kernel flag is in the control-plane env but FLASH_CHALK_SPEC is pinned
    per-run in [worker_env]. The merged effective env resolves both, so the spec is added."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_TRITON_LORA", "1")  # selection from the process env
    spec = _spec_worker_env({"FLASH_CHALK_SPEC": "git+https://github.com/freesolo-co/chalk@dev"})
    assert chalk_extra_pip(spec) == ["git+https://github.com/freesolo-co/chalk@dev"]


def test_chalk_extra_pip_worker_env_overrides_os_env_flag(monkeypatch):
    """[worker_env] wins over os.environ. chalk is default-on, so to turn it OFF for a run, every
    default-on gap-filler (ROPE/TRITON_LORA/EMBED) must be disabled via [worker_env] (selection is
    binary: install chalk or not). The opt-in kernels are default-off and need no flag."""
    from flash.providers.runpod.train import chalk_extra_pip

    _clear_chalk_flags(monkeypatch)
    monkeypatch.setenv("FLASH_CHALK_SPEC", "git+https://github.com/freesolo-co/chalk@main")
    # default-on -> chalk selected, spec appended
    assert chalk_extra_pip() == ["git+https://github.com/freesolo-co/chalk@main"]
    # a per-run [worker_env] disabling every default-on gap-filler turns chalk off for that run
    spec = _spec_worker_env(dict.fromkeys(_DEFAULT_ON_CHALK_FLAGS, "0"))
    assert chalk_extra_pip(spec) == []


def test_build_worker_env_drops_reserved_disagg_parallel(monkeypatch):
    """Review fix: FLASH_DISAGG_PARALLEL is topology-owned. The allocator's required_vram_gb()
    sizes the inference card from the SUBMITTER os.environ (tp divides the server footprint by
    inference_gpus, dp does not). A per-run [worker_env] override applied AFTER sizing would let a
    tp-sized provision start full-footprint dp replicas -> the paid worker OOMs. So it must be
    reserved and NOT forwardable per run (set it via the submitter env instead)."""
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.delenv("FLASH_DISAGG_PARALLEL", raising=False)
    # A per-run override is dropped (run sizing owns it; submitter env is the only valid source).
    spec = _spec_worker_env({"FLASH_DISAGG_PARALLEL": "dp"})
    assert "FLASH_DISAGG_PARALLEL" not in build_worker_env(spec, 0)
    # Sanity: a NON-reserved per-run key still flows through, so the drop is targeted.
    spec2 = _spec_worker_env({"FLASH_DISAGG_PARALLEL": "dp", "FOO_PER_RUN": "1"})
    assert build_worker_env(spec2, 0).get("FOO_PER_RUN") == "1"
    # The submitter env IS the valid channel: a global value is still forwarded.
    monkeypatch.setenv("FLASH_DISAGG_PARALLEL", "tp")
    assert build_worker_env(_spec_worker_env({}), 0).get("FLASH_DISAGG_PARALLEL") == "tp"


def test_build_worker_env_reserves_disaggregated_role_and_rank_keys(monkeypatch):
    """The disaggregated ROLE / TOPOLOGY / RANK keys are control-plane- and launcher-owned and a
    per-run [worker_env] must NOT be able to forward them: FLASH_INFERENCE_GPUS would change the
    effective rollout split away from the sized/billed topology, FLASH_RL_TRAINER_ONLY would make
    the paid launcher skip starting vllm-serve, and RANK/LOCAL_RANK would suppress the only
    artifact-writing process (is_artifact_writer keys off RANK==0)."""
    from flash.providers.runpod.train import build_worker_env

    spec = _spec_worker_env(
        {
            "FLASH_INFERENCE_GPUS": "0",  # would collapse a provisioned split to colocate
            "FLASH_RL_TRAINER_ONLY": "1",  # would make the launcher act as a trainer child
            "RANK": "3",  # would move artifact writing off the real rank 0
            "LOCAL_RANK": "3",
            "FLASH_USER_KNOB": "ok",  # a genuine per-run override still passes through
        }
    )
    env = build_worker_env(spec, 0)
    # None of the per-run override VALUES ("0"/"1"/"3") may leak through for a reserved key.
    forwarded = {env.get(k) for k in ("FLASH_INFERENCE_GPUS", "FLASH_RL_TRAINER_ONLY", "RANK", "LOCAL_RANK")}
    assert forwarded.isdisjoint({"0", "1", "3"}), (
        f"reserved disaggregated role/rank keys must not be overridable via [worker_env]: {forwarded}"
    )
    assert env["FLASH_USER_KNOB"] == "ok"  # non-reserved keys are still forwarded


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


def test_alloc_conf_rl_cedes_to_worker(monkeypatch):
    # The launcher can't know the worker's sleep decision (resolved from model size + context),
    # so RL ships the conservative non-expandable conf plus FLASH_ALLOC_AUTO=1, ceding the final
    # choice to the worker (which upgrades to expandable_segments once it resolves sleep OFF).
    from flash.providers.runpod.train import build_worker_env

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    env = build_worker_env(_spec(), 0)  # grpo
    assert env["FLASH_ALLOC_AUTO"] == "1"
    assert "expandable_segments" not in env["PYTORCH_ALLOC_CONF"]


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


def _launcher_worker(monkeypatch, *, inference_gpus, total_gpus, trainer_only=False):
    """Configure the worker module globals so _is_disaggregated_ddp_launcher can be exercised
    CPU-only: an rl JobSpec with the given inference_gpus, a stubbed nvidia-smi GPU count, and the
    trainer-only role flag toggled. Returns the worker module."""
    import flash.engine.worker as ne
    from flash.spec import JobSpec, TrainSpec

    monkeypatch.setattr(ne, "RUN_MODE", "rl", raising=False)
    monkeypatch.setattr(
        ne,
        "JOB_SPEC",
        JobSpec(
            model="Qwen/Qwen3.5-4B",
            algorithm="grpo",
            train=TrainSpec(steps=10, seeds=(0,), hf_repo="owner/runs", inference_gpus=inference_gpus),
        ),
        raising=False,
    )
    import flash.engine.disaggregated as _d

    monkeypatch.setattr(_d, "detect_total_gpus", lambda env=None: total_gpus)
    monkeypatch.setattr(_d, "trainer_only_mode", lambda env=None: trainer_only)
    # _env_inference_gpus consults FLASH_INFERENCE_GPUS; keep it unset so the spec value is used.
    monkeypatch.delenv("FLASH_INFERENCE_GPUS", raising=False)
    return ne


def test_is_disaggregated_ddp_launcher_only_for_multi_trainer(monkeypatch):
    """The launcher detector (which gates the fla CUDA-probe skip) must be True ONLY for the
    multi-trainer (train_gpus>1) disaggregated launcher and False for the trainer child, the
    single-trainer (1:N) split, and the colocate path — so a single-process trainer still drops
    fla on Hopper while the always-alive launcher never binds an idle, rank-0-starving CUDA context."""
    # 2 train : 1 infer (total 3, inference_gpus 1) -> multi-trainer launcher
    ne = _launcher_worker(monkeypatch, inference_gpus=1, total_gpus=3)
    assert ne._is_disaggregated_ddp_launcher() is True
    # The accelerate child of that same split IS a trainer -> must NOT skip (drops fla rank-local)
    ne = _launcher_worker(monkeypatch, inference_gpus=1, total_gpus=3, trainer_only=True)
    assert ne._is_disaggregated_ddp_launcher() is False
    # 1 train : 1 infer (total 2) -> single-trainer, single process -> not a launcher
    ne = _launcher_worker(monkeypatch, inference_gpus=1, total_gpus=2)
    assert ne._is_disaggregated_ddp_launcher() is False
    # colocate (inference_gpus=0) -> not a launcher
    ne = _launcher_worker(monkeypatch, inference_gpus=0, total_gpus=1)
    assert ne._is_disaggregated_ddp_launcher() is False


def test_is_disaggregated_ddp_launcher_fails_safe_on_gpu_count_error(monkeypatch):
    """When GPU-count detection genuinely fails (node size UNKNOWN) inside the disaggregated config
    (rl mode, not the trainer child, inference_gpus>0), fail SAFE to True, not silently to False.
    Returning False would let _drop_fla_on_hopper run get_device_capability() and bind a retained
    CUDA context on the pinned trainer card for the whole DDP launch — the exact rank-0 OOM hazard
    this guard prevents. Detection failure is the ONLY path that can reach this fallback now: the
    launcher-vs-single-trainer decision is computed deterministically from total-inference_gpus."""
    ne = _launcher_worker(monkeypatch, inference_gpus=1, total_gpus=3)
    import flash.engine.disaggregated as _d

    def _boom(env=None):
        raise RuntimeError("nvidia-smi blew up")

    monkeypatch.setattr(_d, "detect_total_gpus", _boom)
    assert ne._is_disaggregated_ddp_launcher() is True


def test_is_disaggregated_ddp_launcher_single_trainer_drops_fla_when_count_known(monkeypatch):
    """Regression: a 1:1 single-trainer disaggregated run (total=2, inference_gpus=1, train_gpus=1)
    has NO accelerate children — this process IS the in-process trainer and MUST drop Hopper fla.
    The decision is made from total-inference_gpus, so a working GPU count returns False (don't skip
    the fla probe) even though the broad earlier code path would have over-fired to True on error."""
    ne = _launcher_worker(monkeypatch, inference_gpus=1, total_gpus=2)
    assert ne._is_disaggregated_ddp_launcher() is False
    # 1:3 (total=4, inference_gpus=3, train_gpus=1) is also single-trainer -> False
    ne = _launcher_worker(monkeypatch, inference_gpus=3, total_gpus=4)
    assert ne._is_disaggregated_ddp_launcher() is False
