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
    from flash.core.spec import JobSpec, TrainSpec

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
    from flash.providers.runpod.serverless import build_worker_env

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
    from flash.providers.runpod.serverless import build_worker_env

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
    from flash.core.spec import JobSpec, TrainSpec
    from flash.providers.runpod.serverless import build_worker_env

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
            "FLASH_PUBLIC_URL": "https://broker.example",
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
    from flash.providers.runpod.serverless import build_worker_env

    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "FLASH_JUDGE_MODEL"):
        monkeypatch.setenv(key, "control-plane-should-not-forward")
    env = build_worker_env(_spec(), 0)
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "FLASH_JUDGE_MODEL"):
        assert key not in env


def test_build_worker_env_forwards_github_env_source_token(monkeypatch):
    """The worker receives the control-plane token used for managed Freesolo environments."""
    from flash.providers.runpod.serverless import build_worker_env

    monkeypatch.setenv("GITHUB_TOKEN", "ghp-secret")
    assert build_worker_env(_spec(), 0).get("GITHUB_TOKEN") == "ghp-secret"
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


def test_build_worker_env_forwards_only_managed_teacher_capability_for_opd(monkeypatch):
    """opd receives bounded broker transport while provider credentials remain control-plane-only."""
    from flash.core.spec import JobSpec, TrainSpec
    from flash.providers.runpod.serverless import build_worker_env

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
            "FLASH_PUBLIC_URL": "https://broker.example",
            "FLASH_TEACHER_CAPABILITY": "capability-test-value",
        },
    )
    assert env["FLASH_PUBLIC_URL"] == "https://broker.example"
    assert env["FLASH_TEACHER_CAPABILITY"] == "capability-test-value"
    assert "PARASAIL_API_KEY" not in env
    grpo = build_worker_env(_spec(), 0)
    assert "FLASH_PUBLIC_URL" not in grpo
    assert "FLASH_TEACHER_CAPABILITY" not in grpo


def test_build_worker_env_does_not_accept_legacy_teacher_broker_url():
    from flash.core.spec import JobSpec, TrainSpec
    from flash.providers.runpod.serverless import build_worker_env

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
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
    from flash.providers.runpod.serverless import build_worker_env

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
                "FLASH_PUBLIC_URL": "https://broker.example",
                "FLASH_TEACHER_CAPABILITY": "capability-test-value",
            },
        )


def test_build_worker_env_wandb_is_user_runtime_secret_not_control_plane_env(monkeypatch):
    """Provider/platform creds are supplied by the control plane, but W&B belongs to the user.

    WANDB_API_KEY must therefore come from the per-submit runtime secret path, not from the
    control-plane process env.
    """
    from flash.providers.runpod.serverless import build_worker_env

    monkeypatch.setenv("WANDB_API_KEY", "platform-should-not-forward")
    env = build_worker_env(_spec(), 0)
    assert "WANDB_API_KEY" not in env

    env = build_worker_env(_spec(), 0, runtime_secrets={"WANDB_API_KEY": "user-wb"})
    assert env["WANDB_API_KEY"] == "user-wb"


def test_build_worker_env_forwards_declared_environment_runtime_secrets():
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
    from flash.providers.runpod.serverless import build_worker_env

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


def test_build_worker_env_lists_declared_secret_names_for_the_redactors():
    """declared runtime secrets can carry any name (AWS_SECRET_ACCESS_KEY, ...), so the redactors
    cannot rely on the name-shape heuristic; the env carries the applied names explicitly."""
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
    from flash.providers.runpod.serverless import build_worker_env

    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        environment=EnvironmentSpec(id="owner/env", secrets=("AWS_SECRET_ACCESS_KEY",)),
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )

    env = build_worker_env(
        spec,
        0,
        runtime_secrets={"AWS_SECRET_ACCESS_KEY": "aws-user", "WANDB_API_KEY": "user-wb"},
    )

    listed = set(env[SECRET_ENV_KEYS_ENV].split(","))
    assert listed == {"AWS_SECRET_ACCESS_KEY", "WANDB_API_KEY"}
    # a run with no applied secrets carries no list at all.
    assert SECRET_ENV_KEYS_ENV not in build_worker_env(_spec(), 0)


def test_the_redactor_metadata_name_is_reserved_from_declared_secrets():
    """build_worker_env sets FLASH_SECRET_ENV_KEYS last, so a job declaring that exact name would
    have its credential silently overwritten by the generated name list and fail at runtime. it is
    control-plane-owned, so the declaration is rejected loudly instead."""
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
    from flash.core.spec import CONTROL_PLANE_OWNED_ENV_KEYS
    from flash.schema.fields import ConfigError, _environment_secrets

    assert SECRET_ENV_KEYS_ENV in CONTROL_PLANE_OWNED_ENV_KEYS
    with pytest.raises(ConfigError, match="platform-managed key"):
        _environment_secrets([SECRET_ENV_KEYS_ENV])
    # a case variant is a distinct linux env name but not a distinct DECLARATION: build_worker_env
    # tests ownership on the uppercased name, so accepting it here would drop the secret from the
    # worker env without a word and launch the job missing a credential it declared as required.
    # every reserved name is refused across its whole case-space for that reason.
    for variant in (SECRET_ENV_KEYS_ENV.lower(), "Hf_Token", "runpod_api_key"):
        with pytest.raises(ConfigError, match="platform-managed key"):
            _environment_secrets([variant])


def test_declared_secret_names_cannot_contain_the_metadata_delimiter():
    """the name list travels to every redactor comma-joined, so a name containing a comma arrives
    as two unrelated names, the real key goes unrecognized, and its value reaches diagnostics
    verbatim. rejecting the delimiter at declaration keeps that channel unambiguous."""
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
    from flash.providers.runpod.serverless import build_worker_env
    from flash.schema.fields import ConfigError, _environment_secrets

    with pytest.raises(ConfigError, match="invalid environment variable name"):
        _environment_secrets(["FOO,BAR"])
    # a name-shaped secret is still fine; only the delimiter is refused.
    assert _environment_secrets(["AWS_SECRET_ACCESS_KEY"]) == ("AWS_SECRET_ACCESS_KEY",)

    # and the metadata builder fails closed rather than emitting an ambiguous list, for a spec
    # constructed around the parser.
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        environment=EnvironmentSpec(id="owner/env", secrets=("FOO,BAR",)),
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )
    with pytest.raises(RuntimeError, match="delimiter"):
        build_worker_env(spec, 0, runtime_secrets={"FOO,BAR": "leaky"})


def test_the_handlers_inline_redactor_covers_multiline_secret_components():
    """the handler is source-shipped, so it carries its OWN copy of the redactor rather than
    importing the shared one. the child's stdout is sanitized one line at a time, so a PEM key
    never appears whole in any single call and only its component lines can match. extract that
    copy and exercise it directly, since drift here leaks a credential nothing else catches."""
    import ast
    import inspect
    import os
    import re
    import textwrap

    from flash.providers.runpod.serverless import endpoints

    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoints._train_body)))
    # os/re come from _train_body's own local imports, which the handler makes at the top of its
    # body; urllib.parse it imports itself.
    namespace: dict = {"os": os, "re": re}
    for name in ("_needles", "_safe_detail"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<handler>", "exec"), namespace)
    safe_detail = namespace["_safe_detail"]

    pem = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----"
    # DEPLOY_KEY matches no suffix heuristic, so this also covers the declared-name channel.
    secrets = {"DEPLOY_KEY": pem, "FLASH_SECRET_ENV_KEYS": "DEPLOY_KEY"}

    assert (
        safe_detail("ssh auth: MIIEvQIBADANBgkqhkiG9w0BAQEFAASC", secrets) == "ssh auth: <redacted>"
    )
    assert safe_detail(pem, secrets) == "<redacted>"
    # the length floor keeps short structural fragments of a MULTILINE value readable: a short
    # component is punctuation, not a credential.
    assert safe_detail("near }", {"B": "{\n}\nlongenoughcomponent"}) == "near }"
    # a whole declared value below the floor is a credential, though, and must not leak. it is
    # redacted where it stands alone, and only there: as a global needle `ati` would rewrite
    # `authentication`.
    short = {"PIN": "ati", "FLASH_SECRET_ENV_KEYS": "PIN"}
    assert safe_detail("worker rejected pin ati", short) == "worker rejected pin <redacted>"
    assert safe_detail("trainer crashed after validation", short) == (
        "trainer crashed after validation"
    )
    long = {"PIN": "sk-live-abc123456", "FLASH_SECRET_ENV_KEYS": "PIN"}
    assert safe_detail("trainer crashed holding sk-live-abc123456", long) == (
        "trainer crashed holding <redacted>"
    )
    # the word guard is per EDGE: a value whose own edge is punctuation already separates itself,
    # and requiring a non-word character beyond it would leak "/a" out of "https://host/a/repo".
    path_like = {"S": "/a", "FLASH_SECRET_ENV_KEYS": "S"}
    assert safe_detail("https://host/a/repo", path_like) == "https://host<redacted>/repo"


def test_worker_console_always_uploaded_and_no_flag(monkeypatch):
    """The worker console is ALWAYS uploaded — live (periodic) while the worker runs and once more
    when it exits — so every print reaches `flash runs log`, not just a post-mortem tail on
    crash. There is no FLASH_UPLOAD_CONSOLE flag to forget: it is NOT forwarded to the worker (even
    if an operator sets it), and neither worker run_mode path gates the upload."""
    import inspect

    from flash.providers._lifecycle import bootstrap as _instance_bootstrap
    from flash.providers.runpod.serverless import build_worker_env, endpoints

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


def test_removed_keys_cannot_reach_the_worker_through_environment_secrets():
    """declared runtime secrets must not deliver removed optimization keys to the worker.

    the two filters answer different questions and are deliberately disjoint:
    CONTROL_PLANE_OWNED_ENV_KEYS prevents overrides such as SEED, while
    _REMOVED_OPTIMIZATION_ENV blocks dead keys that configure nothing.
    """
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
    from flash.providers._lifecycle.worker import _REMOVED_OPTIMIZATION_ENV
    from flash.providers.runpod.serverless import build_worker_env

    # every removed key, not a chalk special case. FLASH_TRITON_LORA stands in for the rest.
    declared = ["FLASH_CHALK_SPEC", "FLASH_TRITON_LORA", "MY_TOKEN"]
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
        environment=EnvironmentSpec(id="owner/env", secrets=declared),
    )
    supplied = {
        "FLASH_CHALK_SPEC": "freesolo-chalk==0.5.7",
        "FLASH_TRITON_LORA": "1",
        "MY_TOKEN": "keep-me",
    }
    env = build_worker_env(spec, 0, runtime_secrets=supplied)

    # assert on the values this call supplied, not on key presence: some removed names (e.g.
    # PYTORCH_ALLOC_CONF) are legitimately set by flash itself downstream, so "key absent from env"
    # would be asserting something the fix never promised.
    for key in set(supplied) & _REMOVED_OPTIMIZATION_ENV:
        assert env.get(key) != supplied[key], (
            f"{key} reached the worker through [environment].secrets"
        )
    # a live secret still gets through: this blocks dead keys, not runtime secrets generally.
    assert env["MY_TOKEN"] == "keep-me"


def test_build_worker_env_hf_repo_is_per_run(monkeypatch):
    """The worker env's HF_REPO is seeded from the run's [train] hf_repo, NOT the operator's
    HF_REPO env var (which no longer exists). An operator HF_REPO in the process env is
    ignored — the worker reads its own seeded value, sourced from the spec."""
    from flash.core.spec import JobSpec, TrainSpec
    from flash.providers.runpod.serverless import build_worker_env

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
    from flash.providers.runpod.serverless import build_worker_env

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    env = build_worker_env(_spec(), 0)  # grpo
    assert "expandable_segments" not in env["PYTORCH_ALLOC_CONF"]
    assert env["PYTORCH_ALLOC_CONF"] == env["PYTORCH_CUDA_ALLOC_CONF"]
    # no launcher->worker FLASH_ALLOC_AUTO signal anymore (the worker gates on PHASE == "rl")
    assert "FLASH_ALLOC_AUTO" not in env


def test_alloc_conf_default_expandable_for_sft(monkeypatch):
    from flash.core.spec import JobSpec, TrainSpec
    from flash.providers.runpod.serverless import build_worker_env

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


def test_runpod_backoff_preserves_strategies_cap_jitter_and_idempotence(monkeypatch):
    pytest.importorskip("runpod_flash")
    from flash.providers.runpod.serverless import _patch_runpod_backoff

    _patch_runpod_backoff()
    from runpod_flash.core.resources import serverless
    from runpod_flash.core.utils import backoff

    patched = backoff.get_backoff_delay
    assert serverless.get_backoff_delay is patched
    _patch_runpod_backoff()
    assert backoff.get_backoff_delay is patched
    assert serverless.get_backoff_delay is patched

    jitter_bounds: list[tuple[float, float]] = []

    def _uniform(low: float, high: float) -> float:
        jitter_bounds.append((low, high))
        return high

    monkeypatch.setattr("random.uniform", _uniform)
    strategy = backoff.BackoffStrategy
    assert patched(5000, max_seconds=5, jitter=0) == 5
    assert patched(3, base=0.5, max_seconds=10, jitter=0, strategy=strategy.LINEAR) == 2
    assert patched(2, base=0.5, max_seconds=10, jitter=0, strategy=strategy.LOGARITHMIC) == 1
    assert patched(1000, base=0.5, max_seconds=3, jitter=0, strategy=strategy.LINEAR) == 3
    assert patched(1, base=1, max_seconds=10, jitter=0.2) == pytest.approx(2.4)
    assert jitter_bounds[-1] == (0.8, 1.2)
    with pytest.raises(ValueError, match="Unsupported backoff strategy"):
        patched(1, strategy=object())


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


def test_worker_and_control_plane_agree_on_the_error_artifact_name():
    """The process that WRITES the error artifact and the one that READS it must spell it the same.

    These live on opposite sides of a machine boundary -- the worker uploads to HF, the pollers fetch
    from it -- and nothing at runtime ever compares the two names. A disagreement surfaces only as a
    missing artifact, which is indistinguishable from "the worker crashed before uploading": exactly
    the situation the file exists to explain, so the failure erases its own evidence.

    They now share one definition, and this pins that. A literal on each side would make the two
    agree by coincidence of two assertions rather than by construction -- so the assertion is
    writer-against-reader, not either against a string.
    """
    from flash.engine.worker import error_artifact_name as worker_name
    from flash.providers.artifacts.hf import error_artifact_name as plane_name

    for phase in ("sft", "rl", "opd"):
        for attempt in (0, 1, 7):
            assert worker_name(phase, attempt) == plane_name(phase, attempt)

    # and the reader rejects what the writer would refuse to produce, rather than formatting it
    for invalid in ("3", "", True, 1.5, -1, 1 << 63):
        with pytest.raises(ValueError, match="attempt must be"):
            plane_name("sft", invalid)


def _unresolved_source_globals(source: str) -> set[str]:
    import builtins
    import symtable

    root = symtable.symtable(source, "<source>", "exec")
    builtin_names = set(dir(builtins))
    unresolved: set[str] = set()

    def _visit(table) -> None:
        if table.get_type() == "function":
            unresolved.update(
                symbol.get_name()
                for symbol in table.get_symbols()
                if symbol.is_referenced()
                and symbol.is_global()
                and symbol.get_name() not in builtin_names
            )
        for child in table.get_children():
            _visit(child)

    _visit(root)
    return unresolved


def test_train_body_imports_every_name_it_uses():
    """the source-shipped handler must resolve without module globals."""
    import inspect

    from flash.providers.runpod import serverless as train

    source = inspect.getsource(train._train_body)
    assert _unresolved_source_globals(source) == set()
    assert "_CONSOLE_UPLOAD_INTERVAL_S" not in source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            "def outer():\n    def sibling():\n        value = 1\n    def reader():\n        return value\n",
            {"value"},
            id="sibling-local-fails",
        ),
        pytest.param(
            "def outer():\n    value = 1\n    def reader():\n        return value\n",
            set(),
            id="enclosing-local-succeeds",
        ),
        pytest.param(
            "def outer(items):\n    return len(items)\n",
            set(),
            id="builtin-succeeds",
        ),
    ],
)
def test_source_global_check_sensitivity(source, expected):
    assert _unresolved_source_globals(source) == expected


def test_train_body_has_no_prime_install_path():
    import inspect

    from flash.providers.runpod import serverless as train

    src = inspect.getsource(train._train_body)
    assert '"install", "prime"' not in src
    assert 'shutil.which("prime")' not in src


class _FakePipProc:
    """Popen stand-in for the extra_pip tee: an output stream plus one exit code."""

    def __init__(self, output: str = "", returncode: int = 0):
        import io

        self.stdout = io.StringIO(output)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def _extra_pip_input() -> dict:
    return {
        "phase": "sft",
        "seed": 0,
        "hf_repo": "owner/runs",
        "job_spec_json": '{"algorithm": "sft", "run_id": "flash-test-run"}',
        "env": {"GITHUB_TOKEN": "ghp-secret", "PYTHONPATH": ""},
        "extra_pip": ["git+https://github.com/example/some-env-pkg.git@abc123"],
        # an invalid prefix stops the handler right after the pip step, which is what is under test
        "code_prefix": "../code/flash",
        **_run_deadline_fields(),
    }


def test_train_body_extra_pip_uses_worker_env_credentials(monkeypatch):
    import os
    from pathlib import Path

    from flash.providers.runpod.serverless import endpoints

    calls = []
    askpass_paths = []

    def fake_popen(cmd, *, env=None, **_kwargs):
        askpass = Path(env["GIT_ASKPASS"])
        assert askpass.exists()
        assert os.access(askpass, os.X_OK)
        assert "ghp-secret" not in askpass.read_text()
        askpass_paths.append(askpass)
        calls.append({"cmd": cmd, "env": env})
        return _FakePipProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    with pytest.raises(ValueError, match="invalid code_prefix"):
        endpoints._train_body(_extra_pip_input())

    assert len(calls) == 1
    env = calls[0]["env"]
    assert env["GITHUB_TOKEN"] == "ghp-secret"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert askpass_paths
    assert all(not p.exists() for p in askpass_paths)


def test_train_body_extra_pip_ignores_askpass_cleanup_errors(monkeypatch):
    import os
    from pathlib import Path

    from flash.providers.runpod.serverless import endpoints

    askpass_paths = []

    def fake_popen(_cmd, *, env=None, **_kwargs):
        askpass_paths.append(Path(env["GIT_ASKPASS"]))
        return _FakePipProc()

    original_remove = os.remove

    def fake_remove(path):
        if Path(path) in askpass_paths:
            raise PermissionError("locked askpass helper")
        return original_remove(path)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(os, "remove", fake_remove)

    try:
        with pytest.raises(ValueError, match="invalid code_prefix"):
            endpoints._train_body(_extra_pip_input())
    finally:
        for askpass in askpass_paths:
            if askpass.exists():
                original_remove(askpass)

    assert askpass_paths


def _wire_train_body_pip(monkeypatch, results):
    """Patch Popen to replay ``results`` (output, rc) in order; returns the recorded calls."""
    calls = []
    queue = list(results)

    def fake_popen(cmd, *, env=None, **_kwargs):
        calls.append(cmd)
        output, rc = queue.pop(0) if queue else ("", 0)
        return _FakePipProc(output, rc)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    return calls


def test_train_body_extra_pip_retries_a_transient_index_failure(monkeypatch):
    # Same contract as the instance bootstrap: a PyPI blip is infra, not a bad requirement, so the
    # handler retries in place instead of failing the paid run on the first connection error.
    from flash.providers.runpod.serverless import endpoints

    calls = _wire_train_body_pip(
        monkeypatch,
        [
            (
                "WARNING: Retrying (Retry(total=4)) after connection broken by NewConnectionError\n",
                1,
            ),
            ("Successfully installed some-env-pkg-1.0\n", 0),
        ],
    )
    with pytest.raises(ValueError, match="invalid code_prefix"):
        endpoints._train_body(_extra_pip_input())
    assert len(calls) == 2


def test_train_body_extra_pip_resolution_error_stays_terminal(monkeypatch):
    # A bad package spec reached the index fine; retrying it would just burn another attempt.
    from flash.providers.runpod.serverless import endpoints

    calls = _wire_train_body_pip(
        monkeypatch,
        [("ERROR: No matching distribution found for definitely-not-a-package\n", 1)],
    )
    with pytest.raises(RuntimeError, match="extra_pip install failed"):
        endpoints._train_body(_extra_pip_input())
    assert len(calls) == 1  # fails fast, never walks the retry ladder


def test_train_body_extra_pip_build_failure_outranks_earlier_transient_text(monkeypatch):
    # pip warns "Retrying (Retry(" on an index blip, recovers, then fails compiling a wheel. Both
    # lines sit in the same captured tail, so matching transient text alone would call a
    # deterministic failure infra and repeat it three more times for nothing.
    from flash.providers.runpod.serverless import endpoints

    calls = _wire_train_body_pip(
        monkeypatch,
        [
            (
                (
                    "WARNING: Retrying (Retry(total=4)) after connection broken by "
                    "NewConnectionError\n"
                    "Collecting some-env-pkg\n"
                    "  error: subprocess-exited-with-error\n"
                    "ERROR: Failed building wheel for some-env-pkg\n"
                ),
                1,
            )
        ],
    )
    with pytest.raises(RuntimeError, match="extra_pip install failed"):
        endpoints._train_body(_extra_pip_input())
    assert len(calls) == 1  # the build failure names the cause, so no retry ladder


def test_train_body_extra_pip_matches_the_bootstrap_on_git_http_blips(monkeypatch):
    # the two classifiers must agree on what is retriable. A VCS pin fails through git, whose
    # phrasing carries none of the urllib shapes, so a 502 must retry here exactly as it does on
    # the instance bootstrap; a 404 is a bad pin and must still fail fast in both.
    from flash.providers.runpod.serverless import endpoints

    blip = (
        "  Running command git clone --filter=blob:none -q https://github.com/org/repo\n"
        "  fatal: unable to access 'https://github.com/org/repo/': "
        "The requested URL returned error: 502\n"
        "  error: subprocess-exited-with-error\n"
    )
    calls = _wire_train_body_pip(
        monkeypatch, [(blip, 1), ("Successfully installed some-env-pkg-1.0\n", 0)]
    )
    with pytest.raises(ValueError, match="invalid code_prefix"):
        endpoints._train_body(_extra_pip_input())
    assert len(calls) == 2

    # git's DNS wording, which urllib never emits, must retry here too.
    dns = blip.replace(
        "The requested URL returned error: 502", "Could not resolve host: github.com"
    )
    calls = _wire_train_body_pip(
        monkeypatch, [(dns, 1), ("Successfully installed some-env-pkg-1.0\n", 0)]
    )
    with pytest.raises(ValueError, match="invalid code_prefix"):
        endpoints._train_body(_extra_pip_input())
    assert len(calls) == 2

    missing = blip.replace("returned error: 502", "returned error: 404")
    calls = _wire_train_body_pip(monkeypatch, [(missing, 1)])
    with pytest.raises(RuntimeError, match="extra_pip install failed"):
        endpoints._train_body(_extra_pip_input())
    assert len(calls) == 1


def test_train_body_extra_pip_matches_the_bootstrap_on_an_index_outage_footer(monkeypatch):
    # an unreachable index prints the same no-candidate footer a typo'd name does, so that footer
    # alone must not be terminal when the tail also carries a transient marker. A build failure in
    # the same tail still decides it, since pip only reaches one with real content in hand. Both
    # classifiers must agree on this, so the RunPod copy is pinned exactly as the bootstrap is.
    from flash.providers.runpod.serverless import endpoints

    outage = (
        "WARNING: Retrying (Retry(total=4, connect=None)) after connection broken by "
        "NewConnectionError\n"
        "ERROR: Could not find a version that satisfies the requirement requests "
        "(from versions: none)\n"
        "ERROR: No matching distribution found for requests\n"
    )
    calls = _wire_train_body_pip(
        monkeypatch, [(outage, 1), ("Successfully installed some-env-pkg-1.0\n", 0)]
    )
    with pytest.raises(ValueError, match="invalid code_prefix"):
        endpoints._train_body(_extra_pip_input())
    assert len(calls) == 2

    built = outage + "ERROR: Failed building wheel for requests\n"
    calls = _wire_train_body_pip(monkeypatch, [(built, 1)])
    with pytest.raises(RuntimeError, match="extra_pip install failed"):
        endpoints._train_body(_extra_pip_input())
    assert len(calls) == 1


def test_train_body_extra_pip_stops_after_the_bounded_retries(monkeypatch):
    from flash.providers.runpod.serverless import endpoints

    calls = _wire_train_body_pip(monkeypatch, [("read timed out\n", 1)] * 4)
    with pytest.raises(RuntimeError, match="could not reach the package index"):
        endpoints._train_body(_extra_pip_input())
    assert len(calls) == 4  # one attempt plus the three bounded retries


def test_sft_train_keeps_the_optimizations_that_survived_the_trl_deletion():
    """The verl SFT path must still carry the sizing/memory optimizations, not just train.

    The previous version of this test read run_sft's source. That body was trl's and is deleted:
    run_sft now delegates to run_sft_train. Rather than drop the coverage, assert against the module
    that really runs. Three of the old assertions are intentionally NOT reproduced -- kernel
    installation, LoRA+ B-matrix ratio plumbing, and the chunked_nll loss_type -- because they were
    properties of trl's SFTTrainer call, and verl owns its own loss and kernel path.

    the optimizations now live in two modules rather than one. dataset preprocessing moved to
    flash.engine.profiling.sft_workload so estimate construction and training share one implementation,
    and the sizing/memory choices stayed with the trainer that makes them. each assertion reads the
    module that actually owns its behaviour: pointing them all at one module would let a symbol
    disappear from the other and still pass.
    """
    import inspect

    from flash.engine.profiling import sft_workload
    from flash.engine.worker import sft_train
    from flash.engine.worker.entry import sft

    # run_sft is now a pure delegation: no backend selector, no trainer of its own.
    assert "run_sft_train()" in inspect.getsource(sft.run_sft)

    workload_src = inspect.getsource(sft_workload)
    # completion-only supervision survives, as verl's loss_mask rather than trl's completion_mask.
    assert "_pretokenize_completion_only(" in workload_src
    assert "completion_mask_from_ids(" in workload_src
    assert '"loss_mask": tokenized["completion_mask"]' in workload_src

    # sft renders its hydra overrides and child shims in train.sft.config, so the trainer's half of
    # this guard spans both modules. keep these in step when sft_train is split further.
    from flash.engine.worker.train.sft import config as sft_config

    train_src = inspect.getsource(sft_train) + inspect.getsource(sft_config)
    # revision-aware vocab resolution: the worker must size the realized batch through the SAME
    # resolver the cost quote priced with, else a revision-pinned run drifts from its quote.
    assert "resolve_vocab_size(" in train_src
    assert "vocab_size_for(model_id)" not in train_src
    # per-device micro-batch / grad-accum sizing for the large-vocab logits cap.
    assert "sft_grad_accum(" in train_src
    # gradient checkpointing, with the MoE/GDN reentrant rule shared with grpo.
    assert "grad_checkpointing_on(" in train_src
    assert "grpo_use_reentrant(" in train_src
    # LoRA+ survives (verl builds the optimizer itself, but flash still supplies the grouping).
    assert "create_loraplus_optimizer" in train_src


@pytest.mark.parametrize(
    ("console_lines", "terminated"),
    [
        (
            ["worker booting\n", ("x" * 70_000) + "\n", "torch.cuda.OutOfMemoryError: CUDA OOM\n"],
            True,
        ),
        # the same crash with NO newline anywhere: one huge unterminated line, which is what a json
        # blob, a native stack or a stream of progress output actually looks like.
        (["q" * 70_000 + "torch.cuda.OutOfMemoryError: CUDA OOM"], False),
    ],
    ids=["oversized-line-then-rootcause", "single-unterminated-line"],
)
def test_train_body_uploads_console_on_missing_metrics(
    monkeypatch, tmp_path, console_lines, terminated
):
    """The 'crashed before finishing' path (no /tmp/metrics.json) MUST upload the captured console
    even when the worker exited 0 — run_mode only uploads on a non-zero exit, so an OOM/segfault or
    silent early-exit otherwise leaves the failure undebuggable (no metrics, often no error_<phase>,
    and the message points at a console that was never uploaded)."""
    import contextlib
    import os
    import subprocess
    import types

    import huggingface_hub

    from flash.providers.runpod.serverless import endpoints

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
            self.stdout = iter(console_lines)
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
        if terminated:
            assert not uploaded_console.startswith("worker booting\n")
            # the 64k byte boundary fell inside the giant x-line, so that truncated line is dropped
            # whole before redaction: a partial line could hold a credential suffix that no longer
            # value-matches.
            assert "x" not in uploaded_console
            assert uploaded_console == "torch.cuda.OutOfMemoryError: CUDA OOM\n"
        else:
            # a tail that is ONE unterminated line is dropped whole, so this uploads nothing. that
            # costs the root cause on exactly the crash that emits one huge line, and keeping it was
            # tried and reverted: every bound that would let the line through is measured against
            # the credentials this process KNOWS, and a value minted at runtime contributes no
            # needle -- so a margin sized from an unrelated secret leaves a long fragment of it
            # behind. the empty console never leaked.
            assert uploaded_console == ""
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

    from flash.providers.runpod.serverless import endpoints

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
    from flash.providers._lifecycle import bootstrap as instance_bootstrap
    from flash.providers.runpod.serverless import endpoints

    assert endpoints._CONSOLE_UPLOAD_INTERVAL_S == 3600.0
    assert instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S == 3600.0
    steady_state_commits_per_hour = (
        3600.0 / worker._HB_MIN_INTERVAL_S + 3600.0 / endpoints._CONSOLE_UPLOAD_INTERVAL_S
    )
    assert steady_state_commits_per_hour <= 5.0


def test_first_console_snapshot_precedes_stall_teardown():
    import importlib
    import inspect

    from flash.providers._lifecycle import bootstrap_console
    from flash.providers.runpod.serverless import endpoints

    importlib.import_module("flash.providers.runpod.jobs")
    poll_job = importlib.import_module("flash.providers.runpod.job_execution").poll_job
    defaults = inspect.signature(poll_job).parameters
    training_stall_s = defaults["stall_after_s"].default
    setup_grace_s = defaults["setup_grace_s"].default

    for first_snapshot_s in (
        endpoints._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S,
        bootstrap_console._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S,
    ):
        assert first_snapshot_s < training_stall_s
        assert first_snapshot_s < setup_grace_s
    for poll_s in (
        endpoints._CONSOLE_UPLOAD_POLL_S,
        bootstrap_console._CONSOLE_UPLOAD_POLL_S,
    ):
        assert 2 * poll_s < training_stall_s


def _source_shipped_namespace(*names: str, namespace: dict | None = None) -> dict:
    import ast
    import inspect
    import textwrap

    from flash.providers.runpod.serverless import endpoints

    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoints._train_body)))
    nodes = [
        next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        for name in names
    ]
    nodes.sort(key=lambda node: node.lineno)
    result = dict(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<handler>", "exec"), result)
    return result


def _source_shipped_console_progress(*, open_fn=None):
    import json
    import os

    namespace = {"json": json, "os": os}
    if open_fn is not None:
        namespace["open"] = open_fn
    return _source_shipped_namespace("_console_progress", namespace=namespace)["_console_progress"]


@pytest.fixture(params=("instance", "source-shipped"))
def console_progress_impl(request):
    if request.param == "instance":
        from flash.providers._lifecycle import bootstrap_console

        return request.param, bootstrap_console.console_progress, bootstrap_console
    progress = _source_shipped_console_progress()
    return request.param, progress, progress.__globals__


def _heartbeat_line(**fields) -> bytes:
    import json

    payload = {"stage": "train", "step": 1, **fields}
    return f"HEARTBEAT {json.dumps(payload)}\n".encode()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(_heartbeat_line(), (1, 1), id="committed"),
        pytest.param(_heartbeat_line(pending=True), (0, 1), id="pending"),
        pytest.param(_heartbeat_line(throttled=True), (0, 1), id="throttled"),
        pytest.param(_heartbeat_line(liveness=True), (0, 0), id="liveness"),
        pytest.param(
            _heartbeat_line(
                sampled_completions=[
                    {
                        "liveness": 'model text: "liveness":',
                        "pending": 'model text: "pending":',
                        "throttled": 'model text: "throttled":',
                    }
                ]
            ),
            (1, 1),
            id="nested-marker-text-is-inert",
        ),
        pytest.param(
            b"HEARTBEAT {not json}\n" + _heartbeat_line(step=2),
            (1, 1),
            id="malformed-line-isolation",
        ),
        pytest.param(
            b'(raylet) {"stage":"train","pending":false}\n',
            (0, 0),
            id="non-heartbeat-json-is-inert",
        ),
    ],
)
def test_console_progress_parser_matrix(console_progress_impl, tmp_path, content, expected):
    name, progress, _owner = console_progress_impl
    console = tmp_path / f"console_{name}.txt"
    console.write_bytes(content)
    state = {"offset": 0, "partial": b"", "dropping": False}

    assert progress(str(console), state) == (len(content), *expected), name
    assert state["offset"] == len(content), name


def _drive_console_cadence(monkeypatch, provider: str, progress, outcomes=True):
    from flash.providers._lifecycle import bootstrap as instance_bootstrap
    from flash.providers._lifecycle import bootstrap_console

    polls = {"count": 0}
    waits: list[float] = []
    attempts: list[int] = []

    class _Stop:
        def wait(self, seconds: float) -> bool:
            waits.append(seconds)
            if polls["count"] >= len(progress):
                return True
            polls["count"] += 1
            return False

    def _progress(_console, _state):
        return progress[polls["count"] - 1]

    def _outcome() -> bool:
        if callable(outcomes):
            return outcomes(len(attempts))
        if isinstance(outcomes, list):
            return outcomes[min(len(attempts) - 1, len(outcomes) - 1)]
        return outcomes

    if provider == "instance":

        def _upload(_payload, _console, _mode):
            attempts.append(polls["count"])
            return _outcome()

        # the loop resolves the parser in bootstrap_console's own namespace, so patching it on
        # bootstrap would silently miss and the test would pass without exercising the schedule.
        monkeypatch.setattr(bootstrap_console, "console_progress", _progress)
        monkeypatch.setattr(instance_bootstrap, "_upload_console_snapshot", _upload)
        instance_bootstrap._console_upload_loop(
            {}, "/tmp/console.txt", "train", instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S, _Stop()
        )
    else:

        def _upload(_mode):
            attempts.append(polls["count"])
            return _outcome()

        loop = _source_shipped_namespace(
            "_upload_loop",
            namespace={
                "console": "/tmp/console.txt",
                "mode": "train",
                "stop_upload": _Stop(),
                "_console_progress": _progress,
                "_upload_console": _upload,
            },
        )["_upload_loop"]
        loop()
    return waits, attempts


@pytest.mark.parametrize("provider", ["instance", "source-shipped"])
@pytest.mark.parametrize(
    ("progress", "outcomes", "expected"),
    [
        pytest.param(
            [(1000 * (index + 1), 0, 0) for index in range(6)],
            True,
            [5],
            id="600s-no-commit-fallback",
        ),
        pytest.param(
            [(1000 * (index + 1), 1, 1) for index in range(29)],
            True,
            [],
            id="healthy-first-hour-write-budget",
        ),
        pytest.param(
            [(1000 * (index + 1), 1, 1) for index in range(31)],
            True,
            [30],
            id="committed-promotion-hourly",
        ),
        pytest.param(
            [(1000 * (index + 1), int(index < 2), int(index < 2)) for index in range(7)],
            True,
            [6],
            id="pre-teardown-quiet-snapshot",
        ),
        pytest.param(
            [(1000 * (index + 1), int(index == 4), int(index == 4)) for index in range(9)],
            True,
            [5, 9],
            id="startup-does-not-spend-credit",
        ),
        pytest.param(
            [
                (1000 * (index + 1), int(index in {0, 5}), int(index in {0, 5}))
                for index in range(10)
            ],
            True,
            [5, 10],
            id="progress-rearms-wedge-credit",
        ),
        pytest.param(
            [
                (1000 * (index + 1), int(index in {0, 5, 10}), int(index in {0, 5, 10}))
                for index in range(15)
            ],
            True,
            [5, 10],
            id="two-credit-flapping-cap",
        ),
        pytest.param(
            [
                (
                    1000 * (index + 1),
                    int(index < 26 or index in {30, 35}),
                    int(index < 26 or index in {30, 35}),
                )
                for index in range(40)
            ],
            True,
            [30, 35, 40],
            id="due-snapshot-does-not-spend-credit",
        ),
        pytest.param(
            [
                (1000 * (index + 1), int(index == 5), int(index in {0, 5, 6, 7, 8, 9}))
                for index in range(10)
            ],
            True,
            [5, 10],
            id="pending-progress-only-before-first-commit",
        ),
        pytest.param(
            [(1000, 0, 0) for _ in range(6)],
            [False, True],
            [5, 6],
            id="failed-upload-retries-next-poll",
        ),
    ],
)
def test_console_upload_cadence_matrix(monkeypatch, provider, progress, outcomes, expected):
    waits, attempts = _drive_console_cadence(monkeypatch, provider, progress, outcomes)
    assert attempts == expected
    assert set(waits) == {120.0}


def test_console_upload_loop_never_waits_longer_than_interval():
    from flash.providers._lifecycle import bootstrap

    waits: list[float] = []

    class _Stop:
        def wait(self, seconds: float) -> bool:
            waits.append(seconds)
            return True

    bootstrap._console_upload_loop({}, "/tmp/console.txt", "train", 30.0, _Stop())
    assert waits == [30.0]


def test_console_progress_split_boundaries(console_progress_impl, tmp_path):
    name, progress, _owner = console_progress_impl
    state = {"offset": 0, "partial": b"", "dropping": False}
    console = tmp_path / f"split_{name}.txt"
    valid = _heartbeat_line(step=7)
    console.write_bytes(valid[:20])

    first = progress(str(console), state)
    assert first == (20, 0, 0), name
    assert state["offset"] == 20
    assert state["partial"] == valid[:20]

    with open(console, "ab") as handle:
        handle.write(valid[20:] + _heartbeat_line(step=8)[:17])
    second = progress(str(console), state)
    assert second[1:] == (1, 1), name
    assert second[0] == console.stat().st_size
    assert state["offset"] == console.stat().st_size
    assert progress(str(console), state)[1:] == (0, 0), f"{name} recounted caught-up bytes"

    liveness = _heartbeat_line(step=9, liveness=True)
    live_console = tmp_path / f"liveness_{name}.txt"
    live_state = {"offset": 0, "partial": b"", "dropping": False}
    cut = liveness.index(b'"liveness":') - 5
    live_console.write_bytes(liveness[:cut])
    live_first = progress(str(live_console), live_state)
    with open(live_console, "ab") as handle:
        handle.write(liveness[cut:])
    live_second = progress(str(live_console), live_state)
    assert live_first[1] + live_second[1] == 0, name
    assert live_first[2] + live_second[2] == 0, name


def test_console_progress_bounds_multi_chunk_reads(console_progress_impl, tmp_path, monkeypatch):
    from flash.providers._lifecycle import bootstrap_console

    name, progress, owner = console_progress_impl
    read_limit = bootstrap_console._CONSOLE_PROGRESS_READ_LIMIT
    console = tmp_path / f"large_{name}.txt"
    console.write_bytes(b"x" * (2 * read_limit + 123))
    state = {"offset": 0, "partial": b"", "dropping": False}
    real_open = open
    read_sizes: list[int] = []

    class _TrackedFile:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, key):
            return getattr(self.handle, key)

        def read(self, size=-1):
            read_sizes.append(size)
            return self.handle.read(size)

    def _open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        return _TrackedFile(handle) if str(file) == str(console) and mode == "rb" else handle

    if isinstance(owner, dict):
        monkeypatch.setitem(owner, "open", _open)
    else:
        monkeypatch.setattr(owner, "open", _open, raising=False)

    result = progress(str(console), state)
    assert result == (console.stat().st_size, 0, 0), name
    assert read_sizes == [read_limit, read_limit, 123], name
    assert state == {"offset": console.stat().st_size, "partial": b"", "dropping": True}
    assert progress(str(console), state)[1:] == (0, 0)
    assert read_sizes == [read_limit, read_limit, 123], "caught-up bytes were reread"


def test_console_progress_recovers_after_overlong_line(console_progress_impl, tmp_path):
    from flash.providers._lifecycle import bootstrap_console

    name, progress, _owner = console_progress_impl
    console = tmp_path / f"overlong_{name}.txt"
    console.write_bytes(b"x" * (bootstrap_console._CONSOLE_PROGRESS_LINE_LIMIT + 1))
    state = {"offset": 0, "partial": b"", "dropping": False}

    assert progress(str(console), state)[1:] == (0, 0)
    assert state["dropping"] is True
    assert state["partial"] == b""
    with open(console, "ab") as handle:
        handle.write(b"\n" + _heartbeat_line(step=3))
    assert progress(str(console), state)[1:] == (1, 1), name
    assert state["dropping"] is False


def test_console_progress_source_parity_across_writes(tmp_path):
    from flash.providers._lifecycle import bootstrap_console

    source_progress = _source_shipped_console_progress()
    console = tmp_path / "console_parity.txt"
    valid = _heartbeat_line(step=1)
    pending = _heartbeat_line(step=2, pending=True)
    writes = [
        b"HEARTBEAT {not json}\n" + valid[:17],
        valid[17:] + pending,
        b"x" * (bootstrap_console._CONSOLE_PROGRESS_READ_LIMIT + 1),
        b"\n" + valid,
    ]
    instance_state = {"offset": 0, "partial": b"", "dropping": False}
    source_state = {"offset": 0, "partial": b"", "dropping": False}
    console.write_bytes(b"")

    for chunk in writes:
        with open(console, "ab") as handle:
            handle.write(chunk)
        assert bootstrap_console.console_progress(str(console), instance_state) == source_progress(
            str(console), source_state
        )
        assert instance_state == source_state


def test_console_progress_keeps_compacted_managed_heartbeat(tmp_path):
    from flash.engine.worker.io.heartbeat import _console_heartbeat_snapshot
    from flash.providers._lifecycle import bootstrap_console

    snapshot = _console_heartbeat_snapshot(
        {
            "stage": "rl_step",
            "step": 3,
            "sampled_completions": [{"completion": "x" * 100_000}],
        }
    )
    line = f"HEARTBEAT {snapshot}\n".encode()
    assert len(line) <= bootstrap_console._CONSOLE_PROGRESS_LINE_LIMIT
    assert '"sample_count": 1' in snapshot
    assert "sampled_completions" not in snapshot
    console = tmp_path / "console_large_managed_heartbeat.txt"
    console.write_bytes(line)
    state = {"offset": 0, "partial": b"", "dropping": False}
    assert bootstrap_console.console_progress(str(console), state)[1:] == (1, 1)


def test_console_upload_uses_distinct_sanitized_live_and_terminal_artifacts(monkeypatch):
    import contextlib
    import json
    import os
    import re

    import huggingface_hub

    mode = f"artifact_{os.getpid()}_{time.time_ns()}"
    console = f"/tmp/console_{mode}.txt"
    calls: list[dict] = []
    deadline_checks: list[None] = []

    class _Api:
        def __init__(self, token=None):
            assert token == "hf-token"

        def upload_file(self, **kwargs):
            with open(kwargs["path_or_fileobj"], encoding="utf-8") as handle:
                kwargs["payload"] = handle.read()
            calls.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    env = {
        "HF_TOKEN": "hf-token",
        "DEPLOY_KEY": "secret-value-123",
        "FLASH_SECRET_ENV_KEYS": "DEPLOY_KEY",
    }
    namespace = _source_shipped_namespace(
        "_needles",
        "_safe_detail",
        "_upload_console",
        namespace={
            "json": json,
            "os": os,
            "re": re,
            "env": env,
            "input_data": {
                "hf_repo": "org/runs",
                "job_spec_json": json.dumps({"algorithm": "grpo", "run_id": "run-1"}),
            },
            "_require_deadline_allowance": lambda: deadline_checks.append(None) or 100.0,
        },
    )
    upload = namespace["_upload_console"]
    live_tail = console + ".live.tail"
    terminal_tail = console + ".terminal.tail"

    try:
        with open(console, "w", encoding="utf-8") as handle:
            handle.write("prefix\n" + "x" * 65_000 + "\nsecret-value-123 live root cause\n")
        assert upload(mode) is True
        with open(console, "a", encoding="utf-8") as handle:
            handle.write("secret-value-123 terminal root cause\n")
        assert upload(mode, final=True) is True

        assert [call["path_or_fileobj"] for call in calls] == [live_tail, terminal_tail]
        assert [call["path_in_repo"] for call in calls] == [
            f"rl/run-1/console_{mode}_live.txt",
            f"rl/run-1/console_{mode}.txt",
        ]
        assert all(len(call["payload"].encode()) <= 64_000 for call in calls)
        assert all("secret-value-123" not in call["payload"] for call in calls)
        assert all("<redacted>" in call["payload"] for call in calls)
        assert "terminal root cause" not in calls[0]["payload"]
        assert "terminal root cause" in calls[1]["payload"]
        assert len(deadline_checks) == 4
    finally:
        for target in (console, live_tail, terminal_tail):
            with contextlib.suppress(FileNotFoundError):
                os.remove(target)


def test_min_cuda_for_uses_the_gpu_class_floor():
    # the CUDA floor is a property of the GPU class, not of the image tag
    from flash.providers.runpod.serverless.endpoints import min_cuda_for

    assert min_cuda_for("B200") == "13.0"  # blackwell needs cu13 drivers
    assert min_cuda_for("H200") == "12.8"


def test_apply_disk_raises_to_the_requested_floor():
    from types import SimpleNamespace

    from flash.providers.runpod.jobs import apply_disk_gb

    tpl = SimpleNamespace(containerDiskInGb=64)
    cfg = SimpleNamespace(template=tpl)
    apply_disk_gb(cfg, 80)
    assert tpl.containerDiskInGb == 80  # raise-only: the request wins over the smaller default
    apply_disk_gb(cfg, 32)
    assert tpl.containerDiskInGb == 80  # never lowers an already-larger disk


def test_snapshot_weight_validation(tmp_path):
    from flash.engine.worker.io.hf import _snapshot_has_weights

    d = tmp_path / "snap"
    d.mkdir()
    (d / "config.json").write_text("{}")
    assert not _snapshot_has_weights(str(d))  # configs only = stale partial snapshot
    (d / "model.safetensors-00001-of-00001.safetensors").write_text("x")
    assert _snapshot_has_weights(str(d))
