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


def test_runpod_backoff_no_overflow_on_long_runs():
    """DEFECT: runpod_flash computed base*(2**attempt) then clamped, so a long poll loop
    overflowed (~80 min in) and killed a healthy job. The patch caps the exponent first."""
    pytest.importorskip("runpod_flash")
    from flash.providers.runpod.serverless import _patch_runpod_backoff

    _patch_runpod_backoff()
    from runpod_flash.core.utils import backoff

    # Pre-patch this raised OverflowError; now it must return a clamped, finite delay.
    delay = backoff.get_backoff_delay(5000, max_seconds=5)
    assert delay <= 5 * 1.2 + 1e-9
    # the serverless module's imported reference is patched too (that's the real call site)
    from runpod_flash.core.resources import serverless

    assert serverless.get_backoff_delay(100000, max_seconds=5) <= 5 * 1.2 + 1e-9


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


def test_train_body_imports_every_name_it_uses():
    """Flash ships only _train_body's source to the worker, where module-level
    imports are out of scope, so every stdlib/3p name it references must be
    imported inside the function body (else NameError before training)."""
    import ast
    import builtins
    import inspect

    from flash.providers.runpod import serverless as train

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

    # a nested helper may only close over names bound in an ENCLOSING scope. reaching for one bound
    # in a sibling function is a NameError at runtime, and unit tests that exec the helper against a
    # hand-built namespace supply the name themselves, so they pass while the worker would die.
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

    def _walk_scope(scope):
        """Nodes belonging to ``scope`` itself, NOT to functions nested inside it.

        ast.walk would descend into them, so a name bound only in a SIBLING helper would count as
        bound here -- exactly the mistake this check exists to catch, silently passing.
        """
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            yield node
            if not isinstance(node, scopes):
                stack.extend(ast.iter_child_nodes(node))

    def _bound(scope) -> set[str]:
        names = {a.arg for a in ast.walk(scope.args) if isinstance(a, ast.arg)}
        for node in _walk_scope(scope):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                names.add(node.name)  # `except X as e` binds e for the handler's body
            elif isinstance(node, ast.Global):
                names |= set(node.names)
        return names

    builtin_names = set(dir(builtins))

    def _check(scope, visible: set[str]) -> None:
        inner = visible | _bound(scope)
        for node in _walk_scope(scope):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                assert node.id in inner or node.id in builtin_names, (
                    f"{getattr(scope, 'name', '<lambda>')} reads {node.id!r}, "
                    "bound in no enclosing scope"
                )
            elif isinstance(node, scopes):
                _check(node, inner)

    _check(fn, set())


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
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap
    from flash.providers.runpod.serverless import endpoints

    assert endpoints._CONSOLE_UPLOAD_INTERVAL_S == 3600.0
    assert _instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S == 3600.0
    steady_state_commits_per_hour = (
        3600.0 / worker._HB_MIN_INTERVAL_S + 3600.0 / endpoints._CONSOLE_UPLOAD_INTERVAL_S
    )
    assert steady_state_commits_per_hour <= 5.0


def test_first_console_snapshot_precedes_the_stall_teardown():
    """A wedged run must produce a console artifact before the stall classifier tears it down.

    The hourly interval is longer than BOTH stall limits, so a periodic uploader that waits a full
    interval for its first snapshot can never fire for a run that hangs -- which is exactly the run
    whose console is the only evidence of why it hung. Multi-turn OPD wedges died this way: the
    child hard-exits without a traceback, and the artifact that would have carried the stderr was
    never uploaded because teardown always won the race.
    """
    import importlib
    import inspect

    from flash.providers._lifecycle import bootstrap as _instance_bootstrap
    from flash.providers.runpod.serverless import endpoints

    # poll_job's stall defaults are the deadlines the first snapshot has to beat. `jobs` has to be
    # imported first: it is mid-cycle with job_execution, so importing job_execution alone raises
    # ImportError. done through importlib so import sorting cannot reorder the two.
    importlib.import_module("flash.providers.runpod.jobs")
    poll_job = importlib.import_module("flash.providers.runpod.job_execution").poll_job

    stall_defaults = inspect.signature(poll_job).parameters
    training_stall_s = stall_defaults["stall_after_s"].default
    setup_grace_s = stall_defaults["setup_grace_s"].default

    for first_snapshot_s in (
        endpoints._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S,
        _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S,
    ):
        assert first_snapshot_s < training_stall_s
        assert first_snapshot_s < setup_grace_s
        # the defect this guards: the steady interval alone outlives both deadlines.
        assert setup_grace_s < endpoints._CONSOLE_UPLOAD_INTERVAL_S

    # the poll is what actually beats the deadline: it is how soon after output stops the uploader
    # can notice and commit. two polls is the worst case.
    for poll_s in (
        endpoints._CONSOLE_UPLOAD_POLL_S,
        _instance_bootstrap._CONSOLE_UPLOAD_POLL_S,
    ):
        assert 2 * poll_s < training_stall_s

    # _train_body ships as SOURCE to the worker, so it inlines these numbers rather than
    # referencing the constants (test_train_body_imports_every_name_it_uses enforces that). pin the
    # literals here so the shipped uploader cannot drift away from the deadlines above.
    body = inspect.getsource(endpoints._train_body)
    assert f"stop_upload.wait({endpoints._CONSOLE_UPLOAD_POLL_S})" in body
    assert f"due_s, since, quiet_polls = {endpoints._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S}" in body
    assert f"quiet_used or (wedged and ok), 0.0, {endpoints._CONSOLE_UPLOAD_INTERVAL_S}" in body
    # the latch is spent only on silence that bought an upload, not on an already-due snapshot.
    assert "and not quiet_used and not due" in body
    # the sustained-silence threshold and the success-gated watermark, which keep a sparsely
    # logging run inside the shared repo's commit budget and retry a swallowed upload failure.
    assert (
        f"quiet_polls >= {_instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS} and not quiet_used"
        in body
    )
    # the latch arms only after real progress: a cold image is quiet through startup and would
    # otherwise spend it on an empty console, leaving a later hang with no snapshot before teardown
    # (test_..._keeps_its_wedge_credit_through_a_slow_startup covers the behavior on the twin loop).
    assert "progressed = progressed or bool(staged)" in body
    assert "wedged = progressed and quiet_polls" in body
    assert "uploaded_size = size if ok else uploaded_size" in body
    # the wedge signal is STAGED HEARTBEATS, not console bytes: a stuck worker still logs, so a
    # size-based rule never fires for it (test_..._detects_a_wedge_that_keeps_logging).
    assert "quiet_polls = 0 if staged else quiet_polls + 1" in body
    # and liveness pings are SUBTRACTED, not counted: they carry "stage" like every other payload
    # and print every 30s from a daemon, so counting them reads a wedge as progress forever.
    assert "count(b'\"stage\":') - buf.count(b'\"liveness\":')" in body
    assert "max(0, buf.count" in body  # floored: a negative count is truthy and reads as progress


def _drive_instance_upload_loop(
    monkeypatch, sizes: list[int], cycles: int, *, succeed=True, staged=None
) -> tuple[list, list]:
    """Run the real instance loop over a scripted console-size series. Returns (waits, uploads).

    ``succeed`` is the upload result, either a bool or a predicate over the attempt number, so a
    test can script an upload that fails the way hf_upload does: swallowed, returning falsy.

    ``staged`` scripts the STAGED-heartbeat count seen at each poll, which is the progress signal
    the loop keys on. It defaults to "a heartbeat whenever the console grew", the shape of a
    healthy run, so a frozen console still reads as wedged. A test covering a worker that keeps
    logging without making progress passes its own series.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    waits: list[float] = []
    uploads: list[int] = []
    clock = {"i": 0}

    class _Stop:
        def wait(self, seconds: float) -> bool:
            waits.append(seconds)
            return len(waits) > cycles

    def _progress(_console: str, _offset: int) -> tuple[int, int]:
        index = min(clock["i"], len(sizes) - 1)
        clock["i"] += 1
        if staged is not None:
            beats = staged[min(index, len(staged) - 1)]
        else:  # grew since the previous poll -> one staged heartbeat
            beats = 1 if index == 0 or sizes[index] != sizes[index - 1] else 0
        return sizes[index], beats

    def _upload(_payload, _console, _mode) -> bool:
        uploads.append(clock["i"])
        return succeed(len(uploads)) if callable(succeed) else succeed

    monkeypatch.setattr(_instance_bootstrap, "_console_progress", _progress)
    monkeypatch.setattr(_instance_bootstrap, "_upload_console_snapshot", _upload)
    _instance_bootstrap._console_upload_loop({}, "/tmp/console.txt", "train", 3600.0, _Stop())
    return waits, uploads


def test_instance_console_upload_loop_polls_faster_than_it_commits(monkeypatch):
    """A healthy, growing run polls often but still commits only on the hourly boundary.

    The poll cadence must not become the COMMIT cadence: the shared artifact repo budgets 5
    commits/hour and the heartbeat already spends 4 (see
    test_live_console_uploads_are_throttled_for_shared_artifact_repos).
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    interval_s = _instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S
    # run long enough to pass the first commit AND the following hourly one.
    cycles = int((first_s + interval_s) / poll_s) + 1
    # console keeps growing every poll: never quiet, so only the elapsed-interval rule can fire.
    growing = [1000 * (n + 1) for n in range(cycles + 2)]
    waits, uploads = _drive_instance_upload_loop(monkeypatch, growing, cycles=cycles)

    assert waits[0] == poll_s
    assert set(waits) == {poll_s}
    # exactly two commits: the first snapshot and one hourly -- NOT one per poll.
    assert len(uploads) == 2
    assert uploads[0] * poll_s == first_s
    assert (uploads[1] - uploads[0]) * poll_s == interval_s


def test_instance_console_upload_loop_commits_when_a_wedged_run_goes_quiet(monkeypatch):
    """The wedge case: output stops, and the snapshot must land before the stall teardown.

    A run that hangs at 700s is torn down around 1900s while the next hourly snapshot would not be
    due until 4200s, so an interval-only loop uploads a console that PREDATES the hang -- losing
    the last lines, which are the whole diagnostic. Going quiet is what triggers the commit.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    # the run must wedge AFTER its first snapshot -- the case the interval rule cannot cover on its
    # own. a series that wedges BEFORE the first commit proves nothing: the elapsed-interval rule
    # fires there anyway, so the test would still pass with the quiet trigger deleted.
    quiet_polls = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    grow_polls = int(first_s / poll_s) + 1
    cycles = grow_polls + quiet_polls + 6
    wedged = [1000 * (n + 1) for n in range(grow_polls)] + [1000 * grow_polls] * (quiet_polls + 8)
    _waits, uploads = _drive_instance_upload_loop(monkeypatch, wedged, cycles=cycles)

    # two commits: the first snapshot, then one more once output stopped.
    assert len(uploads) == 2, "a run that wedges after its first snapshot must commit again"
    # measured from when OUTPUT STOPPED, which is when the stall classifier starts counting -- not
    # from run start. the deadline is 1200s of no progress, so the lag is what has to fit inside it.
    silence_began_s = grow_polls * poll_s
    assert uploads[1] * poll_s - silence_began_s < 1200.0
    assert uploads[1] * poll_s < _instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S
    # and it must not keep re-uploading identical bytes once the run is silent.
    assert uploads[1] < cycles


def test_instance_console_upload_loop_keeps_a_sparsely_logging_run_in_budget(monkeypatch):
    """A healthy run that logs slower than the poll must not be mistaken for a wedge.

    Momentary quiet is normal: a worker in a teacher call or a compile emits nothing for minutes at
    a time. Treating ONE unchanged sample as the wedge signature commits on nearly every poll, so
    the loop that exists to respect the shared repo's 5 commits/hour spends 10 by itself. Silence
    has to be sustained, and the resulting snapshot spent once, or the rate is unbounded.

    The budget governs the SUSTAINED rate, so that is what is asserted here: the scheduled cadence
    over the run, plus at most one wedge commit for the whole run. Dividing every commit by a short
    window instead would charge that one-off snapshot as if it recurred -- 1.5/hr over 2h for a
    loop that converges to 1.0/hr -- and would fail a correct implementation for being measured
    over too short a horizon.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    interval_s = _instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S
    quiet_polls = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    hours = 2
    cycles = int(hours * 3600.0 / poll_s)
    # alternating phases of sustained silence and output -- each silent phase is long enough to
    # look wedged, so without the one-shot latch every cycle buys another commit.
    phase = quiet_polls + 2
    sizes = [1000 * (n // phase) for n in range(cycles + 2)]
    _waits, uploads = _drive_instance_upload_loop(monkeypatch, sizes, cycles=cycles)

    import flash.engine.worker as worker

    # the scheduled cadence is one per interval after the first snapshot; anything beyond that is
    # the one-shot wedge credit, which the latch caps at one for the entire run.
    scheduled = int(hours * 3600.0 / interval_s) + 1
    assert len(uploads) <= scheduled + 1, f"{len(uploads)} commits exceeds the schedule plus one"

    sustained = 3600.0 / interval_s
    total = 3600.0 / worker._HB_MIN_INTERVAL_S + sustained
    assert total <= 5.0, f"{sustained}/hr console + heartbeat = {total}/hr, budget is 5"


def test_instance_console_upload_loop_saves_the_quiet_snapshot_for_a_real_wedge(monkeypatch):
    """A brief healthy pause must not consume the one quiet snapshot the wedge case needs.

    The quiet commit is taken once per run. If a single unchanged sample triggered it, an ordinary
    pause between log lines would spend it early, and the hang that follows -- the failure this
    whole path exists to capture -- would get no snapshot until the next hourly boundary, long
    after the stall classifier tore the run down.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    quiet_polls = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    grow_polls = int(first_s / poll_s) + 1
    pause_polls = quiet_polls - 1  # a pause too short to be a wedge
    wedge_at = grow_polls + pause_polls + 2
    sizes = [1000 * (n + 1) for n in range(grow_polls)]
    sizes += [1000 * grow_polls] * pause_polls  # brief pause, then output resumes
    sizes += [1000 * (grow_polls + n + 1) for n in range(2)]
    sizes += [1000 * (grow_polls + 2)] * (quiet_polls + 4)  # the real wedge
    cycles = len(sizes)
    _waits, uploads = _drive_instance_upload_loop(monkeypatch, sizes, cycles=cycles)

    after_wedge = [u for u in uploads if u > wedge_at]
    assert after_wedge, "the brief pause consumed the snapshot the real wedge needed"
    # and it lands inside the 1200s training stall, measured from when output actually stopped.
    assert (after_wedge[0] - wedge_at) * poll_s < 1200.0


def test_instance_console_upload_loop_does_not_spend_the_latch_on_a_due_snapshot(monkeypatch):
    """A run quiet across its FIRST scheduled snapshot must keep its wedge credit.

    At that poll the scheduled deadline and the sustained-silence rule are both true, so the upload
    happens either way. Charging the one-shot latch there buys nothing and disarms wedge detection
    for the rest of the run: if output resumes and the worker then truly hangs, the next snapshot is
    a full interval away -- 4200s here, against a 1200s stall teardown -- so the hang is never
    captured. The latch may only be spent when silence bought an upload that was not already due.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    quiet_polls = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    # silent from the start, so the first scheduled snapshot and the wedge rule coincide.
    silent_polls = int(first_s / poll_s)
    assert silent_polls > quiet_polls, "the deadline must land after silence is already sustained"
    sizes = [1000] * silent_polls
    sizes += [1000 * (n + 2) for n in range(3)]  # output resumes
    wedge_at = len(sizes)
    sizes += [1000 * 4] * (quiet_polls + 6)  # then the run wedges for good
    _waits, uploads = _drive_instance_upload_loop(monkeypatch, sizes, cycles=len(sizes))

    after_wedge = [u for u in uploads if u > wedge_at]
    assert after_wedge, "the overlapping scheduled snapshot consumed the real wedge's credit"
    assert (after_wedge[0] - wedge_at) * poll_s < 1200.0


def test_instance_console_upload_loop_keeps_its_wedge_credit_through_a_slow_startup(monkeypatch):
    """A cold image that imports for minutes before its first heartbeat must not spend the latch.

    Startup is silent by nature -- pulling and importing the worker stack outruns QUIET_POLLS (480s)
    before the first scheduled snapshot (600s) is even due -- so quiet accounting that begins at
    process start reads it as a wedge. The commit that buys is an empty console: nothing has run
    yet. Worse, spending the latch there also resets the scheduled deadline to a full interval, so
    the next commit is ~4080s out. A real hang after startup then finds the latch gone AND the
    schedule pushed away, and the 1200s stall teardown destroys the box with no console at all --
    strictly worse than never having had wedge detection.

    So the latch arms only once a staged heartbeat has actually been seen. A startup that never
    reaches one is the stall classifier's 3000s setup grace, not this loop's case.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    quiet_polls = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    # the startup must reach the wedge threshold STRICTLY BEFORE the first snapshot is due, or the
    # `not due` guard already covers it and a passing test would prove nothing about this fix.
    startup_polls = quiet_polls
    assert startup_polls * poll_s < first_s, (
        "startup must go quiet before the first snapshot is due"
    )

    healthy_polls = 10
    wedge_at = startup_polls + healthy_polls
    cycles = wedge_at + quiet_polls + 8
    # the console GROWS throughout: banners during startup, ray chatter during the hang. Only the
    # staged-heartbeat series distinguishes the three phases.
    sizes = [1000 * (n + 1) for n in range(cycles + 2)]
    staged = [0] * startup_polls + [1] * healthy_polls
    staged += [0] * (cycles + 2 - len(staged))
    _waits, uploads = _drive_instance_upload_loop(monkeypatch, sizes, cycles=cycles, staged=staged)

    after_wedge = [u for u in uploads if u > wedge_at]
    assert after_wedge, "the silent startup consumed the latch the post-startup wedge needed"
    assert (after_wedge[0] - wedge_at) * poll_s < 1200.0
    # and the startup must not have moved the scheduled snapshot either: pushing it out to a full
    # interval is half the damage, and it survives even if the latch itself is later restored.
    assert uploads[0] * poll_s == first_s, "a quiet startup delayed the first scheduled snapshot"


def test_instance_console_upload_loop_detects_a_wedge_that_keeps_logging(monkeypatch):
    """A worker stuck after its last staged heartbeat still prints -- and must still be caught.

    Ray warnings, liveness pings and library chatter keep the console GROWING while no real work
    happens, so a wedge rule keyed on console bytes never fires: every poll sees a new size. The
    stall classifier does not care about bytes either -- it advances only on a staged heartbeat and
    tears the run down on that clock. Keying on the same signal is what keeps the two agreeing.

    Measured with a byte-based rule: snapshots at 600s and then 4200s, with teardown near 1800s --
    no console for the hang at all.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    quiet_polls = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    healthy_polls = int(first_s / poll_s) + 1
    cycles = healthy_polls + quiet_polls + 6
    # the console never stops growing, so a size-based rule sees progress at every single poll.
    sizes = [1000 * (n + 1) for n in range(cycles + 2)]
    # but staged heartbeats stop after the healthy prefix: that is the wedge.
    staged = [1] * healthy_polls + [0] * (cycles + 2 - healthy_polls)
    _waits, uploads = _drive_instance_upload_loop(monkeypatch, sizes, cycles=cycles, staged=staged)

    after_wedge = [u for u in uploads if u > healthy_polls]
    assert after_wedge, "a wedged run that keeps logging produced no snapshot"
    # and it lands inside the 1200s training stall, measured from when progress actually stopped.
    assert (after_wedge[0] - healthy_polls) * poll_s < 1200.0


def test_console_progress_counts_staged_heartbeats_incrementally(tmp_path):
    """The wedge signal itself, unmocked: every loop test above patches _console_progress out.

    Three properties, all load-bearing. It must count only heartbeats that REPRESENT progress: every
    payload carries ``"stage"``, liveness pings included (``heartbeat.py`` adds ``"liveness": True``
    to the same flat object), and those print every 30s from a daemon thread -- so a worker wedged
    inside a liveness block would look busy forever. ``poll`` refuses to advance its stall key on
    them for exactly this reason, and disagreeing means the run is torn down with no console. It
    must not report a negative count either, since a bare negative is truthy and reads as progress.
    And it must count only bytes past ``offset``: a console reaching hundreds of MB is rescanned
    every poll otherwise, and a heartbeat from the healthy prefix would be recounted forever.
    """
    import json

    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    def _hb(**kw) -> bytes:
        # the real console line: heartbeat.py prints "HEARTBEAT " + json.dumps(payload).
        payload = {"stage": "train", "ts": 1.0, "run_id": "r", "mode": "train", **kw}
        return f"HEARTBEAT {json.dumps(payload)}\n".encode()

    console = tmp_path / "console_train.txt"
    progress = _hb(step=7)
    pings = _hb(liveness=True) + b"ray warning: worker is idle\n"
    console.write_bytes(progress + pings)

    size, beats = _instance_bootstrap._console_progress(str(console), 0)
    assert (size, beats) == (len(progress) + len(pings), 1)

    # the wedge: heartbeats keep arriving, but every one is a liveness ping.
    with open(console, "ab") as f:
        f.write(pings * 3)
    size2, beats2 = _instance_bootstrap._console_progress(str(console), size)
    assert beats2 == 0, "liveness pings after the last real heartbeat must not read as progress"
    assert size2 == size + len(pings) * 3

    # a poll boundary can split a line and leave the "liveness" half alone in the chunk, so the
    # subtraction goes to -1. It must floor at zero: a bare negative is TRUTHY in python, so it
    # would reset quiet_polls and read the wedge as progress -- the exact bug being fixed, arriving
    # through the fix. The offset is computed from the line, not guessed: a guessed one that
    # happens to include both keys makes this assertion vacuous.
    solo = tmp_path / "console_split.txt"
    ping = _hb(liveness=True)
    solo.write_bytes(ping)
    cut = ping.index(b'"liveness":') - 5  # past "stage", before "liveness"
    assert b'"stage":' not in ping[cut:]
    assert b'"liveness":' in ping[cut:]
    assert _instance_bootstrap._console_progress(str(solo), cut) == (len(ping), 0)

    assert _instance_bootstrap._console_progress(str(tmp_path / "absent.txt"), 0) == (-1, 0)


def test_instance_console_upload_loop_retries_when_an_upload_fails(monkeypatch):
    """hf_upload swallows its exception, so a failed snapshot returns normally.

    Advancing the uploaded-bytes watermark on that return marks bytes as stored that reached no
    repo. For a wedged run producing nothing further, every later poll then sees size == sent and
    skips -- so one transient 500 costs the entire console, which is the only evidence of the hang.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    # the run must produce NO new bytes after the failed upload -- that is the suppressed case. a
    # console still growing supplies a different size on the next poll, so the retry happens by
    # accident and the test would pass with the watermark bug restored.
    grow_polls = int(first_s / poll_s) + 1
    sizes = [1000] * (grow_polls + 14)
    cycles = len(sizes)
    _waits, uploads = _drive_instance_upload_loop(
        monkeypatch, sizes, cycles=cycles, succeed=lambda attempt: attempt > 1
    )

    assert len(uploads) > 1, "a swallowed upload failure must not suppress every later attempt"
    # the retry must land inside the stall window, not at the next hourly boundary.
    assert (uploads[1] - uploads[0]) * poll_s < 1200.0


def test_serverless_console_upload_serializes_the_periodic_and_final_snapshots():
    """The final snapshot must be the last writer, even if a periodic one is still uploading.

    The periodic uploader is a daemon thread joined with a 10s timeout, so a slow snapshot outlives
    the join and races the final upload. Both write the same ``.tail`` file and commit to the same
    repo path; if the older call landed last it would REPLACE the terminal console with bytes
    captured before the failure -- destroying the evidence. Moving the first snapshot to 600s made
    this reachable for an ordinary run, where the hourly-only cadence had made it rare.

    Both callers pass the SAME ``mode``, as the loop and the terminal call sites do. Driving them
    with different modes would be the easy mistake: ``console_{mode}.txt`` already differs, so the
    scratch paths could never collide and the test would pass however the file is named.
    """
    import ast
    import inspect
    import textwrap
    import threading
    import types

    from flash.providers.runpod.serverless import endpoints

    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoints._train_body)))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_upload_console"
    )
    order: list[str] = []
    tails: list[str] = []
    started = threading.Event()

    def _upload_console_locked(_mode: str, _console: str, tail_path: str, _final: bool) -> bool:
        started.set()
        order.append(f"begin:{tail_path}")
        tails.append(tail_path)
        time.sleep(0.3)
        order.append(f"end:{tail_path}")
        return True

    namespace: dict = {
        "os": types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda _p: True)),
        "console_upload_lock": threading.Lock(),
        "console_teardown": threading.Event(),  # not set: teardown has not begun
        "_upload_console_locked": _upload_console_locked,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<handler>", "exec"), namespace)
    upload_console = namespace["_upload_console"]

    periodic = threading.Thread(target=upload_console, args=("train",), daemon=True)
    periodic.start()
    started.wait(5.0)
    upload_console("train", final=True)  # the terminal snapshot, racing the one in flight
    periodic.join(timeout=5.0)

    # never interleaved: each upload completes before the next begins, so the last call wins.
    assert [entry.split(":")[0] for entry in order] == ["begin", "end", "begin", "end"]
    # and each caller staged its own bytes: one shared .tail would let the periodic snapshot
    # overwrite the file the final upload is reading, splicing pre-failure bytes into it.
    assert len(set(tails)) == 2, f"both callers staged the same scratch file: {tails}"


def test_serverless_terminal_console_upload_recommits_after_a_slow_holder_finishes():
    """A periodic upload that outlives the acquire timeout must not be the last writer.

    ``console_teardown`` is checked before the HF call, so it stops a periodic snapshot that has not
    started -- but a request already inside ``upload_file`` cannot be recalled. If it held the lock
    past 120s and completed afterwards, the terminal upload (which proceeds unsynchronized rather
    than skip) commits FIRST and the older, pre-failure snapshot lands on top of it. That is the
    exact evidence-destroying overwrite the serialization is for, arriving through the timeout.

    So a terminal upload that ran without the lock re-acquires afterwards: getting it is proof the
    slow request finished, and ``console_teardown`` blocks any later one, so one more commit ends
    the sequence. The lock double here comes free on the second attempt, modelling a holder that
    was slow rather than wedged (``..._never_yields_to_a_wedged_snapshot`` pins the wedged case,
    where the re-commit must NOT fire).
    """
    import ast
    import inspect
    import textwrap
    import threading
    import types

    from flash.providers.runpod.serverless import endpoints

    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoints._train_body)))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_upload_console"
    )
    order: list[str] = []

    class _SlowHolder:
        """Times out once (the holder is mid-``upload_file``), then frees."""

        def __init__(self) -> None:
            self.attempts = 0

        def acquire(self, timeout: float) -> bool:
            self.attempts += 1
            if self.attempts == 1:
                return False
            order.append("holder finished")
            return True

        def release(self) -> None:
            order.append("released")

    def _locked(mode: str, _console: str, tail_path: str, final: bool) -> bool:
        order.append(f"commit:{'final' if final else 'periodic'}:{tail_path}")
        return True

    lock = _SlowHolder()
    namespace: dict = {
        "os": types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda _p: True)),
        "console_upload_lock": lock,
        "console_teardown": threading.Event(),
        "_upload_console_locked": _locked,
        "print": lambda *_a, **_k: None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<handler>", "exec"), namespace)

    assert namespace["_upload_console"]("train", final=True) is True
    commits = [n for n, entry in enumerate(order) if entry.startswith("commit:")]
    assert len(commits) == 2, f"the terminal upload must re-commit once the holder frees: {order}"
    # positions, not values: both commits render identically, so comparing the strings would find
    # the FIRST one and the assertion would hold however the calls were ordered. The re-commit must
    # follow the holder freeing, or it is no better ordered than the first and the stale snapshot
    # can still land last.
    assert commits[1] > order.index("holder finished")


def test_serverless_terminal_console_upload_never_yields_to_a_wedged_snapshot():
    """``final=True`` must upload even when the lock never comes free.

    ``upload_file`` takes no timeout, so a periodic snapshot wedged on a hung HF request holds the
    lock past the acquire timeout and forever after. Skipping the terminal upload there is the worst
    case, not a safe one: the run is failing, and the bytes explaining WHY are exactly the ones that
    never reach the repo, leaving the pre-failure console as the only record. A periodic snapshot
    still yields -- another is in flight and it has nothing new to say.

    The wait is bounded, not skipped: a HEALTHY in-flight snapshot finishes inside the timeout and
    the terminal one commits after it, which is what
    ``test_serverless_console_upload_serializes_the_periodic_and_final_snapshots`` pins. So the lock
    here is a double that reports the timeout expiring, rather than a real one held for 120s.
    """
    import ast
    import inspect
    import textwrap
    import threading
    import types

    from flash.providers.runpod.serverless import endpoints

    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoints._train_body)))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_upload_console"
    )
    uploaded: list[str] = []
    timeouts: list[float] = []

    class _NeverFree:
        """Acquire always times out; releasing a lock we never took would raise."""

        def acquire(self, timeout: float) -> bool:
            timeouts.append(timeout)
            return False

        def release(self) -> None:
            raise AssertionError("released a lock that was never acquired")

    namespace: dict = {
        "os": types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda _p: True)),
        "console_upload_lock": _NeverFree(),
        "console_teardown": threading.Event(),
        "_upload_console_locked": lambda mode, _c, _t, _f: uploaded.append(mode) or True,
        "print": lambda *_a, **_k: None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<handler>", "exec"), namespace)
    upload_console = namespace["_upload_console"]

    assert upload_console("periodic") is False, "a periodic snapshot must yield to the holder"
    assert upload_console("final", final=True) is True
    # exactly one commit: the re-commit that makes an unsynchronized terminal upload last-writer
    # is conditional on the lock coming free, and here it never does. a second commit would mean
    # the re-commit fires blind, doubling every terminal upload on a genuinely wedged holder.
    assert uploaded == ["final"], "the terminal snapshot must not be skipped for a held lock"
    # bounded wait, so a wedged holder delays the terminal upload rather than suppressing it.
    assert timeouts
    assert all(0 < t <= 120.0 for t in timeouts)


def test_serverless_periodic_console_upload_defers_once_teardown_begins():
    """A periodic snapshot must not commit after the terminal one is under way.

    ``final=True`` deliberately does not wait for the lock, so the two can run concurrently and the
    periodic upload -- captured BEFORE the failure -- could land last and overwrite the terminal
    console with pre-failure bytes, losing the diagnostic. The terminal call sets
    ``console_teardown``, so a periodic snapshot that sees it drops its commit rather than racing.
    """
    import ast
    import inspect
    import textwrap
    import threading
    import types

    from flash.providers.runpod.serverless import endpoints

    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoints._train_body)))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_upload_console"
    )
    uploaded: list[str] = []
    console_teardown = threading.Event()
    console_teardown.set()  # a terminal snapshot has begun

    namespace: dict = {
        "os": types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda _p: True)),
        "console_upload_lock": threading.Lock(),  # free: only the teardown flag may defer
        "console_teardown": console_teardown,
        "_upload_console_locked": lambda mode, _c, _t, _f: uploaded.append(mode) or True,
        "print": lambda *_a, **_k: None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<handler>", "exec"), namespace)
    upload_console = namespace["_upload_console"]

    assert upload_console("train") is False, "a periodic snapshot must defer once teardown began"
    assert uploaded == []
    # the terminal snapshot is exactly the one teardown is running, so it must still commit.
    assert upload_console("train", final=True) is True
    assert uploaded == ["train"]


def test_serverless_periodic_console_upload_drops_a_commit_started_before_teardown(
    tmp_path, monkeypatch
):
    """Teardown can begin while a periodic snapshot is already staging its tail.

    The pre-acquire gate cannot catch that one: it passed before teardown started. Without a
    re-check immediately before the commit, the in-flight periodic upload lands AFTER the terminal
    one and overwrites the failure console with bytes read before the failure -- the exact loss the
    terminal upload exists to prevent. This drives the real ``_upload_console_locked``, since the
    re-check lives there rather than in the caller.
    """
    import ast
    import inspect
    import json
    import os
    import textwrap
    import threading

    from flash.providers.runpod.serverless import endpoints

    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoints._train_body)))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_upload_console_locked"
    )
    console = tmp_path / "console_train.txt"
    console.write_text("line one\nline two\n")
    commits: list[str] = []
    console_teardown = threading.Event()
    console_teardown.set()  # teardown began while this snapshot was staging its tail

    class _Api:
        def __init__(self, token=None):
            pass

        def upload_file(self, *, path_in_repo: str, **_kw) -> None:
            commits.append(path_in_repo)

    # the function imports HfApi from huggingface_hub itself, so a namespace entry is ignored and
    # the real client would make a LIVE network call. patch where it is looked up.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)

    warnings: list[str] = []
    namespace: dict = {
        "os": os,
        "json": json,
        "input_data": {
            "job_spec_json": json.dumps({"algorithm": "opd", "run_id": "r1"}),
            "hf_repo": "org/repo",
        },
        "env": {},
        "console_teardown": console_teardown,
        "_require_deadline_allowance": lambda: None,
        "_safe_detail": lambda text, _env, _limit=0: text,
        "print": lambda *args, **_kw: warnings.append(" ".join(str(a) for a in args)),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<handler>", "exec"), namespace)
    upload_locked = namespace["_upload_console_locked"]

    tail = str(tmp_path / "console_train.txt.tail")
    assert upload_locked("train", str(console), tail, False) is False
    assert commits == [], "a periodic upload must not commit once teardown has begun"
    # the terminal snapshot still commits: it IS the teardown upload.
    assert upload_locked("train", str(console), tail + ".final", True) is True
    assert commits == ["opd/r1/console_train.txt"]
    # a False from a swallowed error would satisfy the first assertion for the wrong reason.
    assert not [w for w in warnings if "warn" in w], warnings


def test_instance_console_upload_loop_never_waits_longer_than_the_interval():
    """A caller passing an interval shorter than the first-snapshot delay must not be lengthened."""
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    waits: list[float] = []

    class _Stop:
        def wait(self, seconds: float) -> bool:
            waits.append(seconds)
            return True  # stop immediately; only the first wait matters here

    _instance_bootstrap._console_upload_loop({}, "/tmp/console.txt", "train", 30.0, _Stop())
    assert waits == [30.0]


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
