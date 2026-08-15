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
    run_code = tmp_path / "runcode"
    real_join = os.path.join

    def mapped_join(*parts):
        joined = real_join(*parts)
        if joined == "/runcode" or joined.startswith("/runcode/"):
            return str(run_code) + joined.removeprefix("/runcode")
        return joined

    monkeypatch.setattr(os.path, "join", mapped_join)
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
                types.SimpleNamespace(
                    path=f"{code_prefix}/providers/_lifecycle/bootstrap_console.py", size=10
                ),
                types.SimpleNamespace(path=f"{code_prefix}/adapters/artifacts.py", size=10),
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
        from pathlib import Path

        download_calls.append({"filename": filename, "local_dir": local_dir, **kw})
        target = run_code / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        relative = filename.removeprefix(code_prefix + "/")
        sources = {
            "providers/_lifecycle/bootstrap_console.py": (
                Path(__file__).resolve().parents[1]
                / "flash/providers/_lifecycle/bootstrap_console.py"
            ),
            "adapters/artifacts.py": (
                Path(__file__).resolve().parents[1] / "flash/adapters/artifacts.py"
            ),
        }
        target.write_text(sources[relative].read_text() if relative in sources else "")
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    class _FakeProc:
        # Worker boots, logs an OOM, then the kernel/clean-exit leaves NO metrics.json.
        def __init__(self, *a, **k):
            assert k["cwd"] == str(run_code / "code/0123456789abcdef0123456789abcdef")
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
            f"{code_prefix}/providers/_lifecycle/bootstrap_console.py",
            f"{code_prefix}/adapters/artifacts.py",
        ]
    finally:
        # _train_body writes the hardcoded /tmp/console_sft.txt(.tail); remove them so this test
        # doesn't leak state across tests (flaky under isolated/parallel runners).
        import shutil

        shutil.rmtree(run_code, ignore_errors=True)
        for _p in (
            "/tmp/console_sft.txt",
            "/tmp/console_sft.txt.live.tail",
            "/tmp/console_sft.txt.final.tail",
        ):
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


def _drive_instance_upload_loop(
    monkeypatch,
    sizes: list[int],
    cycles: int,
    *,
    succeed=True,
    staged=None,
    beats=None,
    scan_cursors=None,
) -> tuple[list, list]:
    """Run the real instance loop over scripted console eofs. Returns (waits, uploads).

    ``succeed`` is the upload result, either a bool or a predicate over the attempt number, so a
    test can script an upload that fails the way hf_upload does: swallowed, returning falsy.

    ``staged`` scripts the STAGED-heartbeat count seen at each poll, which is the progress signal
    the loop keys on. It defaults to "a heartbeat whenever the console grew", the shape of a
    healthy run, so a frozen console still reads as wedged. A test covering a worker that keeps
    logging without making progress passes its own series.

    ``beats`` scripts the arming count separately. The real ``_console_progress`` returns both
    because they answer different questions: only a committed heartbeat resets the wedge timer,
    but any heartbeat -- ``pending`` included -- proves the loop was reached and may arm it. It
    defaults to ``staged``, so a test that does not care about the distinction sees the old
    behaviour; a test covering failed heartbeat uploads scripts arming beats with zero staged.

    ``scan_cursors`` may differ from ``sizes`` to model unterminated bytes: the line scanner stays
    at the preceding newline while the observed eof advances and remains eligible for upload.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap
    from flash.providers._lifecycle import bootstrap_console as _bootstrap_console

    waits: list[float] = []
    uploads: list[int] = []
    clock = {"i": 0}

    class _Stop:
        def wait(self, seconds: float) -> bool:
            waits.append(seconds)
            return len(waits) > cycles

    def _progress(_console: str, _offset: int) -> tuple[int, int, int, int]:
        index = min(clock["i"], len(sizes) - 1)
        clock["i"] += 1
        if staged is not None:
            progressed = staged[min(index, len(staged) - 1)]
        else:  # grew since the previous poll -> one staged heartbeat
            progressed = 1 if index == 0 or sizes[index] != sizes[index - 1] else 0
        arming = progressed if beats is None else beats[min(index, len(beats) - 1)]
        cursor = sizes[index] if scan_cursors is None else scan_cursors[index]
        return cursor, sizes[index], progressed, arming

    def _upload(_payload, _console, _mode) -> bool:
        uploads.append(clock["i"])
        return succeed(len(uploads)) if callable(succeed) else succeed

    monkeypatch.setattr(_bootstrap_console, "_console_progress", _progress)
    monkeypatch.setattr(_instance_bootstrap, "_upload_console_snapshot", _upload)
    _instance_bootstrap._console_upload_loop({}, "/tmp/console.txt", "train", 3600.0, _Stop())
    return waits, uploads


def test_instance_console_upload_loop_polls_faster_than_it_commits(monkeypatch):
    """A healthy, growing run polls often but still commits only on the hourly boundary.

    The poll cadence must not become the COMMIT cadence: this run's artifact cadence budgets 5
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
    the loop that exists to respect this run's 5 commits/hour spends 10 by itself. Silence
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


def test_instance_console_upload_loop_rearms_its_wedge_credit_after_progress_resumes(monkeypatch):
    """A healthy quiet stage must not disarm wedge detection for the whole run.

    A slow ``sft_model_load`` emits one staged transition and then only liveness pings, which
    ``_console_progress`` subtracts -- so it reads as sustained silence and buys a wedge snapshot
    even though nothing is wrong. With a single PERMANENT credit that run can never buy another, so
    a genuine hang later waits for the hourly cadence and the box is torn down at 1200s with no
    failure-era console: exactly the loss this loop exists to prevent, reached through the
    detection built to prevent it.

    Progress after a spent credit means the run RECOVERED, so the next quiet run is a different
    stall and re-arms. The credits bound the total, which is what keeps the re-arm from becoming an
    unbounded commit rate -- ``..._keeps_a_slow_starting_run_in_budget`` measures that ceiling.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    quiet_polls = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    # one staged transition, a long healthy-but-quiet load that spends a credit, one more
    # transition (the load finished), then a permanent wedge.
    healthy_quiet = quiet_polls + 5
    staged = [1] + [0] * healthy_quiet + [1]
    wedge_at = len(staged)
    staged += [0] * (quiet_polls + 8)
    # the console keeps growing throughout: a wedged worker still prints ray warnings, so bytes
    # never signal the stall and only the staged count can.
    sizes = [1000 * (n + 1) for n in range(len(staged))]
    _waits, uploads = _drive_instance_upload_loop(
        monkeypatch, sizes, cycles=len(sizes), staged=staged
    )

    after_wedge = [u for u in uploads if u > wedge_at]
    assert after_wedge, (
        "the healthy quiet stage spent the only wedge credit, so the real hang got no snapshot"
    )
    # and it lands inside the 1200s training stall, measured from when progress actually stopped.
    assert (after_wedge[0] - wedge_at) * poll_s < 1200.0


def test_instance_console_upload_loop_bounds_its_wedge_credits_over_a_flapping_run(monkeypatch):
    """Re-arming must not become an unbounded commit rate.

    The latch re-arms on progress, so a run that alternates a heartbeat with sustained silence
    re-arms on every cycle. Without a cap that shape buys a wedge snapshot each time -- 6/hr
    console plus the heartbeat's 4/hr, double this run's hard 5.0/hour allocation -- and the loop
    written to respect the budget becomes the thing that blows it. The credits are what bound it,
    and they are per RUN, so they never enter the sustained rate.
    """
    import flash.engine.worker as worker
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    quiet_polls = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    interval_s = _instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S
    hours = 4
    cycles = int(hours * 3600.0 / poll_s)
    # adversarial: one heartbeat, then just enough silence to look wedged, forever.
    cycle = [1] + [0] * quiet_polls
    staged = (cycle * (cycles // len(cycle) + 1))[:cycles]
    sizes = [1000 * (n + 1) for n in range(cycles)]
    _waits, uploads = _drive_instance_upload_loop(monkeypatch, sizes, cycles=cycles, staged=staged)

    # the BUDGET, not the credit constant: deriving the allowance from the value under test makes
    # the assertion self-referential -- raising the credits raises the bar with it, so the test can
    # never fail. (Measured: credits=999 commits 24 times at 10/hr and still "passes" that way.)
    scheduled_per_hour = 3600.0 / interval_s
    heartbeat_per_hour = 3600.0 / worker._HB_MIN_INTERVAL_S
    assert scheduled_per_hour + heartbeat_per_hour <= 5.0, "the schedule alone must fit the budget"
    # the credits are a fixed per-run addition, so the total is the scheduled cadence plus a
    # constant -- never one commit per silent cycle, which is what an unbounded re-arm buys.
    unbounded = cycles // len(cycle)
    assert len(uploads) < unbounded, (
        f"{len(uploads)} commits over {hours}h is the unbounded rate: every silent cycle bought one"
    )
    # and the whole run stays inside the ceiling once its one-off credits are amortized.
    assert len(uploads) / hours + heartbeat_per_hour <= 5.0 + 1.0, (
        f"console {len(uploads) / hours}/hr + heartbeat exceeds the 5.0/hr ceiling"
    )


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


def test_instance_console_upload_loop_never_charges_a_credit_to_a_scheduled_snapshot(monkeypatch):
    """Silence coinciding with a DUE snapshot must not spend a credit, even after re-arming.

    The test above covers a run silent from the start, where the ``armed`` rule alone already
    withholds the credit. This one arms the latch first, so only the ``not due`` guard can withhold
    it. Progress re-arms the latch but never refunds ``spent``, so charging an upload that the
    schedule was making anyway drains the per-run cap on snapshots that cost nothing to skip: two
    such coincidences exhaust it, and the genuine wedge afterwards gets no console before the 1200s
    stall teardown. Each scheduled boundary here is approached with output still flowing and goes
    quiet only for exactly the wedge threshold, so no INDEPENDENT wedge fires -- the sole difference
    between guarded and unguarded is what the due polls are charged.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    threshold = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    first_polls = int(_instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S / poll_s)
    interval_polls = int(_instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S / poll_s)
    # the run must be able to reach each deadline with output still flowing, or the silence that
    # coincides with it is indistinguishable from a wedge that was already owed a snapshot.
    assert first_polls > threshold, "the first deadline must land after the wedge threshold"

    sizes: list[int] = []
    staged: list[int] = []
    size = [1000]

    def _phase(progress: int, quiet: int) -> None:
        for _ in range(progress):
            size[0] += 1000
            sizes.append(size[0])
            staged.append(1)
        for _ in range(quiet):
            sizes.append(size[0])
            staged.append(0)

    # each deadline is met with quiet_polls at exactly the threshold: due and wedge coincide.
    _phase(first_polls - threshold, threshold)
    _phase(interval_polls - threshold, threshold)
    _phase(1, 0)  # output resumes, re-arming the latch
    wedge_at = len(sizes)
    _phase(0, threshold + 6)  # then the run wedges for good, far from any deadline
    _waits, uploads = _drive_instance_upload_loop(
        monkeypatch, sizes, cycles=len(sizes), staged=staged
    )

    after_wedge = [u for u in uploads if u > wedge_at]
    assert after_wedge, "the scheduled snapshots drained the credits the real wedge needed"
    assert (after_wedge[0] - wedge_at) * poll_s < 1200.0, (
        "the wedge snapshot landed after the stall teardown would have killed the box"
    )


def test_scheduled_console_snapshot_clears_captured_silence_without_spending_credit(monkeypatch):
    """A due snapshot that captures a quiet wedge must not trigger another upload next poll."""
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    threshold = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    first_polls = int(
        _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
        / _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    )
    assert first_polls == threshold + 1

    # progress arms the loop, then sustained silence overlaps the first scheduled snapshot.
    staged = [1] + [0] * threshold
    sizes = [1000] * first_polls
    # a complete noise line arrives next. it changes the uploaded bytes but is not progress, so a
    # latch left active by the scheduled upload would trigger a redundant emergency snapshot here.
    sizes.append(2000)
    staged.append(0)
    # real progress re-arms the latch, and the later genuine wedge must still spend a credit.
    sizes.append(3000)
    staged.append(1)
    sizes.extend([4000 + poll for poll in range(threshold + 3)])
    staged.extend([0] * (threshold + 3))

    _waits, uploads = _drive_instance_upload_loop(
        monkeypatch, sizes, cycles=len(sizes), staged=staged
    )

    scheduled_poll = first_polls
    noise_poll = scheduled_poll + 1
    genuine_wedge_poll = noise_poll + 1 + threshold
    assert uploads[0] == scheduled_poll
    assert noise_poll not in uploads, (
        "scheduled captured silence caused a redundant emergency upload"
    )
    assert genuine_wedge_poll in uploads, "the scheduled snapshot spent the real wedge's credit"


def test_instance_console_upload_loop_keeps_a_slow_starting_run_in_budget(monkeypatch):
    """A silent setup must not buy the run a faster sustained cadence.

    A tempting fix for "a setup that never emits a heartbeat is torn down at 3000s with only its
    600s snapshot" is to hold the first-snapshot cadence until progress starts. It does close that
    gap -- and it costs one extra commit that lands in the SUSTAINED rate, because a real run has a
    silent setup and then trains for hours: 4h with a 600s setup measures 1.25/hr console + 4/hr
    heartbeat = 5.25/hr against a hard 5.0. The gap is left to the stall classifier instead. This
    pins the rate so that fix cannot be reintroduced without buying rate somewhere else.
    """
    import flash.engine.worker as worker
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    interval_s = _instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    hours = 4
    cycles = int(hours * 3600.0 / poll_s)
    # the realistic shape: a silent setup, then a healthy run that logs for hours.
    setup_polls = int(first_s / poll_s)
    sizes = [1000 * (n + 1) for n in range(cycles + 2)]
    staged = [0] * setup_polls + [1] * (cycles + 2 - setup_polls)
    _waits, uploads = _drive_instance_upload_loop(monkeypatch, sizes, cycles=cycles, staged=staged)

    per_hour = len(uploads) / hours
    total = per_hour + 3600.0 / worker._HB_MIN_INTERVAL_S
    assert total <= 5.0, f"console {per_hour}/hr + heartbeat = {total}/hr, budget is 5"
    # and the cadence must actually converge to the interval, not merely squeak under the cap.
    assert (uploads[-1] - uploads[-2]) * poll_s == interval_s


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


def test_generated_worker_loads_console_runtime_from_downloaded_code(tmp_path, monkeypatch):
    import ast
    import contextlib
    import hashlib
    import json
    import os
    import re
    import shutil
    import subprocess
    import sys
    import types
    from pathlib import Path

    import huggingface_hub

    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile.worker").read_text()
    copy = re.search(
        r"COPY (flash/providers/runpod/serverless/handler\.py) "
        r"(docker/make_rp_handler\.py) /tmp/rpgen/",
        dockerfile,
    )
    assert copy, "worker image must copy the handler source and generator together"
    assert "make_rp_handler.py /tmp/rpgen/handler.py /rp_handler.py" in dockerfile
    generated = tmp_path / "rp_handler.py"
    subprocess.run(
        [sys.executable, str(root / copy.group(2)), str(root / copy.group(1)), str(generated)],
        check=True,
    )
    generated_source = generated.read_text()
    function = next(
        node
        for node in ast.parse(generated_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_train_body"
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(generated), "exec"), namespace)

    run_code = tmp_path / "runcode"
    real_join = os.path.join

    def mapped_join(*parts):
        joined = real_join(*parts)
        if joined == "/runcode" or joined.startswith("/runcode/"):
            return str(run_code) + joined.removeprefix("/runcode")
        return joined

    monkeypatch.setattr(os.path, "join", mapped_join)
    digest = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:32]
    code_prefix = f"code/{digest}/flash"
    code_dir = run_code / "code" / digest
    console_origin = tmp_path / "console_origin.txt"
    artifact_origin = tmp_path / "artifact_origin.txt"
    loop_call = tmp_path / "loop_call.txt"
    uploads: list[str] = []
    module_sources = {
        "providers/_lifecycle/bootstrap_console.py": (
            "import json\nfrom pathlib import Path\n"
            f"Path({str(console_origin)!r}).write_text(__file__)\n"
            "def _run_console_upload_loop(console, interval_s, stop_upload, *, upload):\n"
            f"    Path({str(loop_call)!r}).write_text(json.dumps([__file__, interval_s]))\n"
            "    upload()\n"
        ),
        "adapters/artifacts.py": (
            "from pathlib import Path\n"
            f"Path({str(artifact_origin)!r}).write_text(__file__)\n"
            "def attempt_scoped_artifact_name(kind, phase, attempt):\n"
            "    return f'exact-{kind}-{phase}-attempt{attempt}.txt'\n"
        ),
    }

    class _Api:
        def __init__(self, token=None):
            pass

        def list_repo_tree(self, **_kwargs):
            return [
                types.SimpleNamespace(path=f"{code_prefix}/{relative}", size=1)
                for relative in module_sources
            ]

        def upload_file(self, *, path_in_repo, **_kwargs):
            uploads.append(path_in_repo)

    def _download(*, filename, local_dir, **_kwargs):
        relative = filename.removeprefix(code_prefix + "/")
        target = run_code / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(module_sources[relative])
        return str(target)

    class _Process:
        def __init__(self, *_args, **kwargs):
            assert Path(kwargs["cwd"]) == code_dir
            self.stdout = iter(["worker started\n"])
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    monkeypatch.setattr(subprocess, "Popen", _Process)
    input_data = {
        "phase": "sft",
        "seed": 0,
        "hf_repo": "owner/runs",
        "job_spec_json": '{"algorithm":"sft","run_id":"generated-run"}',
        "env": {"HF_TOKEN": "token", "ATTEMPT": "2"},
        "code_prefix": code_prefix,
        **_run_deadline_fields(),
    }
    try:
        with pytest.raises(RuntimeError, match=r"produced no /tmp/metrics\.json"):
            namespace["_train_body"](input_data)
        assert Path(console_origin.read_text()) == (
            code_dir / "flash/providers/_lifecycle/bootstrap_console.py"
        )
        assert Path(artifact_origin.read_text()) == code_dir / "flash/adapters/artifacts.py"
        called_from, interval_s = json.loads(loop_call.read_text())
        assert Path(called_from) == code_dir / "flash/providers/_lifecycle/bootstrap_console.py"
        assert interval_s == 3600.0
        assert "sft/generated-run/exact-console-sft-attempt2.txt" in uploads
        assert "sft/generated-run/console_sft.txt" in uploads
    finally:
        shutil.rmtree(code_dir, ignore_errors=True)
        for target in (
            "/tmp/console_sft.txt",
            "/tmp/console_sft.txt.live.tail",
            "/tmp/console_sft.txt.final.tail",
        ):
            with contextlib.suppress(FileNotFoundError):
                os.remove(target)


def test_console_progress_counts_staged_heartbeats_incrementally(tmp_path):
    """The wedge signal itself, unmocked: every loop test above patches _console_progress out.

    Three properties, all load-bearing. It must count only heartbeats that REPRESENT progress: every
    payload carries ``"stage"``, liveness pings included (``heartbeat.py`` adds ``"liveness": True``
    to the same flat object), and those print every 30s from a daemon thread -- so a worker wedged
    inside a liveness block would look busy forever. ``poll`` refuses to advance its stall key on
    them for exactly this reason, and disagreeing means the run is torn down with no console. It
    must count only lines a heartbeat actually WROTE, since ray, verl and any library are free to
    emit structured json carrying the same key -- and a wedged worker keeps printing those. And it
    must count only bytes past ``offset``: a console reaching hundreds of MB is rescanned every
    poll otherwise, and a heartbeat from the healthy prefix would be recounted forever.
    """
    import json

    from flash.providers._lifecycle import bootstrap_console as _bootstrap_console

    def _hb(**kw) -> bytes:
        # the real console line: heartbeat.py prints "HEARTBEAT " + json.dumps(payload).
        payload = {"stage": "train", "ts": 1.0, "run_id": "r", "mode": "train", **kw}
        return f"HEARTBEAT {json.dumps(payload)}\n".encode()

    console = tmp_path / "console_train.txt"
    progress = _hb(step=7)
    pings = _hb(liveness=True) + b"ray warning: worker is idle\n"
    console.write_bytes(progress + pings)

    cursor, observed_eof, beats, _any = _bootstrap_console._console_progress(str(console), 0)
    assert (cursor, observed_eof, beats) == (
        len(progress) + len(pings),
        len(progress) + len(pings),
        1,
    )

    # the wedge: heartbeats keep arriving, but every one is a liveness ping.
    with open(console, "ab") as f:
        f.write(pings * 3)
    cursor2, eof2, beats2, _any2 = _bootstrap_console._console_progress(str(console), cursor)
    assert beats2 == 0, "liveness pings after the last real heartbeat must not read as progress"
    assert (cursor2, eof2) == (cursor + len(pings) * 3, observed_eof + len(pings) * 3)

    # a poll boundary can split a line, leaving a chunk that starts mid-payload. Neither half may
    # read as progress on its own: the tail has the disqualifying key but no prefix, and the head
    # has the prefix but not yet the key. The offsets are computed from the line, not guessed: a
    # guessed one that happens to include both keys makes these assertions vacuous.
    solo = tmp_path / "console_split.txt"
    ping = _hb(liveness=True)
    solo.write_bytes(ping)
    cut = ping.index(b'"liveness":') - 5  # past "stage", before "liveness"
    assert b'"stage":' not in ping[cut:]
    assert b'"liveness":' in ping[cut:]
    assert _bootstrap_console._console_progress(str(solo), cut) == (
        len(ping),
        len(ping),
        0,
        0,
    )

    # an UNCOMMITTED heartbeat (upload attempted, did not land) is excluded for the same reason
    # as a liveness ping: the provider's stall clock reads heartbeat.json from HF, so one that
    # never reached HF is not progress it can observe. Counting it resets the wedge timer against a
    # stall clock still anchored to the older committed heartbeat, and the snapshot this whole path
    # exists to take gets scheduled after the teardown has already killed the box.
    pend = tmp_path / "console_pending.txt"
    pend.write_bytes(_hb(step=8, pending=True) * 2)
    assert _bootstrap_console._console_progress(str(pend), 0)[2] == 0
    # but it does count toward arming, the fourth value: the line exists because a heartbeat was
    # produced, so the training loop was reached. If arming used the progress count instead, a run
    # whose heartbeat uploads all fail would never arm and could never buy a wedge snapshot -- the
    # next scheduled one is an interval out, past the setup teardown, so the console is lost.
    assert _bootstrap_console._console_progress(str(pend), 0)[3] == 2

    throttled = tmp_path / "console_throttled.txt"
    throttled.write_bytes(_hb(step=9, throttled=True))
    assert _bootstrap_console._console_progress(str(throttled), 0)[2:] == (0, 1)

    # a liveness ping arms nothing: it prints from a daemon whether or not the training loop is
    # alive, so unlike a pending heartbeat it is not evidence the run ever reached the loop.
    lv_only = tmp_path / "console_liveness_only.txt"
    lv_only.write_bytes(_hb(liveness=True) * 3)
    assert _bootstrap_console._console_progress(str(lv_only), 0)[2:] == (0, 0)

    # a THIRD-PARTY line carrying "stage" is not a heartbeat and must not read as progress. ray,
    # verl and any library are free to emit structured json, and a wedged worker keeps emitting it
    # -- so a bare-substring count would see progress forever and the wedge would never fire. The
    # `HEARTBEAT ` prefix names the producer instead of inferring it from a shared key.
    foreign = tmp_path / "console_foreign.txt"
    foreign.write_bytes(b'(raylet) {"ts":"t","level":"WARN","stage":"rollout","msg":"idle"}\n' * 5)
    assert _bootstrap_console._console_progress(str(foreign), 0)[2] == 0

    # a line split by the poll boundary is counted EXACTLY ONCE, in whichever chunk carries its
    # `HEARTBEAT` prefix. The old subtraction could see the tail half alone, count its "stage" and
    # miss the "liveness" that qualified it; the anchor sits at the start of the line, so the half
    # without it matches nothing rather than being judged on the keys it happens to carry.
    split = tmp_path / "console_split_beat.txt"
    beat = _hb(step=11)
    split.write_bytes(beat[:20])
    head_cursor, _head_eof, head_beats, _ = _bootstrap_console._console_progress(str(split), 0)
    with open(split, "ab") as f:
        f.write(beat[20:])
    tail_cursor, tail_eof, tail_beats, _ = _bootstrap_console._console_progress(
        str(split), head_cursor
    )
    assert head_beats + tail_beats == 1, "a split heartbeat is counted once, not zero or twice"
    assert (tail_cursor, tail_eof) == (len(beat), len(beat))
    assert _bootstrap_console._console_progress(str(split), tail_cursor)[2] == 0, "no recount"

    # and the tail half of a split LIVENESS ping is never counted: the old rule saw its "stage"
    # without the "liveness" that disqualified it, so a wedge read as progress forever.
    lv_split = tmp_path / "console_split_liveness.txt"
    ping = _hb(step=12, liveness=True)
    cut = ping.index(b'"liveness":') - 5
    lv_split.write_bytes(ping[:cut])
    lv_head = _bootstrap_console._console_progress(str(lv_split), 0)
    with open(lv_split, "ab") as f:
        f.write(ping[cut:])
    lv_tail = _bootstrap_console._console_progress(str(lv_split), lv_head[0])
    assert lv_head[2] + lv_tail[2] == 0, "a split liveness ping must never read as progress"

    # the scan cursor stops at the last newline while observed eof exposes unterminated bytes to the
    # upload state machine. when the line completes, it is parsed exactly once from that cursor.
    part = tmp_path / "console_partial.txt"
    whole = _hb(step=3)
    partial = _hb(step=4)
    part.write_bytes(whole + partial[:20])
    first = _bootstrap_console._console_progress(str(part), 0)
    assert first == (len(whole), len(whole) + 20, 1, 1)
    with open(part, "ab") as f:
        f.write(partial[20:])
    completed = _bootstrap_console._console_progress(str(part), first[0])
    assert completed == (len(whole) + len(partial), len(whole) + len(partial), 1, 1)
    assert _bootstrap_console._console_progress(str(part), completed[0])[2:] == (0, 0)

    # one poll drains to its captured tail in bounded chunks, so a chatty console cannot make the
    # heartbeat cursor fall farther behind on every cycle.
    cap = _bootstrap_console._CONSOLE_SCAN_BYTES
    huge = tmp_path / "console_huge_line.txt"
    huge.write_bytes(b"x" * (2 * cap + 17) + b"\n" + _hb(step=99))
    cursor, eof, observed, any_beats = _bootstrap_console._console_progress(str(huge), 0)
    assert (cursor, eof) == (huge.stat().st_size, huge.stat().st_size)
    assert (observed, any_beats) == (1, 1)

    assert _bootstrap_console._console_progress(str(tmp_path / "absent.txt"), 0) == (
        -1,
        -1,
        0,
        0,
    )


def test_instance_console_upload_loop_arms_on_a_heartbeat_whose_upload_failed(monkeypatch):
    """A run whose heartbeat uploads all fail must still be able to buy a wedge snapshot.

    Every heartbeat then prints ``"pending": true``, so the STAGED count is zero forever. Arming on
    that count leaves the latch permanently false: no wedge snapshot is ever bought, and the only
    other upload is the scheduled cadence -- an interval (3600s) after the 600s one, so 4200s, past
    the 3000s no-heartbeat setup teardown. The console covering the hang is lost entirely.

    So the counts are separate, and which one drives the timer depends on whether this run has EVER
    committed a heartbeat. Until it has, a ``pending`` one counts: nothing else exists to contradict
    it, and teardown is the fixed setup grace. Once one has committed, only committed ones count --
    the provider's stall clock is anchored to that commit and already counting down, so treating a
    later pending line as progress would push the snapshot past the box's own death.

    Asserted in three directions, because the failure modes point opposite ways: the pending run
    buys a wedge snapshot AFTER its hang (not burned on the healthy stretch before it); a run with
    no heartbeat at all buys none; and a run that committed and then stopped still buys one.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    hang_polls = 14  # heartbeats stop here; the console keeps growing on ray chatter
    cycles = hang_polls + 12
    # the console never stops growing, so a size-only rule cannot separate these cases: the
    # heartbeat counts are the whole signal.
    sizes = [1000 * (i + 1) for i in range(cycles + 2)]
    # scheduled snapshot at first_s, then an hour out -- past the 3000s setup teardown.
    scheduled_at = {first_s}

    # every heartbeat is pending (uploads all failing), then the worker hangs at hang_polls.
    beats = [1] * hang_polls + [0] * (cycles - hang_polls)
    _waits, uploads = _drive_instance_upload_loop(
        monkeypatch, sizes, cycles=cycles, staged=[0] * cycles, beats=beats
    )
    at_s = {n * poll_s for n in uploads}
    wedge_at = sorted(at_s - scheduled_at)
    assert wedge_at, "a run whose heartbeat uploads all fail must still buy a wedge snapshot"
    # and it must be bought AFTER the hang, not burned on the healthy stretch before it. That is
    # the direction that makes the snapshot worth its credit: an early one shows a working run.
    assert all(t > hang_polls * poll_s for t in wedge_at), wedge_at
    assert all(t < 3000.0 for t in wedge_at), "must land before the setup teardown"

    # the other direction: no heartbeat of ANY kind (liveness pings alone, or startup noise) must
    # not arm, or a slow start spends a credit on a console that has nothing in it yet.
    _w2, uploads2 = _drive_instance_upload_loop(
        monkeypatch, sizes, cycles=cycles, staged=[0] * cycles, beats=[0] * cycles
    )
    assert {n * poll_s for n in uploads2} == {first_s, 2400.0}

    # once a heartbeat has COMMITTED, pending ones must stop counting as progress: the provider's
    # stall clock is anchored to that commit and is already counting down to a teardown, so a
    # pending line must not reset this timer past the box's own death.
    staged3 = [1] * 4 + [0] * (cycles - 4)
    _w3, uploads3 = _drive_instance_upload_loop(
        monkeypatch, sizes, cycles=cycles, staged=staged3, beats=[1] * cycles
    )
    assert sorted({n * poll_s for n in uploads3} - scheduled_at), (
        "a run that stopped committing must still buy a wedge snapshot"
    )


def test_throttled_heartbeat_cannot_push_snapshot_past_stall_teardown(monkeypatch):
    """Console-only progress must preserve the provider's older committed deadline."""
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    stall_s = 1200.0
    cycles = 14
    # the committed heartbeat at poll 4 arms the loop close enough to the first scheduled snapshot
    # that poll 5 has not captured sustained silence. a throttled heartbeat appears at poll 8; it
    # never reached hf, so treating it as committed would reset quiet and push the wedge past teardown.
    sizes = [1000] * 7 + [2000] + [2000] * (cycles - 8)
    staged = [0] * 3 + [1] + [0] * (cycles - 4)
    beats = [0] * 3 + [1] + [0] * 3 + [1] + [0] * (cycles - 8)
    _waits, uploads = _drive_instance_upload_loop(
        monkeypatch,
        sizes,
        cycles=cycles,
        staged=staged,
        beats=beats,
    )

    at_s = [attempt * poll_s for attempt in uploads]
    assert at_s[0] == first_s
    assert at_s[1] == 8 * poll_s
    assert at_s[1] < stall_s


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


def test_instance_console_upload_loop_retries_a_failed_snapshot_before_any_progress(monkeypatch):
    """A failed upload must not advance the DEADLINE either, including before progress starts.

    The uploaded-bytes watermark is only half of it. hf_upload swallows its exception and returns
    falsy, so resetting ``since`` on that return books a snapshot that reached no repo AND pushes
    the next attempt a full interval out -- 4200s, past the 1200s stall and the 3000s setup grace.

    Before any staged heartbeat the wedge path is disarmed by design (a wedge is progress that
    STOPPED), so it cannot supply the retry the way it incidentally did when quiet accounting ran
    from process start. That makes the deadline the ONLY thing standing between a transient 500 at
    the first snapshot and a run that dies with no console at all. The series here never emits a
    heartbeat and never grows, so a retry can only come from the deadline staying due.
    """
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    setup_grace_s = 3000.0  # poll.SETUP_GRACE_S: teardown for a run that never reaches a heartbeat
    grow_polls = int(first_s / poll_s) + 1
    # frozen console AND no heartbeat ever: neither the size change nor the wedge can buy the retry.
    sizes = [1000] * (grow_polls + 20)
    staged = [0] * len(sizes)
    _waits, uploads = _drive_instance_upload_loop(
        monkeypatch,
        sizes,
        cycles=len(sizes),
        succeed=lambda attempt: attempt > 1,
        staged=staged,
    )

    assert len(uploads) > 1, "a failed first snapshot was never retried before teardown"
    assert uploads[1] * poll_s < setup_grace_s, "the retry landed after the setup teardown"
    # one poll later, not one interval later -- the deadline must not have advanced on the failure.
    assert (uploads[1] - uploads[0]) * poll_s <= poll_s


def test_console_uploads_use_observed_eof_when_scan_cursor_matches_sent_eof(monkeypatch):
    """Unterminated bytes remain uploadable on both scheduled and emergency paths."""
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_polls = int(_instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S / poll_s)

    # after the first setup snapshot, an unterminated diagnostic advances eof but leaves the scanner
    # at the successfully sent newline. the second setup deadline must still upload those bytes.
    second_setup_polls = int(1800.0 / poll_s)
    scheduled_eofs = [100] * first_polls + [150] * second_setup_polls
    scheduled_cursors = [100] * len(scheduled_eofs)
    _waits, scheduled = _drive_instance_upload_loop(
        monkeypatch,
        scheduled_eofs,
        cycles=len(scheduled_eofs),
        staged=[0] * len(scheduled_eofs),
        scan_cursors=scheduled_cursors,
    )
    assert scheduled == [first_polls, first_polls + second_setup_polls]

    # a committed heartbeat arms the loop. after its scheduled snapshot, unterminated diagnostics
    # leave scan_cursor equal to sent_eof while observed_eof advances; sustained silence must still
    # buy the emergency snapshot before teardown.
    threshold = _instance_bootstrap._CONSOLE_UPLOAD_QUIET_POLLS
    emergency_eofs = [100 * poll for poll in range(1, first_polls + 1)] + [550] * threshold
    emergency_cursors = [100 * poll for poll in range(1, first_polls + 1)] + [500] * threshold
    staged = [1] * first_polls + [0] * threshold
    _waits, emergency = _drive_instance_upload_loop(
        monkeypatch,
        emergency_eofs,
        cycles=len(emergency_eofs),
        staged=staged,
        scan_cursors=emergency_cursors,
    )
    assert emergency == [first_polls, first_polls + threshold]


def test_instance_console_upload_takes_a_second_setup_snapshot_before_teardown(monkeypatch):
    """Pending heartbeats cannot postpone the last setup diagnostic past the 3000s teardown."""
    from flash.providers._lifecycle import bootstrap as _instance_bootstrap

    poll_s = _instance_bootstrap._CONSOLE_UPLOAD_POLL_S
    first_s = _instance_bootstrap._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    setup_grace_s = 3000.0
    cycles = int(setup_grace_s / poll_s) + 1
    sizes = [1000 * (index + 1) for index in range(cycles + 2)]
    # heartbeat production continues, but no heartbeat upload ever commits. the provider therefore
    # remains on its fixed setup grace even though the console uploader can still reach hf.
    _waits, uploads = _drive_instance_upload_loop(
        monkeypatch,
        sizes,
        cycles=cycles,
        staged=[0] * cycles,
        beats=[1] * cycles,
    )

    at_s = [attempt * poll_s for attempt in uploads]
    assert at_s[0] == first_s
    assert len(at_s) >= 2
    assert at_s[1] < setup_grace_s


def test_serverless_live_and_terminal_console_snapshots_use_distinct_repo_paths(
    tmp_path, monkeypatch
):
    """A live upload that finishes late cannot overwrite the canonical terminal snapshot."""
    import ast
    import inspect
    import json
    import os
    import textwrap
    import threading
    from pathlib import Path

    from flash.providers.runpod.serverless import endpoints

    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoints._train_body)))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_commit_console_snapshot"
    )
    console = tmp_path / "console_train.txt"
    console.write_text("line one\nline two\n")
    commits: list[str] = []

    class _Api:
        def __init__(self, token=None):
            pass

        def upload_file(self, *, path_in_repo: str, **_kw) -> None:
            commits.append(path_in_repo)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    teardown = threading.Event()
    namespace: dict = {
        "os": os,
        "json": json,
        "input_data": {
            "job_spec_json": json.dumps({"algorithm": "opd", "run_id": "r1"}),
            "hf_repo": "org/repo",
        },
        "env": {"ATTEMPT": "2"},
        "console_teardown": teardown,
        "_require_deadline_allowance": lambda: None,
        "_read_console_tail": lambda path: Path(path).read_text(),
        "_safe_detail": lambda text, _env, _limit=0: text,
        "attempt_scoped_artifact_name": lambda kind, phase, attempt: (
            f"{kind}_{phase}_attempt{attempt}.txt"
        ),
        "print": lambda *_args, **_kw: None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<handler>", "exec"), namespace)
    commit = namespace["_commit_console_snapshot"]
    tail = str(tmp_path / "console_train.txt.tail")

    assert commit("train", str(console), tail, False) is True
    assert commit("train", str(console), tail + ".final", True) is True
    assert commits == [
        "opd/r1/console_train_attempt2.txt",
        "opd/r1/console_train.txt",
    ]


def test_serverless_terminal_console_does_not_wait_for_a_stuck_live_upload(tmp_path):
    """The terminal snapshot has its own path, so an unbounded live request cannot block it."""
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
    console = tmp_path / "console_train.txt"
    console.write_text("before failure\n")
    started = threading.Event()
    release = threading.Event()
    calls: list[bool] = []

    def _commit(_mode: str, _console: str, _tail: str, final: bool) -> bool:
        calls.append(final)
        if not final:
            started.set()
            release.wait(5.0)
        return True

    namespace = {
        "os": types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda _p: True)),
        "console_teardown": threading.Event(),
        "_commit_console_snapshot": _commit,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<handler>", "exec"), namespace)
    upload = namespace["_upload_console"]

    live = threading.Thread(target=upload, args=("train",), daemon=True)
    live.start()
    assert started.wait(5.0)
    assert upload("train", final=True) is True
    assert calls == [False, True], "terminal upload waited behind the stuck live request"
    release.set()
    live.join(timeout=5.0)
    assert not live.is_alive()


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
