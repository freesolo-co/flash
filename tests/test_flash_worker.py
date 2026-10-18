"""Regression tests for the Flash worker plumbing fixed in this PR.

Covers:
- build_worker_env sets managed worker defaults without forwarding removed tuning knobs;
- the runpod_flash backoff OverflowError that aborted long runs is patched;
- per-phase error artifact names don't collide (train error survives a later eval error).
"""

from __future__ import annotations

import time

import pytest

import flash.engine.worker.io.heartbeat as worker_heartbeat
from tests._helpers.source_snapshot import valid_source_snapshot

SOURCE_SNAPSHOT = valid_source_snapshot()


def _spec():
    from flash.core.spec import JobSpec, TrainSpec

    return JobSpec(
        model="Qwen/Qwen3.5-9B",
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

    from flash.providers._lifecycle.net.worker import build_worker_env

    knobs = {
        "VLLM_USE_V1": "0",
        "SFT_PER_DEVICE_BS": "4",
    }
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)

    env = build_worker_env(_spec())
    for k in knobs:
        assert k not in env, f"{k} should not be forwarded to worker"
    # fragmentation-safe allocator default is always set
    assert "PYTORCH_CUDA_ALLOC_CONF" in env


def test_build_worker_env_ignores_alloc_conf_override(monkeypatch):
    """flash is fully managed: an operator PYTORCH_CUDA_ALLOC_CONF in the process env does NOT
    override flash's computed allocator conf (RL is non-expandable, sleep-safe)."""
    from flash.providers._lifecycle.net.worker import build_worker_env

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:999")
    env = build_worker_env(_spec())  # grpo -> sleep-safe non-expandable
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
    from flash.providers._lifecycle.net.worker import build_worker_env

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    opd_spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )
    env = build_worker_env(
        opd_spec,
        runtime_secrets={
            "FLASH_PUBLIC_URL": "https://broker.example",
            "FLASH_TEACHER_CAPABILITY": "capability-test-value",
        },
    )
    assert "expandable_segments" not in env["PYTORCH_CUDA_ALLOC_CONF"]
    assert "expandable_segments" not in env["PYTORCH_ALLOC_CONF"]
    # GRPO still ships the sleep-safe non-expandable conf.
    grpo_env = build_worker_env(_spec())
    assert "expandable_segments" not in grpo_env["PYTORCH_CUDA_ALLOC_CONF"]


def test_build_worker_env_does_not_forward_judge_creds(monkeypatch):
    """flash is fully managed: reward-judge creds and the judge-model id are NOT hardcoded
    control-plane forwards. An env that needs a judge provider key declares it as an
    [environment].secrets entry (forwarded via runtime_secrets); the env's own default judge model
    otherwise applies. A stray control-plane OPENROUTER_API_KEY / OPENAI_API_KEY / FLASH_JUDGE_MODEL
    must NOT leak into every worker."""
    from flash.providers._lifecycle.net.worker import build_worker_env

    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "FLASH_JUDGE_MODEL"):
        monkeypatch.setenv(key, "control-plane-should-not-forward")
    env = build_worker_env(_spec())
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "FLASH_JUDGE_MODEL"):
        assert key not in env


def test_build_worker_env_forwards_github_only_for_private_vcs_pip(monkeypatch):
    from flash.providers._lifecycle.net.worker import build_worker_env

    monkeypatch.setenv("GITHUB_TOKEN", "ghp-secret")
    monkeypatch.setenv("GIT_ASKPASS", "/tmp/operator-askpass")
    env = build_worker_env(_spec())
    assert env["GITHUB_TOKEN"] == "ghp-secret"
    assert "GIT_ASKPASS" not in env


def test_build_worker_env_forwards_only_managed_teacher_capability_for_opd(monkeypatch):
    """opd receives bounded broker transport while provider credentials remain control-plane-only."""
    from flash.core.spec import JobSpec, TrainSpec
    from flash.providers._lifecycle.net.worker import build_worker_env

    monkeypatch.setenv("PARASAIL_API_KEY", "platform-managed-parasail")
    opd_spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )
    env = build_worker_env(
        opd_spec,
        runtime_secrets={
            "FLASH_PUBLIC_URL": "https://broker.example",
            "FLASH_TEACHER_CAPABILITY": "capability-test-value",
        },
    )
    assert env["FLASH_PUBLIC_URL"] == "https://broker.example"
    assert env["FLASH_TEACHER_CAPABILITY"] == "capability-test-value"
    assert "PARASAIL_API_KEY" not in env
    grpo = build_worker_env(_spec())
    assert "FLASH_PUBLIC_URL" not in grpo
    assert "FLASH_TEACHER_CAPABILITY" not in grpo


def test_build_worker_env_does_not_accept_legacy_teacher_broker_url():
    from flash.core.spec import JobSpec, TrainSpec
    from flash.providers._lifecycle.net.worker import build_worker_env

    opd_spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )

    with pytest.raises(RuntimeError, match="control-panel teacher transport is missing"):
        build_worker_env(
            opd_spec,
            runtime_secrets={
                "FLASH_TEACHER_BROKER_URL": "https://broker.example",
                "FLASH_TEACHER_CAPABILITY": "capability-test-value",
            },
        )


def test_build_worker_env_rejects_managed_teacher_byo_names():
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
    from flash.providers._lifecycle.net.worker import build_worker_env

    opd_spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        environment=EnvironmentSpec(id="org/env", secrets=("PARASAIL_API_KEY",)),
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )
    with pytest.raises(ValueError, match="managed teacher credential names"):
        build_worker_env(
            opd_spec,
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
    from flash.providers._lifecycle.net.worker import build_worker_env

    monkeypatch.setenv("WANDB_API_KEY", "platform-should-not-forward")
    env = build_worker_env(_spec())
    assert "WANDB_API_KEY" not in env

    env = build_worker_env(_spec(), runtime_secrets={"WANDB_API_KEY": "user-wb"})
    assert env["WANDB_API_KEY"] == "user-wb"


def test_build_worker_env_forwards_declared_environment_runtime_secrets():
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
    from flash.providers._lifecycle.net.worker import build_worker_env

    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        environment=EnvironmentSpec(id="owner/env", secrets=("SERPAPI_API_KEY",)),
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )

    env = build_worker_env(
        spec,
        runtime_secrets={
            "SERPAPI_API_KEY": "serp-user",
            "UNDECLARED_API_KEY": "must-not-forward",
        },
    )
    assert env["SERPAPI_API_KEY"] == "serp-user"
    assert "UNDECLARED_API_KEY" not in env


def test_build_worker_env_lists_declared_secret_names_for_the_redactors(monkeypatch, tmp_path):
    """the producer's exact applied-name metadata is also the verl child scrub contract."""
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
    from flash.engine.worker.train.entry.sft_train import _build_verl_child_env
    from flash.providers._lifecycle.net.worker import build_worker_env

    declared = (
        "AWS_SECRET_ACCESS_KEY",
        "CUDA_SECRET",
        "FLA_CREDENTIAL",
        "PYTHONPATH",
        "WANDB_USER_SECRET",
    )
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        environment=EnvironmentSpec(id="owner/env", secrets=declared),
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )
    supplied = {name: f"synthetic-{name.lower()}" for name in declared}
    supplied["WANDB_API_KEY"] = "synthetic-wandb-key"
    env = build_worker_env(spec, runtime_secrets=supplied)

    listed = set(env[SECRET_ENV_KEYS_ENV].split(","))
    assert listed == {*declared, "WANDB_API_KEY"}
    for name in listed:
        assert name in env
    # a run with no applied secrets carries no list at all.
    assert SECRET_ENV_KEYS_ENV not in build_worker_env(_spec())

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("FLA_TILELANG", "0")
    monkeypatch.setenv("WANDB_MODE", "offline")
    child = _build_verl_child_env(shim_dir=str(tmp_path), wandb_enabled=True)

    for name in declared:
        if name != "PYTHONPATH":
            assert name not in child
    assert child["PYTHONPATH"] == str(tmp_path)
    assert SECRET_ENV_KEYS_ENV not in child
    assert "WANDB_API_KEY" in child
    assert child["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert child["FLA_TILELANG"] == "0"
    assert child["WANDB_MODE"] == "offline"


def test_the_redactor_metadata_name_is_reserved_from_declared_secrets():
    """build_worker_env sets FLASH_SECRET_ENV_KEYS last, so a job declaring that exact name would
    have its credential silently overwritten by the generated name list and fail at runtime. it is
    control-plane-owned, so the declaration is rejected loudly instead."""
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
    from flash.core.spec import CONTROL_PLANE_OWNED_ENV_KEYS
    from flash.schema.fields import ConfigError, _environment_secrets

    assert SECRET_ENV_KEYS_ENV in CONTROL_PLANE_OWNED_ENV_KEYS
    with pytest.raises(ConfigError, match="platform-managed key"):
        _environment_secrets([SECRET_ENV_KEYS_ENV], "grpo")
    # a case variant is a distinct linux env name but not a distinct DECLARATION: build_worker_env
    # tests ownership on the uppercased name, so accepting it here would drop the secret from the
    # worker env without a word and launch the job missing a credential it declared as required.
    # every reserved name is refused across its whole case-space for that reason.
    for variant in (SECRET_ENV_KEYS_ENV.lower(), "Hf_Token", "runpod_api_key"):
        with pytest.raises(ConfigError, match="platform-managed key"):
            _environment_secrets([variant], "grpo")


def test_grpo_worker_env_keeps_native_thread_policy_managed():
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
    from flash.core.grpo import GRPO_NATIVE_THREAD_ENV
    from flash.core.spec import EnvironmentSpec, JobSpec
    from flash.providers._lifecycle.net.worker import build_worker_env

    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        environment=EnvironmentSpec(id="owner/project/env", secrets=("OMP_NUM_THREADS",)),
    )
    env = build_worker_env(spec, runtime_secrets={"OMP_NUM_THREADS": "999"})

    assert env["OMP_NUM_THREADS"] == GRPO_NATIVE_THREAD_ENV["OMP_NUM_THREADS"]
    assert "OMP_NUM_THREADS" not in set(env.get(SECRET_ENV_KEYS_ENV, "").split(","))


def test_sft_worker_env_forwards_declared_native_thread_secret():
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
    from flash.core.spec import EnvironmentSpec, JobSpec
    from flash.providers._lifecycle.net.worker import build_worker_env

    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        environment=EnvironmentSpec(id="owner/project/env", secrets=("OMP_NUM_THREADS",)),
    )
    env = build_worker_env(spec, runtime_secrets={"OMP_NUM_THREADS": "7"})

    assert env["OMP_NUM_THREADS"] == "7"
    assert "OMP_NUM_THREADS" in set(env[SECRET_ENV_KEYS_ENV].split(","))


def test_declared_secret_names_cannot_contain_the_metadata_delimiter():
    """the name list travels to every redactor comma-joined, so a name containing a comma arrives
    as two unrelated names, the real key goes unrecognized, and its value reaches diagnostics
    verbatim. rejecting the delimiter at declaration keeps that channel unambiguous."""
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
    from flash.providers._lifecycle.net.worker import build_worker_env
    from flash.schema.fields import ConfigError, _environment_secrets

    with pytest.raises(ConfigError, match="invalid environment variable name"):
        _environment_secrets(["FOO,BAR"], "grpo")
    # a name-shaped secret is still fine; only the delimiter is refused.
    assert _environment_secrets(["AWS_SECRET_ACCESS_KEY"], "grpo") == ("AWS_SECRET_ACCESS_KEY",)

    # and the metadata builder fails closed rather than emitting an ambiguous list, for a spec
    # constructed around the parser.
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        environment=EnvironmentSpec(id="owner/env", secrets=("FOO,BAR",)),
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )
    with pytest.raises(RuntimeError, match="delimiter"):
        build_worker_env(spec, runtime_secrets={"FOO,BAR": "leaky"})


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

    import flash.providers.runpod.serverless.endpoints as endpoints

    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoints._train_body)))
    # os/re come from _train_body's own local imports, which the handler makes at the top of its
    # body; urllib.parse it imports itself.
    namespace: dict = {"os": os, "re": re}
    for name in ("_percent_pattern", "_needles", "_safe_detail"):
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
    punctuation = {"PIN": ".", "FLASH_SECRET_ENV_KEYS": "PIN"}
    assert safe_detail("module.py: failed at /tmp/a.py", punctuation) == (
        "module.py: failed at /tmp/a.py"
    )
    assert safe_detail("token=.", punctuation) == "token=<redacted>"
    shaped = {
        "KEYED_PIN": ";",
        "BEARER_PIN": "!",
        "FLASH_SECRET_ENV_KEYS": "KEYED_PIN,BEARER_PIN",
    }
    assert safe_detail("token=;", shaped) == "token=<redacted>"
    assert safe_detail("Bearer !", shaped) == "Bearer <redacted>"
    overlapping_shape = {"KEY": "token", "PIN": ";", "FLASH_SECRET_ENV_KEYS": "KEY,PIN"}
    detail = safe_detail("token=;", overlapping_shape)
    assert detail == "<redacted>"
    assert ";" not in detail
    overlapping_shape.update({"KEY": "Bearer", "PIN": "!"})
    detail = safe_detail("Bearer !", overlapping_shape)
    assert detail == "<redacted>"
    assert "!" not in detail
    for secret, encoded in ((".", "%2E"), ("-", "%2D"), ("~", "%7E"), ("/", "%2f")):
        mapping = {"PIN": secret, "FLASH_SECRET_ENV_KEYS": "PIN"}
        assert safe_detail(f"encoded {encoded}", mapping) == "encoded <redacted>"
    literal_case = {"PIN": "A/B", "FLASH_SECRET_ENV_KEYS": "PIN"}
    assert safe_detail("encoded A%2fB", literal_case) == "encoded <redacted>"
    assert safe_detail("encoded a%2fb", literal_case) == "encoded a%2fb"
    for secret, case_variant in (
        ("A%2FB", "A%2fB"),
        ("literal%2Fsecret", "literal%2fsecret"),
    ):
        mapping = {"PIN": secret, "FLASH_SECRET_ENV_KEYS": "PIN"}
        assert safe_detail(f"literal {secret}", mapping) == "literal <redacted>"
        assert safe_detail(f"literal {case_variant}", mapping) == f"literal {case_variant}"
    overlap = {
        "LONG_TOKEN": "a%2Fb%2B",
        "SHORT_TOKEN": "a/b+c&d",
        "FLASH_SECRET_ENV_KEYS": "LONG_TOKEN,SHORT_TOKEN",
    }
    detail = safe_detail("fetch failed for a%2Fb%2Bc%26d", overlap)
    assert detail == "fetch failed for <redacted>"
    for secret in ("a%2Fb%2B", "c%26d", "a/b+c&d"):
        assert secret not in detail


def test_all_worker_redactors_share_the_same_secret_corpus(monkeypatch):
    """the canonical, bootstrap, and source-shipped redactors must stay behaviorally identical."""
    import ast
    import inspect
    import os
    import re
    import textwrap

    import flash.providers.runpod.serverless.endpoints as endpoints
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV, sanitize_diagnostic
    from flash.providers._lifecycle.bootstrapping.secrets import (
        _safe_detail as bootstrap_safe_detail,
    )

    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoints._train_body)))
    namespace: dict = {"os": os, "re": re}
    for name in ("_percent_pattern", "_needles", "_safe_detail"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<handler>", "exec"), namespace)
    handler_safe_detail = namespace["_safe_detail"]

    def canonical_safe_detail(text: str, secrets: dict[str, str]) -> str:
        with monkeypatch.context() as environment:
            environment.setenv(SECRET_ENV_KEYS_ENV, ",".join(secrets))
            for key, secret in secrets.items():
                environment.setenv(key, secret)
            return sanitize_diagnostic(text, limit=1000)

    multiline_secret = (
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----"
    )
    redacted_prefix = "token=<redacted> "
    cases = [
        (
            "comma-delimited-semicolon",
            "token=;, next",
            {"PIN_SECRET": ";"},
            "token=<redacted>, next",
        ),
        (
            "comma-delimited-comma",
            "token=,, next",
            {"PIN_SECRET": ","},
            "token=<redacted>, next",
        ),
        (
            "comma-delimited-bearer",
            "Bearer !, next",
            {"PIN_SECRET": "!"},
            "Bearer <redacted>, next",
        ),
        (
            "semicolon-delimited-keyed",
            "token=;; next",
            {"PIN_SECRET": ";"},
            "token=<redacted>; next",
        ),
        (
            "semicolon-delimited-bearer",
            "Bearer !; next",
            {"PIN_SECRET": "!"},
            "Bearer <redacted>; next",
        ),
        ("non-delimited-keyed", "token=;next", {"PIN_SECRET": ";"}, "token=;next"),
        ("non-delimited-comma", "token=,next", {"PIN_SECRET": ","}, "token=,next"),
        ("non-delimited-bearer", "Bearer !next", {"PIN_SECRET": "!"}, "Bearer !next"),
        ("keyed-punctuation", "token=;", {"PIN_SECRET": ";"}, "token=<redacted>"),
        ("bearer-punctuation", "Bearer !", {"PIN_SECRET": "!"}, "Bearer <redacted>"),
        (
            "ordinary-punctuation",
            "module.py: values a,b; failed at /tmp/a.py",
            {"PIN_SECRET": "."},
            "module.py: values a,b; failed at /tmp/a.py",
        ),
        (
            "encoded-literal-case",
            "literal A%2FB",
            {"PIN_SECRET": "A%2FB"},
            "literal <redacted>",
        ),
        (
            "encoded-literal-case-control",
            "literal A%2fB",
            {"PIN_SECRET": "A%2FB"},
            "literal A%2fB",
        ),
        (
            "encoded-raw-case",
            "encoded A%2fB",
            {"PIN_SECRET": "A/B"},
            "encoded <redacted>",
        ),
        (
            "encoded-raw-case-control",
            "encoded a%2fb",
            {"PIN_SECRET": "A/B"},
            "encoded a%2fb",
        ),
        (
            "multiline-component",
            "ssh auth: MIIEvQIBADANBgkqhkiG9w0BAQEFAASC",
            {"DEPLOY_SECRET": multiline_secret},
            "ssh auth: <redacted>",
        ),
        (
            "long-bounded-output",
            "token=; " + "x" * 1200,
            {"PIN_SECRET": ";"},
            redacted_prefix + "x" * (1000 - len(redacted_prefix)),
        ),
    ]
    for label, secret, encoded in (
        ("dot", ".", "%2E"),
        ("dash", "-", "%2D"),
        ("tilde", "~", "%7E"),
        ("slash", "/", "%2f"),
    ):
        cases.append(
            (
                f"generated-percent-{label}",
                f"encoded {encoded}",
                {"PIN_SECRET": secret},
                "encoded <redacted>",
            )
        )

    for label, text, secrets, expected in cases:
        actual = (
            canonical_safe_detail(text, secrets),
            bootstrap_safe_detail(text, limit=1000, secrets=secrets),
            handler_safe_detail(text, secrets, 1000),
        )
        assert actual == (expected, expected, expected), label


def test_worker_console_always_uploaded_and_no_flag(monkeypatch):
    """The worker console is ALWAYS uploaded — live (periodic) while the worker runs and once more
    when it exits — so every print reaches `flash runs log`, not just a post-mortem tail on
    crash. There is no FLASH_UPLOAD_CONSOLE flag to forget: it is NOT forwarded to the worker (even
    if an operator sets it), and neither worker run_mode path gates the upload."""
    import inspect

    import flash.providers.runpod.serverless.endpoints as endpoints
    from flash.providers._lifecycle.bootstrapping import bootstrap as _instance_bootstrap
    from flash.providers._lifecycle.net.worker import build_worker_env

    # the flag is gone — setting it in the control-plane env does not reach the worker
    monkeypatch.setenv("FLASH_UPLOAD_CONSOLE", "1")
    assert "FLASH_UPLOAD_CONSOLE" not in build_worker_env(_spec())

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
    from flash.providers._lifecycle.net.worker import _REMOVED_OPTIMIZATION_ENV, build_worker_env

    # every removed key, not a chalk special case. FLASH_TRITON_LORA stands in for the rest.
    declared = ["FLASH_CHALK_SPEC", "FLASH_TRITON_LORA", "MY_TOKEN"]
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
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
    env = build_worker_env(spec, runtime_secrets=supplied)

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
    from flash.providers._lifecycle.net.worker import build_worker_env

    # an operator HF_REPO in the env must NOT leak into the worker env
    monkeypatch.setenv("HF_REPO", "operator/default")
    per_run = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="myorg/runs"),
        seed=0,
    )
    assert build_worker_env(per_run)["HF_REPO"] == "myorg/runs"
    # still the per-run value even with no operator HF_REPO at all
    monkeypatch.delenv("HF_REPO", raising=False)
    assert build_worker_env(per_run)["HF_REPO"] == "myorg/runs"


def test_alloc_conf_rl_is_non_expandable(monkeypatch):
    # vLLM sleep-mode CuMemAllocator is incompatible with expandable_segments, so RL ships the
    # sleep-SAFE non-expandable conf; the worker upgrades to expandable_segments at boot once it
    # resolves sleep OFF for the model/context (engine.worker.finalize_alloc_conf_for_sleep). The
    # conf is deterministic — there is no launcher sleep/alloc knob.
    from flash.providers._lifecycle.net.worker import build_worker_env

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    env = build_worker_env(_spec())  # grpo
    assert "expandable_segments" not in env["PYTORCH_ALLOC_CONF"]
    assert env["PYTORCH_ALLOC_CONF"] == env["PYTORCH_CUDA_ALLOC_CONF"]
    # no launcher->worker FLASH_ALLOC_AUTO signal anymore (the worker gates on PHASE == "rl")
    assert "FLASH_ALLOC_AUTO" not in env


def test_alloc_conf_default_expandable_for_sft(monkeypatch):
    from flash.core.spec import JobSpec, TrainSpec
    from flash.providers._lifecycle.net.worker import build_worker_env

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, max_examples=2),
        seed=0,
    )
    env = build_worker_env(spec)
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_runpod_backoff_preserves_strategies_cap_jitter_and_idempotence(monkeypatch):
    pytest.importorskip("runpod_flash")
    from flash.providers.runpod.serverless.endpoints import _patch_runpod_backoff

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
    from flash.engine.worker.io.hf import error_artifact_name

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
    from flash.engine.worker.io.hf import error_artifact_name, ray_log_artifact_name

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
    from flash.engine.worker.io.hf import error_artifact_name as worker_name
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

    import flash.providers.runpod.serverless.endpoints as train

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

    import flash.providers.runpod.serverless.endpoints as train

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


def _extra_pip_input(monkeypatch) -> dict:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_kwargs: "/source.zip")
    monkeypatch.setattr(
        "flash.snapshot.archive.materialize_verified_archive_file",
        lambda *_args: None,
    )
    monkeypatch.setattr("importlib.util.spec_from_file_location", lambda *_args: None)
    return {
        "phase": "sft",
        "seed": 0,
        "hf_repo": "owner/runs",
        "job_spec_json": '{"algorithm": "sft", "run_id": "flash-test-run"}',
        "env": {"GITHUB_TOKEN": "ghp-secret", "PYTHONPATH": ""},
        "extra_pip": ["git+https://github.com/example/some-env-pkg.git@abc123"],
        "source_snapshot": SOURCE_SNAPSHOT,
        "run_id": "flash-test-run",
        "attempt": 0,
        **_run_deadline_fields(),
    }


def test_train_body_source_verification_failure_prevents_pip(monkeypatch):
    import flash.providers.runpod.serverless.endpoints as endpoints
    from flash.snapshot import archive as source_snapshot

    input_data = _extra_pip_input(monkeypatch)
    pip_calls = []
    monkeypatch.setattr(
        source_snapshot,
        "materialize_verified_archive_file",
        lambda *_args: (_ for _ in ()).throw(
            source_snapshot.SourceSnapshotError("source verification failed")
        ),
    )
    monkeypatch.setattr(
        "subprocess.Popen", lambda *args, **kwargs: pip_calls.append((args, kwargs))
    )

    with pytest.raises(source_snapshot.SourceSnapshotError, match="verification"):
        endpoints._train_body(input_data)
    assert pip_calls == []


@pytest.mark.parametrize(
    ("status", "retriable"),
    [(401, False), (403, False), (404, False), (429, True), (500, True), (599, True)],
)
def test_train_body_source_fetch_http_classification(monkeypatch, status, retriable):
    import types

    import huggingface_hub

    import flash.providers.runpod.serverless.endpoints as endpoints

    input_data = _extra_pip_input(monkeypatch)
    input_data["extra_pip"] = []
    calls = []

    class FetchError(RuntimeError):
        pass

    def fail_download(**_kwargs):
        calls.append(status)
        error = FetchError("private upstream response")
        error.response = types.SimpleNamespace(status_code=status, headers={})
        raise error

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail_download)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="pinned flash source") as raised:
        endpoints._train_body(input_data)
    assert getattr(raised.value, "flash_retriable", False) is retriable
    assert len(calls) == (6 if retriable else 1)


def test_train_body_extra_pip_uses_worker_env_credentials(monkeypatch):
    import os
    from pathlib import Path

    import flash.providers.runpod.serverless.endpoints as endpoints

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

    monkeypatch.setenv("GITHUB_TOKEN", "operator-secret")
    monkeypatch.setenv("GIT_ASKPASS", "/tmp/operator-askpass")
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="could not load downloaded module"):
        endpoints._train_body(_extra_pip_input(monkeypatch))

    assert len(calls) == 1
    env = calls[0]["env"]
    assert env["GITHUB_TOKEN"] == "ghp-secret"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert askpass_paths
    assert all(not path.exists() for path in askpass_paths)


def test_train_body_extra_pip_ignores_askpass_cleanup_errors(monkeypatch):
    import os
    from pathlib import Path

    import flash.providers.runpod.serverless.endpoints as endpoints

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
        with pytest.raises(RuntimeError, match="could not load downloaded module"):
            endpoints._train_body(_extra_pip_input(monkeypatch))
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
    import flash.providers.runpod.serverless.endpoints as endpoints

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
    with pytest.raises(RuntimeError, match="could not load downloaded module"):
        endpoints._train_body(_extra_pip_input(monkeypatch))
    assert len(calls) == 2


def test_train_body_extra_pip_resolution_error_stays_terminal(monkeypatch):
    # A bad package spec reached the index fine; retrying it would just burn another attempt.
    import flash.providers.runpod.serverless.endpoints as endpoints

    calls = _wire_train_body_pip(
        monkeypatch,
        [("ERROR: No matching distribution found for definitely-not-a-package\n", 1)],
    )
    with pytest.raises(RuntimeError, match="extra_pip install failed"):
        endpoints._train_body(_extra_pip_input(monkeypatch))
    assert len(calls) == 1  # fails fast, never walks the retry ladder


def test_train_body_extra_pip_build_failure_outranks_earlier_transient_text(monkeypatch):
    # pip warns "Retrying (Retry(" on an index blip, recovers, then fails compiling a wheel. Both
    # lines sit in the same captured tail, so matching transient text alone would call a
    # deterministic failure infra and repeat it three more times for nothing.
    import flash.providers.runpod.serverless.endpoints as endpoints

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
        endpoints._train_body(_extra_pip_input(monkeypatch))
    assert len(calls) == 1  # the build failure names the cause, so no retry ladder


def test_train_body_extra_pip_matches_the_bootstrap_on_git_http_blips(monkeypatch):
    # the two classifiers must agree on what is retriable. A VCS pin fails through git, whose
    # phrasing carries none of the urllib shapes, so a 502 must retry here exactly as it does on
    # the instance bootstrap; a 404 is a bad pin and must still fail fast in both.
    import flash.providers.runpod.serverless.endpoints as endpoints

    blip = (
        "  Running command git clone --filter=blob:none -q https://github.com/org/repo\n"
        "  fatal: unable to access 'https://github.com/org/repo/': "
        "The requested URL returned error: 502\n"
        "  error: subprocess-exited-with-error\n"
    )
    calls = _wire_train_body_pip(
        monkeypatch, [(blip, 1), ("Successfully installed some-env-pkg-1.0\n", 0)]
    )
    with pytest.raises(RuntimeError, match="could not load downloaded module"):
        endpoints._train_body(_extra_pip_input(monkeypatch))
    assert len(calls) == 2

    # git's DNS wording, which urllib never emits, must retry here too.
    dns = blip.replace(
        "The requested URL returned error: 502", "Could not resolve host: github.com"
    )
    calls = _wire_train_body_pip(
        monkeypatch, [(dns, 1), ("Successfully installed some-env-pkg-1.0\n", 0)]
    )
    with pytest.raises(RuntimeError, match="could not load downloaded module"):
        endpoints._train_body(_extra_pip_input(monkeypatch))
    assert len(calls) == 2

    missing = blip.replace("returned error: 502", "returned error: 404")
    calls = _wire_train_body_pip(monkeypatch, [(missing, 1)])
    with pytest.raises(RuntimeError, match="extra_pip install failed"):
        endpoints._train_body(_extra_pip_input(monkeypatch))
    assert len(calls) == 1


def test_train_body_extra_pip_matches_the_bootstrap_on_an_index_outage_footer(monkeypatch):
    # an unreachable index prints the same no-candidate footer a typo'd name does, so that footer
    # alone must not be terminal when the tail also carries a transient marker. A build failure in
    # the same tail still decides it, since pip only reaches one with real content in hand. Both
    # classifiers must agree on this, so the RunPod copy is pinned exactly as the bootstrap is.
    import flash.providers.runpod.serverless.endpoints as endpoints

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
    with pytest.raises(RuntimeError, match="could not load downloaded module"):
        endpoints._train_body(_extra_pip_input(monkeypatch))
    assert len(calls) == 2

    built = outage + "ERROR: Failed building wheel for requests\n"
    calls = _wire_train_body_pip(monkeypatch, [(built, 1)])
    with pytest.raises(RuntimeError, match="extra_pip install failed"):
        endpoints._train_body(_extra_pip_input(monkeypatch))
    assert len(calls) == 1


def test_train_body_extra_pip_stops_after_the_bounded_retries(monkeypatch):
    import flash.providers.runpod.serverless.endpoints as endpoints

    calls = _wire_train_body_pip(monkeypatch, [("read timed out\n", 1)] * 4)
    with pytest.raises(RuntimeError, match="could not reach the package index"):
        endpoints._train_body(_extra_pip_input(monkeypatch))
    assert len(calls) == 4  # one attempt plus the three bounded retries


def test_sft_train_keeps_the_optimizations_that_survived_the_trl_deletion():
    """The verl SFT path must still carry the sizing/memory optimizations, not just train.

    The previous version of this test read run_sft's source. That body was trl's and is deleted:
    run_sft now delegates to run_sft_train. Rather than drop the coverage, assert against the module
    that really runs. Two of the old assertions are intentionally NOT reproduced -- kernel
    installation and the chunked_nll loss_type -- because they were properties of trl's SFTTrainer
    call, and verl owns its own loss and kernel path.

    the optimizations now live in three modules rather than one. dataset preprocessing moved to
    flash.engine.profiling.sft_workload so estimate construction and training share one implementation,
    image-row tokenization lives in flash.engine.profiling.sft_image_rows, sizing and memory choices
    stayed with the trainer and hydra config, and LoRA+ grouping moved to the external child plugin.
    each assertion reads the module that actually owns its behaviour: pointing them all at one module
    would let a symbol disappear from another and still pass.
    """
    import inspect

    from flash.engine.profiling import sft_image_rows, sft_workload
    from flash.engine.worker.entry import sft
    from flash.engine.worker.train.entry import sft_train

    # run_sft is now a pure delegation: no backend selector, no trainer of its own.
    assert "run_sft_train()" in inspect.getsource(sft.run_sft)

    workload_src = inspect.getsource(sft_workload)
    image_rows_src = inspect.getsource(sft_image_rows)
    # completion-only supervision survives, as verl's loss_mask rather than trl's completion_mask.
    assert "_pretokenize_completion_only(" in workload_src
    assert "completion_mask_from_ids(" in image_rows_src
    assert '"loss_mask": tokenized["completion_mask"]' in workload_src

    # sft renders its hydra overrides and child shims in train.sft.config, so the trainer's half of
    # this guard spans both modules. keep these in step when sft_train is split further.
    from flash.engine.worker.train.sft.child import plugin as sft_plugin
    from flash.engine.worker.train.sft.setup import config as sft_config

    train_src = inspect.getsource(sft_train) + inspect.getsource(sft_config)
    plugin_src = inspect.getsource(sft_plugin)
    # revision-aware vocab resolution: the worker must size the realized batch through the SAME
    # resolver the cost quote priced with, else a revision-pinned run drifts from its quote.
    assert "resolve_vocab_size(" in train_src
    assert "vocab_size_for(model_id)" not in train_src
    # per-device micro-batch / grad-accum sizing for the large-vocab logits cap.
    assert "sft_grad_accum(" in train_src
    # gradient checkpointing, with the MoE/GDN reentrant rule shared with grpo.
    assert "grad_checkpointing_on(" in train_src
    assert "grpo_use_reentrant(" in train_src
    # LoRA+ survives in the external child plugin that owns verl's optimizer grouping.
    assert "create_loraplus_optimizer" in plugin_src


@pytest.fixture
def _serialized_sft_console():
    import fcntl

    # both parametrizations exercise the production hardcoded /tmp/console_sft.txt path.
    with open("/tmp/flash-test-sft-console.lock", "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


@pytest.mark.usefixtures("_serialized_sft_console")
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

    import huggingface_hub

    import flash.providers.runpod.serverless.endpoints as endpoints

    monkeypatch.setenv("GITHUB_TOKEN", "operator-secret")
    monkeypatch.setenv("GIT_ASKPASS", "/tmp/operator-askpass")
    run_code = tmp_path / "runcode"
    late_marker = tmp_path / "late-live-attempted"
    real_join = os.path.join

    def mapped_join(*parts):
        joined = real_join(*parts)
        if joined == "/runcode" or joined.startswith("/runcode/"):
            return str(run_code) + joined.removeprefix("/runcode")
        return joined

    monkeypatch.setattr(os.path, "join", mapped_join)
    download_calls = []

    uploads = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def upload_file(self, **kw):
            uploads.append(kw)
            if str(kw.get("path_in_repo", "")).endswith("/console_sft.txt"):
                time.sleep(0.05)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    def fake_hf_hub_download(**kwargs):
        download_calls.append(kwargs)
        return str(tmp_path / "source.zip")

    def materialize(_archive_path, _descriptor, destination):
        target = run_code / "flash-test-run-attempt-7"
        assert destination == str(target)
        console = target / "flash/providers/_lifecycle/bootstrapping/console.py"
        console.parent.mkdir(parents=True, exist_ok=True)
        console.write_text(
            "import threading, time\n"
            "def _run_console_upload_loop(console, interval, stop, *, upload):\n"
            "    upload()\n"
            "    def late():\n"
            "        stop.wait(); time.sleep(0.01); upload()\n"
            f"        open({str(late_marker)!r}, 'w').write('1')\n"
            "    threading.Thread(target=late, daemon=True).start()\n"
            "    stop.wait()\n"
        )
        artifacts = target / "flash/adapters/artifacts.py"
        artifacts.parent.mkdir(parents=True, exist_ok=True)
        artifacts.write_text(
            "def attempt_scoped_artifact_name(kind, phase, attempt):\n"
            "    return f'exact_{kind}_{phase}_attempt{attempt}.txt'\n"
        )

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    def materialization_path(root, run_id, attempt):
        assert root == "/runcode"
        return run_code / f"{run_id}-attempt-{attempt}"

    monkeypatch.setattr("flash.snapshot.archive.attempt_materialization_path", materialization_path)
    monkeypatch.setattr("flash.snapshot.archive.materialize_verified_archive_file", materialize)

    class _FakeProc:
        # Worker boots, logs an OOM, then the kernel/clean-exit leaves NO metrics.json.
        def __init__(self, *a, **k):
            assert k["cwd"] == str(run_code / "flash-test-run-attempt-7")
            assert "GITHUB_TOKEN" not in k["env"]
            assert "GIT_ASKPASS" not in k["env"]
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
        "env": {
            "HF_TOKEN": "tok",
            "GITHUB_TOKEN": "payload-secret",
            "GIT_ASKPASS": "/tmp/payload-askpass",
            "PYTHONPATH": "",
            "ATTEMPT": "999",
        },
        "source_snapshot": SOURCE_SNAPSHOT,
        "run_id": "flash-test-run",
        "attempt": 7,
        **_run_deadline_fields(),
    }

    try:
        with pytest.raises(RuntimeError, match=r"produced no /tmp/metrics\.json"):
            endpoints._train_body(input_data)

        paths = [upload["path_in_repo"] for upload in uploads]
        assert paths == [
            "sft/flash-test-run/exact_console_sft_attempt7.txt",
            "sft/flash-test-run/console_sft.txt",
        ]
        assert late_marker.exists(), (
            "the late live callback must run after terminal teardown begins"
        )
        with open(uploads[-1]["path_or_fileobj"], encoding="utf-8") as f:
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
        assert len(download_calls) == 1
        assert download_calls[0]["filename"] == SOURCE_SNAPSHOT["archive_path"]
        assert download_calls[0]["revision"] == SOURCE_SNAPSHOT["revision"]
    finally:
        # _train_body writes hardcoded console paths; remove them for parallel runs.
        import shutil

        shutil.rmtree(run_code, ignore_errors=True)
        for _p in (
            "/tmp/console_sft.txt",
            "/tmp/console_sft.txt.live.tail",
            "/tmp/console_sft.txt.final.tail",
        ):
            with contextlib.suppress(FileNotFoundError):
                os.remove(_p)


def test_train_body_rejects_malformed_source_descriptor_before_download(monkeypatch):
    import huggingface_hub

    import flash.providers.runpod.serverless.endpoints as endpoints

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **k: pytest.fail("download should not run with an invalid source descriptor"),
    )
    with pytest.raises(RuntimeError, match="descriptor"):
        endpoints._train_body(
            {
                "phase": "sft",
                "seed": 0,
                "hf_repo": "owner/runs",
                "job_spec_json": '{"algorithm": "sft", "run_id": "flash-test-run"}',
                "env": {"HF_TOKEN": "tok"},
                "source_snapshot": {"invalid": True},
                "run_id": "flash-test-run",
                "attempt": 0,
                **_run_deadline_fields(),
            }
        )


def test_live_console_uploads_are_throttled_for_shared_artifact_repos():
    import flash.providers.runpod.serverless.endpoints as endpoints
    from flash.providers._lifecycle.bootstrapping import bootstrap as instance_bootstrap

    assert endpoints._CONSOLE_UPLOAD_INTERVAL_S == 3600.0
    assert instance_bootstrap._CONSOLE_UPLOAD_INTERVAL_S == 3600.0
    steady_state_commits_per_hour = (
        3600.0 / worker_heartbeat._HB_MIN_INTERVAL_S + 3600.0 / endpoints._CONSOLE_UPLOAD_INTERVAL_S
    )
    assert steady_state_commits_per_hour <= 5.0


def test_first_console_snapshot_precedes_stall_teardown():
    import importlib
    import inspect

    from flash.providers._lifecycle.bootstrapping import console as bootstrap_console

    importlib.import_module("flash.providers.runpod.execution.jobs")
    poll_job = importlib.import_module("flash.providers.runpod.execution.polling").poll_job
    defaults = inspect.signature(poll_job).parameters
    training_stall_s = defaults["stall_after_s"].default
    setup_grace_s = defaults["setup_grace_s"].default

    # the serverless handler loads this exact module rather than shipping its own copy, so these
    # constants have one home and both providers are bound by the same margin.
    assert training_stall_s > bootstrap_console._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    assert setup_grace_s > bootstrap_console._CONSOLE_UPLOAD_FIRST_SNAPSHOT_S
    assert training_stall_s > 2 * bootstrap_console._CONSOLE_UPLOAD_POLL_S


def test_console_heartbeat_stays_flat_so_the_scanner_can_match_on_substrings(tmp_path):
    """the console scanner matches marker keys as raw substrings, which is only sound because the
    producer compacts every nested field away first.

    a nested ``{"pending": ...}`` anywhere in the line would read as an uncommitted beat and make a
    healthy run look wedged. that cannot happen while the producer replaces list values with counts,
    so this pins the producer side of that contract rather than the scanner's.
    """
    import json

    from flash.engine.worker.io.heartbeat import _console_heartbeat_snapshot
    from flash.providers._lifecycle.bootstrapping import console as bootstrap_console

    snapshot = _console_heartbeat_snapshot(
        {
            "stage": "rl_step",
            "step": 3,
            "sampled_completions": [{"completion": "x" * 100_000, "pending": True}],
            "metrics_last": [{"throttled": True}],
        }
    )
    assert '"samples_count": 1' in snapshot
    assert '"metrics_last_count": 1' in snapshot
    assert "sampled_completions" not in snapshot
    assert "metrics_last" not in snapshot.replace('"metrics_last_count"', "")
    # no nested container survives, so no marker substring can come from anything but a real marker.
    assert not any(isinstance(value, (dict, list)) for value in json.loads(snapshot).values())

    console = tmp_path / "console_large_managed_heartbeat.txt"
    console.write_bytes(f"HEARTBEAT {snapshot}\n".encode())
    assert bootstrap_console._console_progress(str(console), 0)[2:] == (1, 1)


def test_min_cuda_for_uses_the_gpu_class_floor():
    # the CUDA floor is a property of the GPU class, not of the image tag
    from flash.providers.runpod.serverless.endpoints import min_cuda_for

    assert min_cuda_for("B200") == "13.0"  # blackwell needs cu13 drivers
    assert min_cuda_for("H200") == "12.8"


def test_apply_disk_raises_to_the_requested_floor():
    from types import SimpleNamespace

    from flash.providers.runpod.execution.job_execution import apply_disk_gb

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
