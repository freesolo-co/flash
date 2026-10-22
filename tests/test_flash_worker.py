"""Regression tests for the Flash worker plumbing fixed in this PR.

Covers:
- build_worker_env sets managed worker defaults without forwarding removed tuning knobs;
- the another CLI package backoff OverflowError that aborted long runs is patched;
- per-phase error artifact names don't collide (train error survives a later eval error).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

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


def test_build_worker_env_lists_declared_secret_names_for_the_redactors(
    worker_context, monkeypatch, tmp_path
):
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
    child = _build_verl_child_env(wandb_enabled=True, context=worker_context)

    for name in declared:
        if name != "PYTHONPATH":
            assert name not in child
    assert child["PYTHONPATH"] == str(Path(__file__).resolve().parents[1])
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
    for variant in (SECRET_ENV_KEYS_ENV.lower(), "Hf_Token", "modal_token_secret"):
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
    stay with the trainer and hydra config; the external child plugin owns the runtime patches.
    each assertion reads the module that actually owns its behaviour: pointing them all at one module
    would let a symbol disappear from another and still pass.
    """
    import inspect

    from flash.engine.profiling import sft_image_rows, sft_workload
    from flash.engine.worker.entry import worker
    from flash.engine.worker.train.entry import sft_train

    # Dispatch goes straight to the trainer; preparation has no worker entrypoint dependency.
    assert "partial(sft_entry.run_sft_train, context=context)" in inspect.getsource(
        worker._run_worker_mode
    )

    workload_src = inspect.getsource(sft_workload)
    image_rows_src = inspect.getsource(sft_image_rows)
    # completion-only supervision survives, as verl's loss_mask rather than trl's completion_mask.
    assert "_pretokenize_completion_only(" in workload_src
    assert "completion_mask_from_ids(" in image_rows_src
    assert '"loss_mask": tokenized["completion_mask"]' in workload_src

    # sft renders its hydra overrides and child shims in train.sft.config, so the trainer's half of
    # this guard spans both modules. keep these in step when sft_train is split further.
    from flash.engine.worker.train.core.child import options as runtime_options
    from flash.engine.worker.train.sft.setup import config as sft_config

    train_src = inspect.getsource(sft_train) + inspect.getsource(sft_config)
    engine_src = Path(runtime_options.__file__).with_name("engine.py").read_text()
    # revision-aware vocab resolution: the worker must size the realized batch through the SAME
    # resolver the cost quote priced with, else a revision-pinned run drifts from its quote.
    assert "resolve_vocab_size(" in train_src
    assert "vocab_size_for(model_id)" not in train_src
    # per-device micro-batch / grad-accum sizing for the large-vocab logits cap.
    assert "sft_grad_accum(" in train_src
    # gradient checkpointing, with the MoE/GDN reentrant rule shared with grpo.
    assert "grad_checkpointing_on(" in train_src
    assert "grpo_use_reentrant(" in train_src
    assert "get_linear_schedule_with_warmup" in engine_src
    assert "optimizer = super()._build_optimizer(module)" in engine_src


@pytest.fixture
def _serialized_sft_console():
    import fcntl

    # both parametrizations exercise the production hardcoded /tmp/console_sft.txt path.
    with open("/tmp/flash-test-sft-console.lock", "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


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


def test_snapshot_weight_validation(tmp_path):
    from flash.engine.worker.io.hf import _snapshot_has_weights

    d = tmp_path / "snap"
    d.mkdir()
    (d / "config.json").write_text("{}")
    assert not _snapshot_has_weights(str(d))  # configs only = stale partial snapshot
    (d / "model.safetensors-00001-of-00001.safetensors").write_text("x")
    assert _snapshot_has_weights(str(d))
