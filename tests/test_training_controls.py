from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

import flash.engine.worker.io.heartbeat as worker_heartbeat
import flash.engine.worker.io.hf as worker_hf
import flash.engine.worker.runtime.state as worker_state
from flash.core.grpo import GRPO_NATIVE_THREAD_ENV
from flash.core.spec import FIXED_SEED, EnvironmentSpec, JobSpec, TrainSpec
from flash.engine.plan.steps import (
    final_save_due,
    resolve_update_horizon,
    sft_update_steps,
    validate_save_steps,
)
from flash.schema import ConfigError, spec_and_train_keys_from_file, spec_from_dict

_BASE_TOML = """
model = "Qwen/Qwen3.5-9B"
algorithm = "grpo"

[environment]
id = "freesolo/example-project/gsm8k"

[train]
epochs = 1
max_examples = 8
group_size = 2
"""


def test_toml_seed_defaults_to_42_and_round_trips(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text(_BASE_TOML)

    spec = spec_and_train_keys_from_file(str(path), run_id="seed-default")[0]

    assert spec.seed == FIXED_SEED == 42
    assert JobSpec.from_json(spec.to_json()).seed == 42
    assert spec.to_dict()["seed"] == 42


def test_toml_seed_and_save_at_steps_parse(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text(
        "seed = 987654321\n" + _BASE_TOML + "max_steps = 12\nsave_at_steps = [1, 4, 12]\n"
    )

    spec = spec_and_train_keys_from_file(str(path), run_id="seed-explicit")[0]

    assert spec.seed == 987654321
    assert spec.train.max_steps == 12
    assert spec.train.save_at_steps == (1, 4, 12)
    assert JobSpec.from_json(spec.to_json()) == spec


@pytest.mark.parametrize("value", [True, -1, 2**63, 1.5, "7"])
def test_jobspec_seed_validation(value):
    with pytest.raises((TypeError, ValueError), match="seed"):
        JobSpec(seed=value)


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_max_steps_rejects_non_integer_types_across_direct_and_json_construction(value):
    with pytest.raises((TypeError, ValueError), match=r"train\.max_steps"):
        TrainSpec(max_steps=value)
    with pytest.raises((TypeError, ValueError), match=r"train\.max_steps"):
        JobSpec.from_dict({"train": {"max_steps": value}})


def test_max_steps_normalizes_non_positive_to_derived_fallback():
    # positive is authoritative; absent or non-positive (zero, negative) canonicalizes to none so one
    # sentinel means "use the derived update count" (the user contract: non-positive keeps fallback).
    assert TrainSpec(max_steps=None).max_steps is None
    assert TrainSpec(max_steps=0).max_steps is None
    assert TrainSpec(max_steps=-4).max_steps is None
    assert TrainSpec(max_steps=3).max_steps == 3


@pytest.mark.parametrize("toml_value", ["true", "1.0", '"1"'])
def test_toml_max_steps_rejects_non_integer_types(tmp_path, toml_value):
    path = tmp_path / "invalid-max-steps.toml"
    path.write_text(_BASE_TOML + f"max_steps = {toml_value}\n")

    with pytest.raises(ConfigError, match=r"train\.max_steps"):
        spec_and_train_keys_from_file(str(path))


def test_toml_max_steps_non_positive_uses_derived_fallback(tmp_path):
    # a negative max_steps is not an error: it keeps the derived horizon (canonicalized to null).
    path = tmp_path / "neg-max-steps.toml"
    path.write_text(_BASE_TOML + "max_steps = -3\n")

    assert spec_and_train_keys_from_file(str(path))[0].train.max_steps is None


@pytest.mark.parametrize(
    ("required_saves", "match"),
    [
        ("1", "list of integers"),
        ("[true]", "entries must be integers"),
        ("[1.0]", "entries must be integers"),
        ("[0]", "entries must be positive"),
        ("[2, 1]", "strictly increasing"),
        ("[1, 1]", "strictly increasing"),
    ],
)
def test_toml_save_at_steps_validation(tmp_path, required_saves, match):
    path = tmp_path / "invalid.toml"
    path.write_text(_BASE_TOML + f"save_at_steps = {required_saves}\n")

    with pytest.raises(ConfigError, match=match):
        spec_and_train_keys_from_file(str(path))


def test_save_at_steps_reject_steps_beyond_positive_max_steps(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text(_BASE_TOML + "max_steps = 5\nsave_at_steps = [2, 6]\n")

    with pytest.raises(ConfigError, match=r"beyond train\.max_steps"):
        spec_and_train_keys_from_file(str(path))

    with pytest.raises(ValueError, match=r"beyond train\.max_steps"):
        TrainSpec(max_steps=5, save_at_steps=(2, 6))

    assert TrainSpec(max_steps=5).save_at_steps == ()
    # a non-positive max_steps is not a positive horizon, so exact saves alongside it are rejected.
    with pytest.raises(ValueError, match=r"requires positive train\.max_steps"):
        TrainSpec(max_steps=-1, save_at_steps=(2,))


def test_save_at_steps_require_explicit_positive_max_steps(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text(_BASE_TOML + "save_at_steps = [2]\n")

    with pytest.raises(ConfigError, match=r"requires positive train\.max_steps"):
        spec_and_train_keys_from_file(str(path))

    with pytest.raises(ValueError, match=r"requires positive train\.max_steps"):
        TrainSpec(save_at_steps=(2,))


def test_authoritative_update_horizon_and_fallback_behavior():
    assert resolve_update_horizon(7, 11) == 11
    assert resolve_update_horizon(7, 1) == 1
    assert resolve_update_horizon(7, 0) == 7
    assert resolve_update_horizon(7, -3) == 7
    assert resolve_update_horizon(7, None) == 7


def test_sft_update_horizon_uses_packed_blocks_when_bfd_is_active():
    assert (
        sft_update_steps(
            epochs=1,
            example_count=100,
            examples_per_update=4,
            packed_block_count=25,
        )
        == 7
    )


def test_final_save_is_kept_unless_exact_save_steps_exclude_it():
    assert final_save_due(9, ())
    assert not final_save_due(9, (2, 7))


def test_save_step_validation_rejects_unreachable_runtime_step():
    validate_save_steps((2, 7), 7)
    with pytest.raises(ValueError, match="exceeds the 6-update horizon"):
        validate_save_steps((2, 7), 6)


@pytest.mark.parametrize("interval", [0, -1, -20])
def test_nonpositive_save_every_is_rejected(interval):
    with pytest.raises(ValueError, match=r"train\.save_every must be positive"):
        TrainSpec(save_every=interval)


def test_positive_save_every_is_preserved():
    assert TrainSpec(save_every=20).save_every == 20


def test_resume_first_companion_retries_without_reuploading_full_state(monkeypatch, tmp_path):

    calls = {"resume": 0, "deployable": 0}

    class Api:
        def upload_folder(self, **kwargs):
            assert kwargs["path_in_repo"].endswith("/checkpoint/checkpoint-4")
            calls["resume"] += 1

        def list_repo_files(self, **_kwargs):
            return []

    def publish_once_then_succeed():
        calls["deployable"] += 1
        if calls["deployable"] == 1:
            raise ConnectionError("hf deployable upload unavailable")

    monkeypatch.setattr(worker_state, "HF_REPO", "org/runs")
    monkeypatch.setattr(worker_hf, "hf_api", lambda: Api())
    monkeypatch.setattr(worker_heartbeat, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_hf, "_CKPT_UPLOAD_BACKOFF_S", 0.0)

    assert worker_hf.upload_resume_checkpoint(
        4, str(tmp_path), after_upload=publish_once_then_succeed
    )
    assert calls == {"resume": 1, "deployable": 2}


def test_resume_checkpoint_failure_heartbeat_surfaces_sanitized_error(monkeypatch, tmp_path):

    secret = "hf_checkpoint_upload_secret"
    calls = 0
    heartbeats: list[tuple[str, dict]] = []

    class Api:
        def upload_folder(self, **_kwargs):
            nonlocal calls
            calls += 1
            raise PermissionError(f"hf quota refused credential {secret} " + "x" * 400)

    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setattr(worker_state, "HF_REPO", "org/runs")
    monkeypatch.setattr(worker_hf, "hf_api", lambda: Api())
    monkeypatch.setattr(
        worker_heartbeat, "heartbeat", lambda stage, **kwargs: heartbeats.append((stage, kwargs))
    )
    monkeypatch.setattr(worker_hf, "_CKPT_UPLOAD_BACKOFF_S", 0.0)

    assert not worker_hf.upload_resume_checkpoint(50, str(tmp_path))

    failed = [fields for stage, fields in heartbeats if stage == "checkpoint_upload_failed"]
    assert calls == worker_hf._CKPT_UPLOAD_RETRIES
    assert len(failed) == 1
    assert failed[0]["step"] == 50
    failure = failed[0]["checkpoint_failure"]
    assert failure["step"] == 50
    assert failure["operation"] == "resume"
    assert failure["error"].startswith("PermissionError: hf quota refused")
    assert "<redacted>" in failure["error"]
    assert secret not in failure["error"]
    assert len(failure["error"]) <= 300


@pytest.mark.parametrize("failure_stage", ["before", "after"])
def test_resume_checkpoint_companion_failure_still_raises(monkeypatch, tmp_path, failure_stage):

    calls = {"resume": 0, "companion": 0}
    heartbeats: list[tuple[str, dict]] = []

    class Api:
        def upload_folder(self, **_kwargs):
            calls["resume"] += 1

        def list_repo_files(self, **_kwargs):
            return []

    def fail_companion():
        calls["companion"] += 1
        raise RuntimeError(f"{failure_stage} companion failed")

    callbacks = {f"{failure_stage}_upload": fail_companion}
    monkeypatch.setattr(worker_state, "HF_REPO", "org/runs")
    monkeypatch.setattr(worker_hf, "hf_api", lambda: Api())
    monkeypatch.setattr(
        worker_heartbeat, "heartbeat", lambda stage, **kwargs: heartbeats.append((stage, kwargs))
    )
    monkeypatch.setattr(worker_hf, "_CKPT_UPLOAD_BACKOFF_S", 0.0)

    with pytest.raises(RuntimeError, match=f"{failure_stage} companion failed"):
        worker_hf.upload_resume_checkpoint(50, str(tmp_path), **callbacks)

    failed = [fields for stage, fields in heartbeats if stage == "checkpoint_upload_failed"]
    assert calls["companion"] == worker_hf._CKPT_UPLOAD_RETRIES
    assert calls["resume"] == (0 if failure_stage == "before" else 1)
    assert len(failed) == 1
    failure = failed[0]["checkpoint_failure"]
    assert failure["step"] == 50
    assert failure["operation"] == failure_stage
    assert f"{failure_stage} companion failed" in failure["error"]


def test_upload_failure_cause_reaches_the_rendered_run_log():
    """The payload is not the deliverable: `get_logs()` shows only what the formatter renders.

    The formatter emits a fixed list of numeric metric keys, so a stage whose entire content is an
    explanation rather than a measurement committed the cause and printed `stage=... step=50` alone.
    """
    from flash.providers._lifecycle.instances.poll import _format_heartbeat

    line = _format_heartbeat(
        {
            "stage": "checkpoint_upload_failed",
            "step": 50,
            "checkpoint_failure": {
                "step": 50,
                "operation": "resume",
                "error": "PermissionError: hf quota refused credential <redacted>",
            },
        }
    )

    assert "stage=checkpoint_upload_failed" in line
    assert "step=50" in line
    assert "checkpoint_failure_step=50" in line
    assert "checkpoint_failure_stage=resume" in line
    assert "checkpoint_error=PermissionError: hf quota refused credential <redacted>" in line


def test_a_failure_cause_cannot_rewrite_the_log_it_is_printed_in():
    """This text is an exception message, so it is not ours: a hook or provider response wrote it.

    The same line already neutralizes child output and sampled completions. Surfacing the cause
    raw would let a `\\x1b[2J` clear the screen or a `\\r` overwrite the line the user is reading
    the failure in -- and the failure report is exactly when the log has to stay legible.
    """
    from flash.providers._lifecycle.instances.poll import _format_heartbeat

    line = _format_heartbeat(
        {
            "stage": "checkpoint_upload_failed",
            "step": 50,
            "checkpoint_failure": {
                "step": 50,
                "operation": "resume",
                "error": "boom\x1b[2Jwiped\roverwrite",
            },
        }
    )

    assert "\x1b" not in line
    assert "\r" not in line
    assert "checkpoint_error=boom\\x1b[2Jwiped\\x0doverwrite" in line


def test_a_heartbeat_without_a_failure_cause_is_unchanged():
    """The failure fields are additive: an ordinary step line must not grow an empty `error=`."""
    from flash.providers._lifecycle.instances.poll import _format_heartbeat

    line = _format_heartbeat(
        {"stage": "sft_step", "step": 12, "error": "   ", "checkpoint_failure": {}}
    )

    assert line == "worker: stage=sft_step step=12"


def test_an_upload_failure_is_critical_so_an_in_flight_commit_cannot_drop_it():
    """This heartbeat is raised from inside a long HF upload -- when a ping is otherwise dropped.

    `_HB_UPLOAD_IN_FLIGHT` is set for the duration of that upload, and an unforced non-critical
    stage returns not-due immediately. The failure is reported once and nothing restates it, so the
    one case that produces this heartbeat is the case that would discard it.
    """
    from flash.engine.worker.io.heartbeat import _is_critical_stage, _is_terminal_stage

    assert _is_critical_stage("checkpoint_upload_failed")
    # criticality and checkpoint carry use one terminal predicate.
    assert _is_terminal_stage("done")
    assert _is_terminal_stage("error_oom")
    assert not _is_terminal_stage("checkpoint_upload_failed")
    assert _is_critical_stage("done")
    assert _is_critical_stage("error_oom")
    assert not _is_critical_stage("sft_step")
    assert not _is_critical_stage("rl_step")


def test_seed_training_rngs_initializes_all_supported_generators(monkeypatch):
    from flash.engine.worker.runtime import rng

    calls: list[tuple[str, int]] = []
    fake_numpy = SimpleNamespace(
        random=SimpleNamespace(seed=lambda value: calls.append(("numpy", value)))
    )
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda value: calls.append(("cuda", value)),
    )
    fake_torch = SimpleNamespace(
        manual_seed=lambda value: calls.append(("torch", value)),
        cuda=fake_cuda,
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(rng.random, "seed", lambda value: calls.append(("python", value)))

    full_seed = 2**32 + 7
    rng.seed_training_rngs(full_seed)

    assert calls == [
        ("python", full_seed),
        ("numpy", 7),
        ("torch", full_seed),
        ("cuda", full_seed),
    ]
    assert rng.backend_seed(full_seed) == 7


def test_seed_host_rngs_reaches_the_dataset_generators_without_importing_torch(monkeypatch):
    """host seeding reaches python and numpy without importing the model stack.

    environment code may consume python's or numpy's global generator while building training rows.
    torch is the opposite case: importing it here would make a host-only helper load the model stack.
    """
    from flash.engine.worker.runtime import rng

    calls: list[tuple[str, int]] = []
    fake_numpy = SimpleNamespace(
        random=SimpleNamespace(seed=lambda value: calls.append(("numpy", value)))
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(rng.random, "seed", lambda value: calls.append(("python", value)))

    # any import of torch from here on is the defect, whether or not it is installed.
    real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("seed_host_rngs imported torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded)

    full_seed = 2**32 + 7
    rng.seed_host_rngs(full_seed)

    assert calls == [("python", full_seed), ("numpy", 7)]


def test_worker_seed_prefers_jobspec_when_present():
    from flash.engine.worker.runtime.state import _resolve_worker_seed

    assert _resolve_worker_seed(SimpleNamespace(seed=123), "999") == 123
    assert _resolve_worker_seed(None, "999") == 999
    assert _resolve_worker_seed(None, None) == 42
    assert _resolve_worker_seed(None, "invalid") == 42
    assert _resolve_worker_seed(None, "-1") == 42


def test_provider_worker_env_emits_authoritative_spec_seed():
    """The spec is the only seed channel, so no caller can supply a second one to disagree with."""
    import inspect

    from flash.providers._lifecycle.net.worker import build_worker_env

    spec = JobSpec(model="model", seed=987)
    assert build_worker_env(spec)["SEED"] == "987"
    assert "seed" not in inspect.signature(build_worker_env).parameters


def test_every_attempt_of_a_run_shares_the_one_authoritative_seed():
    """A run owns one seed; an attempt is a fresh host for that same run, never a new identity."""
    from flash.providers._lifecycle.net.worker import build_worker_env

    spec = JobSpec(model="model", seed=987)
    seeds = {build_worker_env(spec)["SEED"] for _ in range(3)}
    assert seeds == {"987"}


def test_sft_under_ran_only_fails_a_genuine_under_run():
    from flash.engine.worker.entry.sft import sft_under_ran

    # a real under-run (fewer updates than the quoted horizon) fails loudly.
    assert sft_under_ran(9, 10)
    # a fresh run landed exactly on the horizon passes.
    assert not sft_under_ran(10, 10)
    # a resume from a checkpoint past a lowered horizon did zero new steps yet is fully trained.
    assert not sft_under_ran(12, 10)


def test_grpo_under_ran_only_fails_a_genuine_under_run():
    # verl has no grpo_under_ran helper: run_rl_train compares the checkpoint dir's step count to
    # expected_steps inline. same invariant, so assert on the comparison rather than a symbol --
    # STRICTLY less-than, so a resume that overshoots a lowered horizon (12 >= 10) still finalizes
    # instead of failing a fully-trained policy.
    import ast
    import inspect
    import textwrap

    from flash.engine.worker.train.entry import rl_train

    tree = ast.parse(textwrap.dedent(inspect.getsource(rl_train.run_rl_train)))
    compares = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "steps_run"
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == "expected_steps"
    ]
    assert len(compares) == 1
    assert isinstance(compares[0].ops[0], ast.Lt)


@pytest.mark.parametrize("bad", [[], False, "", 0])
def test_from_dict_rejects_falsy_non_object_train(bad):
    # strict schema: an explicit non-object train must raise, not silently become an empty table.
    with pytest.raises(TypeError, match="train must be an object"):
        JobSpec.from_dict({"train": bad})


@pytest.mark.parametrize("payload", [{}, {"train": None}, {"train": {}}])
def test_from_dict_defaults_missing_credit_assignment(payload):
    assert JobSpec.from_dict(payload).train.credit_assignment == "per_episode"


def test_from_dict_rejects_removed_legacy_train_seeds():
    with pytest.raises(ValueError, match=r"train has unknown key\(s\): seeds"):
        JobSpec.from_dict({"train": {"seeds": [1, 2]}})


def test_from_dict_rejects_misspelled_train_key():
    with pytest.raises(ValueError, match=r"train has unknown key\(s\): max_step"):
        JobSpec.from_dict({"train": {"max_step": 10}})


def test_runtime_secret_cannot_override_control_plane_seed():
    from flash.providers._lifecycle.net.worker import build_worker_env

    # a directly-constructed spec can declare a seed env secret (the toml schema rejects it, but
    # json/direct construction does not). the built worker env must still hold the canonical seed.
    spec = JobSpec(model="m", seed=987, environment=EnvironmentSpec(id="e", secrets=("SEED",)))
    env = build_worker_env(spec, runtime_secrets={"SEED": "7"})
    assert env["SEED"] == "987"


def test_provider_worker_env_carries_control_plane_resume_revision():
    from flash.providers._lifecycle.net.worker import build_worker_env
    from flash.teacher.retry_contract import OPD_RESUME_REVISION_ENV

    spec = JobSpec(model="m", algorithm="opd", seed=987)
    env = build_worker_env(
        spec,
        runtime_secrets={
            OPD_RESUME_REVISION_ENV: "a" * 40,
            "FLASH_PUBLIC_URL": "https://broker.example",
            "FLASH_TEACHER_CAPABILITY": "capability-test-value",
        },
    )
    assert env[OPD_RESUME_REVISION_ENV] == "a" * 40


def test_toml_environment_secrets_reject_control_plane_seed(tmp_path):
    path = tmp_path / "seed-secret.toml"
    path.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "freesolo/example-project/gsm8k"\n'
        'secrets = ["SEED"]\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 8\n"
        "group_size = 2\n"
    )
    with pytest.raises(ConfigError, match="platform-managed key"):
        spec_and_train_keys_from_file(str(path))


def _spec_with_declared_secret(algorithm: str, secret: str) -> dict:
    return {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": algorithm,
        "environment": {"id": "owner/project/env", "secrets": [secret]},
        "train": {"epochs": 1},
    }


@pytest.mark.parametrize("secret", GRPO_NATIVE_THREAD_ENV)
@pytest.mark.parametrize("algorithm", ["sft", "opd"])
def test_non_grpo_authoring_accepts_grpo_native_thread_secrets(algorithm, secret):
    spec = spec_from_dict(_spec_with_declared_secret(algorithm, secret))

    assert spec.environment.secrets == (secret,)


@pytest.mark.parametrize("secret", GRPO_NATIVE_THREAD_ENV)
def test_grpo_authoring_rejects_native_thread_secrets(secret):
    with pytest.raises(
        ConfigError,
        match=rf"\[environment\] secrets must not include platform-managed key\(s\): {secret}",
    ):
        spec_from_dict(_spec_with_declared_secret("grpo", secret))


@pytest.mark.parametrize("algorithm", ["sft", "opd", "grpo"])
def test_all_algorithms_reject_unconditional_control_plane_secrets(algorithm):
    with pytest.raises(
        ConfigError,
        match=r"\[environment\] secrets must not include platform-managed key\(s\): SEED",
    ):
        spec_from_dict(_spec_with_declared_secret(algorithm, "SEED"))
