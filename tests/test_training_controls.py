from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from flash.engine.steps import (
    configure_trainer_save_schedule,
    final_save_due,
    resolve_update_horizon,
    save_step_due,
    sft_update_steps,
    validate_save_steps,
)
from flash.schema import ConfigError, spec_from_file
from flash.spec import FIXED_SEED, EnvironmentSpec, JobSpec, TrainSpec

_BASE_TOML = """
model = "Qwen/Qwen3.5-0.8B"
algorithm = "grpo"

[environment]
id = "freesolo/gsm8k"

[train]
epochs = 1
max_examples = 8
group_size = 2
"""


def test_toml_seed_defaults_to_42_and_round_trips(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text(_BASE_TOML)

    spec = spec_from_file(str(path), run_id="seed-default")

    assert spec.seed == FIXED_SEED == 42
    assert JobSpec.from_json(spec.to_json()).seed == 42
    assert spec.to_dict()["seed"] == 42


def test_toml_seed_and_save_at_steps_parse(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text(
        "seed = 987654321\n" + _BASE_TOML + "max_steps = 12\nsave_at_steps = [1, 4, 12]\n"
    )

    spec = spec_from_file(str(path), run_id="seed-explicit")

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
        spec_from_file(str(path))


def test_toml_max_steps_non_positive_uses_derived_fallback(tmp_path):
    # a negative max_steps is not an error: it keeps the derived horizon (canonicalized to null).
    path = tmp_path / "neg-max-steps.toml"
    path.write_text(_BASE_TOML + "max_steps = -3\n")

    assert spec_from_file(str(path)).train.max_steps is None


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
        spec_from_file(str(path))


def test_save_at_steps_reject_steps_beyond_positive_max_steps(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text(_BASE_TOML + "max_steps = 5\nsave_at_steps = [2, 6]\n")

    with pytest.raises(ConfigError, match=r"beyond train\.max_steps"):
        spec_from_file(str(path))

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
        spec_from_file(str(path))

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


def test_exact_required_saves_suppress_periodic_trainer_saves():
    config = {"save_steps": 5}
    configure_trainer_save_schedule(config, (2, 7))
    assert config == {"save_steps": 5, "save_strategy": "no"}

    fallback = {"save_steps": 5}
    configure_trainer_save_schedule(fallback, ())
    assert fallback == {"save_steps": 5}


def test_opd_checkpoint_schedule_uses_required_saves_or_periodic_fallback():
    assert save_step_due(2, (2, 7), 5)
    assert save_step_due(7, (2, 7), 5)
    assert not save_step_due(5, (2, 7), 5)
    assert save_step_due(5, (), 5)
    assert not save_step_due(4, (), 5)
    assert final_save_due(9, ())
    assert not final_save_due(9, (2, 7))


def test_save_step_validation_rejects_unreachable_runtime_step():
    validate_save_steps((2, 7), 7)
    with pytest.raises(ValueError, match="exceeds the 6-update horizon"):
        validate_save_steps((2, 7), 6)


def test_checkpoint_upload_callback_requests_only_exact_save_steps(monkeypatch):
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from flash.engine.worker.hf import make_checkpoint_upload_callback

    callback = make_checkpoint_upload_callback((2, 7))
    for step, expected in ((1, False), (2, True), (3, False), (7, True)):
        control = SimpleNamespace(should_save=False)
        returned = callback.on_step_end(None, SimpleNamespace(global_step=step), control)
        assert returned is control
        assert control.should_save is expected


def test_required_save_fails_when_trainer_checkpoint_is_missing(monkeypatch, tmp_path):
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    callback = worker.make_checkpoint_upload_callback((4,))

    with pytest.raises(RuntimeError, match="no trainer checkpoint directory"):
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            SimpleNamespace(global_step=4),
            SimpleNamespace(),
        )


def test_permanent_required_save_failure_is_not_retried(monkeypatch, tmp_path):
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import flash.engine.worker as worker
    from flash.engine.worker import hf as worker_hf

    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    calls = 0

    def fail_permanently(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise worker_hf.RequiredSaveError("missing deployable adapter")

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    monkeypatch.setattr(worker_hf, "publish_deployable_checkpoint", fail_permanently)
    callback = worker.make_checkpoint_upload_callback((4,))

    with pytest.raises(worker_hf.RequiredSaveError, match="missing deployable adapter"):
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            SimpleNamespace(global_step=4),
            SimpleNamespace(),
        )

    assert calls == 1


def test_resume_upload_failure_preserves_published_required_save(monkeypatch, tmp_path):
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import flash.engine.worker as worker
    from flash.engine.worker import hf as worker_hf

    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    calls = {"deployable": 0, "resume": 0}

    class ResumeFailingApi:
        def upload_folder(self, **kwargs):
            if kwargs["path_in_repo"].endswith("/checkpoints/step-4/adapter"):
                calls["deployable"] += 1
                return
            calls["resume"] += 1
            raise RuntimeError("resume upload unavailable")

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    monkeypatch.setattr(worker, "hf_api", lambda: ResumeFailingApi())
    monkeypatch.setattr(worker, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_hf, "_CKPT_UPLOAD_BACKOFF_S", 0.0)
    callback = worker.make_checkpoint_upload_callback((4,))

    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=4),
        SimpleNamespace(),
    )

    assert calls == {"deployable": 1, "resume": 3}


def test_resumed_trainer_credits_required_saves_verified_on_hf(monkeypatch, tmp_path):
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    # both required saves are durably present on hf, so resume credits them and completeness passes.
    monkeypatch.setattr(worker, "hf_api", lambda: SimpleNamespace(file_exists=lambda **k: True))
    callback = worker.make_checkpoint_upload_callback((10, 20))
    control = SimpleNamespace()
    state = SimpleNamespace(global_step=20)

    assert callback.on_train_begin(SimpleNamespace(), state, control) is control
    callback.on_train_end(SimpleNamespace(output_dir=str(tmp_path)), state, control)


def test_resumed_required_save_missing_on_hf_fails_completeness(monkeypatch, tmp_path):
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    # the pre-resume worker advanced past step 10 but never durably published its deployable, so
    # resume must not credit step 10 from the restored counter alone: on_train_end fails loudly.
    present = {20}
    monkeypatch.setattr(
        worker,
        "hf_api",
        lambda: SimpleNamespace(
            file_exists=lambda **k: any(f"step-{s}/" in k["filename"] for s in present)
        ),
    )
    callback = worker.make_checkpoint_upload_callback((10, 20))
    control = SimpleNamespace()
    state = SimpleNamespace(global_step=20)

    callback.on_train_begin(SimpleNamespace(), state, control)
    with pytest.raises(RuntimeError, match="required saves were not durably published"):
        callback.on_train_end(SimpleNamespace(output_dir=str(tmp_path)), state, control)


def test_resumed_required_save_hf_lookup_outage_is_retriable(monkeypatch, tmp_path):
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")

    # a transient hf lookup outage during resume verification must surface as retriable infra, not be
    # misread as a permanently-missing required save (which would permanently fail a complete run).
    def boom(**kwargs):
        raise ConnectionError("hf 503")

    monkeypatch.setattr(worker, "hf_api", lambda: SimpleNamespace(file_exists=boom))
    callback = worker.make_checkpoint_upload_callback((10, 20))
    state = SimpleNamespace(global_step=20)

    with pytest.raises(worker.RetriableInfraError, match="could not verify required save step"):
        callback.on_train_begin(SimpleNamespace(), state, SimpleNamespace())


def test_required_save_fails_when_publication_never_lands(monkeypatch, tmp_path):
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import flash.engine.worker as worker
    from flash.engine.worker import hf as worker_hf

    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")

    class FailingApi:
        def upload_folder(self, **kwargs):
            raise RuntimeError("hf unavailable")

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    monkeypatch.setattr(worker, "hf_api", lambda: FailingApi())
    monkeypatch.setattr(worker, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_hf, "_CKPT_UPLOAD_BACKOFF_S", 0.0)
    callback = worker.make_checkpoint_upload_callback((4,))

    with pytest.raises(worker.RetriableInfraError, match="not durably published"):
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            SimpleNamespace(global_step=4),
            SimpleNamespace(),
        )


def test_seed_training_rngs_initializes_all_supported_generators(monkeypatch):
    from flash.engine.worker import rng

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


def test_worker_seed_prefers_jobspec_when_present():
    from flash.engine.worker import _resolve_worker_seed

    assert _resolve_worker_seed(SimpleNamespace(seed=123), "999") == 123
    assert _resolve_worker_seed(None, "999") == 999
    assert _resolve_worker_seed(None, None) == 42
    assert _resolve_worker_seed(None, "invalid") == 42
    assert _resolve_worker_seed(None, "-1") == 42


@pytest.mark.parametrize("key", ["SEED", "seed", "Run_Id", "HF_REPO", "flash_arm"])
def test_worker_env_rejects_control_plane_owned_keys(key, tmp_path):
    with pytest.raises(ValueError, match="control-plane key"):
        JobSpec(worker_env={key: "override"})

    path = tmp_path / "reserved-worker-env.toml"
    path.write_text(_BASE_TOML + f'\n[worker_env]\n{key} = "override"\n')
    with pytest.raises(ConfigError, match="control-plane key"):
        spec_from_file(str(path))


def test_provider_worker_env_emits_authoritative_spec_seed():
    from flash.providers._worker import build_worker_env

    spec = JobSpec(model="model", seed=987)
    assert build_worker_env(spec, 987)["SEED"] == "987"
    with pytest.raises(ValueError, match=r"does not match JobSpec\.seed"):
        build_worker_env(spec, 42)


def test_lifecycle_rejects_seed_mismatch_before_provider_work():
    from flash.runner.lifecycle import _submit_seed_supervised

    spec = JobSpec(model="model", seed=987)
    with pytest.raises(ValueError, match=r"does not match JobSpec\.seed"):
        _submit_seed_supervised(spec, 42, SimpleNamespace())


def test_sft_train_tokens_scale_with_completed_authoritative_updates():
    from flash.engine.worker.sft import sft_completed_train_tokens

    assert sft_completed_train_tokens(1_000, 2, 20, 20) == 2_000
    assert sft_completed_train_tokens(1_000, 2, 20, 5) == 500
    assert sft_completed_train_tokens(1_000, 2, 20, 30) == 3_000


def test_sft_under_ran_only_fails_a_genuine_under_run():
    from flash.engine.worker.sft import sft_under_ran

    # a real under-run (fewer updates than the authoritative horizon) fails loudly.
    assert sft_under_ran(9, 10, 10)
    # a fresh run landed exactly on the horizon passes.
    assert not sft_under_ran(10, 10, 10)
    # a resume from a checkpoint past a lowered horizon did zero new steps yet is fully trained.
    assert not sft_under_ran(12, 10, 10)
    # non-positive max_steps keeps the derived-fallback behavior (never authoritative under-run).
    assert not sft_under_ran(9, 10, 0)


def test_grpo_under_ran_only_fails_a_genuine_under_run():
    from flash.engine.worker.rl import grpo_under_ran

    assert grpo_under_ran(9, 10)
    assert not grpo_under_ran(10, 10)
    assert not grpo_under_ran(12, 10)


@pytest.mark.parametrize("bad", [[], False, "", 0])
def test_from_dict_rejects_falsy_non_object_train(bad):
    # strict schema: an explicit non-object train must raise, not silently become an empty table.
    with pytest.raises(TypeError, match="train must be an object"):
        JobSpec.from_dict({"train": bad})


def test_from_dict_defaults_omitted_or_null_train_to_empty():
    assert JobSpec.from_dict({}).seed == FIXED_SEED
    assert JobSpec.from_dict({"train": None}).train.max_steps is None


def test_from_dict_rejects_removed_legacy_train_seeds():
    with pytest.raises(ValueError, match=r"train has unknown key\(s\): seeds"):
        JobSpec.from_dict({"train": {"seeds": [1, 2]}})


def test_from_dict_rejects_misspelled_train_key():
    with pytest.raises(ValueError, match=r"train has unknown key\(s\): max_step"):
        JobSpec.from_dict({"train": {"max_step": 10}})


def test_runtime_secret_cannot_override_control_plane_seed():
    from flash.providers._worker import build_worker_env

    # a directly-constructed spec can declare a seed env secret (the toml schema rejects it, but
    # json/direct construction does not). the built worker env must still hold the canonical seed.
    spec = JobSpec(model="m", seed=987, environment=EnvironmentSpec(id="e", secrets=("SEED",)))
    env = build_worker_env(spec, 987, runtime_secrets={"SEED": "7"})
    assert env["SEED"] == "987"


def test_toml_environment_secrets_reject_control_plane_seed(tmp_path):
    path = tmp_path / "seed-secret.toml"
    path.write_text(
        'model = "Qwen/Qwen3.5-0.8B"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "freesolo/gsm8k"\n'
        'secrets = ["SEED"]\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 8\n"
        "group_size = 2\n"
    )
    with pytest.raises(ConfigError, match="platform-managed key"):
        spec_from_file(str(path))
