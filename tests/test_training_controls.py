from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace

import pytest

from flash.engine.steps import (
    checkpoint_step_due,
    configure_trainer_checkpoint_schedule,
    final_checkpoint_due,
    resolve_update_horizon,
    sft_update_steps,
    validate_checkpoint_horizon,
)
from flash.schema import ConfigError, spec_from_file
from flash.spec import FIXED_SEED, JobSpec, TrainSpec

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


def test_toml_seed_and_checkpoint_landmarks_parse(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text(
        "seed = 987654321\n" + _BASE_TOML + "max_steps = 12\ncheckpoint_landmarks = [1, 4, 12]\n"
    )

    spec = spec_from_file(str(path), run_id="seed-explicit")

    assert spec.seed == 987654321
    assert spec.train.max_steps == 12
    assert spec.train.checkpoint_landmarks == (1, 4, 12)
    assert JobSpec.from_json(spec.to_json()) == spec


@pytest.mark.parametrize("value", [True, -1, 2**63, 1.5, "7"])
def test_jobspec_seed_validation(value):
    with pytest.raises((TypeError, ValueError), match="seed"):
        JobSpec(seed=value)


@pytest.mark.parametrize(
    ("landmarks", "match"),
    [
        ("1", "list of integers"),
        ("[true]", "entries must be integers"),
        ("[1.0]", "entries must be integers"),
        ("[0]", "entries must be positive"),
        ("[2, 1]", "strictly increasing"),
        ("[1, 1]", "strictly increasing"),
    ],
)
def test_toml_checkpoint_landmark_validation(tmp_path, landmarks, match):
    path = tmp_path / "invalid.toml"
    path.write_text(_BASE_TOML + f"checkpoint_landmarks = {landmarks}\n")

    with pytest.raises(ConfigError, match=match):
        spec_from_file(str(path))


def test_checkpoint_landmarks_reject_steps_beyond_positive_max_steps(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text(_BASE_TOML + "max_steps = 5\ncheckpoint_landmarks = [2, 6]\n")

    with pytest.raises(ConfigError, match=r"beyond train\.max_steps"):
        spec_from_file(str(path))

    with pytest.raises(ValueError, match=r"beyond train\.max_steps"):
        TrainSpec(max_steps=5, checkpoint_landmarks=(2, 6))

    assert TrainSpec(max_steps=5).checkpoint_landmarks == ()
    with pytest.raises(ValueError, match=r"requires positive train\.max_steps"):
        TrainSpec(max_steps=-1, checkpoint_landmarks=(2,))


def test_checkpoint_landmarks_require_explicit_positive_max_steps(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text(_BASE_TOML + "checkpoint_landmarks = [2]\n")

    with pytest.raises(ConfigError, match=r"requires positive train\.max_steps"):
        spec_from_file(str(path))

    with pytest.raises(ValueError, match=r"requires positive train\.max_steps"):
        TrainSpec(checkpoint_landmarks=(2,))


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


def test_exact_landmarks_suppress_periodic_trainer_saves():
    config = {"save_steps": 5}
    configure_trainer_checkpoint_schedule(config, (2, 7))
    assert config == {"save_steps": 5, "save_strategy": "no"}

    fallback = {"save_steps": 5}
    configure_trainer_checkpoint_schedule(fallback, ())
    assert fallback == {"save_steps": 5}


def test_opd_checkpoint_schedule_uses_landmarks_or_periodic_fallback():
    assert checkpoint_step_due(2, (2, 7), 5)
    assert checkpoint_step_due(7, (2, 7), 5)
    assert not checkpoint_step_due(5, (2, 7), 5)
    assert checkpoint_step_due(5, (), 5)
    assert not checkpoint_step_due(4, (), 5)
    assert final_checkpoint_due(9, ())
    assert not final_checkpoint_due(9, (2, 7))


def test_checkpoint_horizon_rejects_unreachable_runtime_landmark():
    validate_checkpoint_horizon((2, 7), 7)
    with pytest.raises(ValueError, match="exceeds the 6-update horizon"):
        validate_checkpoint_horizon((2, 7), 6)


def test_trainer_landmark_callback_requests_only_exact_steps(monkeypatch):
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from flash.engine.worker.hf import make_checkpoint_landmark_callback

    callback = make_checkpoint_landmark_callback((2, 7))
    for step, expected in ((1, False), (2, True), (3, False), (7, True)):
        control = SimpleNamespace(should_save=False)
        returned = callback.on_step_end(None, SimpleNamespace(global_step=step), control)
        assert returned is control
        assert control.should_save is expected


def test_required_landmark_fails_when_trainer_checkpoint_is_missing(monkeypatch, tmp_path):
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


def test_permanent_landmark_failure_is_not_retried(monkeypatch, tmp_path):
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
        raise worker_hf.CheckpointLandmarkError("missing deployable adapter")

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    monkeypatch.setattr(worker_hf, "publish_deployable_checkpoint", fail_permanently)
    callback = worker.make_checkpoint_upload_callback((4,))

    with pytest.raises(worker_hf.CheckpointLandmarkError, match="missing deployable adapter"):
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            SimpleNamespace(global_step=4),
            SimpleNamespace(),
        )

    assert calls == 1


def test_resumed_trainer_credits_landmarks_at_or_before_restored_step(monkeypatch, tmp_path):
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    callback = worker.make_checkpoint_upload_callback((10, 20))
    control = SimpleNamespace()
    state = SimpleNamespace(global_step=20)

    assert callback.on_train_begin(SimpleNamespace(), state, control) is control
    callback.on_train_end(SimpleNamespace(output_dir=str(tmp_path)), state, control)


def test_required_landmark_fails_when_publication_never_lands(monkeypatch, tmp_path):
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


def test_sft_and_grpo_use_shared_exact_landmark_schedule():
    from flash.engine.worker import rl, sft

    for run_function in (sft.run_sft, rl.run_rl):
        source = inspect.getsource(run_function)
        assert "configure_trainer_checkpoint_schedule" in source
        assert "make_checkpoint_landmark_callback(checkpoint_landmarks)" in source
        assert "make_checkpoint_upload_callback(checkpoint_landmarks)" in source


def test_each_training_path_seeds_before_environment_and_model_construction():
    from flash.engine.worker import opd, rl, sft

    checks = (
        (sft.run_sft, "sft_model = _w.prepare_fresh_lora_base"),
        (rl.run_rl, "init_model, init_peft = _w._init_adapter_model"),
        (opd.run_opd, "model, rollout_model_source = _student_model"),
    )
    for run_function, model_constructor in checks:
        source = inspect.getsource(run_function)
        first_seed = source.index("seed_training_rngs(_w.SEED)")
        second_seed = source.index("seed_training_rngs(_w.SEED)", first_seed + 1)
        assert first_seed < source.index("require_active_env")
        assert second_seed < source.index(model_constructor)


def test_worker_seed_prefers_jobspec_and_defaults_compatibly():
    from flash.engine.worker import _resolve_worker_seed

    assert _resolve_worker_seed(SimpleNamespace(seed=123), "999") == 123
    assert _resolve_worker_seed(None, "999") == 999
    assert _resolve_worker_seed(None, None) == 42
    assert _resolve_worker_seed(None, "invalid") == 42
    assert _resolve_worker_seed(None, "-1") == 42


def test_provider_worker_env_emits_and_reserves_selected_seed():
    from flash.providers._worker import build_worker_env

    spec = JobSpec(model="model", worker_env={"SEED": "1"})
    env = build_worker_env(spec, 987)

    assert env["SEED"] == "987"
