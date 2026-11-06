"""Recovery scheduling follows elapsed training time, not a fixed step interval."""

import pytest

from flash.engine.worker.train.core.child.checkpoint_schedule import CheckpointClock


@pytest.mark.parametrize(("seconds", "expected"), [(60, [1, 11, 21]), (300, list(range(1, 22, 2)))])
def test_cadence_tracks_step_duration_and_excludes_save_overhead(seconds, expected):
    now = [0]
    timer = CheckpointClock(clock=lambda: now[0])
    saves = []
    for step in range(1, 22):
        now[0] += seconds
        if timer.due(step, final_step=100):
            saves.append(step)
            now[0] += 90  # slow checkpoint write is not a training step
            timer.saved()
    assert saves == expected


def test_resume_baseline_explicit_and_final_saves():
    now = [0]
    timer = CheckpointClock(clock=lambda: now[0])
    for step in [500, 501, 502, 503]:
        now[0] += 1
        due = timer.due(step, required=(502,), final_step=503)
        assert due is (step != 501)
        if due:
            timer.saved()
        assert not timer.due(step, final_step=503)


def test_slowing_steps_shorten_interval():
    now = [0]
    timer = CheckpointClock(clock=lambda: now[0])
    assert timer.due(1, final_step=100)
    timer.saved()
    now[0] = 60
    assert not timer.due(2, final_step=100)
    now[0] = 360
    assert timer.due(3, final_step=100)  # another five-minute step would exceed ten minutes


@pytest.mark.parametrize("seconds", [600, 601, 900, 3600])
def test_steps_at_or_above_ten_minutes_save_after_every_step(seconds):
    now = [0]
    timer = CheckpointClock(clock=lambda: now[0])
    for step in range(1, 6):
        now[0] += seconds
        assert timer.due(step, final_step=100)
        timer.saved()


def test_greedy_schedule_immediately_adapts_when_steps_speed_up():
    now = [0]
    timer = CheckpointClock(clock=lambda: now[0])
    saves = []
    # Two slow steps establish checkpoints, then one-minute steps fill a fresh
    # ten-minute budget without retaining the old, slower duration estimate.
    for step, seconds in enumerate([900, 900] + [60] * 20, start=1):
        now[0] += seconds
        if timer.due(step, final_step=100):
            saves.append(step)
            timer.saved()
    assert saves == [1, 2, 12, 22]


def test_greedy_schedule_saves_before_a_step_would_overrun_remaining_budget():
    now = [0]
    timer = CheckpointClock(clock=lambda: now[0])
    saves = []
    for step in range(1, 8):
        now[0] += 240
        if timer.due(step, final_step=100):
            saves.append(step)
            timer.saved()
    assert saves == [1, 3, 5, 7]  # 8 minutes fits; another 4-minute step would not.


def test_distributed_save_uses_rank_zero_decision(monkeypatch):
    import sys
    from types import SimpleNamespace

    from flash.engine.worker.train.core.child import checkpoint_schedule

    class Flag:
        value = 0

        def item(self):
            return self.value

    flag = Flag()
    broadcasts = []

    def broadcast(value, src):
        broadcasts.append(src)
        value.value = 1

    dist = SimpleNamespace(
        is_initialized=lambda: True, get_backend=lambda: "gloo", broadcast=broadcast
    )
    torch = SimpleNamespace(distributed=dist, tensor=lambda *a, **kw: flag, int32="int32")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    owner = SimpleNamespace(_flash_checkpoint_clock=SimpleNamespace(due=lambda *a, **kw: False))
    assert checkpoint_schedule.should_save(owner, 2, (), 100, distributed=True)
    assert broadcasts == [0]


def test_sft_recovery_upload_is_independent_of_requested_exports(monkeypatch, tmp_path):
    from flash.core.spec import GpuSpec, JobSpec
    from flash.engine.worker.runtime import state
    from flash.engine.worker.train.sft.setup import checkpoints

    context = state.WorkerContext(config=state.WorkerConfig(spec=JobSpec(gpu=GpuSpec(spot=True))))
    source = tmp_path / "checkpoints" / "global_step_2"
    source.mkdir(parents=True)
    (source / "trainer-state").write_text("recoverable")
    uploaded = []

    def upload(step, directory, **kwargs):
        from pathlib import Path

        assert (Path(directory) / "trainer-state").read_text() == "recoverable"
        uploaded.append(step)
        from flash.engine.worker.io.artifacts import ArtifactCommit, PublicationResult

        return PublicationResult(resume=ArtifactCommit("checkpoint-2", "commit"))

    monkeypatch.setattr(checkpoints._worker_hf, "upload_resume_checkpoint", upload)
    watcher = checkpoints._VerlCheckpointWatcher(
        local_dir=str(source.parent),
        export_root=str(tmp_path / "exports"),
        python_bin="unused",
        model_id="test/model",
        model_revision="commit",
        required_steps=(10,),
        context=context,
    )
    watcher._publish(2, str(source))
    watcher.persistence.close()
    assert uploaded == [2]
    assert watcher.lifecycle.facts(2).resume_uploaded
    assert not watcher.lifecycle.facts(2).deployable_published


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(60, [1, 11, 21]), (300, list(range(1, 22, 2))), (900, list(range(1, 22)))],
)
def test_actual_trainer_save_hook_uses_greedy_clock(monkeypatch, algorithm, seconds, expected):
    from types import SimpleNamespace

    from flash.engine.worker.train.core.child.options import RuntimeOptions
    from tests.test_verl_child_plugins import _import_recipe, _modules

    class Checkpoint:
        def __init__(self, **kwargs):
            self.saved = []

        def save_checkpoint(self, step):
            self.saved.append(step)

        def _save_checkpoint(self):
            self.saved.append(self.global_steps)

    _modules(
        monkeypatch,
        {
            "torch": {"distributed": SimpleNamespace(is_initialized=lambda: False)},
            "torch.distributed": {"is_initialized": lambda: False},
            "torch.utils.data": {"DistributedSampler": object},
            "torchdata.stateful_dataloader": {"StatefulDataLoader": object},
            "verl.trainer.sft_trainer": {"SFTTrainer": object},
            "verl.utils.checkpoint": {"CheckpointHandler": Checkpoint},
            "verl.utils.dataset.dataset_utils": {"SFTTensorCollator": object},
            "flash.engine.worker.train.sft.child.dataset": {"FlashTokenizedSFTDataset": object},
            "flash.engine.worker.train.sft.child.loop": {"fit_sft": lambda _: None},
            "omegaconf": {"OmegaConf": object},
            "verl.trainer.ppo.utils": {"Role": object},
            "verl.trainer.ppo.v1.trainer_sync": {"PPOTrainerSync": Checkpoint},
            "flash.engine.worker.train.core.child.queue": {"_init_transfer_queue": lambda _: None},
            "flash.engine.worker.train.core.child.replay": {"BoundedReplayBuffer": object},
            "flash.engine.worker.train.core.child.tracking": {
                "publish_tracking_link": lambda _: None
            },
        },
    )
    name = "sft" if algorithm == "sft" else "core"
    recipe = _import_recipe(monkeypatch, f"flash.engine.worker.train.{name}.child.trainer")
    runtime = RuntimeOptions(algorithm, 100, "ready")
    if algorithm == "sft":
        owner = recipe.FlashCheckpointHandler(runtime=runtime)
    else:
        owner = object.__new__(recipe.FlashSyncTrainer)
        owner.runtime = runtime
        owner.saved = []
    now = [0]
    owner._flash_checkpoint_clock = CheckpointClock(clock=lambda: now[0])
    monkeypatch.setenv("FLASH_SPOT_CHECKPOINT_SECONDS", "600")
    for step in range(1, 22):
        now[0] += seconds
        if algorithm == "sft":
            owner.save_checkpoint(step)
        else:
            owner.global_steps = step
            owner._save_checkpoint()
    assert owner.saved == expected
