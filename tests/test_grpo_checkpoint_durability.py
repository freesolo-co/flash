"""grpo checkpoint durability, resume evidence, and publication gates."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

import flash.engine.worker.train.entry.rl_train_runner as rl_runner
import flash.engine.worker.train.rl.launch.checkpoints as rl_checkpoints
from flash.engine.worker.train.entry import backend_common, rl_train


def test_resume_uploader_publishes_required_steps_and_reports_missing(tmp_path, monkeypatch):
    # the deployable at a required step is the whole point of save_at_steps. a resume-state upload
    # alone leaves the step resumable but not servable, which is the gap this closes.
    published: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "publish_deployable_checkpoint",
        lambda d, s, **kw: published.append((d, s, kw.get("required", False))),
    )
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "upload_resume_checkpoint", lambda *a, **kw: True
    )
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "hf_upload_folder", lambda *a, **kw: True, raising=False
    )
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "write_base_model_provenance", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        rl_checkpoints,
        "export_peft_adapter",
        lambda actor, destination, **kwargs: _write_internal_adapter(Path(destination)),
    )
    monkeypatch.setattr(rl_checkpoints, "stamp_adapter_dir_provenance", lambda *a, **kw: None)

    local_dir = tmp_path / "ckpt"
    (local_dir / "global_step_10" / "actor").mkdir(parents=True)
    (local_dir / "global_step_5" / "actor").mkdir(parents=True)
    (local_dir / "latest_checkpointed_iteration.txt").write_text("10")

    class _Tok:
        def save_pretrained(self, path):
            pass

    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(10, 20),
        export_root=str(tmp_path / "exports"),
        python_bin="python",
        model_id="Qwen/Qwen3.5-9B",
        model_revision="rev",
        preprocessor=_Tok(),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    uploader.allow_deployable_publication()
    uploader.stop()

    # step 10 was required and completed, so it published as a REQUIRED deployable. step 5 is a gcd
    # by-product verl wrote on the way there; it is resume state only and must not be published.
    assert [(step, required) for _, step, required in published] == [(10, True)]
    # step 20 never completed, so the run must fail rather than silently ship an incomplete set.
    with pytest.raises(RuntimeError, match="required saves were not durably published: \\[20\\]"):
        uploader.raise_if_incomplete()


def test_resume_credits_required_steps_already_durable_on_hf(tmp_path, monkeypatch):
    # a resumed run never re-saves a step it trained past. without crediting the earlier required
    # steps a retry that resumes at 20 would report step 10 missing and fail a successful run.
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "_deployable_adapter_on_hf", lambda step: step == 10
    )

    class _Tok:
        def save_pretrained(self, path):
            pass

    uploader = rl_checkpoints._VerlResumeUploader(
        str(tmp_path),
        resume_step=20,
        required_steps=(10, 15, 25),
        preprocessor=_Tok(),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.credit_durable_required_steps(20)

    # step 10 is verified on hf, so it is credited. step 15 is below the resume point but its
    # adapter never landed, so it stays uncredited and completeness still catches it. step 25 is
    # ahead of the resume point and is this run's job to publish.
    assert uploader.lifecycle.deployable_published_steps == {10}
    with pytest.raises(RuntimeError, match=r"not durably published: \[15, 25\]"):
        uploader.raise_if_incomplete()


def test_resume_step_is_not_credited_without_a_durable_adapter(tmp_path, monkeypatch):
    # a preempted worker can advance past a required step without its deployable ever reaching hf,
    # so the restored step counter alone must never credit a required save.
    monkeypatch.setattr(rl_checkpoints._worker_hf, "_deployable_adapter_on_hf", lambda step: False)

    uploader = rl_checkpoints._VerlResumeUploader(
        str(tmp_path),
        resume_step=10,
        required_steps=(10,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.credit_durable_required_steps(10)

    assert uploader.lifecycle.deployable_published_steps == set()


def test_restore_verl_resume_is_a_noop_without_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_checkpoints._worker_hf, "hf_resume_checkpoint", lambda *a, **k: None)
    assert (
        rl_checkpoints._restore_verl_resume(str(tmp_path), world_size=1, expected_fsdp_generation=2)
        == 0
    )
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


def test_restore_verl_resume_stages_the_checkpoint_where_verl_looks(tmp_path, monkeypatch):
    src = tmp_path / "checkpoint-7"
    (src / "actor").mkdir(parents=True)
    (src / "actor" / "model.safetensors").write_text("weights")
    # this test is about staging mechanics, so provide one complete native fsdp2 rank.
    (src / "actor" / "fsdp_config.json").write_text(
        json.dumps({"FSDP_version": 2, "world_size": 1})
    )
    for kind in ("model", "optim", "extra_state"):
        (src / "actor" / f"{kind}_world_size_1_rank_0.pt").write_bytes(b"shard")
    _write_resume_manifest(src, 7, positive=3)
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(rl_checkpoints._worker_hf, "hf_resume_checkpoint", lambda *a, **k: str(src))

    assert (
        rl_checkpoints._restore_verl_resume(
            str(local_dir), world_size=1, expected_fsdp_generation=2
        )
        == 7
    )
    # verl discovers the checkpoint through this marker plus the global_step_N layout.
    assert (local_dir / "latest_checkpointed_iteration.txt").read_text().strip() == "7"
    assert (local_dir / "global_step_7" / "actor" / "model.safetensors").read_text() == "weights"


def test_restore_verl_resume_rejects_an_unparseable_checkpoint_path(tmp_path, monkeypatch):
    bad = tmp_path / "not-a-checkpoint"
    bad.mkdir()
    monkeypatch.setattr(rl_checkpoints._worker_hf, "hf_resume_checkpoint", lambda *a, **k: str(bad))
    with pytest.raises(RuntimeError, match="invalid GRPO resume checkpoint path"):
        rl_checkpoints._restore_verl_resume(
            str(tmp_path / "ckpt"), world_size=1, expected_fsdp_generation=2
        )


class _AlwaysReadyMetricEvidence:
    prior_positive_grad_step = 1

    def set_prior_positive_step(self, step, *, checkpoint_step):
        self.prior_positive_grad_step = step

    def checkpoint_manifest_evidence(self, checkpoint_step):
        return True, 1 if checkpoint_step >= 1 else None


def _always_ready_metric_evidence():
    return _AlwaysReadyMetricEvidence()


def _write_step(local_dir, step):
    d = local_dir / f"global_step_{step}"
    (d / "actor").mkdir(parents=True)
    (local_dir / "latest_checkpointed_iteration.txt").write_text(str(step))
    return d


def test_resume_uploader_uploads_each_completed_step(tmp_path):
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    seen = []
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir), resume_step=0, metric_evidence=_always_ready_metric_evidence()
    )

    original = rl_checkpoints._worker_hf.upload_resume_checkpoint
    rl_checkpoints._worker_hf.upload_resume_checkpoint = lambda step, path, **k: seen.append(
        int(step)
    )
    try:
        _write_step(local_dir, 4)
        uploader.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and 4 not in seen:
            time.sleep(0.05)
        _write_step(local_dir, 8)
        while time.monotonic() < deadline and 8 not in seen:
            time.sleep(0.05)
        uploader.stop()
    finally:
        rl_checkpoints._worker_hf.upload_resume_checkpoint = original
    assert seen == [4, 8]


def test_resume_uploader_skips_the_step_it_resumed_from(tmp_path):
    # that checkpoint is already durable on hf; re-uploading it wastes the upload slot.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    seen = []
    original = rl_checkpoints._worker_hf.upload_resume_checkpoint
    rl_checkpoints._worker_hf.upload_resume_checkpoint = lambda step, path, **k: seen.append(
        int(step)
    )
    try:
        _write_step(local_dir, 5)
        uploader = rl_checkpoints._VerlResumeUploader(
            str(local_dir), resume_step=5, metric_evidence=_always_ready_metric_evidence()
        )
        uploader.start()
        time.sleep(0.5)
        uploader.stop()
    finally:
        rl_checkpoints._worker_hf.upload_resume_checkpoint = original
    assert seen == []


def test_resume_uploader_never_fails_the_run_on_an_upload_error(tmp_path):
    # the policy is still trained and published; a failed resume upload only costs restart distance.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()

    def boom(step, path, **k):
        raise RuntimeError("hf is down")

    original = rl_checkpoints._worker_hf.upload_resume_checkpoint
    rl_checkpoints._worker_hf.upload_resume_checkpoint = boom
    try:
        _write_step(local_dir, 2)
        uploader = rl_checkpoints._VerlResumeUploader(
            str(local_dir), resume_step=0, metric_evidence=_always_ready_metric_evidence()
        )
        uploader.start()
        time.sleep(0.5)
        uploader.stop()  # must not raise
    finally:
        rl_checkpoints._worker_hf.upload_resume_checkpoint = original
    assert 2 in uploader.lifecycle.discovered_steps


def test_grpo_gradient_check_rejects_all_zero_actor_gradients():
    with pytest.raises(RuntimeError, match="zero actor gradient norm"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [1.0, 1.0], [0.0, 0.0], {1: 0.0, 2: 0.0}, expected_steps=range(1, 3)
        )


def test_grpo_gradient_check_accepts_positive_norm_despite_zero_advantage_spread():
    rl_checkpoints._check_grpo_had_a_gradient(
        [0.875, 1.0],
        [0.0, 0.0],
        {1: 0.0076509942300617695, 2: 0.0},
        expected_steps=range(1, 3),
    )


def test_positive_advantage_spread_does_not_override_all_zero_gradients():
    with pytest.raises(RuntimeError, match="zero actor gradient norm"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [0.4, 0.6], [1.5, 0.0], {1: 0.0, 2: 0.0}, expected_steps=range(1, 3)
        )


def test_grpo_gradient_check_rejects_reward_metrics_without_advantage_metrics():
    with pytest.raises(RuntimeError, match="no advantage metrics"):
        rl_checkpoints._check_grpo_had_a_gradient([1.0], [], {1: 1.0}, expected_steps=range(1, 2))


def test_grpo_gradient_check_still_rejects_an_unconsulted_reward_bridge():
    with pytest.raises(RuntimeError, match="never consulted"):
        rl_checkpoints._check_grpo_had_a_gradient([], [], {}, expected_steps=range(1, 2))


def test_advantage_bounds_remain_diagnostic_when_gradient_evidence_is_positive():
    line = (
        "step:1 - critic/rewards/mean:0.875 - critic/advantages/max:0.125 - "
        "critic/advantages/min:0.125 - actor/grad_norm:0.0076509942300617695"
    )
    adv_max = backend_common.parse_verl_metric(line, "critic/advantages/max")
    adv_min = backend_common.parse_verl_metric(line, "critic/advantages/min")
    grad_norm = backend_common.parse_verl_metric(line, "actor/grad_norm")
    assert adv_max - adv_min == 0.0
    rl_checkpoints._check_grpo_had_a_gradient(
        [0.875], [0.0], {1: grad_norm}, expected_steps=range(1, 2)
    )


def test_grpo_gradient_check_combines_prior_and_current_positive_evidence():
    rl_checkpoints._check_grpo_had_a_gradient(
        [1.0],
        [0.0],
        {10: 0.0},
        resume_step=9,
        prior_positive_step=4,
        expected_steps=range(10, 11),
    )
    with pytest.raises(RuntimeError, match="no prior positive-gradient evidence"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [1.0], [0.0], {10: 0.0}, resume_step=9, expected_steps=range(10, 11)
        )
    rl_checkpoints._check_grpo_had_a_gradient(
        [1.0], [0.0], {10: 0.25}, resume_step=9, expected_steps=range(10, 11)
    )
    with pytest.raises(RuntimeError, match=r"missing=\[10\]"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [1.0],
            [0.0],
            {},
            resume_step=9,
            prior_positive_step=4,
            expected_steps=range(10, 11),
        )
    with pytest.raises(RuntimeError, match="not finite and nonnegative"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [1.0],
            [0.0],
            {10: -1.0},
            resume_step=9,
            prior_positive_step=4,
            expected_steps=range(10, 11),
        )


def test_terminal_metric_evidence_rejects_missing_nonfinite_negative_and_bad_range():
    with pytest.raises(RuntimeError, match=r"missing=\[2\], extra=\[\]"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [0.5, 0.5], [1.0, 1.0], {1: 1.0}, expected_steps=range(1, 3)
        )
    for value in (float("nan"), float("inf"), -0.1):
        with pytest.raises(RuntimeError, match="not finite and nonnegative"):
            rl_checkpoints._check_grpo_had_a_gradient(
                [0.5], [1.0], {1: value}, expected_steps=range(1, 2)
            )
    with pytest.raises(RuntimeError, match=r"missing=\[1\], extra=\[3\]"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [0.5], [1.0], {3: 1.0}, expected_steps=range(1, 2)
        )


def test_terminal_advantage_evidence_still_rejects_missing_and_nonfinite_bounds():
    missing = rl_runner._StepMetricState()
    missing.reward_history[:] = [0.5, 0.5]
    missing.adv_spread_history[:] = [1.0]
    missing.advantage_bounds[1] = (-0.5, 0.5)
    missing.grad_norms = {1: 1.0, 2: 1.0}
    with pytest.raises(RuntimeError, match=r"missing=\[2\], extra=\[\]"):
        rl_runner._validate_rl_child(0, missing, 0, 2, None)

    nonfinite = rl_runner._StepMetricState()
    nonfinite.reward_history.append(0.5)
    nonfinite.adv_spread_history.append(1.0)
    nonfinite.advantage_bounds[1] = (0.0, float("inf"))
    nonfinite.grad_norms[1] = 1.0
    with pytest.raises(RuntimeError, match="not finite and ordered"):
        rl_runner._validate_rl_child(0, nonfinite, 0, 1, None)


def test_grpo_gradient_check_requires_prior_evidence_for_an_already_complete_resume():
    rl_checkpoints._check_grpo_had_a_gradient(
        [],
        [],
        {},
        resume_step=5,
        prior_positive_step=3,
        already_complete=True,
        expected_steps=range(6, 6),
    )
    with pytest.raises(RuntimeError, match="no durable positive actor gradient evidence"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [], [], {}, resume_step=5, already_complete=True, expected_steps=range(6, 6)
        )
    with pytest.raises(RuntimeError, match="invalid prior"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [],
            [],
            {},
            resume_step=5,
            prior_positive_step=6,
            already_complete=True,
            expected_steps=range(6, 6),
        )
    with pytest.raises(RuntimeError, match="never consulted"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [], [], {}, expected_steps=range(6, 7), resume_step=5
        )


def test_resume_uploader_publication_latch_starts_closed_and_opens_explicitly():
    uploader = rl_checkpoints._VerlResumeUploader(
        "/nonexistent",
        resume_step=0,
        required_steps=(1,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    assert uploader._deployable_allowed() is False
    uploader.allow_deployable_publication()
    assert uploader._deployable_allowed() is True


def test_terminal_validation_opens_the_uploader_latch_before_the_final_drain():
    verdict_source = inspect.getsource(rl_runner._validate_rl_child)
    assert "resume_uploader.allow_deployable_publication()" in verdict_source
    assert verdict_source.index("_check_grpo_had_a_gradient(") < verdict_source.index(
        "resume_uploader.allow_deployable_publication()"
    )
    assert verdict_source.index("_finalize_advantage_evidence(") < verdict_source.index(
        "resume_uploader.allow_deployable_publication()"
    )
    assert verdict_source.index(
        "resume_uploader.allow_deployable_publication()"
    ) < verdict_source.index("resume_uploader.stop()")


def _patch_stage_and_publish(monkeypatch, staged: list[int], published: list[int]) -> None:
    """record staging and publication separately, without running model_merger or touching hf.

    they are patched as two seams because the production code separates them: staging is bounded by
    verl's checkpoint retention, publication by the terminal validation latch.
    """

    def stage(self, step, path):
        staged.append(int(step))
        adapter_dir = f"{path}-adapter"
        os.makedirs(adapter_dir, exist_ok=True)
        Path(adapter_dir, "adapter_config.json").write_text("{}")
        Path(adapter_dir, "adapter_model.safetensors").write_bytes(b"weights")
        return adapter_dir

    monkeypatch.setattr(rl_checkpoints._VerlResumeUploader, "_stage_deployable", stage)
    monkeypatch.setattr(
        rl_checkpoints._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (
            published.append(int(step)),
            self.lifecycle.mark_deployable_published(step),
        )[0],
    )


def _write_internal_adapter(adapter_dir: Path, *, weights: str = "single") -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}")
    if weights == "single":
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    elif weights == "sharded":
        for index in (1, 2):
            name = f"adapter_model-{index:05d}-of-00002.safetensors"
            (adapter_dir / name).write_bytes(f"shard-{index}".encode())
        (adapter_dir / "adapter_model.safetensors.index.json").write_text("{}")
    elif weights == "incomplete-sharded":
        (adapter_dir / "adapter_model-00001-of-00002.safetensors").write_bytes(b"shard-1")
        (adapter_dir / "adapter_model.safetensors.index.json").write_text("{}")
    else:
        raise AssertionError(f"unknown adapter weight shape {weights}")


def _write_resume_manifest(
    checkpoint: Path, step: int, required=(), positive=1, *, attempt=0
) -> None:
    (checkpoint / "_flash_resume_manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "checkpoint_step": step,
                "checkpoint_attempt": attempt,
                "required_adapters": [
                    {"step": required_step, "attempt": attempt} for required_step in required
                ],
                "first_positive_grad_step": positive,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _metric_state(*norms: tuple[int, float], resume_step: int = 0, prior=None):
    state = rl_runner._StepMetricState(resume_step=resume_step)
    state.set_prior_positive_step(prior, checkpoint_step=resume_step)
    for step, value in norms:
        state.record_grad_norm(step, value)
    return state


def test_internal_required_adapters_upload_once_and_restore_from_latest_manifest(
    tmp_path, monkeypatch
):
    local = tmp_path / "local"
    local.mkdir()
    remote = tmp_path / "remote" / "checkpoint"
    remote.mkdir(parents=True)
    internal_uploads = []
    native_uploads = []
    published = []

    def stage(self, step, _path):
        adapter = tmp_path / "exports-one" / f"step-{step}"
        _write_internal_adapter(adapter)
        return str(adapter)

    def upload_internal(source, subpath, required=False):
        assert required is True
        internal_uploads.append(subpath)
        destination = tmp_path / "remote" / subpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        return True

    def upload_native(step, source, **kwargs):
        destination = remote / f"checkpoint-{step}"
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source, destination)
        for old in remote.glob("checkpoint-*"):
            if old != destination:
                shutil.rmtree(old)
        native_uploads.append(step)
        callback = kwargs.get("after_upload")
        if callback:
            callback()
        return True

    monkeypatch.setattr(rl_checkpoints._VerlResumeUploader, "_stage_deployable", stage)
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "hf_upload_folder", upload_internal, raising=False
    )
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "upload_resume_checkpoint", upload_native, raising=False
    )
    monkeypatch.setattr(
        rl_checkpoints._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, path: (
            published.append(step),
            self.lifecycle.mark_deployable_published(step),
        )[0],
    )

    uploader = rl_checkpoints._VerlResumeUploader(
        str(local),
        resume_step=0,
        required_steps=(2, 4),
        metric_evidence=_metric_state((1, 0.0), (2, 0.5), (3, 0.0), (4, 0.0)),
        export_root=str(tmp_path / "exports-one"),
    )
    try:
        for step in (2, 4):
            native = _write_step(local, step)
            (native / "actor" / "fsdp_config.json").write_text(
                '{"FSDP_version": 2, "world_size": 1}'
            )
            for kind in ("model", "optim", "extra_state"):
                (native / "actor" / f"{kind}_world_size_1_rank_0.pt").write_bytes(b"shard")
            uploader.start() if step == 2 else None
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and step not in native_uploads:
                time.sleep(0.01)
        uploader.stop()
    finally:
        if uploader._thread.is_alive():
            uploader.stop()

    assert internal_uploads == [
        "checkpoint/required-adapters/attempt-0/step-2/adapter",
        "checkpoint/required-adapters/attempt-0/step-4/adapter",
    ]
    latest = remote / "checkpoint-4"
    manifest = json.loads((latest / "_flash_resume_manifest.json").read_text())
    assert manifest["required_adapters"] == [
        {"step": 2, "attempt": 0},
        {"step": 4, "attempt": 0},
    ]
    assert manifest["first_positive_grad_step"] == 2
    assert not list(latest.rglob("adapter_model.safetensors"))

    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "hf_resume_checkpoint", lambda **kwargs: str(latest)
    )
    monkeypatch.setattr(rl_checkpoints._worker_hf, "_deployable_adapter_on_hf", lambda step: False)
    restored = tmp_path / "restored"
    restored.mkdir()
    resume_step = rl_checkpoints._restore_verl_resume(
        str(restored),
        world_size=1,
        expected_fsdp_generation=2,
        required_steps=(2, 4),
    )
    assert resume_step == 4
    second_state = _metric_state(resume_step=4)
    second = rl_checkpoints._VerlResumeUploader(
        str(restored),
        resume_step=4,
        required_steps=(2, 4),
        metric_evidence=second_state,
        export_root=str(tmp_path / "exports-two"),
    )
    second.credit_durable_required_steps(4)
    second.restore_staged_adapters(4)
    assert second_state.prior_positive_grad_step == 2
    assert sorted(second.staged_adapters) == [2, 4]
    second.allow_deployable_publication()
    second._publish_ready()
    assert published == [2, 4]


def test_attempt_scoped_internal_adapter_survives_an_incompatible_retry_window(
    tmp_path, monkeypatch
):
    remote = tmp_path / "remote"
    checkpoint = remote / "checkpoint" / "checkpoint-2"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (actor / "fsdp_config.json").write_text('{"FSDP_version": 2, "world_size": 1}')
    for kind in ("model", "optim", "extra_state"):
        (actor / f"{kind}_world_size_1_rank_0.pt").write_bytes(b"shard")
    _write_resume_manifest(checkpoint, 2, required=(2,), positive=2, attempt=0)
    old_adapter = remote / "checkpoint" / "required-adapters" / "attempt-0" / "step-2" / "adapter"
    _write_internal_adapter(old_adapter)
    (old_adapter / "adapter_model.safetensors").write_bytes(b"old-trajectory")
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "hf_resume_checkpoint", lambda **kwargs: str(checkpoint)
    )

    incompatible_local = tmp_path / "incompatible"
    incompatible_local.mkdir()
    assert (
        rl_checkpoints._restore_verl_resume(
            str(incompatible_local),
            world_size=2,
            expected_fsdp_generation=2,
            required_steps=(2,),
        )
        == 0
    )

    monkeypatch.setattr(rl_checkpoints._worker_state, "ATTEMPT", 1)

    def upload_internal(source, subpath, required=False):
        assert required is True
        destination = remote / subpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        return True

    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "hf_upload_folder", upload_internal, raising=False
    )
    new_adapter = tmp_path / "new-adapter"
    _write_internal_adapter(new_adapter)
    (new_adapter / "adapter_model.safetensors").write_bytes(b"new-trajectory")
    new_attempt = rl_checkpoints._VerlResumeUploader(
        str(incompatible_local),
        resume_step=0,
        required_steps=(2,),
        metric_evidence=_metric_state((1, 0.5), (2, 0.0)),
    )
    new_attempt._make_required_adapter_durable(2, str(new_adapter))

    assert (old_adapter / "adapter_model.safetensors").read_bytes() == b"old-trajectory"
    assert (
        remote
        / "checkpoint"
        / "required-adapters"
        / "attempt-1"
        / "step-2"
        / "adapter"
        / "adapter_model.safetensors"
    ).read_bytes() == b"new-trajectory"

    compatible_local = tmp_path / "compatible"
    compatible_local.mkdir()
    assert (
        rl_checkpoints._restore_verl_resume(
            str(compatible_local),
            world_size=1,
            expected_fsdp_generation=2,
            required_steps=(2,),
        )
        == 2
    )
    restored = compatible_local / "_flash_required_adapter_store" / "step-2" / "adapter"
    assert (restored / "adapter_model.safetensors").read_bytes() == b"old-trajectory"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.update(extra=True),
        lambda manifest: manifest.update(version=3),
        lambda manifest: manifest.update(version=2.0),
        lambda manifest: manifest.update(checkpoint_step=True),
        lambda manifest: manifest.update(checkpoint_attempt=-1),
        lambda manifest: manifest.update(checkpoint_attempt=True),
        lambda manifest: manifest.update(
            required_adapters=[{"step": 2, "attempt": 0}, {"step": 2, "attempt": 1}]
        ),
        lambda manifest: manifest.update(
            required_adapters=[{"step": 2, "attempt": 0}, {"step": 1, "attempt": 0}]
        ),
        lambda manifest: manifest.update(required_adapters=[{"step": 0, "attempt": 0}]),
        lambda manifest: manifest.update(required_adapters=[{"step": 5, "attempt": 0}]),
        lambda manifest: manifest.update(required_adapters=[{"step": 2, "attempt": -1}]),
        lambda manifest: manifest.update(required_adapters=[{"step": 2, "attempt": True}]),
        lambda manifest: manifest.update(required_adapters=[{"step": 2, "attempt": 1}]),
        lambda manifest: manifest.update(required_adapters=[{"step": 2}]),
        lambda manifest: manifest.update(first_positive_grad_step=0),
        lambda manifest: manifest.update(first_positive_grad_step=5),
        lambda manifest: manifest.update(first_positive_grad_step=True),
    ],
)
def test_resume_manifest_rejects_unknown_noncanonical_and_out_of_range_facts(tmp_path, mutate):
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    manifest = {
        "version": 2,
        "checkpoint_step": 4,
        "checkpoint_attempt": 0,
        "required_adapters": [{"step": 1, "attempt": 0}, {"step": 2, "attempt": 0}],
        "first_positive_grad_step": 1,
    }
    mutate(manifest)
    (checkpoint / "_flash_resume_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="GRPO resume manifest"):
        rl_checkpoints._read_resume_manifest(
            str(checkpoint), checkpoint_step=4, required_steps=(1, 2)
        )


@pytest.mark.parametrize(
    "payload",
    [
        (
            b'{"version":2,"version":2,"checkpoint_step":4,"checkpoint_attempt":0,'
            b'"required_adapters":[],"first_positive_grad_step":1}'
        ),
        (
            b'{"version":2,"checkpoint_step":4,"checkpoint_attempt":0,'
            b'"required_adapters":[{"step":2,"step":2,"attempt":0}],'
            b'"first_positive_grad_step":1}'
        ),
    ],
)
def test_resume_manifest_rejects_duplicate_keys_at_every_depth(tmp_path, payload):
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "_flash_resume_manifest.json").write_bytes(payload)

    with pytest.raises(RuntimeError, match=r"unreadable _flash_resume_manifest\.json"):
        rl_checkpoints._read_resume_manifest(
            str(checkpoint), checkpoint_step=4, required_steps=(2,)
        )


def test_resume_manifest_rejects_invalid_utf8(tmp_path):
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "_flash_resume_manifest.json").write_bytes(b"{\xff}")

    with pytest.raises(RuntimeError, match=r"unreadable _flash_resume_manifest\.json"):
        rl_checkpoints._read_resume_manifest(str(checkpoint), checkpoint_step=4, required_steps=())


def test_required_adapter_namespace_rejects_canonical_collision_and_step_symlink(tmp_path):
    checkpoint = tmp_path / "checkpoint" / "checkpoint-4"
    checkpoint.mkdir(parents=True)
    root = checkpoint.parent / "required-adapters"
    attempt = root / "attempt-0"
    _write_internal_adapter(attempt / "step-2" / "adapter")
    _write_internal_adapter(attempt / "step-02" / "adapter")
    with pytest.raises(RuntimeError, match="namespace collision"):
        rl_checkpoints._internal_adapter_sources(str(checkpoint), ((2, 0),))

    shutil.rmtree(attempt / "step-02")
    outside = tmp_path / "outside-step"
    _write_internal_adapter(outside / "adapter")
    shutil.rmtree(attempt / "step-2")
    (attempt / "step-2").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="namespace step"):
        rl_checkpoints._internal_adapter_sources(str(checkpoint), ((2, 0),))


def test_checkpoint_waits_for_its_own_metric_and_preemption_keeps_previous_remote(
    tmp_path, monkeypatch
):
    local = tmp_path / "local"
    local.mkdir()
    state = _metric_state()
    uploaded = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **kwargs: uploaded.append(step),
        raising=False,
    )
    checkpoint = _write_step(local, 1)
    uploader = rl_checkpoints._VerlResumeUploader(str(local), resume_step=0, metric_evidence=state)
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and uploader._blocked_evidence_step != 1:
            time.sleep(0.01)
        assert uploaded == []
        assert not (checkpoint / "_flash_resume_manifest.json").exists()
    finally:
        uploader.stop()
    assert uploaded == []


def test_checkpoint_manifest_is_written_only_after_metric_ingestion(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    state = _metric_state()
    uploaded = []

    def upload(step, path, **kwargs):
        uploaded.append(
            (step, json.loads((Path(path) / "_flash_resume_manifest.json").read_text()))
        )

    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "upload_resume_checkpoint", upload, raising=False
    )
    _write_step(local, 1)
    uploader = rl_checkpoints._VerlResumeUploader(str(local), resume_step=0, metric_evidence=state)
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and uploader._blocked_evidence_step != 1:
            time.sleep(0.01)
        assert uploaded == []
        state.record_grad_norm(1, 0.25)
        while time.monotonic() < deadline and not uploaded:
            time.sleep(0.01)
    finally:
        uploader.stop()
    assert uploaded[0][1]["first_positive_grad_step"] == 1


def test_required_adapter_upload_failure_blocks_later_native_replacement(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    native = []
    monkeypatch.setattr(
        rl_checkpoints._VerlResumeUploader,
        "_stage_deployable",
        lambda self, step, path: str(tmp_path / f"step-{step}"),
    )
    _write_internal_adapter(tmp_path / "step-2")
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "hf_upload_folder",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("hf down")),
        raising=False,
    )
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **kwargs: native.append(step),
        raising=False,
    )
    _write_step(local, 2)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local),
        resume_step=0,
        required_steps=(2,),
        metric_evidence=_metric_state((2, 0.5)),
    )
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and 2 not in uploader.staged_adapters:
            time.sleep(0.01)
        _write_step(local, 3)
    finally:
        uploader.stop()
    assert native == []
    with pytest.raises(RuntimeError, match=r"not durably published: \[2\]"):
        uploader.raise_if_incomplete()


def test_required_save_debt_prevents_a_later_checkpoint_from_replacing_recoverable_state(
    tmp_path, monkeypatch
):
    local = tmp_path / "local"
    local.mkdir()
    uploaded = []
    upload_started = threading.Event()
    release_upload = threading.Event()

    def blocked_upload(step, path, **kwargs):
        uploaded.append(step)
        if step == 1:
            upload_started.set()
            assert release_upload.wait(timeout=10)
        callback = kwargs.get("after_upload")
        if callback is not None:
            callback()

    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        blocked_upload,
        raising=False,
    )
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local),
        resume_step=0,
        required_steps=(2,),
        metric_evidence=_metric_state((1, 0.5), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)),
    )
    _write_step(local, 1)
    uploader.start()
    try:
        assert upload_started.wait(timeout=10)
        _write_step(local, 2)
        shutil.rmtree(local / "global_step_2")
        for step in (3, 4, 5):
            _write_step(local, step)
        release_upload.set()
    finally:
        release_upload.set()
        uploader.stop()
    assert uploaded == [1]
    with pytest.raises(RuntimeError, match=r"not durably published: \[2\]"):
        uploader.raise_if_incomplete()


def test_resume_manifest_and_internal_tree_fail_closed(tmp_path, monkeypatch):
    remote_root = tmp_path / "remote" / "checkpoint"
    checkpoint = remote_root / "checkpoint-4"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (actor / "fsdp_config.json").write_text('{"FSDP_version": 2, "world_size": 1}')
    for kind in ("model", "optim", "extra_state"):
        (actor / f"{kind}_world_size_1_rank_0.pt").write_bytes(b"shard")
    _write_resume_manifest(checkpoint, 4, required=(2,), positive=2)
    internal = remote_root / "required-adapters" / "attempt-0" / "step-2" / "adapter"
    _write_internal_adapter(internal, weights="sharded")
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "hf_resume_checkpoint", lambda **kwargs: str(checkpoint)
    )

    local = tmp_path / "local"
    local.mkdir()
    assert (
        rl_checkpoints._restore_verl_resume(
            str(local),
            world_size=1,
            expected_fsdp_generation=2,
            required_steps=(2,),
        )
        == 4
    )
    assert (local / "_flash_required_adapter_store" / "step-2" / "adapter").is_dir()

    (internal / "adapter_model-00002-of-00002.safetensors").unlink()
    broken = tmp_path / "broken"
    broken.mkdir()
    with pytest.raises(RuntimeError, match="required adapter"):
        rl_checkpoints._restore_verl_resume(
            str(broken),
            world_size=1,
            expected_fsdp_generation=2,
            required_steps=(2,),
        )


def test_internal_adapter_validation_accepts_complete_sharded_weights(tmp_path):
    adapter = tmp_path / "adapter"
    _write_internal_adapter(adapter, weights="sharded")
    rl_checkpoints._validate_internal_adapter_dir(str(adapter), label="step-2")


def test_internal_adapter_validation_rejects_incomplete_sharded_weights(tmp_path):
    adapter = tmp_path / "adapter"
    _write_internal_adapter(adapter, weights="incomplete-sharded")
    with pytest.raises(RuntimeError, match="missing adapter config or weights"):
        rl_checkpoints._validate_internal_adapter_dir(str(adapter), label="step-2")


@pytest.mark.parametrize("missing", ["config", "weights"])
def test_internal_adapter_validation_rejects_missing_required_file(tmp_path, missing):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    if missing != "config":
        (adapter / "adapter_config.json").write_text("{}")
    if missing != "weights":
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="missing adapter config or weights"):
        rl_checkpoints._validate_internal_adapter_dir(str(adapter), label="step-2")


def test_internal_adapter_validation_rejects_root_and_nested_symlinks(tmp_path):
    outside = tmp_path / "outside"
    _write_internal_adapter(outside)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="not a real directory"):
        rl_checkpoints._validate_internal_adapter_dir(str(root_link), label="step-2")

    nested = outside / "nested"
    nested.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("{}")
    (nested / "escape.json").symlink_to(target)
    with pytest.raises(RuntimeError, match=r"symlink 'escape.json'"):
        rl_checkpoints._validate_internal_adapter_dir(str(outside), label="step-2")


def test_published_required_adapter_is_not_restored_or_republished(tmp_path, monkeypatch):
    local = tmp_path / "local"
    checkpoint = local / "global_step_4"
    checkpoint.mkdir(parents=True)
    _write_resume_manifest(checkpoint, 4, required=(2,), positive=2)
    _write_internal_adapter(local / "_flash_required_adapter_store" / "step-2" / "adapter")
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "_deployable_adapter_on_hf", lambda step: step == 2
    )
    published = []
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local),
        resume_step=4,
        required_steps=(2,),
        metric_evidence=_metric_state(resume_step=4),
        export_root=str(tmp_path / "exports"),
    )
    uploader.credit_durable_required_steps(4)
    uploader.restore_staged_adapters(4)
    monkeypatch.setattr(uploader, "_publish_staged", lambda step, path: published.append(step))
    uploader.allow_deployable_publication()
    uploader._publish_ready()
    assert uploader.staged_adapters == {}
    assert published == []


def test_start_resume_uploader_hydrates_manifest_before_thread_start(tmp_path, monkeypatch):
    local = tmp_path / "local"
    checkpoint = local / "global_step_4"
    checkpoint.mkdir(parents=True)
    _write_resume_manifest(checkpoint, 4, required=(2,), positive=2)
    _write_internal_adapter(local / "_flash_required_adapter_store" / "step-2" / "adapter")
    started = []
    monkeypatch.setattr(rl_checkpoints._worker_hf, "_deployable_adapter_on_hf", lambda step: False)
    monkeypatch.setattr(
        rl_checkpoints._VerlResumeUploader,
        "start",
        lambda self: started.append(
            (sorted(self.staged_adapters), self.metric_evidence.prior_positive_grad_step)
        ),
    )
    state = _metric_state(resume_step=4)
    uploader = rl_train._start_resume_uploader(
        local_dir=str(local),
        resume_step=4,
        inp={
            "save_at_steps": (2,),
            "model_id": "Qwen/Qwen3.5-9B",
            "model_revision": "rev",
            "multimodal": False,
        },
        workdir=str(tmp_path),
        python_bin="python",
        preprocessor=object(),
        metric_evidence=state,
    )
    assert started == [([2], 2)]
    assert sorted(uploader.staged_adapters) == [2]


def test_no_required_saves_write_only_the_small_resume_manifest(tmp_path):
    checkpoint = _write_step(tmp_path, 1)
    state = _metric_state((1, 0.5))
    uploaded = []
    uploader = rl_checkpoints._VerlResumeUploader(
        str(tmp_path), resume_step=0, required_steps=(), metric_evidence=state
    )
    original = rl_checkpoints._worker_hf.upload_resume_checkpoint
    rl_checkpoints._worker_hf.upload_resume_checkpoint = lambda step, path, **kwargs: (
        uploaded.append(step)
    )
    try:
        uploader.start()
        uploader.stop()
    finally:
        rl_checkpoints._worker_hf.upload_resume_checkpoint = original
    assert uploaded == [1]
    assert (
        json.loads((checkpoint / "_flash_resume_manifest.json").read_text())["required_adapters"]
        == []
    )
    assert not (checkpoint / "_flash_required_adapters").exists()


def test_withheld_required_step_still_uploads_resume_state_exactly_once(tmp_path, monkeypatch):
    # withholding gates PUBLICATION only. the resume upload is internal retry scaffolding, and with
    # exact save_at_steps these are often the only on-disk checkpoints -- skipping it would leave a run
    # preempted before the first nonzero spread with nothing to resume from. neither the upload nor the
    # staging may repeat on every 0.5s sweep while the step waits for the gate.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    uploaded: list[int] = []
    staged: list[int] = []
    published: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: uploaded.append(int(step)),
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, staged, published)
    _write_step(local_dir, 3)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(3,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and 3 not in uploaded:
            time.sleep(0.05)
        # the gate is shut, so publication is withheld -- but resume state IS durable, and the
        # adapter is already staged out of verl's reach.
        assert uploaded == [3]
        assert staged == [3]
        assert published == []
        time.sleep(1.5)  # several sweeps: neither the upload nor the export may repeat
        assert uploaded == [3]
        assert staged == [3]
        uploader.allow_deployable_publication()
        while time.monotonic() < deadline and not published:
            time.sleep(0.05)
        assert published == [3]
        assert uploaded == [3]
        assert staged == [3]
    finally:
        uploader.stop()
    uploader.raise_if_incomplete()


def test_positive_early_gradient_never_publishes_before_terminal_validation(tmp_path, monkeypatch):
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    staged: list[int] = []
    published: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: True,
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, staged, published)
    _write_step(local_dir, 1)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(1,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not staged:
            time.sleep(0.01)
        assert staged == [1]
        time.sleep(0.6)
        assert published == []
    finally:
        uploader.stop()


def test_invalid_later_step_never_releases_an_early_positive_required_save(tmp_path, monkeypatch):
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    staged: list[int] = []
    published: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: True,
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, staged, published)
    _write_step(local_dir, 1)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(1,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not staged:
        time.sleep(0.01)

    state = rl_runner._StepMetricState()
    state.reward_history[:] = [0.5, 0.5]
    state.adv_spread_history[:] = [1.0]
    state.advantage_bounds[1] = (-0.5, 0.5)
    state.grad_norms[1] = 0.25
    with pytest.raises(RuntimeError, match=r"missing=\[2\]"):
        rl_runner._validate_rl_child(0, state, 0, 2, uploader)
    uploader.stop()

    assert staged == [1]
    assert published == []


def test_clean_terminal_validation_releases_staged_required_save(tmp_path, monkeypatch):
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    staged: list[int] = []
    published: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: True,
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, staged, published)
    _write_step(local_dir, 1)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(1,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not staged:
        time.sleep(0.01)
    assert published == []

    state = rl_runner._StepMetricState()
    state.reward_history.append(0.5)
    state.adv_spread_history.append(1.0)
    state.advantage_bounds[1] = (-0.5, 0.5)
    state.grad_norms[1] = 0.25
    rl_runner._validate_rl_child(0, state, 0, 1, uploader)

    assert published == [1]
    uploader.raise_if_incomplete()


def test_resumed_new_required_step_waits_for_terminal_validation(tmp_path, monkeypatch):
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    staged: list[int] = []
    published: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: True,
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, staged, published)
    _write_step(local_dir, 2)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=1,
        required_steps=(2,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not staged:
        time.sleep(0.01)
    assert published == []

    state = rl_runner._StepMetricState(resume_step=1)
    state.set_prior_positive_step(1, checkpoint_step=1)
    state.reward_history.append(0.5)
    state.adv_spread_history.append(0.0)
    state.advantage_bounds[2] = (0.0, 0.0)
    state.grad_norms[2] = 0.0
    rl_runner._validate_rl_child(0, state, 1, 2, uploader)

    assert published == [2]
    uploader.raise_if_incomplete()


def test_a_gate_already_open_publishes_the_deployable_before_the_resume_upload(
    tmp_path, monkeypatch
):
    """the opposite publication order to the withheld case above, and equally legitimate.

    with gradient evidence already present, a required step is staged and published on the sweep
    that finds it, BEFORE its resume upload runs. the withheld case reaches the same two facts in
    the other order. the lifecycle records them independently precisely so neither ordering has to
    be called the canonical one.
    """
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    order: list[str] = []
    staged: list[int] = []
    published: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: (order.append("resume"), k["after_upload"]())[0],
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, staged, published)
    _write_step(local_dir, 3)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(3,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.allow_deployable_publication()
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not published:
            time.sleep(0.05)
    finally:
        uploader.stop()

    assert published == [3]
    assert order.index("resume") == len(order) - 1, "the resume upload must not precede the publish"
    facts = uploader.lifecycle.facts(3)
    assert facts.staged
    assert facts.deployable_published
    assert facts.resume_uploaded
    uploader.raise_if_incomplete()


def test_a_failed_resume_upload_is_not_recorded_as_durable_and_stays_non_fatal(
    tmp_path, monkeypatch
):
    """a resume upload that raises leaves resume_uploaded unset without failing the run.

    grpo treats resume state as internal retry scaffolding: losing it costs restart distance, not
    the policy. the attempt is still recorded so a permanently failing upload cannot respin every
    0.5s, which is why "attempted" and "uploaded" have to be two different things.
    """
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    attempts: list[int] = []

    def exploding_upload(step, path, **kwargs):
        attempts.append(int(step))
        raise RuntimeError("hub is down")

    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "upload_resume_checkpoint", exploding_upload, raising=False
    )
    _patch_stage_and_publish(monkeypatch, [], [])
    _write_step(local_dir, 2)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not attempts:
            time.sleep(0.05)
        time.sleep(1.0)  # several sweeps: the failed upload must not be retried on each one
    finally:
        uploader.stop()

    assert attempts == [2], f"a permanently failing upload respun: {attempts}"
    assert not uploader.lifecycle.facts(2).resume_uploaded
    assert uploader.lifecycle.facts(2).discovered
    uploader.raise_if_incomplete()  # non-fatal: no required save was owed


def test_a_permanently_withheld_step_fails_the_run_and_does_not_hang_stop(tmp_path, monkeypatch):
    # the gate never opening must not wedge stop() waiting for a step it will never release, and the
    # run must still fail rather than silently ship without the customer's requested deployable.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: True,
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, [], [])
    _write_step(local_dir, 3)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(3,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    time.sleep(0.5)
    uploader.stop()  # must return, not hang
    with pytest.raises(RuntimeError, match="required saves were not durably published"):
        uploader.raise_if_incomplete()


def test_gate_opening_just_before_stop_still_publishes_rather_than_failing_on_timing(
    tmp_path, monkeypatch
):
    # the drain loop samples the gate once per sweep. if the main thread records the run's first
    # terminal validation can open the latch immediately before stop. publishing nothing would fail a
    # genuinely trained run for no reason but thread scheduling, so the stop sweep must observe it.
    # still act on the gate as it stands then.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    published: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: True,
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, [], published)
    _write_step(local_dir, 3)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(3,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not uploader.staged_adapters:
        time.sleep(0.01)
    uploader.allow_deployable_publication()
    uploader.stop()
    assert published == [3]
    uploader.raise_if_incomplete()


def test_resumed_required_step_can_still_publish_its_withheld_deployable(tmp_path, monkeypatch):
    # a previous worker resume-uploads a required checkpoint while withholding its adapter behind the
    # terminal latch, so the step is durable as resume state but not published. seeding the lifecycle
    # with resume_step would hide it from _pending forever, and completeness would then fail a run on
    # the one step this worker is both able and allowed to publish.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    published: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: True,
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, [], published)
    _write_step(local_dir, 4)
    # resumed at exactly the required step, and no adapter on hf for it, so it stays uncredited.
    monkeypatch.setattr(rl_checkpoints._worker_hf, "_deployable_adapter_on_hf", lambda step: False)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=4,
        required_steps=(4,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.credit_durable_required_steps(4)
    uploader.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not uploader.staged_adapters:
        time.sleep(0.01)
    assert published == []
    uploader.allow_deployable_publication()
    uploader.stop()
    assert published == [4]
    uploader.raise_if_incomplete()


def test_checkpoint_appearing_at_stop_is_uploaded_before_the_exit(tmp_path, monkeypatch):
    # verl advances latest_checkpointed_iteration.txt right up to the moment the child exits, so the
    # newest resume checkpoint can appear after the drain's last scan but before stop(). exiting
    # without sweeping that checkpoint would drop durable work a preemption then has to redo. resume
    # upload is not gated, and with the terminal latch shut nothing may be published.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    uploaded: list[int] = []
    staged: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: uploaded.append(int(step)),
        raising=False,
    )

    def stage(self, step, path):
        staged.append(int(step))
        adapter_dir = f"{path}-adapter"
        os.makedirs(adapter_dir, exist_ok=True)
        Path(adapter_dir, "adapter_config.json").write_text("{}")
        Path(adapter_dir, "adapter_model.safetensors").write_bytes(b"weights")
        return adapter_dir

    monkeypatch.setattr(rl_checkpoints._VerlResumeUploader, "_stage_deployable", stage)
    monkeypatch.setattr(
        rl_checkpoints._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (_ for _ in ()).throw(AssertionError("gate is shut")),
    )
    # the checkpoint must become visible AFTER a sweep has already decided what to scan, with stop
    # already set -- writing it between sweeps does not discriminate, because the next top-of-loop
    # scan picks it up either way. the tracker read is that boundary: _pending only accepts steps at
    # or below the value it returns, so a step written right after that read is invisible to the
    # sweep holding it and visible to the next one.
    real_completed = rl_checkpoints._VerlResumeUploader._completed_step
    raced = [False]

    def _completed_then_race(self):
        value = real_completed(self)
        if not raced[0]:
            raced[0] = True
            # verl finishes step 5 and advances its tracker here, then the child exits and the main
            # thread calls stop() -- all after this sweep already read the pre-step-5 tracker.
            _write_step(local_dir, 5)
            self._stop.set()
        return value

    monkeypatch.setattr(
        rl_checkpoints._VerlResumeUploader, "_completed_step", _completed_then_race, raising=True
    )
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(5,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and uploader._thread.is_alive():
        time.sleep(0.01)
    uploader.stop()
    assert uploaded == [5]
    # staged out of verl's reach on the same sweep, so the gate opening later can still publish it.
    assert staged == [5]


def test_required_step_publishes_after_verl_prunes_its_checkpoint(tmp_path, monkeypatch):
    # verl keeps only max_actor_ckpt_to_keep=3 actor checkpoints, so with four or more required steps
    # written before terminal validation it deletes the earliest source while its
    # deployable is still withheld. deferring the EXPORT until the gate opens would then leave that
    # step unpublishable and fail an otherwise valid run, so the export is staged under export_root
    # (flash's own workdir, outside verl's retention) and only the upload waits for the gate.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    staged: list[int] = []
    published: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: True,
        raising=False,
    )
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "hf_upload_folder", lambda *a, **kw: True, raising=False
    )

    def _stage_requiring_its_source(self, step, path):
        # the real _stage_deployable runs model_merger over <path>/actor, so it cannot succeed once
        # verl has pruned that directory. asserting it here is what makes this test fail on the
        # actual defect -- an unpublishable required step -- rather than on bookkeeping.
        if not os.path.isdir(path):
            raise AssertionError(f"staged step {step} after verl pruned {path}")
        staged.append(int(step))
        adapter = Path(f"{path}-adapter")
        _write_internal_adapter(adapter)
        return str(adapter)

    monkeypatch.setattr(
        rl_checkpoints._VerlResumeUploader, "_stage_deployable", _stage_requiring_its_source
    )
    monkeypatch.setattr(
        rl_checkpoints._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (
            published.append(int(step)),
            self.lifecycle.mark_deployable_published(step),
        )[0],
    )
    for step in (1, 2, 3, 4):
        (local_dir / f"global_step_{step}").mkdir()
    _write_step(local_dir, 4)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(1, 2, 3, 4),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and len(staged) < 4:
            time.sleep(0.01)
        # nothing may be servable yet: the gate is still shut.
        assert published == []
        # verl prunes the oldest checkpoints now that step 4 has landed -- exactly what strands a
        # step whose export was deferred until the gate opened.
        for step in (1, 2):
            shutil.rmtree(local_dir / f"global_step_{step}")
        uploader.allow_deployable_publication()
        while time.monotonic() < deadline and len(published) < 4:
            time.sleep(0.01)
    finally:
        uploader.stop()
    # every required step publishes, including the two whose verl checkpoints no longer exist. this
    # is asserted before `staged` so a deferred export fails here, on the unpublishable step, rather
    # than on the bookkeeping that led to it.
    assert published == [1, 2, 3, 4]
    uploader.raise_if_incomplete()
    assert staged == [1, 2, 3, 4]


def test_staging_failure_does_not_strand_an_earlier_publishable_step(tmp_path, monkeypatch):
    # a sweep can find several new checkpoints at once, and exporting one of them can fail (a corrupt
    # shard, a full disk, an OOM in model_merger). publishing only after the whole sweep finished let
    # that failure abort the thread with earlier, fully exported adapters still local-only -- and the
    # same window swallows a preemption during the resume upload that runs between the two. each step
    # is therefore made durable as soon as it is staged and permitted.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    published: list[int] = []
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: True,
        raising=False,
    )
    monkeypatch.setattr(
        rl_checkpoints._worker_hf, "hf_upload_folder", lambda *a, **kw: True, raising=False
    )

    def _stage_failing_on_step_2(self, step, path):
        if int(step) == 2:
            raise RuntimeError("model_merger ran out of memory")
        adapter = Path(f"{path}-adapter")
        _write_internal_adapter(adapter)
        return str(adapter)

    monkeypatch.setattr(
        rl_checkpoints._VerlResumeUploader, "_stage_deployable", _stage_failing_on_step_2
    )
    monkeypatch.setattr(
        rl_checkpoints._VerlResumeUploader,
        "_publish_staged",
        lambda self, step, adapter_dir: (
            published.append(int(step)),
            self.lifecycle.mark_deployable_published(step),
        )[0],
    )
    for step in (1, 2):
        (local_dir / f"global_step_{step}").mkdir()
    _write_step(local_dir, 2)
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(1, 2),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.allow_deployable_publication()
    uploader.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and uploader._error is None:
            time.sleep(0.01)
    finally:
        uploader.stop()
    # step 1 was exported before step 2 failed, so it must already be durable. the run still fails --
    # step 2 was required -- but a retry does not have to redo step 1, and step 1 is servable.
    assert published == [1]
    with pytest.raises(RuntimeError, match="verl resume uploader failed"):
        uploader.raise_if_incomplete()


def test_zero_gradient_is_reported_before_a_withheld_required_save(tmp_path, monkeypatch):
    # a zero-gradient run withholds every required deployable by design. checking completeness first
    # would raise on artifacts the gate is deliberately holding, reporting a checkpoint-publication
    # failure -- the symptom -- instead of the constant reward signal that caused it.
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    _write_step(local_dir, 6)
    monkeypatch.setattr(
        rl_checkpoints._worker_hf,
        "upload_resume_checkpoint",
        lambda step, path, **k: True,
        raising=False,
    )
    _patch_stage_and_publish(monkeypatch, [], [])
    uploader = rl_checkpoints._VerlResumeUploader(
        str(local_dir),
        resume_step=0,
        required_steps=(6,),
        metric_evidence=_always_ready_metric_evidence(),
    )
    uploader.start()
    uploader.stop()
    # both failures are live: the deployable was withheld, and the run produced only zero actor
    # gradients. the gradient verdict must be the one that speaks.
    with pytest.raises(RuntimeError, match="zero actor gradient norm"):
        rl_checkpoints._check_grpo_had_a_gradient(
            [0.5, 0.5], [0.0, 0.0], {1: 0.0, 2: 0.0}, expected_steps=range(1, 3)
        )
    with pytest.raises(RuntimeError, match="required saves were not durably published"):
        uploader.raise_if_incomplete()
    # ordering is asserted at the call site: the verdict precedes stop()/raise_if_incomplete().
    # match on the call name alone -- the argument list spans several lines, so pinning an argument
    # would make this fail on a reformat rather than on a reordering, which is the real invariant.
    source = inspect.getsource(rl_runner._validate_rl_child)
    assert source.count("_check_grpo_had_a_gradient(") == 1
    assert source.count("resume_uploader.raise_if_incomplete()") == 1
    verdict = source.index("_check_grpo_had_a_gradient(")
    completeness = source.index("resume_uploader.raise_if_incomplete()")
    assert verdict < completeness
