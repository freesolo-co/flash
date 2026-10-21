"""Offline policy tests for the explicit upstream integration."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.test_verl_child_plugins import _import_recipe, _modules


@pytest.mark.parametrize("source", [None, "fork", "wrong_commit", "upstream"])
def test_installed_source_must_match_exact_upstream_pin(monkeypatch, source):
    from flash.engine.worker.train.core.child import upstream
    from flash.engine.worker.train.entry.backend_common import VERL_REQUIREMENT_URL

    url, commit = VERL_REQUIREMENT_URL.removeprefix("git+").rsplit("@", 1)
    metadata = {"url": url, "vcs_info": {"vcs": "git", "commit_id": commit}}
    if source == "fork":
        metadata["url"] = "https://github.com/freesolo-co/verl"
    elif source == "wrong_commit":
        metadata["vcs_info"]["commit_id"] = "0" * 40
    payload = None if source is None else json.dumps(metadata)
    monkeypatch.setattr(
        upstream.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda filename: payload),
    )
    if source == "upstream":
        upstream.validate_upstream()
    else:
        with pytest.raises(RuntimeError, match="pinned unmodified upstream"):
            upstream.validate_upstream()


@pytest.mark.parametrize("loaded", [False, True])
def test_server_requires_training_adapter_and_preserves_exact_finish_reason(monkeypatch, loaded):
    calls = []

    class Server:
        async def generate(self, *args, **kwargs):
            calls.append("generated")
            return "tokens"

        def _postprocess_output(self, output, final_res, sampling_params):
            return output

    _modules(
        monkeypatch,
        {
            "ray": {"remote": lambda cls: cls},
            "verl.workers.rollout.vllm_rollout.utils": {"VLLM_LORA_INT_ID": 123},
            "verl.workers.rollout.vllm_rollout.vllm_async_server": {
                "vLLMHttpServer": Server,
                "vLLMReplica": object,
            },
        },
    )
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.core.child.rollout")
    server = recipe.FlashVLLMServer()
    server.lora_as_adapter = True

    async def list_loras():
        return [123] if loaded else []

    server.engine = SimpleNamespace(list_loras=list_loras)
    if loaded:
        assert asyncio.run(server.generate()) == "tokens"
        assert calls == ["generated"]
    else:
        with pytest.raises(RuntimeError, match="refusing base-model generation"):
            asyncio.run(server.generate())
        assert not calls
    output = SimpleNamespace(extra_fields={})
    server._postprocess_output(
        output, SimpleNamespace(outputs=[SimpleNamespace(finish_reason="length")]), {}
    )
    assert output.extra_fields == {"finish_reason": "length"}


@pytest.mark.parametrize(("resume", "fail"), [(0, False), (2, False), (4, False), (0, True)])
def test_sft_loop_owns_step_horizon_checkpoints_and_tracking(monkeypatch, resume, fail):
    events = []

    class Tracking:
        def __init__(self, **kwargs):
            pass

        def log(self, *, data, step):
            events.append(("log", step))

        def finish(self, exit_code):
            events.append(("finish", exit_code))

    _modules(
        monkeypatch,
        {
            "omegaconf": {"OmegaConf": SimpleNamespace(to_container=lambda *a, **kw: {})},
            "tensordict": {"NonTensorData": object},
            "verl.utils": {"tensordict_utils": object},
            "flash.engine.worker.train.core.child.tracking": {"OwnedTracking": Tracking},
        },
    )
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.sft.child.loop")

    def train_batch(trainer, batch, step):
        if fail:
            raise ValueError("optimizer failed")
        events.append(("train", step))
        return {"train/global_tokens": 3}

    monkeypatch.setattr(recipe, "_train_batch", train_batch)
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            trainer=SimpleNamespace(
                project_name="flash", experiment_name="run", logger=[], total_epochs=2
            )
        ),
        engine=SimpleNamespace(
            is_mp_src_rank_with_outputs=lambda: True, get_data_parallel_rank=lambda: 0
        ),
        resume_global_step=resume,
        total_training_steps=4,
        steps_per_epoch=2,
        train_sampler=SimpleNamespace(set_epoch=lambda epoch: events.append(("epoch", epoch))),
        train_dataloader=["batch1", "batch2"],
        save_freq=2,
        ckpt_handler=SimpleNamespace(save_checkpoint=lambda step: events.append(("save", step))),
    )
    if fail:
        with pytest.raises(ValueError, match="optimizer failed"):
            recipe.fit_sft(trainer)
        assert events == [("epoch", 0), ("finish", 1)]
    else:
        recipe.fit_sft(trainer)
        assert [step for kind, step in events if kind == "train"] == list(range(resume + 1, 5))
        assert [step for kind, step in events if kind == "save"] == list(range(resume + 2, 5, 2))
        assert events[-1] == ("finish", 0)
