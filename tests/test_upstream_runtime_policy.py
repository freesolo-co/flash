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


def test_rollout_versions_are_isolated_across_reused_concurrent_episodes():
    from flash.engine.worker.train.core.child.observations import ObservedRolloutClient

    class Client:
        async def generate(self, version):
            await asyncio.sleep(0)
            return SimpleNamespace(extra_fields={"global_steps": version})

    observed = ObservedRolloutClient(Client())

    async def run(versions):
        with observed.episode():
            for version in versions:
                await observed.generate(version)
            return observed.annotate(SimpleNamespace(extra_fields={})).extra_fields

    async def exercise():
        assert await run([1, 2]) == {"min_global_steps": 1, "max_global_steps": 2}
        assert await asyncio.gather(run([5, 6]), run([20, 21])) == [
            {"min_global_steps": 5, "max_global_steps": 6},
            {"min_global_steps": 20, "max_global_steps": 21},
        ]
        with pytest.raises(RuntimeError, match="episode scope"):
            await observed.generate(99)
        with pytest.raises(ValueError, match="failed"), observed.episode():
            raise ValueError("failed")
        assert await run([30]) == {"min_global_steps": 30, "max_global_steps": 30}

    asyncio.run(exercise())


@pytest.mark.parametrize("failed_property", ["run", "url", "id", None])
@pytest.mark.parametrize("owned", [False, True])
def test_tracking_link_lookup_is_optional(monkeypatch, capsys, failed_property, owned):
    class Run:
        @property
        def url(self):
            if failed_property == "url":
                raise RuntimeError("offline")
            return "https://wandb.example/run"

        @property
        def id(self):
            if failed_property == "id":
                raise RuntimeError("offline")
            return "run-id"

    class Logger:
        @property
        def run(self):
            if failed_property == "run":
                raise RuntimeError("offline")
            return Run()

    class Tracking:
        def __init__(self):
            self.logger = {"wandb": Logger()}

    _modules(monkeypatch, {"verl.utils.tracking": {"Tracking": Tracking}})
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.core.child.tracking")
    if owned:
        recipe.OwnedTracking()
    else:
        recipe.publish_tracking_link(Tracking())
    output = capsys.readouterr().out
    if failed_property:
        assert output == ""
    else:
        assert json.loads(output.removeprefix("FLASH_WANDB_LINK ")) == {
            "wandb_url": "https://wandb.example/run",
            "wandb_id": "run-id",
        }


@pytest.mark.parametrize("multimodal", [False, True])
def test_opd_parent_passes_model_checkpointing_and_vision_policy(
    worker_context, monkeypatch, multimodal
):
    from flash.engine.worker.train.entry import opd_train_runner as runner

    request = SimpleNamespace(
        model_id="Qwen/Qwen3.5-9B",
        knobs=SimpleNamespace(stop_sequences=(), save_at_steps=()),
        structured_outputs=None,
        model_vocab_size=248320,
        multi_turn=False,
        max_turns=1,
    )
    workload = SimpleNamespace(
        runtime_dir="/runtime",
        update_horizon=2,
        exclude_modules=None,
        multimodal=multimodal,
        **dict.fromkeys(
            (
                "mutation_failure_path",
                "score_delivery_failure_path",
                "rollout_failure_path",
                "abandonment_failure_path",
                "resample_failure_path",
                "cycle_commit_failure_path",
            ),
            "",
        ),
    )
    runtime = SimpleNamespace(
        bridge=SimpleNamespace(url="http://teacher", token="test"),
        loggers=[],
        gdn_reset_arch="qwen3_5",
    )
    monkeypatch.setattr(runner._opd_overrides, "_build_opd_child_env", lambda **kwargs: kwargs)
    child = runner._build_child_env(
        request, SimpleNamespace(max_model_len=256), workload, runtime, (), context=worker_context
    )
    options = json.loads(child["launch_config"])
    assert options["reentrant_checkpointing"] is True
    assert options["multimodal"] is multimodal


@pytest.mark.parametrize("lengths", [(89, 89), (19, 23), (4, 4), (1, 1, 1, 1)])
def test_multimodal_batch_decoding_preserves_each_position_axis(lengths):
    import torch

    from flash.engine.worker.train.core.child.batch import normalize_position_ids

    expected = [torch.arange(4 * n).reshape(4, n) + 1000 * i for i, n in enumerate(lengths)]
    queued = torch.nested.as_nested_tensor(expected, layout=torch.jagged)
    data = {
        "input_ids": torch.nested.as_nested_tensor(
            [torch.arange(n) for n in lengths], layout=torch.jagged
        ),
        "position_ids": queued,
    }
    normalize_position_ids(data)
    normalize_position_ids(data)
    for actual, reference in zip(data["position_ids"].unbind(), expected, strict=True):
        torch.testing.assert_close(actual, reference)


def test_multimodal_batch_decoding_rejects_mismatched_token_lengths():
    import torch

    from flash.engine.worker.train.core.child.batch import normalize_position_ids

    data = {
        "input_ids": torch.nested.as_nested_tensor([torch.arange(9)] * 2, layout=torch.jagged),
        "position_ids": torch.nested.as_nested_tensor([torch.ones(4, 7)] * 2, layout=torch.jagged),
    }
    with pytest.raises(ValueError, match="do not match token lengths"):
        normalize_position_ids(data)
