from __future__ import annotations

import sys
import types
import weakref
from dataclasses import dataclass

from flash.engine.worker.resident import ResidentBase, ResidentJobState, base_fingerprint
from flash.engine.worker_entrypoint import WORKER_FAILURE_LINE
from flash.engine.worker_resident import run_jobs
from flash.spec import JobSpec


@dataclass
class _FakeParameter:
    values: tuple[float, ...]
    requires_grad: bool = True
    shape: tuple[int, ...] = (2,)
    dtype: str = "float32"

    def _resident_fingerprint_bytes(self) -> bytes:
        return repr(self.values).encode()


class _FakeHookHandle:
    def __init__(self, hooks, hook_id) -> None:
        self._hooks = hooks
        self._hook_id = hook_id

    def remove(self) -> None:
        self._hooks.pop(self._hook_id, None)


class _FakeEmbedding:
    def __init__(self) -> None:
        self._forward_hooks = {}
        self._next_hook_id = 0

    def register_forward_hook(self, hook):
        hook_id = self._next_hook_id
        self._next_hook_id += 1
        self._forward_hooks[hook_id] = hook
        return _FakeHookHandle(self._forward_hooks, hook_id)


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.weight = _FakeParameter((1.0, 2.0))
        self.evaluated = False
        self.gradient_checkpointing = False
        self.embedding = _FakeEmbedding()
        self._require_grads_hooks = []

    def named_parameters(self):
        return [("weight", self.weight)]

    def eval(self):
        self.evaluated = True
        return self

    def get_input_embeddings(self):
        return self.embedding

    def enable_input_require_grads(self):
        def make_inputs_require_grads(_module, _inputs, _output):
            return None

        handle = self.embedding.register_forward_hook(make_inputs_require_grads)
        self._require_grads_hooks = [handle]
        self._require_grads_hook = handle

    def disable_input_require_grads(self):
        for handle in self._require_grads_hooks:
            handle.remove()
        self._require_grads_hooks = []
        if hasattr(self, "_require_grads_hook"):
            del self._require_grads_hook

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False


class _FakeTokenizer:
    pass


class _FakePeftModel:
    def __init__(self, base: _FakeModel) -> None:
        self.base = base
        self.peft_config = {"default": object()}
        self.unloaded = False

    def named_parameters(self):
        return [
            *self.base.named_parameters(),
            ("layer.lora_A.default.weight", _FakeParameter((3.0, 4.0))),
        ]

    def get_base_model(self):
        return self.base

    def unload(self):
        self.peft_config.clear()
        self.unloaded = True
        return self.base


class _FakeTrainer:
    def __init__(self, model) -> None:
        self.model = model
        self.optimizer = object()
        self.lr_scheduler = object()
        self.train_dataset = object()
        self.eval_dataset = object()
        self.data_collator = object()


class _FakeDataset:
    def __init__(self) -> None:
        self.cleaned = False

    def cleanup_cache_files(self):
        self.cleaned = True


def _holder(loads, tokenizers, cache_calls):
    def load_model(model_id, loader_kind, kwargs):
        model = _FakeModel(f"{model_id}:{len(loads)}")
        loads.append((model_id, loader_kind, kwargs, model))
        return model

    def load_tokenizer(model_id, revision):
        tokenizer = _FakeTokenizer()
        tokenizers.append((model_id, revision, tokenizer))
        return tokenizer

    return ResidentBase(
        model_loader=load_model,
        tokenizer_loader=load_tokenizer,
        gpu_arch=lambda: (9, 0),
        empty_cuda_cache=lambda: cache_calls.append(True),
    )


def _get(holder, model_id="fake/model", *, revision="rev-a", attn="sdpa"):
    return holder.get_base(
        model_id,
        revision=revision,
        model_init_kwargs={
            "dtype": "bfloat16",
            "device_map": None,
            "attn_implementation": attn,
        },
        context_length=128,
        loader_kind="causal-lm",
    )


def test_causal_loader_uses_trl_default_model_resolution(monkeypatch):
    calls = []
    expected = object()

    def create_model_from_path(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return expected

    trl_module = types.ModuleType("trl")
    trainer_module = types.ModuleType("trl.trainer")
    utils_module = types.ModuleType("trl.trainer.utils")
    utils_module.create_model_from_path = create_model_from_path
    monkeypatch.setitem(sys.modules, "trl", trl_module)
    monkeypatch.setitem(sys.modules, "trl.trainer", trainer_module)
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils_module)
    kwargs = {"dtype": "bfloat16", "device_map": None, "revision": "rev-a"}

    loaded = ResidentBase._load_model("fake/model", "causal-lm", kwargs)

    assert loaded is expected
    assert calls == [("fake/model", kwargs)]
    assert "trust_remote_code" not in calls[0][1]


def test_get_base_caches_matching_compatibility_key():
    loads = []
    tokenizers = []
    cache_calls = []
    holder = _holder(loads, tokenizers, cache_calls)

    first_model, first_tokenizer = _get(holder)
    second_model, second_tokenizer = _get(holder)

    assert second_model is first_model
    assert second_tokenizer is first_tokenizer
    assert len(loads) == 1
    assert len(tokenizers) == 1
    assert first_model.weight.requires_grad is False
    assert "trust_remote_code" not in loads[0][2]


def test_get_base_key_change_evicts_and_reloads():
    loads = []
    tokenizers = []
    cache_calls = []
    holder = _holder(loads, tokenizers, cache_calls)

    first_model, _ = _get(holder)
    second_model, _ = _get(holder, attn="flash_attention_2")

    assert second_model is not first_model
    assert len(loads) == 2
    assert len(tokenizers) == 1
    assert cache_calls == [True, True]


def test_kwargs_only_revision_changes_compatibility_and_tokenizer_keys():
    loads = []
    tokenizers = []
    cache_calls = []
    holder = _holder(loads, tokenizers, cache_calls)

    first_model, _ = holder.get_base(
        "fake/model",
        model_init_kwargs={"dtype": "bfloat16", "device_map": None, "revision": "rev-a"},
        context_length=128,
        loader_kind="causal-lm",
    )
    second_model, _ = holder.get_base(
        "fake/model",
        model_init_kwargs={"dtype": "bfloat16", "device_map": None, "revision": "rev-b"},
        context_length=128,
        loader_kind="causal-lm",
    )

    assert second_model is not first_model
    assert [load[2]["revision"] for load in loads] == ["rev-a", "rev-b"]
    assert [tokenizer[1] for tokenizer in tokenizers] == ["rev-a", "rev-b"]


def test_reset_after_job_unloads_adapter_and_preserves_fingerprint():
    loads = []
    tokenizers = []
    cache_calls = []
    holder = _holder(loads, tokenizers, cache_calls)
    base, _ = _get(holder)
    before = base_fingerprint(base)
    wrapped = _FakePeftModel(base)
    trainer = _FakeTrainer(wrapped)
    dataset = _FakeDataset()

    holder.reset_after_job(wrapped, trainer=trainer, dataset=dataset)

    assert wrapped.unloaded is True
    assert wrapped.peft_config == {}
    assert base_fingerprint(base) == before
    assert trainer.model is None
    assert trainer.optimizer is None
    assert trainer.train_dataset is None
    assert dataset.cleaned is True


def test_reset_after_job_removes_accumulated_input_grad_hooks():
    loads = []
    tokenizers = []
    cache_calls = []
    holder = _holder(loads, tokenizers, cache_calls)
    base, _ = _get(holder)

    for _cycle in range(3):
        base.gradient_checkpointing = True
        base.enable_input_require_grads()
        base.enable_input_require_grads()
        assert len(base.embedding._forward_hooks) == 2

        holder.reset_after_job(_FakePeftModel(base))

        assert base.gradient_checkpointing is False
        assert base.embedding._forward_hooks == {}
        assert base._require_grads_hooks == []
        assert not hasattr(base, "_require_grads_hook")


def _job(run_id: str) -> JobSpec:
    return JobSpec(
        model="fake/model",
        algorithm="sft",
        run_id=run_id,
        model_revision="a" * 40,
        model_policy="allow",
    )


def test_resident_loop_runs_two_jobs_with_one_base_load(monkeypatch):
    import flash.engine.worker as worker

    loads = []
    tokenizers = []
    cache_calls = []
    holder = _holder(loads, tokenizers, cache_calls)
    models = []
    monkeypatch.setattr(worker, "wandb_finish", lambda **_kwargs: None)

    def run_job(_worker, resident):
        model, _ = resident.get_base(
            "fake/model",
            revision="a" * 40,
            model_init_kwargs={"dtype": "bfloat16", "device_map": None},
            context_length=128,
            loader_kind="causal-lm",
        )
        models.append(model)
        wrapped = _FakePeftModel(model)
        return ResidentJobState(model=wrapped, trainer=_FakeTrainer(wrapped))

    status = run_jobs(
        [_job("job-one"), _job("job-two")],
        resident=holder,
        run_job=run_job,
        prepare_process=False,
    )

    assert status == 0
    assert models[1] is models[0]
    assert len(loads) == 1


def test_resident_loop_releases_completed_state_before_key_change_load(monkeypatch):
    import flash.engine.worker as worker

    load_count = 0
    collected_before_reload = []
    wrapper_ref = None
    base_ref = None

    def load_model(model_id, _loader_kind, _kwargs):
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            collected_before_reload.append((wrapper_ref() is None, base_ref() is None))
        return _FakeModel(f"{model_id}:{load_count}")

    holder = ResidentBase(
        model_loader=load_model,
        tokenizer_loader=lambda _model_id, _revision: _FakeTokenizer(),
        gpu_arch=lambda: (9, 0),
        empty_cuda_cache=lambda: None,
    )
    calls = 0
    monkeypatch.setattr(worker, "wandb_finish", lambda **_kwargs: None)

    def run_job(_worker, resident):
        nonlocal calls, wrapper_ref, base_ref
        calls += 1
        revision = "rev-a" if calls == 1 else "rev-b"
        base, _ = resident.get_base(
            "fake/model",
            revision=revision,
            model_init_kwargs={"dtype": "bfloat16", "device_map": None},
            context_length=128,
            loader_kind="causal-lm",
        )
        wrapped = _FakePeftModel(base)
        if calls == 1:
            wrapper_ref = weakref.ref(wrapped)
            base_ref = weakref.ref(base)
        return ResidentJobState(model=wrapped, trainer=_FakeTrainer(wrapped))

    status = run_jobs(
        [_job("job-one"), _job("job-two")],
        resident=holder,
        run_job=run_job,
        prepare_process=False,
    )

    assert status == 0
    assert collected_before_reload == [(True, True)]


def test_resident_loop_stops_after_job_error(monkeypatch, capsys):
    import flash.engine.worker as worker

    loads = []
    tokenizers = []
    cache_calls = []
    holder = _holder(loads, tokenizers, cache_calls)
    calls = []
    monkeypatch.setattr(worker, "wandb_finish", lambda **_kwargs: None)

    def fail_job(_worker, _resident):
        calls.append(True)
        raise RuntimeError("private failure detail")

    status = run_jobs(
        [_job("job-one"), _job("job-two")],
        resident=holder,
        run_job=fail_job,
        prepare_process=False,
    )

    captured = capsys.readouterr()
    assert status == 1
    assert len(calls) == 1
    assert captured.err.strip() == WORKER_FAILURE_LINE
    assert "private failure detail" not in captured.err
