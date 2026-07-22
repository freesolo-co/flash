"""cpu coverage for the OpenRLHF GRPO foundation."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib.util
import json
import logging
import math
import os
import sys
import types
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.engine.worker import grpo_openrlhf, rl

_OPENRLHF_SOURCE = Path(os.environ.get("FLASH_TEST_OPENRLHF_SOURCE", "/mnt/resource/openrlhf-src"))
requires_openrlhf_source = pytest.mark.skipif(
    not _OPENRLHF_SOURCE.joinpath("openrlhf/cli/train_ppo_ray.py").is_file(),
    reason="pinned OpenRLHF source is unavailable",
)
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is unavailable in offline CI",
)


def _value(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]


def _config(**overrides) -> grpo_openrlhf.OpenRLHFGRPOConfig:
    values = {
        "model_path": "/cache/models--Qwen--Qwen3.5-0.8B/snapshots/" + "a" * 40,
        "dataset_path": "/work/train.jsonl",
        "reward_url": "http://127.0.0.1:1234/reward/token",
        "output_dir": "/work/final",
        "checkpoint_dir": "/work/checkpoints",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_revision": "a" * 40,
        "max_length": 4096,
        "max_completion": 320,
        "prompts_per_step": 16,
        "group_size": 8,
        "scheduled_prompt_count": 48,
        "learning_rate": 1e-5,
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": 42,
        "lora_rank": 32,
        "lora_alpha": 64,
        "kl_coef": 0.0,
        "save_every": -1,
        "save_at_steps": (),
        "gpu_count": 2,
        "qwen35_language_model_only": True,
        "fp8_kv": False,
        "warmstart_adapter": "",
        "resume": False,
        "actor_attn_implementation": "flash_attention_3",
    }
    values.update(overrides)
    return grpo_openrlhf.OpenRLHFGRPOConfig(**values)


def _install_resolve_inputs_fakes(monkeypatch, *, save_every=None):
    class _Env:
        multi_turn = False
        is_tool_env = False

        def dataset(self):
            return [{"answer": "ok"}]

        def prompt_messages(self, _example):
            return [{"role": "user", "content": "prompt"}]

    class _Tokenizer:
        pad_token = None
        eos_token = "</s>"

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_ids=[1, 2])

    train = SimpleNamespace(
        init_from_adapter=False,
        save_at_steps=(),
        save_every=save_every,
        stop_sequences=(),
        structured_outputs="",
        credit_assignment="per_episode",
        batch_size=1,
        learning_rate=0.0,
        lora_rank=0,
        lora_alpha=0,
        max_context_tokens=16,
        max_examples=0,
        epochs=1,
        max_steps=None,
    )
    spec = SimpleNamespace(
        algorithm="grpo",
        train=train,
        model="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        seed=7,
        gpu=SimpleNamespace(count=1),
    )
    recipe = SimpleNamespace(
        rl=SimpleNamespace(
            prompts_per_step=8,
            group_size=8,
            sampling_temperature=0.7,
            learning_rate=1e-5,
            max_completion_len_thinking=8,
            max_completion_len=8,
            max_prompt_len=8,
            num_epochs=1,
            sampling_top_p=0.95,
        ),
        lora=SimpleNamespace(dropout=0.0, rank=32, alpha=64),
    )
    monkeypatch.setattr(grpo_openrlhf, "RECIPE", recipe)
    monkeypatch.setattr(grpo_openrlhf._w, "JOB_SPEC", spec)
    monkeypatch.setattr(grpo_openrlhf._w, "THINKING", False)
    monkeypatch.setattr(grpo_openrlhf._w, "require_active_env", lambda: _Env())
    monkeypatch.setattr(
        grpo_openrlhf._w,
        "grpo_overrides",
        lambda: {
            "group_size": 2,
            "temperature": 0.0,
            "thinking_length_penalty_coef": 0.0,
            "kl_penalty_coef": 0.0,
            "max_tokens": 4,
        },
    )
    monkeypatch.setattr(grpo_openrlhf._w, "load_tokenizer", lambda *_args, **_kwargs: _Tokenizer())
    monkeypatch.setattr(grpo_openrlhf._w, "resolve_grpo_prompts_per_step", lambda value, _n: value)
    return train


def _hierarchize(namespace):
    root = SimpleNamespace()
    for dotted_name, value in vars(namespace).items():
        target = root
        parts = dotted_name.split(".")
        for part in parts[:-1]:
            child = getattr(target, part, None)
            if child is None:
                child = SimpleNamespace()
                setattr(target, part, child)
            target = child
        setattr(target, parts[-1], value)
    return root


def _parse_with_pinned_openrlhf(monkeypatch, args: list[str]):
    source_path = _OPENRLHF_SOURCE / "openrlhf/cli/train_ppo_ray.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    main_block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    final_call = main_block.body[-1]
    assert isinstance(final_call, ast.Expr)
    assert isinstance(final_call.value, ast.Call)
    assert isinstance(final_call.value.func, ast.Name)
    assert final_call.value.func.id == "train"
    main_block.body[-1] = ast.Assign(
        targets=[ast.Name(id="_flash_captured_args", ctx=ast.Store())],
        value=ast.Name(id="args", ctx=ast.Load()),
    )
    ast.fix_missing_locations(tree)

    ray = types.ModuleType("ray")
    ray.util = types.ModuleType("ray.util")
    ray.util.placement_group = types.ModuleType("ray.util.placement_group")
    ray.util.placement_group.placement_group = lambda *_args, **_kwargs: None
    openrlhf = types.ModuleType("openrlhf")
    openrlhf.__path__ = []
    trainer = types.ModuleType("openrlhf.trainer")
    trainer.__path__ = []
    trainer_ray = types.ModuleType("openrlhf.trainer.ray")
    trainer_ray.__path__ = []
    trainer_ray.create_vllm_engines = object()
    launcher = types.ModuleType("openrlhf.trainer.ray.launcher")
    launcher.RayActorGroup = object
    launcher.ReferenceModelActor = object
    launcher.RewardModelActor = object
    ppo_actor = types.ModuleType("openrlhf.trainer.ray.ppo_actor")
    ppo_actor.PolicyModelActor = object
    ppo_critic = types.ModuleType("openrlhf.trainer.ray.ppo_critic")
    ppo_critic.CriticModelActor = object
    utils = types.ModuleType("openrlhf.utils")
    utils.__path__ = []
    utils.get_strategy = lambda *_args, **_kwargs: None
    config = types.ModuleType("openrlhf.utils.config")
    config.hierarchize = _hierarchize

    modules = {
        "ray": ray,
        "ray.util": ray.util,
        "ray.util.placement_group": ray.util.placement_group,
        "openrlhf": openrlhf,
        "openrlhf.trainer": trainer,
        "openrlhf.trainer.ray": trainer_ray,
        "openrlhf.trainer.ray.launcher": launcher,
        "openrlhf.trainer.ray.ppo_actor": ppo_actor,
        "openrlhf.trainer.ray.ppo_critic": ppo_critic,
        "openrlhf.utils": utils,
        "openrlhf.utils.config": config,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", [str(source_path), *args])
    namespace = {"__name__": "__main__", "__file__": str(source_path)}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["_flash_captured_args"]


def _source_module(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _package(monkeypatch, name: str):
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules[parent_name]
        monkeypatch.setattr(parent, child_name, module, raising=False)
    return module


def _install_pinned_sitecustomize_modules(
    monkeypatch, *, language_model_only=False, sm120_vllm_backend=False
):
    torch = pytest.importorskip("torch")
    openrlhf = _package(monkeypatch, "openrlhf")
    models = _package(monkeypatch, "openrlhf.models")
    trainer = _package(monkeypatch, "openrlhf.trainer")
    trainer_ray = _package(monkeypatch, "openrlhf.trainer.ray")
    ppo_utils = _package(monkeypatch, "openrlhf.trainer.ppo_utils")
    utils = _package(monkeypatch, "openrlhf.utils")
    del openrlhf, trainer

    ray = types.ModuleType("ray")

    class _RayActorClass:
        def __init__(self, modified_class):
            self.__ray_metadata__ = SimpleNamespace(modified_class=modified_class)
            for name, value in vars(modified_class).items():
                if callable(value):
                    setattr(self, name, value)

    def remote(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _RayActorClass(args[0])
        return lambda target: _RayActorClass(target)

    ray.remote = remote
    ray.get = lambda value: value
    ray._private = SimpleNamespace(
        services=SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
    )
    monkeypatch.setitem(sys.modules, "ray", ray)

    deepspeed = types.ModuleType("deepspeed")

    class _GatheredParameters:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return None

    deepspeed.zero = SimpleNamespace(GatheredParameters=_GatheredParameters)
    deepspeed.module_inject = SimpleNamespace(
        layers=SimpleNamespace(GatherReplacedLayerParams=_GatheredParameters)
    )
    deepspeed.initialize_calls = []

    def deepspeed_initialize(*args, **kwargs):
        deepspeed.initialize_calls.append((args, kwargs))
        return "engine", kwargs.get("optimizer"), None, "scheduler"

    deepspeed.initialize = deepspeed_initialize
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)

    model_utils = types.ModuleType("openrlhf.models.utils")

    def masked_mean(tensor, mask, dim=None):
        numerator = (tensor * mask).sum(dim=dim)
        denominator = mask.sum(dim=dim).clamp(min=1)
        return numerator / denominator

    model_utils.masked_mean = masked_mean
    model_utils.compute_approx_kl = lambda action, base, **_kwargs: action - base
    monkeypatch.setitem(sys.modules, "openrlhf.models.utils", model_utils)
    monkeypatch.setattr(models, "utils", model_utils, raising=False)
    loss_module = _source_module(
        monkeypatch,
        "openrlhf.models.loss",
        _OPENRLHF_SOURCE / "openrlhf/models/loss.py",
    )
    monkeypatch.setattr(models, "loss", loss_module, raising=False)
    models.PolicyLoss = loss_module.PolicyLoss
    models.aggregate_loss = loss_module.aggregate_loss

    class _Actor:
        def __init__(self, *_args, **_kwargs):
            self.model = torch.nn.Linear(2, 2, bias=False)
            self.packing_samples = False
            self.temperature = 1.0

        def forward(self, *_args, **_kwargs):
            raise NotImplementedError

    models.Actor = _Actor
    actor_module = types.ModuleType("openrlhf.models.actor")
    actor_module.Actor = _Actor
    monkeypatch.setitem(sys.modules, "openrlhf.models.actor", actor_module)

    peft = types.ModuleType("peft")
    peft.loads = []
    peft.adapter_loads = []
    peft.zero_adapter = False
    peft.load_result = SimpleNamespace(missing_keys=[], unexpected_keys=[])

    class _FakePeftModel(torch.nn.Module):
        def __init__(self, base, *, is_trainable):
            super().__init__()
            self.base = base
            self.proj = torch.nn.Module()
            self.proj.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False)})
            self.proj.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False)})
            with torch.no_grad():
                self.proj.lora_B["default"].weight.fill_(0.0 if peft.zero_adapter else 1.0)
            self.is_trainable = is_trainable

        @classmethod
        def from_pretrained(cls, base, path, *, adapter_name, is_trainable, key_mapping):
            assert adapter_name == "default"
            assert key_mapping is None
            model = cls(base, is_trainable=is_trainable)
            model.load_adapter(
                path,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                key_mapping=key_mapping,
            )
            peft.loads.append((path, is_trainable, model))
            return model

        def load_adapter(self, path, *, adapter_name, is_trainable, key_mapping):
            peft.adapter_loads.append((path, adapter_name, is_trainable, key_mapping))
            return peft.load_result

    peft.PeftModel = _FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", peft)

    utils.get_tokenizer = lambda *_args, **_kwargs: None
    utils_utils = types.ModuleType("openrlhf.utils.utils")

    def zero_pad_sequences(sequences, side="right", value=0, stack=False):
        max_length = max(sequence.shape[-1] for sequence in sequences)
        padded = []
        for sequence in sequences:
            width = max_length - sequence.shape[-1]
            padding = (0, width) if side == "right" else (width, 0)
            padded.append(torch.nn.functional.pad(sequence, padding, value=value))
        return torch.stack(padded) if stack else padded

    utils_utils.zero_pad_sequences = zero_pad_sequences
    monkeypatch.setitem(sys.modules, "openrlhf.utils.utils", utils_utils)

    experience_module = _source_module(
        monkeypatch,
        "openrlhf.trainer.ppo_utils.experience",
        _OPENRLHF_SOURCE / "openrlhf/trainer/ppo_utils/experience.py",
    )
    ppo_utils.Experience = experience_module.Experience

    logging_utils = types.ModuleType("openrlhf.utils.logging_utils")
    logging_utils.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "openrlhf.utils.logging_utils", logging_utils)

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = type("SamplingParams", (), {})

    class _EngineArgs:
        def __init__(
            self,
            *,
            kv_cache_dtype=None,
            language_model_only=False,
            attention_backend=None,
        ):
            self.kv_cache_dtype = kv_cache_dtype
            self.language_model_only = language_model_only
            self.attention_backend = attention_backend

    class _AsyncEngineArgs:
        def __init__(
            self,
            *,
            kv_cache_dtype=None,
            language_model_only=False,
            attention_backend=None,
        ):
            self.kv_cache_dtype = kv_cache_dtype
            self.language_model_only = language_model_only
            self.attention_backend = attention_backend

    vllm.EngineArgs = _EngineArgs
    vllm.AsyncEngineArgs = _AsyncEngineArgs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    vllm_engine = types.ModuleType("openrlhf.trainer.ray.vllm_engine")
    vllm_engine.batch_vllm_engine_call = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ray.vllm_engine", vllm_engine)
    samples_module = _source_module(
        monkeypatch,
        "openrlhf.trainer.ppo_utils.samples_generator",
        _OPENRLHF_SOURCE / "openrlhf/trainer/ppo_utils/samples_generator.py",
    )
    ppo_utils.samples_generator = samples_module

    deepspeed_strategy = types.ModuleType("openrlhf.utils.deepspeed")
    deepspeed_strategy.__path__ = []
    deepspeed_strategy.DeepspeedStrategy = type("DeepspeedStrategy", (), {})
    monkeypatch.setitem(sys.modules, "openrlhf.utils.deepspeed", deepspeed_strategy)
    deepspeed_helpers = types.ModuleType("openrlhf.utils.deepspeed.deepspeed_utils")
    deepspeed_helpers.offload_deepspeed_states = lambda *_args, **_kwargs: None
    deepspeed_helpers.reload_deepspeed_states = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.utils.deepspeed.deepspeed_utils",
        deepspeed_helpers,
    )
    distributed_util = types.ModuleType("openrlhf.utils.distributed_util")
    distributed_util.stateless_init_process_group = lambda *_args, **_kwargs: None
    distributed_util.torch_dist_barrier_and_cuda_sync = lambda: None
    monkeypatch.setitem(sys.modules, "openrlhf.utils.distributed_util", distributed_util)
    loss_utils = types.ModuleType("openrlhf.utils.loss_utils")
    loss_utils.get_loss_batch_info = lambda *_args, **_kwargs: {}
    loss_utils.iter_grad_accum_global_norm = lambda *args, **_kwargs: args[0]
    monkeypatch.setitem(sys.modules, "openrlhf.utils.loss_utils", loss_utils)
    vlm_utils = types.ModuleType("openrlhf.utils.vlm_utils")
    vlm_utils.merge_mm_train_inputs = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "openrlhf.utils.vlm_utils", vlm_utils)

    ppo_utils.NaiveReplayBuffer = type("NaiveReplayBuffer", (), {})
    launcher = types.ModuleType("openrlhf.trainer.ray.launcher")
    launcher.BaseModelActor = type("BaseModelActor", (), {})

    class _ReferenceModelActor:
        def init_model_from_pretrained(self, *_args, **_kwargs):
            self.actor = models.Actor("base", lora_rank=0)
            return self.actor

    launcher.ReferenceModelActor = ray.remote(_ReferenceModelActor)
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ray.launcher", launcher)
    ray_utils = types.ModuleType("openrlhf.trainer.ray.utils")
    ray_utils.get_physical_gpu_id = lambda: 0
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ray.utils", ray_utils)
    ppo_actor_module = _source_module(
        monkeypatch,
        "openrlhf.trainer.ray.ppo_actor",
        _OPENRLHF_SOURCE / "openrlhf/trainer/ray/ppo_actor.py",
    )
    trainer_ray.ppo_actor = ppo_actor_module

    ppo_trainer_module = types.ModuleType("openrlhf.trainer.ppo_trainer")

    class _PPOTrainer:
        def fit(self, *_args, **_kwargs):
            return None

        def save_logs_and_checkpoints(self, global_step, *_args, **_kwargs):
            self.observed_save_steps.append(self.args.ckpt.save_steps)
            if global_step % self.args.ckpt.save_steps == 0:
                return self.policy.save_checkpoint(f"global_step{global_step}")
            return None

    ppo_trainer_module.PPOTrainer = ray.remote(_PPOTrainer)
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_trainer", ppo_trainer_module)

    agent_module = types.ModuleType("openrlhf.utils.agent")

    class _SingleTurnAgentExecutor:
        async def execute(self, *_args, **_kwargs):
            return {
                "reward": [0.0],
                "scores": [0.0],
                "extra_logs": {"format": [1.0]},
            }

    agent_module.SingleTurnAgentExecutor = _SingleTurnAgentExecutor
    monkeypatch.setitem(sys.modules, "openrlhf.utils.agent", agent_module)

    monkeypatch.setenv("FLASH_OPENRLHF_MAX_RESPONSE_LENGTH", "5")
    if language_model_only:
        monkeypatch.setenv("FLASH_OPENRLHF_LANGUAGE_MODEL_ONLY", "1")
    else:
        monkeypatch.delenv("FLASH_OPENRLHF_LANGUAGE_MODEL_ONLY", raising=False)
    if sm120_vllm_backend:
        monkeypatch.setenv("FLASH_OPENRLHF_SM120_VLLM_BACKEND", "1")
        monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    else:
        monkeypatch.delenv("FLASH_OPENRLHF_SM120_VLLM_BACKEND", raising=False)
    namespace = {"__name__": "sitecustomize", "__file__": "sitecustomize.py"}
    exec(compile(grpo_openrlhf._sitecustomize_source(), "sitecustomize.py", "exec"), namespace)
    return namespace, loss_module, ppo_actor_module, samples_module, experience_module.Experience


def test_build_openrlhf_grpo_args_maps_flash_recipe():
    args = grpo_openrlhf.build_openrlhf_grpo_args(_config())

    assert _value(args, "--actor.model_name_or_path").endswith("/" + "a" * 40)
    assert _value(args, "--reward.remote_url").startswith("http://127.0.0.1:")
    assert _value(args, "--data.prompt_dataset") == "/work/train.jsonl"
    assert _value(args, "--data.input_key") == "input"
    assert _value(args, "--data.label_key") == "label"
    assert _value(args, "--data.max_len") == "4096"
    assert _value(args, "--rollout.max_new_tokens") == "320"
    assert _value(args, "--rollout.batch_size") == "16"
    assert _value(args, "--rollout.n_samples_per_prompt") == "8"
    assert _value(args, "--train.batch_size") == "128"
    assert _value(args, "--train.max_epochs") == "1"
    assert _value(args, "--algo.advantage.estimator") == "dr_grpo"
    assert args[args.index("--reward.clip_range") + 1 : args.index("--reward.clip_range") + 3] == [
        "-1000000000",
        "1000000000",
    ]
    assert _value(args, "--actor.adam.lr") == "1e-05"
    assert args[args.index("--actor.adam.betas") + 1 : args.index("--actor.adam.betas") + 3] == [
        "0.9",
        "0.999",
    ]
    assert _value(args, "--actor.lr_scheduler") == "constant"
    assert _value(args, "--ds.lora.rank") == "32"
    assert _value(args, "--ds.lora.alpha") == "64"
    assert _value(args, "--ds.lora.target_modules") == "all-linear"
    assert _value(args, "--ds.zero_stage") == "3"
    assert _value(args, "--ds.ring_attn_size") == "1"
    assert _value(args, "--actor.num_gpus_per_node") == "2"
    assert _value(args, "--vllm.num_engines") == "2"
    assert "--train.colocate_all" in args
    assert "--vllm.enforce_eager" in args
    assert _value(args, "--ds.attn_implementation") == "flash_attention_3"
    assert "--train.full_determinism_enable" not in args
    assert "--actor.gradient_checkpointing_reentrant" in args
    assert _value(args, "--algo.kl.init_coef") == "0.0"
    assert "--algo.advantage.is_correction_enable" in args
    assert args[
        args.index("--algo.advantage.is_correction_threshold") + 1 : args.index(
            "--algo.advantage.is_correction_threshold"
        )
        + 3
    ] == ["0.0", "2.0"]
    assert _value(args, "--algo.advantage.is_correction_type") == "tis"


@requires_openrlhf_source
def test_pinned_openrlhf_parser_accepts_full_generated_args(monkeypatch):
    parsed = _parse_with_pinned_openrlhf(
        monkeypatch,
        grpo_openrlhf.build_openrlhf_grpo_args(_config()),
    )

    assert parsed.reward.clip_range == [-1_000_000_000.0, 1_000_000_000.0]
    assert parsed.actor.adam.betas == [0.9, 0.999]
    assert parsed.actor.gradient_checkpointing_reentrant is True
    assert parsed.actor.num_gpus_per_node == 2
    assert parsed.rollout.n_samples_per_prompt == 8


def test_build_openrlhf_grpo_args_uses_reentrant_only_for_required_families():
    gdn_args = grpo_openrlhf.build_openrlhf_grpo_args(_config())
    dense_args = grpo_openrlhf.build_openrlhf_grpo_args(_config(model_id="meta-llama/Llama-3.2-1B"))

    assert "--actor.gradient_checkpointing_reentrant" in gdn_args
    assert "--actor.gradient_checkpointing_reentrant" not in dense_args


def test_build_openrlhf_grpo_args_enables_fresh_policy_kl():
    args = grpo_openrlhf.build_openrlhf_grpo_args(_config(kl_coef=0.03))

    assert _value(args, "--algo.kl.init_coef") == "0.03"
    assert "--algo.kl.use_loss" in args
    assert _value(args, "--algo.kl.estimator") == "k2"
    assert _value(args, "--ref.num_nodes") == "1"
    assert _value(args, "--ref.num_gpus_per_node") == "2"


def test_build_openrlhf_grpo_args_maps_exact_saves_and_resume():
    args = grpo_openrlhf.build_openrlhf_grpo_args(_config(save_at_steps=(3, 7), resume=True))

    assert _value(args, "--ckpt.save_steps") == "-1"
    assert "--ckpt.load_enable" in args


def test_build_openrlhf_grpo_args_rejects_degenerate_group():
    with pytest.raises(ValueError, match="group_size greater than 1"):
        grpo_openrlhf.build_openrlhf_grpo_args(_config(group_size=1))


def test_resolve_single_turn_inputs_preserves_explicit_zero_values(monkeypatch):
    _install_resolve_inputs_fakes(monkeypatch)

    inputs = grpo_openrlhf._resolve_single_turn_inputs()

    assert inputs["temperature"] == 0.0
    assert inputs["think_penalty"] == 0.0
    assert inputs["kl_coef"] == 0.0
    assert inputs["learning_rate"] == 0.0
    assert inputs["lora_rank"] == 0
    assert inputs["lora_alpha"] == 0
    assert inputs["save_every"] == -1


def test_resolve_single_turn_inputs_rejects_unpublished_periodic_saves(monkeypatch):
    _install_resolve_inputs_fakes(monkeypatch, save_every=5)

    with pytest.raises(RuntimeError, match="periodic checkpoints are not uploaded"):
        grpo_openrlhf._resolve_single_turn_inputs()


def test_dr_grpo_fixed_length_normalization_matches_trl_formula():
    losses = [[1.0, 2.0, 100.0], [3.0, 4.0, 5.0]]
    masks = [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]

    actual = grpo_openrlhf.dr_grpo_fixed_length_normalize(losses, masks, 5)

    assert actual == pytest.approx((1.0 + 2.0) / (2 * 5))
    token_mean = (1.0 + 2.0) / 2
    assert actual != pytest.approx(token_mean)


def test_token_level_tis_math_matches_trl_token_truncate_formula():
    losses = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    masks = [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    log_ratios = [[0.0, math.log(4.0), -math.log(2.0)], [math.log(1.5), 0.0, 0.0]]

    actual = grpo_openrlhf.tis_weighted_dr_grpo_normalize(
        losses,
        masks,
        log_ratios,
        5,
    )

    expected = (1.0 * 1.0 + 2.0 * 2.0 + 4.0 * 1.5) / (2 * 5)
    assert actual == pytest.approx(expected)


def test_completion_split_matches_openrlhf_tokenizer_canonicalization():
    class _Ids(list):
        def tolist(self):
            return list(self)

    class _Tokenizer:
        def __call__(self, *, text, add_special_tokens, return_tensors):
            assert text == "raw prompt"
            assert add_special_tokens is False
            assert return_tensors == "pt"
            return {"input_ids": [_Ids([1, 2])]}

        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [1, 2]
            assert skip_special_tokens is False
            return "canonical prompt"

    completion = grpo_openrlhf.completion_from_tokenizer_query(
        _Tokenizer(), "canonical promptcompletion", "raw prompt"
    )

    assert completion == "completion"


def test_reward_bridge_round_trip_preserves_label_reward_and_metrics():
    calls = []

    def score(label, completion, prompt):
        calls.append((label, completion, prompt))
        return grpo_openrlhf.RewardResult(1.25, 1.25, {"format": 0.5})

    with grpo_openrlhf.RewardBridge(
        score, samples_per_step=1, first_step=4, token="test-token"
    ) as bridge:
        response = grpo_openrlhf.post_reward_request(
            bridge.url,
            {
                "query": ["rendered promptcompletion"],
                "prompts": ["rendered prompt"],
                "labels": [7],
            },
        )

    assert response == {
        "rewards": [1.25],
        "scores": [1.25],
        "extra_logs": {"format": [0.5]},
    }
    assert calls == [(7, "completion", "rendered prompt")]
    assert bridge.rewards == [1.25]
    assert bridge.drain_sampled_completions(4) == [
        {
            "prompt_tail": "rendered prompt",
            "completion": "completion",
            "reward": 1.25,
            "generated_at_step": 4,
        }
    ]
    assert bridge.drain_sampled_completions(5) == []


def test_reward_bridge_keeps_samples_with_their_generation_step():
    def score(label, _completion, _prompt):
        return grpo_openrlhf.RewardResult(float(label), float(label), {})

    with grpo_openrlhf.RewardBridge(
        score, samples_per_step=2, first_step=7, token="step-token"
    ) as bridge:
        for label in range(4):
            grpo_openrlhf.post_reward_request(
                bridge.url,
                {
                    "query": [f"promptcompletion-{label}"],
                    "prompts": ["prompt"],
                    "labels": [label],
                },
            )

        step_7 = bridge.drain_sampled_completions(7)
        step_8 = bridge.drain_sampled_completions(8)

    assert [sample["completion"] for sample in step_7] == ["completion-0", "completion-1"]
    assert [sample["generated_at_step"] for sample in step_7] == [7, 7]
    assert [sample["completion"] for sample in step_8] == ["completion-2", "completion-3"]
    assert [sample["generated_at_step"] for sample in step_8] == [8, 8]


def test_reward_bridge_rejects_bad_auth_and_scoring_failures():
    def fail(_label, _completion, _prompt):
        raise RuntimeError("environment unavailable")

    with grpo_openrlhf.RewardBridge(
        fail, samples_per_step=1, first_step=1, token="right-token"
    ) as bridge:
        payload = {"query": ["pc"], "prompts": ["p"], "labels": [0]}
        wrong_url = bridge.url.replace("right-token", "wrong-token")
        with pytest.raises(urllib.error.HTTPError) as auth_error:
            grpo_openrlhf.post_reward_request(wrong_url, payload)
        with pytest.raises(urllib.error.HTTPError) as score_error:
            grpo_openrlhf.post_reward_request(bridge.url, payload)

    assert auth_error.value.code == 401
    assert score_error.value.code == 500
    assert bridge.call_count == 0


def test_score_single_turn_maps_environment_exception_to_zero(monkeypatch):
    class _Env:
        def reward(self, _graded, _example, _state):
            raise ValueError("bad completion")

    monkeypatch.setattr(grpo_openrlhf._w, "graded_text", lambda text, **_kwargs: text)
    result = grpo_openrlhf.score_single_turn(
        _Env(),
        "answer",
        {"id": 1},
        tokenizer=object(),
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )

    assert result == grpo_openrlhf.RewardResult(0.0, 0.0, {})


def test_child_env_excludes_environment_and_provider_secrets(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    monkeypatch.setenv("PYTHONPATH", "/parent/flash-code")
    monkeypatch.setenv("USER_ENV_SECRET", "do-not-forward")
    monkeypatch.setenv("RUNPOD_API_KEY", "do-not-forward")
    monkeypatch.setenv("HF_TOKEN", "do-not-forward")

    child = grpo_openrlhf.build_openrlhf_child_env(
        plugin_dir="/work/plugin",
        max_response_length=320,
        language_model_only=True,
        sm120_vllm_backend=True,
        fp8_kv=True,
        warmstart_adapter="/work/incoming-adapter",
        save_at_steps=(3, 7),
        actor_attn_implementation="sdpa",
    )

    assert child["PATH"] == "/usr/bin"
    assert child["NCCL_DEBUG"] == "WARN"
    assert child["PYTHONPATH"] == "/work/plugin"
    assert child["FLASH_OPENRLHF_MAX_RESPONSE_LENGTH"] == "320"
    assert child["FLASH_OPENRLHF_LANGUAGE_MODEL_ONLY"] == "1"
    assert child["FLASH_OPENRLHF_SM120_VLLM_BACKEND"] == "1"
    assert child["FLASH_OPENRLHF_FP8_KV"] == "1"
    assert child["FLASH_OPENRLHF_WARMSTART_ADAPTER"] == "/work/incoming-adapter"
    assert child["FLASH_OPENRLHF_SAVE_AT_STEPS"] == "3,7"
    assert child["FLASH_OPENRLHF_ATTN_IMPLEMENTATION"] == "sdpa"
    assert "USER_ENV_SECRET" not in child
    assert "RUNPOD_API_KEY" not in child
    assert "HF_TOKEN" not in child


def test_sitecustomize_carries_fail_closed_and_fixed_length_hooks():
    source = grpo_openrlhf._sitecustomize_source()

    compile(source, "sitecustomize.py", "exec")
    assert "ActorPPOTrainer.training_step = _flash_training_step" in source
    assert "_Actor.forward = _flash_actor_forward" in source
    assert "for start in range(0, flat_hidden.shape[0], int(chunk_size))" in source
    assert "32k gpu validation pending" in source
    assert "loss.shape[0] * _MAX_RESPONSE_LENGTH" in source
    assert "_ppo_actor_module.aggregate_loss = _flash_aggregate_loss" in source
    assert "experience.action_mask.zero_()" in source
    assert "_ActorPPOTrainer.broadcast_to_vllm = _flash_broadcast_to_vllm" in source
    assert 'for key in ("reward", "scores")' in source
    assert "returned invalid {key}" in source
    assert "_flash_patch_engine_args(vllm.AsyncEngineArgs" in source
    assert "_flash_patch_engine_args(vllm.EngineArgs" in source
    assert "_NON_LM_PARAMETER_SEGMENTS" in source
    assert "_flash_attention_context" in source
    assert 'target_modules"] = "all-linear"' in source
    assert 'vllm_is_truncated_threshold"] = [0.0, 2.0]' in source
    assert '_engine_arg_defaults["kv_cache_dtype"] = "fp8"' in source
    assert "for name, value in _engine_arg_defaults.items()" in source
    assert "PeftModel.from_pretrained" in source
    assert 'getattr(metadata, "modified_class", actor_class)' in source
    assert "_ReferenceModelActorImpl.init_model_from_pretrained = _flash_reference_init" in source
    assert "_PPOTrainerImpl.fit = _flash_ppo_fit" in source
    assert "_PPOTrainerImpl.save_logs_and_checkpoints = _flash_save_logs_and_checkpoints" in source
    assert "_PolicyModelActorImpl.save_checkpoint = _flash_actor_save_checkpoint" in source
    assert "PagedAdamW8bit" in source
    assert 'config["zero_allow_untested_optimizer"] = True' in source
    assert "self.broadcast_to_vllm()" in source
    assert "[flash-openrlhf-checkpoint]" in source


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_normalizes_singleton_reward_batches(monkeypatch):
    _install_pinned_sitecustomize_modules(monkeypatch)
    executor_type = sys.modules["openrlhf.utils.agent"].SingleTurnAgentExecutor

    output = asyncio.run(executor_type().execute())

    assert output == {"reward": 0.0, "scores": 0.0, "extra_logs": {"format": 1.0}}


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_patches_sync_and_async_language_only_engine_args(monkeypatch):
    _install_pinned_sitecustomize_modules(monkeypatch, language_model_only=True)
    vllm = sys.modules["vllm"]

    assert vllm.EngineArgs().language_model_only is True
    assert vllm.AsyncEngineArgs().language_model_only is True


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_replaces_actor_adamw_with_paged_8bit(monkeypatch):
    pytest.importorskip("torch")
    _install_pinned_sitecustomize_modules(monkeypatch)
    bitsandbytes = types.ModuleType("bitsandbytes")

    class _PagedAdamW8bit:
        def __init__(self, parameters, **kwargs):
            self.parameters = parameters
            self.kwargs = kwargs

    bitsandbytes.optim = SimpleNamespace(PagedAdamW8bit=_PagedAdamW8bit)
    monkeypatch.setitem(sys.modules, "bitsandbytes", bitsandbytes)
    deepspeed = sys.modules["deepspeed"]
    parameters = [object()]
    config = {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        }
    }

    _, optimizer, _, _ = deepspeed.initialize(
        model=object(),
        model_parameters=parameters,
        optimizer=None,
        config=config,
    )

    assert isinstance(optimizer, _PagedAdamW8bit)
    assert optimizer.parameters is parameters
    assert optimizer.kwargs == {
        "lr": 1e-5,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    forwarded = deepspeed.initialize_calls[-1][1]
    assert forwarded["model_parameters"] is None
    assert "optimizer" not in forwarded["config"]
    assert forwarded["config"]["zero_allow_untested_optimizer"] is True


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_pins_sm120_vllm_attention_backend(monkeypatch):
    _install_pinned_sitecustomize_modules(monkeypatch, sm120_vllm_backend=True)
    vllm = sys.modules["vllm"]

    assert vllm.EngineArgs().attention_backend == "FLASHINFER"
    assert vllm.AsyncEngineArgs().attention_backend == "FLASHINFER"


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_combines_policy_and_kl_with_one_dr_grpo_denominator(monkeypatch):
    torch = pytest.importorskip("torch")
    namespace, loss_module, ppo_actor_module, _, Experience = _install_pinned_sitecustomize_modules(
        monkeypatch
    )
    del namespace
    action_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    policy_tokens = torch.tensor([[1.0, 2.0, 99.0], [3.0, 88.0, 77.0], [66.0, 55.0, 44.0]])
    kl_tokens = torch.tensor([[0.5, 1.0, 9.0], [1.5, 8.0, 7.0], [6.0, 5.0, 4.0]])

    class _ActorLoss:
        def __call__(self, *_args, action_mask, **_kwargs):
            loss = loss_module.aggregate_loss(policy_tokens, action_mask)
            zero = torch.tensor(0.0)
            return loss, zero, zero, None

    class _ActorModel:
        def train(self):
            return None

        def __call__(self, sequences, action_mask, **_kwargs):
            return torch.zeros_like(action_mask), SimpleNamespace()

    class _Strategy:
        ring_attn_group = None

        def backward(self, loss, *_args):
            self.backward_loss = loss.detach()

        def optimizer_step(self, *_args, **_kwargs):
            return None

        def get_grad_norm(self, _actor):
            return torch.tensor(0.0)

    trainer = object.__new__(ppo_actor_module.ActorPPOTrainer)
    trainer.actor = _ActorModel()
    trainer.actor_loss_fn = _ActorLoss()
    trainer.actor_optim = object()
    trainer.actor_scheduler = SimpleNamespace(get_last_lr=lambda: [1e-5])
    trainer.ema_model = None
    trainer.strategy = _Strategy()
    trainer.aux_loss = False
    trainer.replay_buffer = SimpleNamespace()
    trainer.args = SimpleNamespace(
        actor=SimpleNamespace(entropy_coef=None),
        algo=SimpleNamespace(kl=SimpleNamespace(use_loss=True, init_coef=0.03, estimator="k2")),
        train=SimpleNamespace(dynamic_batch_enable=False),
    )
    ppo_actor_module.compute_approx_kl = lambda *_args, **_kwargs: kl_tokens
    experience = Experience(
        sequences=torch.ones((3, 4), dtype=torch.long),
        attention_mask=torch.ones((3, 4), dtype=torch.long),
        action_mask=action_mask,
        action_log_probs=torch.zeros_like(action_mask),
        base_action_log_probs=torch.zeros_like(action_mask),
        rollout_log_probs=torch.zeros_like(action_mask),
        advantages=torch.ones_like(action_mask),
        info={},
    )
    kl_coef = 0.4

    trainer.training_step(experience, kl_coef, step=0, loss_batch_info={})

    expected = ((policy_tokens + kl_coef * kl_tokens) * action_mask).sum() / (3 * 5)
    assert trainer.strategy.backward_loss.item() == pytest.approx(expected.item())
    assert ppo_actor_module.aggregate_loss is loss_module.aggregate_loss


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_applies_token_tis_inside_fixed_dr_grpo_loss(monkeypatch):
    torch = pytest.importorskip("torch")
    namespace, loss_module, _, _, _ = _install_pinned_sitecustomize_modules(monkeypatch)
    loss_fn = loss_module.PolicyLoss(
        enable_vllm_is_correction=False,
        vllm_is_truncated_threshold=None,
    )
    action_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    zeros = torch.zeros_like(action_mask)
    rollout_log_probs = -torch.tensor([[0.0, math.log(4.0)], [math.log(1.5), 0.0]])
    advantages = torch.ones_like(action_mask)
    token = namespace["_fixed_dr_grpo_loss"].set(True)
    try:
        actual, _, _, _ = loss_fn(
            zeros,
            zeros,
            advantages,
            action_mask=action_mask,
            rollout_log_probs=rollout_log_probs,
        )
    finally:
        namespace["_fixed_dr_grpo_loss"].reset(token)

    assert loss_fn.enable_vllm_is_correction is True
    assert loss_fn.vllm_is_truncated_threshold == [0.0, 2.0]
    assert loss_fn.vllm_is_correction_type == "tis"
    assert actual.item() == pytest.approx(-(1.0 + 2.0 + 1.5) / (2 * 5))


@requires_openrlhf_source
@requires_torch
def test_chunked_action_logprobs_match_full_policy_loss_and_gradients(monkeypatch):
    torch = pytest.importorskip("torch")
    namespace, loss_module, _, _, _ = _install_pinned_sitecustomize_modules(monkeypatch)
    torch.manual_seed(17)
    batch, actions, hidden_size, vocab_size = 3, 5, 7, 13
    temperature = 0.7
    action_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    labels = torch.randint(vocab_size, (batch, actions))
    initial_hidden = torch.randn(batch, actions, hidden_size)
    initial_weight = torch.randn(vocab_size, hidden_size)
    initial_bias = torch.randn(vocab_size)
    old_log_probs = torch.randn(batch, actions)
    tis_log_ratios = torch.tensor(
        [
            [0.0, math.log(4.0), math.log(1.5), -math.log(2.0), 0.0],
            [math.log(3.0), 0.0, math.log(1.25), 0.0, 0.0],
            [math.log(1.75), 0.0, 0.0, 0.0, 0.0],
        ]
    )
    rollout_log_probs = old_log_probs - tis_log_ratios
    advantages = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75, -1.0],
            [0.5, 1.25, -0.75, 0.0, 0.0],
            [-1.5, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    base_log_probs = torch.randn(batch, actions)
    kl_coef = 0.03
    entropy_coef = 0.02

    def combined_loss(action_log_probs, entropy):
        loss_fn = loss_module.PolicyLoss(
            enable_vllm_is_correction=False,
            vllm_is_truncated_threshold=None,
        )
        token = namespace["_fixed_dr_grpo_loss"].set(True)
        try:
            policy_loss, _, _, _ = loss_fn(
                action_log_probs,
                old_log_probs,
                advantages,
                action_mask=action_mask,
                rollout_log_probs=rollout_log_probs,
            )
            kl_tokens = (action_log_probs.float() - base_log_probs.float()).square() / 2.0
            kl_loss = namespace["_flash_aggregate_loss"](kl_tokens, action_mask)
            entropy_loss = namespace["_flash_aggregate_loss"](entropy, action_mask)
        finally:
            namespace["_fixed_dr_grpo_loss"].reset(token)
        assert loss_fn.enable_vllm_is_correction is True
        assert loss_fn.vllm_is_truncated_threshold == [0.0, 2.0]
        assert loss_fn.vllm_is_correction_type == "tis"
        return policy_loss + kl_coef * kl_loss - entropy_coef * entropy_loss

    def make_projection():
        projection = torch.nn.Linear(hidden_size, vocab_size)
        with torch.no_grad():
            projection.weight.copy_(initial_weight)
            projection.bias.copy_(initial_bias)
        return projection

    full_hidden = initial_hidden.clone().requires_grad_(True)
    full_projection = make_projection()
    full_raw_logits = full_projection(full_hidden).float()
    full_logits = full_raw_logits / temperature
    full_logsumexp = torch.logsumexp(full_logits, dim=-1)
    full_selected = full_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    full_action_log_probs = (full_selected - full_logsumexp) * action_mask
    full_probabilities = torch.softmax(full_raw_logits, dim=-1)
    full_entropy = torch.logsumexp(full_raw_logits, dim=-1) - (
        full_probabilities * full_raw_logits
    ).sum(dim=-1)
    full_loss = combined_loss(full_action_log_probs, full_entropy)
    full_loss.backward()
    expected_gradients = (
        full_hidden.grad.detach().clone(),
        full_projection.weight.grad.detach().clone(),
        full_projection.bias.grad.detach().clone(),
    )

    for chunk_size in (1, 4, batch * actions):
        chunked_hidden = initial_hidden.clone().requires_grad_(True)
        chunked_projection = make_projection()
        projected_rows = []

        def recording_projection(
            hidden,
            projection=chunked_projection,
            rows=projected_rows,
        ):
            rows.append(hidden.shape[0])
            return projection(hidden)

        saved_shapes = []

        def pack_saved(tensor, shapes=saved_shapes):
            shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack_saved, lambda tensor: tensor):
            action_log_probs, entropy = namespace["_flash_chunked_action_log_probs"](
                chunked_hidden,
                labels,
                action_mask,
                recording_projection,
                temperature,
                chunk_size=chunk_size,
                return_entropy=True,
            )
            loss = combined_loss(action_log_probs, entropy)
        loss.backward()

        torch.testing.assert_close(action_log_probs, full_action_log_probs, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(entropy, full_entropy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(loss, full_loss, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(chunked_hidden.grad, expected_gradients[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            chunked_projection.weight.grad,
            expected_gradients[1],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            chunked_projection.bias.grad,
            expected_gradients[2],
            atol=1e-5,
            rtol=1e-5,
        )
        assert max(projected_rows) <= chunk_size
        assert sum(projected_rows) == 2 * batch * actions
        assert not any(len(shape) == 2 and shape[-1] == vocab_size for shape in saved_shapes)


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_actor_projects_only_action_positions(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_pinned_sitecustomize_modules(monkeypatch)
    actor_type = sys.modules["openrlhf.models.actor"].Actor
    hidden_size, vocab_size = 6, 17

    class _Output(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    class _OutputHead(torch.nn.Linear):
        def __init__(self):
            super().__init__(hidden_size, vocab_size)
            self.projected_rows = []

        def forward(self, hidden):
            self.projected_rows.append(hidden.shape[0])
            return super().forward(hidden)

    class _ToyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            self.lm_head = _OutputHead()

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, attention_mask, position_ids):
            assert attention_mask.shape == input_ids.shape
            assert position_ids.shape == input_ids.shape
            hidden = self.embed(input_ids)
            return _Output(logits=self.lm_head(hidden), hidden_states=hidden)

    actor = actor_type("base")
    actor.model = _ToyCausalLM()
    actor.packing_samples = False
    actor.temperature = 0.8
    sequences = torch.randint(vocab_size, (2, 9))
    attention_mask = torch.ones_like(sequences)
    action_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])

    action_log_probs, output = actor.forward(
        sequences,
        action_mask,
        attention_mask=attention_mask,
        return_output=True,
        return_entropy=True,
    )

    with torch.no_grad():
        hidden = actor.model.embed(sequences)[:, -action_mask.shape[1] - 1 : -1]
        logits = actor.model.lm_head(hidden).float() / actor.temperature
        labels = sequences[:, -action_mask.shape[1] :]
        expected = (
            logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        ) * action_mask
    torch.testing.assert_close(action_log_probs, expected, atol=1e-5, rtol=1e-5)
    action_log_probs.sum().backward()
    assert output.logits is None
    assert output.entropy.shape == action_mask.shape
    assert actor.model.embed.weight.grad is not None
    assert actor.model.lm_head.projected_rows == [
        action_mask.numel(),
        sequences.shape[0],
        action_mask.numel(),
    ]
    assert actor.model.lm_head.projected_rows[0] < sequences.numel()


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_masks_truncated_response_but_retains_row(monkeypatch):
    torch = pytest.importorskip("torch")
    _, _, _, samples_module, _ = _install_pinned_sitecustomize_modules(monkeypatch)
    generator = object.__new__(samples_module.SamplesGenerator)
    response = {
        "observation_tokens": [10, 20, 21],
        "action_ranges": [(1, 3)],
        "rollout_log_probs": None,
        "reward": 1.0,
        "scores": 1.0,
        "prompt": "prompt",
        "label": 0,
        "truncated": True,
    }

    experience = generator._process_response_into_experience(response, max_len=8)

    assert experience.action_mask.shape == (1, 2)
    assert torch.count_nonzero(experience.action_mask).item() == 0
    assert experience.truncated.tolist() == [True]


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_maps_fp8_kv_to_vllm_engine_args(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("FLASH_OPENRLHF_FP8_KV", "1")
    _install_pinned_sitecustomize_modules(monkeypatch)

    engine_args = sys.modules["vllm"].AsyncEngineArgs()

    assert engine_args.kv_cache_dtype == "fp8"


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_loads_warm_policy_and_incoming_kl_reference(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("FLASH_OPENRLHF_WARMSTART_ADAPTER", "/work/incoming-adapter")
    namespace, _, _, _, _ = _install_pinned_sitecustomize_modules(monkeypatch)
    actor_class = sys.modules["openrlhf.models"].Actor

    policy = actor_class("base", lora_rank=8, target_modules=["all-linear"])
    reference_wrapper = namespace["_ReferenceModelActor"]
    reference_class = reference_wrapper.__ray_metadata__.modified_class
    reference = reference_class()
    reference.init_model_from_pretrained(None, "base")
    loads = sys.modules["peft"].loads

    assert reference_class.init_model_from_pretrained is namespace["_flash_reference_init"]
    assert reference_wrapper.init_model_from_pretrained is not namespace["_flash_reference_init"]
    assert loads[0][0:2] == ("/work/incoming-adapter", True)
    assert loads[1][0:2] == ("/work/incoming-adapter", False)
    assert sys.modules["peft"].adapter_loads == [
        ("/work/incoming-adapter", "default", True, None),
        ("/work/incoming-adapter", "default", False, None),
    ]
    assert policy.model.is_trainable is True
    assert reference.actor.model.is_trainable is False
    assert all(not parameter.requires_grad for parameter in reference.actor.model.parameters())

    trainer_wrapper = namespace["_PPOTrainer"]
    trainer_class = trainer_wrapper.__ray_metadata__.modified_class
    trainer = trainer_class()
    broadcasts = []
    trainer.broadcast_to_vllm = lambda: broadcasts.append(True)
    trainer.fit(global_step=0)

    assert trainer_class.fit is namespace["_flash_ppo_fit"]
    assert trainer_wrapper.fit is not namespace["_flash_ppo_fit"]
    assert broadcasts == [True]


@pytest.mark.parametrize(
    ("load_result", "zero_adapter", "message"),
    [
        (
            SimpleNamespace(missing_keys=["proj.lora_B.default.weight"], unexpected_keys=[]),
            False,
            "load was incomplete",
        ),
        (SimpleNamespace(missing_keys=[], unexpected_keys=[]), True, "all-zero LoRA delta"),
    ],
)
@requires_openrlhf_source
@requires_torch
def test_sitecustomize_warmstart_validates_single_weight_load(
    monkeypatch, load_result, zero_adapter, message
):
    pytest.importorskip("torch")
    monkeypatch.setenv("FLASH_OPENRLHF_WARMSTART_ADAPTER", "/work/incoming-adapter")
    _install_pinned_sitecustomize_modules(monkeypatch)
    peft = sys.modules["peft"]
    peft.load_result = load_result
    peft.zero_adapter = zero_adapter
    actor_class = sys.modules["openrlhf.models"].Actor

    with pytest.raises(RuntimeError, match=message):
        actor_class("base", lora_rank=8, target_modules=["all-linear"])

    assert peft.adapter_loads == [("/work/incoming-adapter", "default", True, None)]


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_exact_save_reaches_exported_policy_actor_and_acknowledges(
    monkeypatch, tmp_path
):
    pytest.importorskip("torch")
    monkeypatch.setenv("FLASH_OPENRLHF_SAVE_AT_STEPS", "3")
    namespace, _, ppo_actor_module, _, _ = _install_pinned_sitecustomize_modules(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    saved = []

    class _Strategy:
        def __init__(self):
            self.args = SimpleNamespace(
                ckpt=SimpleNamespace(path=str(checkpoint_dir), max_num=2, max_mem=0)
            )

        def save_ckpt(self, model, path, tag, max_num, max_mem, client_states, **kwargs):
            saved.append((model, path, tag, max_num, max_mem, client_states, kwargs))

        def is_rank_0(self):
            return True

    policy_wrapper = ppo_actor_module.PolicyModelActor
    policy_class = policy_wrapper.__ray_metadata__.modified_class
    policy = object.__new__(policy_class)
    policy.strategy = _Strategy()
    policy.actor = SimpleNamespace(model=object())
    policy.disable_ds_ckpt = False
    policy.save_hf_ckpt = False
    markers = []

    def acknowledge_marker(message, *_args, **kwargs):
        if message.startswith("[flash-openrlhf-checkpoint] "):
            assert kwargs == {"flush": True}
            marker = json.loads(message.removeprefix("[flash-openrlhf-checkpoint] "))
            markers.append(marker)
            Path(marker["ack_path"]).touch()

    namespace["print"] = acknowledge_marker
    trainer_wrapper = namespace["_PPOTrainer"]
    trainer_class = trainer_wrapper.__ray_metadata__.modified_class
    trainer = trainer_class()
    trainer.args = SimpleNamespace(ckpt=SimpleNamespace(save_steps=float("inf")))
    trainer.observed_save_steps = []
    trainer.policy = policy

    trainer.save_logs_and_checkpoints(3, {}, {})

    assert trainer_class.save_logs_and_checkpoints is namespace["_flash_save_logs_and_checkpoints"]
    assert (
        trainer_wrapper.save_logs_and_checkpoints
        is not namespace["_flash_save_logs_and_checkpoints"]
    )
    assert policy_class.save_checkpoint is namespace["_flash_actor_save_checkpoint"]
    assert policy_wrapper.save_checkpoint is not namespace["_flash_actor_save_checkpoint"]
    assert trainer.observed_save_steps == [3]
    assert trainer.args.ckpt.save_steps == float("inf")
    assert saved[0][1:3] == (str(checkpoint_dir / "_actor"), "global_step3")
    assert markers == [
        {
            "step": 3,
            "tag": "global_step3",
            "checkpoint_dir": str(checkpoint_dir),
            "adapter_dir": str(checkpoint_dir / "global_step3_hf"),
            "ack_path": str(checkpoint_dir / ".flash-uploaded-global_step3"),
        }
    ]
    assert Path(markers[0]["ack_path"]).is_file()


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_nonzero_rank_waits_for_checkpoint_ack(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.setenv("FLASH_OPENRLHF_SAVE_AT_STEPS", "3")
    namespace, _, ppo_actor_module, _, _ = _install_pinned_sitecustomize_modules(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    saved = []

    class _Strategy:
        def __init__(self):
            self.args = SimpleNamespace(
                ckpt=SimpleNamespace(path=str(checkpoint_dir), max_num=2, max_mem=0)
            )

        def save_ckpt(self, model, path, tag, max_num, max_mem, client_states, **kwargs):
            saved.append((model, path, tag, max_num, max_mem, client_states, kwargs))

        def is_rank_0(self):
            return False

    policy_class = ppo_actor_module.PolicyModelActor.__ray_metadata__.modified_class
    policy = object.__new__(policy_class)
    policy.strategy = _Strategy()
    policy.actor = SimpleNamespace(model=object())
    policy.disable_ds_ckpt = False
    policy.save_hf_ckpt = False
    ack_path = checkpoint_dir / ".flash-uploaded-global_step3"
    sleeps = []

    def release_rank(_seconds):
        sleeps.append(_seconds)
        ack_path.touch()

    monotonic_values = iter((0.0, 1.0))
    namespace["time"] = SimpleNamespace(
        monotonic=lambda: next(monotonic_values),
        sleep=release_rank,
    )
    markers = []
    namespace["print"] = lambda *args, **kwargs: markers.append((args, kwargs))

    policy.save_checkpoint("global_step3")

    assert saved[0][1:3] == (str(checkpoint_dir / "_actor"), "global_step3")
    assert sleeps == [0.1]
    assert ack_path.is_file()
    assert markers == []


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_lora_sync_emits_only_merged_base_model_keys(monkeypatch):
    torch = pytest.importorskip("torch")
    namespace, _, _, _, _ = _install_pinned_sitecustomize_modules(monkeypatch)

    class _ToyLora(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(3, 2, bias=False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 1, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})
            self.active_adapters = ["default"]
            self.lora_variant = {}
            self.lora_bias = {"default": False}
            self.scaling = {"default": 2.0}

        def get_delta_weight(self, adapter):
            return self.lora_B[adapter].weight @ self.lora_A[adapter].weight * self.scaling[adapter]

    class _WrappedBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _ToyLora()
            self.norm = torch.nn.LayerNorm(2)

    class _PeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = _WrappedBase()

        def get_base_model(self):
            return self.base_model

    class _ExpectedBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 2, bias=False)
            self.norm = torch.nn.LayerNorm(2)

    model = _PeftModel()
    entries = namespace["_flash_lora_sync_entries"](model)
    by_name = {name: entry for name, *entry in entries}
    expected_names = set(dict(_ExpectedBase().named_parameters()))

    assert set(by_name) == expected_names
    assert not any(
        "base_model" in name or "base_layer" in name or "lora_" in name for name in by_name
    )
    parameter, module, adapters, _sources = by_name["proj.weight"]
    merged = namespace["_flash_materialize_sync_weight"](parameter, module, adapters)
    expected = parameter + module.get_delta_weight("default")
    assert torch.allclose(merged, expected)


def test_per_step_checkpoint_publishes_deployable_and_complete_resume_sidecars(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(grpo_openrlhf._w, "HF_REPO", "owner/artifacts")
    checkpoint_dir = tmp_path / "checkpoints"
    tag = "global_step3"
    actor_state = checkpoint_dir / "_actor" / tag
    adapter_source = checkpoint_dir / f"{tag}_hf"
    actor_state.mkdir(parents=True)
    adapter_source.mkdir(parents=True)
    (actor_state / "model_states.pt").write_bytes(b"model")
    (actor_state / "optim_states.pt").write_bytes(b"optimizer")
    (actor_state / "scheduler_states.pt").write_bytes(b"scheduler")
    (actor_state / "client_state.json").write_text(
        json.dumps(
            {
                "episode": 0,
                "global_step": 3,
                "total_consumed_prompts": 12,
                "data_loader_state_dict": {"cursor": 12},
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "_actor" / "latest").write_text(tag, encoding="utf-8")
    ack_path = checkpoint_dir / f".flash-uploaded-{tag}"
    marker = {
        "step": 3,
        "tag": tag,
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_dir": str(adapter_source),
        "ack_path": str(ack_path),
    }

    def fake_export(_source, destination, *_args):
        os.makedirs(destination, exist_ok=True)
        Path(destination, "adapter_config.json").write_text("{}", encoding="utf-8")
        Path(destination, "adapter_model.safetensors").write_bytes(b"adapter")

    class _Tokenizer:
        def save_pretrained(self, destination):
            Path(destination, "tokenizer.json").write_text("{}", encoding="utf-8")

    published = []
    uploaded = []
    monkeypatch.setattr(grpo_openrlhf, "export_openrlhf_adapter", fake_export)
    monkeypatch.setattr(
        grpo_openrlhf._w, "write_base_model_provenance", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        grpo_openrlhf._w,
        "publish_deployable_checkpoint",
        lambda path, step, **kwargs: published.append((path, step, kwargs)),
    )

    def fake_upload(step, staged_resume, *, before_upload):
        before_upload()
        staged = Path(staged_resume)
        metadata = json.loads((staged / "resume_metadata.json").read_text(encoding="utf-8"))
        assert metadata == {
            "backend": "openrlhf",
            "checkpoint_tag": tag,
            "global_step": 3,
            "schema_version": 1,
        }
        assert (staged / "_actor" / "latest").read_text(encoding="utf-8") == tag
        assert (staged / "_actor" / tag / "model_states.pt").read_bytes() == b"model"
        assert (staged / "_actor" / tag / "optim_states.pt").read_bytes() == b"optimizer"
        assert (staged / "_actor" / tag / "scheduler_states.pt").read_bytes() == b"scheduler"
        uploaded.append((step, staged_resume))
        return True

    monkeypatch.setattr(grpo_openrlhf._w, "upload_resume_checkpoint", fake_upload)

    step = grpo_openrlhf._publish_openrlhf_checkpoint(
        marker,
        checkpoint_dir=str(checkpoint_dir),
        adapter_workdir=str(tmp_path / "deployables"),
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        tokenizer=_Tokenizer(),
        python_bin="/openrlhf/python",
        required_steps=(3,),
    )

    assert step == 3
    assert ack_path.is_file()
    assert published[0][1:] == (3, {"required": True})
    assert uploaded[0][0] == 3
    assert grpo_openrlhf._openrlhf_resume_step(str(checkpoint_dir)) == 3


def test_required_checkpoint_failed_resume_upload_is_retriable(monkeypatch, tmp_path):
    monkeypatch.setattr(grpo_openrlhf._w, "HF_REPO", "owner/artifacts")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    tag = "global_step3"
    ack_path = checkpoint_dir / f".flash-uploaded-{tag}"
    marker = {
        "step": 3,
        "tag": tag,
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_dir": str(checkpoint_dir / f"{tag}_hf"),
        "ack_path": str(ack_path),
    }
    staged_resume = tmp_path / "staged-resume"

    def stage_resume(*_args):
        staged_resume.mkdir()
        return str(staged_resume)

    def fail_upload(_step, _staged_resume, *, before_upload):
        before_upload()
        return False

    monkeypatch.setattr(grpo_openrlhf, "export_openrlhf_adapter", lambda *_args: None)
    monkeypatch.setattr(grpo_openrlhf, "_stage_openrlhf_resume_checkpoint", stage_resume)
    monkeypatch.setattr(
        grpo_openrlhf._w, "write_base_model_provenance", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        grpo_openrlhf._w, "publish_deployable_checkpoint", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(grpo_openrlhf._w, "upload_resume_checkpoint", fail_upload)

    with pytest.raises(
        grpo_openrlhf._w.RetriableInfraError,
        match="required OpenRLHF save step 3 full-state checkpoint was not durable",
    ):
        grpo_openrlhf._publish_openrlhf_checkpoint(
            marker,
            checkpoint_dir=str(checkpoint_dir),
            adapter_workdir=str(tmp_path / "deployables"),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="a" * 40,
            tokenizer=SimpleNamespace(save_pretrained=lambda _path: None),
            python_bin="/openrlhf/python",
            required_steps=(3,),
        )

    assert not ack_path.exists()


def test_openrlhf_resume_step_rejects_incomplete_download(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-3"
    checkpoint_dir.mkdir()

    with pytest.raises(RuntimeError, match="missing its latest tag"):
        grpo_openrlhf._openrlhf_resume_step(str(checkpoint_dir))


def test_resume_credits_only_deployables_verified_on_hf(monkeypatch):
    checked = []

    def deployed(step):
        checked.append(step)
        return step == 2

    monkeypatch.setattr(grpo_openrlhf, "_deployable_adapter_on_hf", deployed)

    published = grpo_openrlhf._verified_openrlhf_published_steps((2, 4, 6), 4)

    assert published == {2}
    assert checked == [2, 4]


@requires_openrlhf_source
@requires_torch
def test_sitecustomize_language_only_sync_excludes_vlm_weights(monkeypatch):
    torch = pytest.importorskip("torch")
    namespace, _, _, _, _ = _install_pinned_sitecustomize_modules(
        monkeypatch, language_model_only=True
    )

    class _ToyLora(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(2, 2, bias=False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})
            self.active_adapters = ["default"]
            self.lora_variant = {}
            self.lora_bias = {"default": False}

        def get_delta_weight(self, _adapter):
            return self.lora_B["default"].weight @ self.lora_A["default"].weight

    class _Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.proj = _ToyLora()
            self.visual = torch.nn.Module()
            self.visual.proj = torch.nn.Linear(2, 2, bias=False)

    class _PeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = _Base()

        def get_base_model(self):
            return self.base_model

    names = {name for name, *_rest in namespace["_flash_lora_sync_entries"](_PeftModel())}

    assert "language_model.proj.weight" in names
    assert not any(name.startswith("visual.") for name in names)


def test_run_rl_dispatches_to_openrlhf(monkeypatch):
    calls = []
    monkeypatch.setenv("FLASH_RL_BACKEND", "openrlhf")
    monkeypatch.setattr(grpo_openrlhf, "run_rl_openrlhf", lambda: calls.append(True))

    rl.run_rl()

    assert calls == [True]


def test_run_rl_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("FLASH_RL_BACKEND", "unknown")

    with pytest.raises(RuntimeError, match="not a known grpo backend"):
        rl.run_rl()


@pytest.mark.parametrize("actor_attn_implementation", ["flash_attention_3", None])
def test_run_rl_openrlhf_launches_mock_subprocess_and_exports(
    monkeypatch, tmp_path, actor_attn_implementation
):
    class _Tokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, *, text, add_special_tokens, return_tensors):
            assert text == "prompt"
            assert add_special_tokens is False
            assert return_tensors == "pt"
            return {"input_ids": [[1]]}

        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [1]
            assert skip_special_tokens is False
            return "prompt"

        def save_pretrained(self, directory):
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "tokenizer.json"), "w") as tokenizer_file:
                tokenizer_file.write("{}")

    class _Env:
        def scores_breakdown(self, graded, example, state):
            assert graded == "completion"
            assert example == {"answer": "completion"}
            assert state is None
            return {"total": 2.0, "exact": 1.0}

    fake_job = SimpleNamespace(
        gpu=SimpleNamespace(type="H100", exact_type="", count=1),
    )
    events = []

    @contextlib.contextmanager
    def fake_liveness(name, *_args, **_kwargs):
        events.append(f"enter:{name}")
        yield
        events.append(f"exit:{name}")

    monkeypatch.setattr(grpo_openrlhf._w, "JOB_SPEC", fake_job)
    monkeypatch.setattr(grpo_openrlhf._w, "SEED", 42)
    monkeypatch.setattr(grpo_openrlhf._w, "THINKING", False)
    heartbeats = []
    monkeypatch.setattr(
        grpo_openrlhf._w,
        "heartbeat",
        lambda stage, **kwargs: heartbeats.append((stage, kwargs)),
    )
    monkeypatch.setattr(grpo_openrlhf._w, "prefetch_model", lambda *_args, **_kwargs: 1.5)
    monkeypatch.setattr(grpo_openrlhf._w, "hf_resume_checkpoint", lambda: None)
    monkeypatch.setattr(grpo_openrlhf._w, "is_vl_checkpoint", lambda *_args: False)
    monkeypatch.setattr(grpo_openrlhf._w, "graded_text", lambda text, **_kwargs: text)
    monkeypatch.setattr(grpo_openrlhf, "wait_for_gpu", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grpo_openrlhf, "setup_perf_backends", lambda: None)
    monkeypatch.setattr(grpo_openrlhf, "_is_sm120", lambda: True)
    monkeypatch.setattr(grpo_openrlhf, "optimal_attn_impl", lambda: actor_attn_implementation)
    monkeypatch.setattr(
        grpo_openrlhf, "seed_training_rngs", lambda seed: events.append(f"seed:{seed}")
    )
    monkeypatch.setattr(grpo_openrlhf, "gpu_diagnostics", lambda: {})
    monkeypatch.setattr(grpo_openrlhf, "liveness_heartbeat", fake_liveness)
    monkeypatch.setattr(
        grpo_openrlhf, "resolve_openrlhf_python", lambda _workdir: "/openrlhf/python"
    )
    monkeypatch.setattr(
        grpo_openrlhf,
        "_resolve_cached_model_snapshot",
        lambda *_args: "/cache/snapshots/" + "a" * 40,
    )

    def fake_resolve_inputs():
        assert events[-1] == "enter:rl_data_loading"
        events.append("resolved-inputs")
        return {
            "env": _Env(),
            "tokenizer": _Tokenizer(),
            "model_id": "Qwen/Qwen3.5-0.8B",
            "model_revision": "a" * 40,
            "prompts": [
                {
                    "rendered": "prompt",
                    "example": {"answer": "completion"},
                    "example_idx": 0,
                }
            ],
            "prompts_per_step": 1,
            "group_size": 2,
            "temperature": 1.0,
            "top_p": 1.0,
            "think_penalty": 0.0,
            "kl_coef": 0.0,
            "learning_rate": 1e-5,
            "lora_rank": 8,
            "lora_alpha": 16,
            "max_completion": 32,
            "max_length": 128,
            "prompt_opened_thinking": False,
            "steps": 2,
            "save_every": -1,
            "save_at_steps": (),
            "gpu_count": 1,
            "seed": 42,
            "fp8_kv": True,
            "warmstart_adapter": "",
        }

    monkeypatch.setattr(grpo_openrlhf, "_resolve_single_turn_inputs", fake_resolve_inputs)
    launched = {}

    def fake_training(python_bin, args, **kwargs):
        launched["python"] = python_bin
        launched["args"] = args
        launched["env"] = kwargs["env"]
        reward_url = _value(args, "--reward.remote_url")
        reward = grpo_openrlhf.post_reward_request(
            reward_url,
            {"query": ["promptcompletion"], "prompts": ["prompt"], "labels": [0]},
        )
        assert reward["rewards"] == [2.0]
        assert "step_pattern" not in kwargs
        kwargs["on_step"](1)
        return 0

    monkeypatch.setattr(grpo_openrlhf, "run_openrlhf_training", fake_training)
    exports = []

    def fake_export(checkpoint, adapter, model_id, revision, python_bin):
        os.makedirs(adapter, exist_ok=True)
        exports.append((checkpoint, adapter, model_id, revision, python_bin))

    monkeypatch.setattr(grpo_openrlhf, "export_openrlhf_adapter", fake_export)
    monkeypatch.setattr(
        grpo_openrlhf._w, "write_base_model_provenance", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(grpo_openrlhf._w, "hf_upload_folder", lambda *_args, **_kwargs: True)
    published = []
    monkeypatch.setattr(
        grpo_openrlhf._w,
        "publish_deployable_checkpoint",
        lambda *args, **_kwargs: published.append(args),
    )
    metadata = []
    monkeypatch.setattr(
        grpo_openrlhf._w, "write_train_meta", lambda **kwargs: metadata.append(kwargs)
    )
    monkeypatch.setattr(grpo_openrlhf, "time", SimpleNamespace(time=lambda: 10.0))

    grpo_openrlhf.run_rl_openrlhf()

    assert launched["python"] == "/openrlhf/python"
    assert _value(launched["args"], "--algo.advantage.estimator") == "dr_grpo"
    if actor_attn_implementation:
        assert _value(launched["args"], "--ds.attn_implementation") == actor_attn_implementation
        assert launched["env"]["FLASH_OPENRLHF_ATTN_IMPLEMENTATION"] == actor_attn_implementation
    else:
        assert "--ds.attn_implementation" not in launched["args"]
        assert "FLASH_OPENRLHF_ATTN_IMPLEMENTATION" not in launched["env"]
    assert _value(launched["args"], "--ckpt.save_steps") == "-1"
    assert launched["env"]["FLASH_OPENRLHF_MAX_RESPONSE_LENGTH"] == "32"
    assert launched["env"]["FLASH_OPENRLHF_FP8_KV"] == "1"
    assert launched["env"]["FLASH_OPENRLHF_SM120_VLLM_BACKEND"] == "1"
    assert launched["env"]["HF_HUB_OFFLINE"] == "1"
    assert events.index("seed:42") < events.index("enter:rl_data_loading")
    assert exports
    assert exports[0][2:4] == ("Qwen/Qwen3.5-0.8B", "a" * 40)
    assert published[0][1] == 2
    rl_step = next(kwargs for stage, kwargs in heartbeats if stage == "rl_step")
    assert rl_step["step"] == 1
    assert rl_step["sampled_completions"][0]["completion"] == "completion"
    assert rl_step["sampled_completions"][0]["reward"] == 2.0
    assert metadata[0]["notes"]["backend"] == "openrlhf"
    assert metadata[0]["notes"]["steps"] == 2
    assert metadata[0]["notes"]["reward_history"] == [2.0]
