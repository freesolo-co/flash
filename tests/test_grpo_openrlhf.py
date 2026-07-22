"""cpu coverage for the OpenRLHF GRPO foundation."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import logging
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
        "save_every": 20,
        "gpu_count": 2,
        "qwen35_language_model_only": True,
    }
    values.update(overrides)
    return grpo_openrlhf.OpenRLHFGRPOConfig(**values)


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


def _install_pinned_sitecustomize_modules(monkeypatch):
    torch = pytest.importorskip("torch")
    openrlhf = _package(monkeypatch, "openrlhf")
    models = _package(monkeypatch, "openrlhf.models")
    trainer = _package(monkeypatch, "openrlhf.trainer")
    trainer_ray = _package(monkeypatch, "openrlhf.trainer.ray")
    ppo_utils = _package(monkeypatch, "openrlhf.trainer.ppo_utils")
    utils = _package(monkeypatch, "openrlhf.utils")
    del openrlhf, trainer

    ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda target: target

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
            pass

    models.Actor = _Actor
    actor_module = types.ModuleType("openrlhf.models.actor")
    actor_module.Actor = _Actor
    monkeypatch.setitem(sys.modules, "openrlhf.models.actor", actor_module)

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

    agent_module = types.ModuleType("openrlhf.utils.agent")

    class _SingleTurnAgentExecutor:
        async def execute(self, *_args, **_kwargs):
            return {"reward": 0.0, "scores": 0.0, "extra_logs": {}}

    agent_module.SingleTurnAgentExecutor = _SingleTurnAgentExecutor
    monkeypatch.setitem(sys.modules, "openrlhf.utils.agent", agent_module)

    monkeypatch.setenv("FLASH_OPENRLHF_MAX_RESPONSE_LENGTH", "5")
    monkeypatch.delenv("FLASH_OPENRLHF_LANGUAGE_MODEL_ONLY", raising=False)
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
    assert _value(args, "--actor.lr_scheduler") == "constant"
    assert _value(args, "--ds.lora.rank") == "32"
    assert _value(args, "--ds.lora.alpha") == "64"
    assert _value(args, "--ds.lora.target_modules") == "all-linear"
    assert _value(args, "--ds.zero_stage") == "3"
    assert _value(args, "--actor.num_gpus_per_node") == "2"
    assert _value(args, "--vllm.num_engines") == "2"
    assert "--train.colocate_all" in args
    assert "--vllm.enforce_eager" in args
    assert "--ds.attn_implementation" in args
    assert "--actor.gradient_checkpointing_reentrant" in args
    assert _value(args, "--algo.kl.init_coef") == "0.0"
    assert "--algo.advantage.is_correction_enable" not in args


@requires_openrlhf_source
def test_pinned_openrlhf_parser_accepts_full_generated_args(monkeypatch):
    parsed = _parse_with_pinned_openrlhf(
        monkeypatch,
        grpo_openrlhf.build_openrlhf_grpo_args(_config()),
    )

    assert parsed.reward.clip_range == [-1_000_000_000.0, 1_000_000_000.0]
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


def test_build_openrlhf_grpo_args_rejects_degenerate_group():
    with pytest.raises(ValueError, match="group_size greater than 1"):
        grpo_openrlhf.build_openrlhf_grpo_args(_config(group_size=1))


def test_dr_grpo_fixed_length_normalization_matches_trl_formula():
    losses = [[1.0, 2.0, 100.0], [3.0, 4.0, 5.0]]
    masks = [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]

    actual = grpo_openrlhf.dr_grpo_fixed_length_normalize(losses, masks, 5)

    assert actual == pytest.approx((1.0 + 2.0) / (2 * 5))
    token_mean = (1.0 + 2.0) / 2
    assert actual != pytest.approx(token_mean)


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

    with grpo_openrlhf.RewardBridge(score, token="test-token") as bridge:
        response = grpo_openrlhf.post_reward_request(
            bridge.url,
            {
                "query": ["rendered promptcompletion"],
                "prompts": ["rendered prompt"],
                "labels": [7],
            },
        )

    assert response == {"rewards": 1.25, "scores": 1.25, "extra_logs": {"format": 0.5}}
    assert calls == [(7, "completion", "rendered prompt")]
    assert bridge.rewards == [1.25]


def test_reward_bridge_rejects_bad_auth_and_scoring_failures():
    def fail(_label, _completion, _prompt):
        raise RuntimeError("environment unavailable")

    with grpo_openrlhf.RewardBridge(fail, token="right-token") as bridge:
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
    monkeypatch.setenv("USER_ENV_SECRET", "do-not-forward")
    monkeypatch.setenv("RUNPOD_API_KEY", "do-not-forward")
    monkeypatch.setenv("HF_TOKEN", "do-not-forward")

    child = grpo_openrlhf.build_openrlhf_child_env(
        plugin_dir="/work/plugin",
        max_response_length=320,
        language_model_only=True,
    )

    assert child["PATH"] == "/usr/bin"
    assert child["NCCL_DEBUG"] == "WARN"
    assert child["FLASH_OPENRLHF_MAX_RESPONSE_LENGTH"] == "320"
    assert child["FLASH_OPENRLHF_LANGUAGE_MODEL_ONLY"] == "1"
    assert "USER_ENV_SECRET" not in child
    assert "RUNPOD_API_KEY" not in child
    assert "HF_TOKEN" not in child


def test_sitecustomize_carries_fail_closed_and_fixed_length_hooks():
    source = grpo_openrlhf._sitecustomize_source()

    compile(source, "sitecustomize.py", "exec")
    assert "ActorPPOTrainer.training_step = _flash_training_step" in source
    assert "loss.shape[0] * _MAX_RESPONSE_LENGTH" in source
    assert "_ppo_actor_module.aggregate_loss = _flash_aggregate_loss" in source
    assert "experience.action_mask.zero_()" in source
    assert "_ActorPPOTrainer.broadcast_to_vllm = _flash_broadcast_to_vllm" in source
    assert 'for key in ("reward", "scores")' in source
    assert "returned invalid {key}" in source
    assert "language_model_only" in source
    assert 'target_modules"] = "all-linear"' in source


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


def test_run_rl_openrlhf_launches_mock_subprocess_and_exports(monkeypatch, tmp_path):
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
    monkeypatch.setattr(grpo_openrlhf._w, "JOB_SPEC", fake_job)
    monkeypatch.setattr(grpo_openrlhf._w, "SEED", 42)
    monkeypatch.setattr(grpo_openrlhf._w, "THINKING", False)
    monkeypatch.setattr(grpo_openrlhf._w, "heartbeat", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grpo_openrlhf._w, "prefetch_model", lambda *_args, **_kwargs: 1.5)
    monkeypatch.setattr(grpo_openrlhf._w, "is_vl_checkpoint", lambda *_args: False)
    monkeypatch.setattr(grpo_openrlhf._w, "graded_text", lambda text, **_kwargs: text)
    monkeypatch.setattr(grpo_openrlhf, "wait_for_gpu", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grpo_openrlhf, "setup_perf_backends", lambda: None)
    monkeypatch.setattr(grpo_openrlhf, "gpu_diagnostics", lambda: {})
    monkeypatch.setattr(
        grpo_openrlhf, "liveness_heartbeat", lambda *_args, **_kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        grpo_openrlhf, "resolve_openrlhf_python", lambda _workdir: "/openrlhf/python"
    )
    monkeypatch.setattr(
        grpo_openrlhf,
        "_resolve_cached_model_snapshot",
        lambda *_args: "/cache/snapshots/" + "a" * 40,
    )
    monkeypatch.setattr(
        grpo_openrlhf,
        "_resolve_single_turn_inputs",
        lambda: {
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
            "save_every": 1,
            "gpu_count": 1,
            "seed": 42,
        },
    )
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
        assert reward["rewards"] == 2.0
        kwargs["on_step"](2)
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
    monkeypatch.setattr(
        grpo_openrlhf._w, "publish_deployable_checkpoint", lambda *_args, **_kwargs: None
    )
    metadata = []
    monkeypatch.setattr(
        grpo_openrlhf._w, "write_train_meta", lambda **kwargs: metadata.append(kwargs)
    )
    monkeypatch.setattr(grpo_openrlhf, "time", SimpleNamespace(time=lambda: 10.0))

    grpo_openrlhf.run_rl_openrlhf()

    assert launched["python"] == "/openrlhf/python"
    assert _value(launched["args"], "--algo.advantage.estimator") == "dr_grpo"
    assert launched["env"]["FLASH_OPENRLHF_MAX_RESPONSE_LENGTH"] == "32"
    assert launched["env"]["HF_HUB_OFFLINE"] == "1"
    assert exports
    assert exports[0][2:4] == ("Qwen/Qwen3.5-0.8B", "a" * 40)
    assert metadata[0]["notes"]["backend"] == "openrlhf"
    assert metadata[0]["notes"]["reward_history"] == [2.0]
