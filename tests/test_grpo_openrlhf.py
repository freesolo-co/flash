"""cpu coverage for the OpenRLHF GRPO foundation."""

from __future__ import annotations

import contextlib
import os
import urllib.error
from types import SimpleNamespace

import pytest

from flash.engine.worker import grpo_openrlhf, rl


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
    assert _value(args, "--reward.clip_range") == "-inf"
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
    assert _value(args, "--algo.kl.init_coef") == "0.0"
    assert "--algo.advantage.is_correction_enable" not in args


def test_build_openrlhf_grpo_args_enables_fresh_policy_kl():
    args = grpo_openrlhf.build_openrlhf_grpo_args(_config(kl_coef=0.03))

    assert _value(args, "--algo.kl.init_coef") == "0.03"
    assert "--algo.kl.use_loss" in args
    assert _value(args, "--algo.kl.estimator") == "k2"


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
    assert 'for key in ("reward", "scores")' in source
    assert "returned invalid {key}" in source
    assert "language_model_only" in source
    assert "target_modules\"] = \"all-linear\"" in source


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
    monkeypatch.setattr(grpo_openrlhf, "liveness_heartbeat", lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setattr(grpo_openrlhf, "resolve_openrlhf_python", lambda _workdir: "/openrlhf/python")
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
