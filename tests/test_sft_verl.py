"""CPU unit tests for the sft->verl override builder."""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace

import pytest

from flash.engine.worker.sft_verl import (
    build_sft_verl_messages_rows,
    build_sft_verl_overrides,
    render_loraplus_shim,
)


def _cfg(**over):
    base = {
        "train_files": "/w/train.parquet",
        "val_files": "/w/val.parquet",
        "train_batch_size": 32,
        "max_length": 32768,
        "micro_batch": 1,
        "max_token_len_per_gpu": 8192,
        "model_path": "Qwen/Qwen3-4B",
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "ulysses_sp_size": 2,
        "lr": 1e-4,
        "warmup_ratio": 0.03,
        "optimizer_impl": "bitsandbytes.optim",
        "optimizer_name": "PagedAdamW8bit",
        "local_dir": "/w/ckpt",
        "save_freq": 50,
        "n_gpus_per_node": 2,
        "seed": 42,
        "project_name": "flash-sft",
        "experiment_name": "run-xyz",
        "loop_epochs": 4,
        "total_training_steps": 120,
    }
    base.update(over)
    return base


def _as_map(ov):
    return dict(s.split("=", 1) for s in ov)


def test_overrides_cover_the_32k_lora_sp_surface():
    m = _as_map(build_sft_verl_overrides(_cfg()))
    # long context + packing + sequence parallel + liger are the 32k-enabling knobs.
    assert m["data.train_batch_size"] == "32"
    assert m["data.max_length"] == "32768"
    assert m["model.use_remove_padding"] == "true"
    assert m["engine.ulysses_sequence_parallel_size"] == "2"
    assert m["model.use_liger"] == "true"
    assert m["engine.strategy"] == "fsdp2"
    assert m["engine.model_dtype"] == "bfloat16"
    assert m["engine.seed"] == "42"
    # sequence-parallel degree and gpu count both come from gpu.count.
    assert m["trainer.n_gpus_per_node"] == "2"
    assert m["trainer.nnodes"] == "1"
    # messages-based dataset (verl computes the role mask); no prompt/response keys.
    assert m["data.messages_key"] == "messages"
    assert m["data.use_dynamic_bsz"] == "true"
    # lora on the immutable base; path key is model.path (not partial_pretrain).
    assert m["model.path"] == "Qwen/Qwen3-4B"
    assert m["model.lora_rank"] == "16"
    # lr renders as a plain decimal hydra parses as a float (1e-4 -> "0.0001", not scientific).
    assert m["optim.lr"] == "0.0001"
    assert m["optim.lr_warmup_steps_ratio"] == "0.03"
    assert m["optim.lr_scheduler_type"] == "linear"
    assert m["optim.optimizer"] == "PagedAdamW8bit"
    assert m["trainer.seed"] == "42"
    assert m["trainer.total_epochs"] == "4"
    assert "data.train_files=/w/train.parquet" in build_sft_verl_overrides(_cfg())


def test_small_lr_renders_fixed_point_not_scientific():
    # 5e-5 would str() as "5e-05"; hydra should get plain decimal.
    m = _as_map(build_sft_verl_overrides(_cfg(lr=5e-5)))
    assert m["optim.lr"] == "0.00005"


def test_target_modules_list_renders_as_hydra_list():
    m = _as_map(build_sft_verl_overrides(_cfg(target_modules=["q_proj", "v_proj"])))
    assert m["model.target_modules"] == "[q_proj,v_proj]"


def test_epochs_path_when_no_steps():
    m = _as_map(
        build_sft_verl_overrides(
            _cfg(total_training_steps=None, total_epochs=3, loop_epochs=3)
        )
    )
    assert m["trainer.total_epochs"] == "3"
    assert m["trainer.total_training_steps"] == "null"


def test_steps_xor_epochs_is_enforced():
    with pytest.raises(ValueError, match="exactly one"):
        build_sft_verl_overrides(_cfg(total_training_steps=120, total_epochs=3))
    with pytest.raises(ValueError, match="exactly one"):
        build_sft_verl_overrides(_cfg(total_training_steps=None, total_epochs=None))


def test_missing_required_key_raises():
    bad = _cfg()
    del bad["model_path"]
    with pytest.raises(KeyError, match="model_path"):
        build_sft_verl_overrides(bad)


def test_messages_rows_concatenate_prompt_and_completion():
    prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    completion = [{"role": "assistant", "content": "a"}]
    rows = build_sft_verl_messages_rows([(prompt, completion)])
    assert rows == [{"messages": [*prompt, *completion]}]


def test_messages_rows_drop_empty_completion():
    prompt = [{"role": "user", "content": "u"}]
    rows = build_sft_verl_messages_rows(
        [
            (prompt, [{"role": "assistant", "content": "a"}]),
            (prompt, []),
        ],
        enable_thinking=True,
    )
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["role"] == "assistant"
    assert rows[0]["enable_thinking"] is True


def test_loraplus_shim_patches_verl_fsdp_optimizer_with_ratio():
    source = render_loraplus_shim(16)
    assert "create_loraplus_optimizer" in source
    assert "loraplus_lr_ratio=16.0" in source
    assert "FSDPEngine._build_optimizer" in source
    assert "_FlashFSDPEngine._build_optimizer = _flash_build_loraplus_optimizer" in source


def test_loraplus_shim_is_absent_when_ratio_is_not_greater_than_one():
    assert render_loraplus_shim(1) == ""


def test_sft_sitecustomize_preserves_exact_saves_seed_and_reentrant_gc():
    from flash.engine.worker.sft_verl import _render_sft_sitecustomize

    source = _render_sft_sitecustomize(
        seed=43,
        loraplus_ratio=16,
        save_at_steps=(3, 7),
        total_steps=9,
        reentrant_gradient_checkpointing=True,
    )
    assert "_flash_seed = 43" in source
    assert "frozenset((3, 7))" in source
    assert "step != _flash_total_steps" in source
    assert "get_linear_schedule_with_warmup" in source
    assert 'kwargs["drop_last"] = False' in source
    assert '"use_reentrant": True' in source
    assert "create_loraplus_optimizer" in source


def test_checkpoint_watcher_exports_and_uploads_required_step(monkeypatch, tmp_path):
    import flash.engine.worker as worker
    from flash.engine.worker import sft_verl

    checkpoint_dir = tmp_path / "checkpoints" / "global_step_5"
    actor_dir = checkpoint_dir / "actor"
    actor_dir.mkdir(parents=True)
    exported = []
    published = []
    uploaded = []

    def fake_export(actor, adapter, **kwargs):
        exported.append((actor, adapter, kwargs))
        os.makedirs(adapter, exist_ok=True)

    monkeypatch.setattr(sft_verl, "_export_checkpoint_adapter", fake_export)
    monkeypatch.setattr(
        worker,
        "publish_deployable_checkpoint",
        lambda adapter, step, **kwargs: published.append((adapter, step, kwargs)),
    )

    def fake_upload(step, checkpoint, **kwargs):
        kwargs["before_upload"]()
        uploaded.append((step, checkpoint))
        return True

    monkeypatch.setattr(worker, "upload_resume_checkpoint", fake_upload)
    watcher = sft_verl._VerlCheckpointWatcher(
        local_dir=str(tmp_path / "checkpoints"),
        export_root=str(tmp_path / "exports"),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(5,),
    )

    watcher._publish(5, str(checkpoint_dir))

    assert exported[0][0] == str(actor_dir)
    assert published[0][1] == 5
    assert published[0][2]["required"] is True
    assert uploaded == [(5, str(checkpoint_dir))]
    assert watcher.processed_steps == {5}


def test_resume_credits_only_required_saves_that_are_durable(monkeypatch):
    import flash.engine.worker as worker
    from flash.engine.worker import sft_verl

    class Api:
        def file_exists(self, *, filename, **kwargs):
            return "/step-3/" in filename

    monkeypatch.setattr(worker, "HF_REPO", "owner/artifacts")
    monkeypatch.setattr(worker, "hf_prefix", lambda: "sft/run")
    monkeypatch.setattr(worker, "hf_api", Api)

    assert sft_verl._durable_required_save_steps((3, 5, 9), 5) == {3}


def test_cached_model_path_uses_prefetched_snapshot_for_mutable_refs(monkeypatch):
    from flash.engine.worker import sft_verl

    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.append(kwargs) or "/cache/snapshot/commit",
    )

    assert sft_verl._cached_model_path("org/model", "") == "/cache/snapshot/commit"
    assert calls == [{"repo_id": "org/model", "revision": None, "local_files_only": True}]


def test_run_sft_verl_orchestrates_subprocess_export_and_metadata(monkeypatch):
    import flash.engine.worker as worker
    from flash.engine.worker import sft_verl

    spec = SimpleNamespace(
        model="Qwen/Qwen3.5-0.8B",
        model_revision="",
        gpu=SimpleNamespace(type="RTX 4090", exact_type="", count=2),
        train=SimpleNamespace(
            epochs=1,
            learning_rate=5e-5,
            batch_size=2,
            max_context_tokens=1024,
            max_examples=2,
            max_steps=2,
            save_at_steps=(),
            save_every=50,
            init_from_adapter="",
        ),
        wandb=SimpleNamespace(project=None, run_name=None),
    )

    class Env:
        id = "owner/env"
        package_root = None
        multi_turn = False

        def dataset(self):
            return [{"prompt": "one"}, {"prompt": "two"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["prompt"]}]

        def sft_completion(self, example):
            return [{"role": "assistant", "content": "answer"}]

    class LoraConfig:
        r = 16
        lora_alpha = 32
        target_modules = "all-linear"

    class Optimizer:
        pass

    Optimizer.__module__ = "torch.optim"
    Optimizer.__name__ = "AdamW"

    class PeakSampler:
        def start(self):
            return self

        def stop_gb(self):
            return 12.5

    class Watcher:
        def __init__(self, **kwargs):
            self.processed_steps = set()

        def start(self):
            return None

        def stop(self, *, require_complete):
            assert require_complete is True

        def raise_if_failed(self):
            return None

    captured = {"heartbeats": [], "published": [], "uploads": []}
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(worker, "SEED", 7)
    monkeypatch.setattr(worker, "RUN_ID", "test-sft-verl-orchestration")
    monkeypatch.setattr(worker, "THINKING", False)
    monkeypatch.setattr(worker, "JOB_SPEC", spec)
    monkeypatch.setattr(worker, "seed_training_rngs", lambda seed: None)
    monkeypatch.setattr(worker, "require_active_env", Env)
    monkeypatch.setattr(
        worker,
        "heartbeat",
        lambda stage, **fields: captured["heartbeats"].append((stage, fields)),
    )
    monkeypatch.setattr(worker, "gpu_diagnostics", lambda **kwargs: {})
    monkeypatch.setattr(worker, "wait_for_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "setup_perf_backends", lambda: None)
    monkeypatch.setattr(worker, "prefetch_model", lambda *args, **kwargs: 1.25)
    monkeypatch.setattr(worker, "make_lora", lambda model_id: LoraConfig())
    monkeypatch.setattr(worker, "grad_checkpointing_on", lambda *args, **kwargs: True)
    monkeypatch.setattr(worker, "grpo_use_reentrant", lambda model_id: False)
    monkeypatch.setattr(worker, "loraplus_optimizer_cls", lambda name: (Optimizer, {}))
    monkeypatch.setattr(worker, "fused_optim_name", lambda: "paged_adamw_8bit")
    monkeypatch.setattr(worker, "backend_seed", lambda seed: seed)
    monkeypatch.setattr(worker, "wandb_run_name", lambda: "flash-sft-test")
    monkeypatch.setattr(worker, "_reset_peak_gpu", lambda: None)
    monkeypatch.setattr(worker, "_GpuPeakSampler", PeakSampler)
    monkeypatch.setattr(
        worker,
        "hf_upload_folder",
        lambda local, remote, required=False: captured["uploads"].append(
            (local, remote, required)
        ),
    )
    monkeypatch.setattr(
        worker,
        "publish_deployable_checkpoint",
        lambda adapter, step, **kwargs: captured["published"].append((adapter, step)),
    )
    monkeypatch.setattr(
        worker,
        "write_train_meta",
        lambda **kwargs: captured.__setitem__("meta", kwargs),
    )
    monkeypatch.setattr(worker, "free_gpu", lambda *args: None)
    monkeypatch.setattr(sft_verl, "liveness_heartbeat", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(sft_verl, "_model_arch_dims", lambda *args, **kwargs: (1024, 24))
    monkeypatch.setattr(sft_verl, "resolve_verl_python", lambda workdir: "/venv/bin/python")
    monkeypatch.setattr(sft_verl, "_cached_model_path", lambda model, revision: model)
    monkeypatch.setattr(sft_verl, "_restore_verl_resume", lambda local_dir: 0)
    monkeypatch.setattr(sft_verl, "_VerlCheckpointWatcher", Watcher)
    monkeypatch.setattr(
        sft_verl,
        "latest_global_step_dir",
        lambda local_dir: (os.path.join(local_dir, "global_step_2", "actor"), 2),
    )

    def fake_export(actor_dir, adapter_dir, **kwargs):
        os.makedirs(adapter_dir, exist_ok=True)
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as file:
            file.write("{}")
        with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "wb") as file:
            file.write(b"adapter")

    monkeypatch.setattr(sft_verl, "_export_checkpoint_adapter", fake_export)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        captured["command"] = command
        captured["child_env"] = env
        on_line("step:1 - train/loss:1.25 - train/grad_norm:0.5 - train/lr:0.00005 - train/global_tokens:8\n")
        on_step(1)
        heartbeat()
        on_line("[lora+] optimizer enabled (B-matrix LR ratio=16, cls=AdamW)\n")
        on_line("step:2 - train/loss:1.0 - train/global_tokens:8\n")
        on_step(2)
        return 0

    monkeypatch.setattr(sft_verl, "run_verl_training", fake_training)

    sft_verl.run_sft_verl(spec)

    assert captured["command"][:3] == ["/venv/bin/python", "-m", "torch.distributed.run"]
    assert "--nproc-per-node=2" in captured["command"]
    assert "verl.trainer.sft_trainer" in captured["command"]
    assert captured["child_env"]["PYTHONPATH"].split(os.pathsep)[0].endswith("/shim")
    assert captured["child_env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["child_env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert captured["uploads"][0][1:] == ("adapter", True)
    assert captured["published"][0][1] == 2
    assert captured["meta"]["step"] == 2
    assert captured["meta"]["train_tokens"] == 16
    assert captured["meta"]["notes"]["loss_curve"] == [1.25, 1.0]
    assert captured["meta"]["notes"]["loraplus_applied"] is True
