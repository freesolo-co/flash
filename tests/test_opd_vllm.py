from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest


def test_opd_lora_rank_uses_maximum_default_and_pattern_rank():
    from flash.engine.worker.opd_vllm import opd_lora_rank

    model = SimpleNamespace(
        peft_config={
            "first": SimpleNamespace(r=16, rank_pattern={"a": 64}),
            "second": SimpleNamespace(r=32, rank_pattern={"b": 8}),
        }
    )
    assert opd_lora_rank(model) == 64


def test_opd_lora_rank_reads_dict_valued_r():
    # Some PEFT configs express per-module ranks via a dict-valued `r` (no rank_pattern); the
    # max-rank path must honor it, else a rank-64 continued adapter initializes vLLM too small.
    from flash.engine.worker.opd_vllm import opd_lora_rank

    model = SimpleNamespace(peft_config={"only": SimpleNamespace(r={"q_proj": 64, "v_proj": 16})})
    assert opd_lora_rank(model) == 64


def test_opd_vllm_engine_syncs_versioned_lora_and_generates(monkeypatch, tmp_path):
    """OPD's vLLM engine must generate with the latest saved adapter, not a stale LoRA."""
    from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine

    class _SamplingParams:
        last_kwargs: ClassVar[dict] = {}

        def __init__(self, **kwargs):
            _SamplingParams.last_kwargs = dict(kwargs)

    class _LoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    class _FakeLLM:
        last_kwargs: ClassVar[dict] = {}
        last_generate: ClassVar[dict] = {}

        def __init__(self, **kwargs):
            _FakeLLM.last_kwargs = dict(kwargs)
            self.removed = []
            self.llm_engine = SimpleNamespace(remove_lora=self.removed.append)

        def generate(self, prompts, *, sampling_params, lora_request, use_tqdm):
            _FakeLLM.last_generate = {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "lora_request": lora_request,
                "use_tqdm": use_tqdm,
            }
            comp = SimpleNamespace(
                token_ids=[3, 4],
                text="ok",
                finish_reason="stop",
                stop_reason=None,
            )
            return [SimpleNamespace(outputs=[comp])]

    vllm = types.ModuleType("vllm")
    vllm.LLM = _FakeLLM
    vllm.SamplingParams = _SamplingParams
    lora_pkg = types.ModuleType("vllm.lora")
    req_mod = types.ModuleType("vllm.lora.request")
    req_mod.LoRARequest = _LoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_pkg)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", req_mod)

    class _Model:
        def save_pretrained(self, path):
            (tmp_path / "seen.txt").write_text(path)

    engine = OpdVllmRolloutEngine(
        model_source="base-or-merged",
        model_revision="refs/pr/123",
        max_model_len=2048,
        temperature=0.7,
        top_p=0.9,
        stop_sequences=("</answer>",),
        lora_rank=16,
        gpu_memory_utilization=0.42,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=8192,
        attention_backend="TRITON_ATTN",
        mm_encoder_attn_backend="TORCH_SDPA",
        enforce_eager=True,
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
        seed=123,
        adapter_root=str(tmp_path / "sync"),
    )
    engine.sync_from_model(_Model())
    first = engine._lora_request
    engine.sync_from_model(_Model())
    second = engine._lora_request
    out = engine.generate([[1, 2]], max_tokens=5)

    assert _FakeLLM.last_kwargs["model"] == "base-or-merged"
    assert _FakeLLM.last_kwargs["revision"] == "refs/pr/123"
    assert _FakeLLM.last_kwargs["enable_lora"] is True
    assert _FakeLLM.last_kwargs["max_lora_rank"] == 16
    assert _FakeLLM.last_kwargs["gpu_memory_utilization"] == 0.42
    assert _FakeLLM.last_kwargs["kv_cache_dtype"] == "fp8"
    assert _FakeLLM.last_kwargs["max_num_batched_tokens"] == 8192
    assert _FakeLLM.last_kwargs["attention_backend"] == "TRITON_ATTN"
    assert _FakeLLM.last_kwargs["mm_encoder_attn_backend"] == "TORCH_SDPA"
    assert "enable_tower_connector_lora" not in _FakeLLM.last_kwargs
    assert _FakeLLM.last_kwargs["enforce_eager"] is True
    assert _FakeLLM.last_kwargs["compilation_config"] == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }
    assert _FakeLLM.last_kwargs["seed"] == 123
    assert first.lora_int_id == 1
    assert second.lora_int_id == 2
    assert second.lora_path.endswith("adapter-000002")
    assert not (tmp_path / "sync" / "adapter-000001").exists()
    assert (tmp_path / "sync" / "adapter-000002").exists()
    assert engine._sync_dirs == [second.lora_path]
    assert _FakeLLM.last_generate["prompts"] == [{"prompt_token_ids": [1, 2]}]
    assert _FakeLLM.last_generate["lora_request"] is second
    assert _FakeLLM.last_generate["use_tqdm"] is False
    assert _SamplingParams.last_kwargs["stop"] == ["</answer>"]
    assert _SamplingParams.last_kwargs["include_stop_str_in_output"] is True
    assert "logit_bias" not in _SamplingParams.last_kwargs
    assert out[0].token_ids == [3, 4]
    assert out[0].terminated is True


def test_opd_vllm_engine_chunks_generate_by_rollout_batch_size(monkeypatch, tmp_path):
    from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine

    class _SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class _LoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    class _FakeLLM:
        batch_sizes: ClassVar[list[int]] = []
        last_kwargs: ClassVar[dict] = {}
        request_seeds: ClassVar[list[int]] = []

        def __init__(self, **kwargs):
            _FakeLLM.last_kwargs = dict(kwargs)

        def generate(self, prompts, *, sampling_params, lora_request, use_tqdm):
            _FakeLLM.batch_sizes.append(len(prompts))
            _FakeLLM.request_seeds.extend(params.kwargs["seed"] for params in sampling_params)
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=[3],
                            text="ok",
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ]
                )
                for _ in prompts
            ]

    vllm = types.ModuleType("vllm")
    vllm.LLM = _FakeLLM
    vllm.SamplingParams = _SamplingParams
    lora_pkg = types.ModuleType("vllm.lora")
    req_mod = types.ModuleType("vllm.lora.request")
    req_mod.LoRARequest = _LoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_pkg)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", req_mod)

    class _Model:
        def save_pretrained(self, path):
            (tmp_path / "seen.txt").write_text(path)

    engine = OpdVllmRolloutEngine(
        model_source="base",
        max_model_len=2048,
        temperature=0.7,
        top_p=0.9,
        rollout_batch_size=2,
        adapter_root=str(tmp_path / "sync"),
    )
    engine.sync_from_model(_Model())

    out = engine.generate(
        [[1], [2], [3], [4], [5]], max_tokens=5, request_seeds=[11, 12, 13, 14, 15]
    )

    assert "revision" not in _FakeLLM.last_kwargs
    assert _FakeLLM.batch_sizes == [2, 2, 1]
    assert _FakeLLM.request_seeds == [11, 12, 13, 14, 15]
    assert [item.token_ids for item in out] == [[3], [3], [3], [3], [3]]


def test_opd_vllm_engine_recasts_enginecore_startup_oom(monkeypatch):
    from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine

    class _SamplingParams:
        pass

    class _LoRARequest:
        pass

    class _FakeLLM:
        def __init__(self, **_kwargs):
            raise RuntimeError("EngineCore initialization failed")

    class _OutOfMemoryError(RuntimeError):
        pass

    class _Cuda:
        OutOfMemoryError = _OutOfMemoryError

        @staticmethod
        def mem_get_info():
            return 2 * 1024**3, 32 * 1024**3

    vllm = types.ModuleType("vllm")
    vllm.LLM = _FakeLLM
    vllm.SamplingParams = _SamplingParams
    lora_pkg = types.ModuleType("vllm.lora")
    req_mod = types.ModuleType("vllm.lora.request")
    req_mod.LoRARequest = _LoRARequest
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = _Cuda
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_pkg)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", req_mod)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    with pytest.raises(_OutOfMemoryError, match="vLLM EngineCore startup OOM"):
        OpdVllmRolloutEngine(
            model_source="base",
            max_model_len=2048,
            temperature=0.7,
            top_p=0.9,
            gpu_memory_utilization=0.50,
        )


def test_opd_vllm_output_uses_stop_vs_length_for_skip_semantics():
    from flash.engine.worker.opd import OpdKnobs, _gen_from_vllm_output
    from flash.engine.worker.opd_vllm import OpdVllmOutput

    class _Tok:
        eos_token_id = 99

        def decode(self, ids, skip_special_tokens=True):
            table = {1: "ok", 2: "</answer>", 99: "" if skip_special_tokens else "<eos>"}
            return "".join(table.get(int(i), "") for i in ids)

    knobs = OpdKnobs(stop_sequences=("</answer>",))
    eos_ids = frozenset({99})
    truncated = _gen_from_vllm_output(
        OpdVllmOutput([1], "ok", finish_reason="length"), _Tok(), knobs, eos_ids
    )
    assert truncated.truncated is True
    assert truncated.finish_reason == "length"
    assert truncated.stop_reason is None

    stopped = _gen_from_vllm_output(
        OpdVllmOutput([1, 2], "ok</answer>", finish_reason="stop"), _Tok(), knobs, eos_ids
    )
    assert stopped.truncated is False
    assert stopped.skip is False
    assert stopped.completion_ids == [1]
    assert stopped.completion_text == "ok"

    length_stopped = _gen_from_vllm_output(
        OpdVllmOutput([1, 2], "ok</answer>", finish_reason="length"), _Tok(), knobs, eos_ids
    )
    assert length_stopped.truncated is False
    assert length_stopped.skip is False
    assert length_stopped.completion_ids == [1]
    assert length_stopped.completion_text == "ok"
    assert length_stopped.finish_reason == "length"

    length_eos = _gen_from_vllm_output(
        OpdVllmOutput([1, 99], "ok", finish_reason="length"),
        _Tok(),
        OpdKnobs(),
        eos_ids,
    )
    assert length_eos.truncated is False
    assert length_eos.skip is False
    assert length_eos.completion_ids == [1, 99]
    assert length_eos.completion_text == "ok"
    assert length_eos.terminal_eos_id == 99
    stop_over_eos_over_length = _gen_from_vllm_output(
        OpdVllmOutput([1, 99, 2], "ok</answer>", finish_reason="length", stop_reason="</answer>"),
        _Tok(),
        knobs,
        eos_ids,
    )
    assert stop_over_eos_over_length.truncated is False
    assert stop_over_eos_over_length.completion_ids == [1, 99]
    assert stop_over_eos_over_length.completion_text == "ok"
    assert stop_over_eos_over_length.terminal_eos_id == 99


def _install_opd_kwargs_test_gpu(monkeypatch, *, card_gb=80):
    from flash.engine import vram
    from flash.engine.worker import gpu_setup

    class _Cuda:
        @staticmethod
        def get_device_capability():
            return (8, 9)

        @staticmethod
        def get_device_properties(_idx):
            return SimpleNamespace(total_memory=card_gb * 1024**3)

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = _Cuda
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setattr(gpu_setup, "force_vllm_backend_for_sm120", lambda: None)
    monkeypatch.setattr(gpu_setup, "force_vit_sdpa_on_blackwell", lambda: None)
    monkeypatch.setattr(vram, "resolve_params_b", lambda _model_id: 4.0)
    monkeypatch.setattr(vram, "colocate_kv_util", lambda *args, **kwargs: 0.37)


def test_opd_vllm_kwargs_floors_hybrid_mamba_budget_on_sub_140gb_card(monkeypatch):
    from flash.engine.worker.opd_vllm import opd_vllm_kwargs

    _install_opd_kwargs_test_gpu(monkeypatch)

    out = opd_vllm_kwargs(
        "Qwen/Qwen3.6-35B-A3B",
        SimpleNamespace(prompts_per_step=4, group_size=1),
        256,
    )

    assert out["max_num_seqs"] == 4
    assert out["max_num_batched_tokens"] == 1072


def test_opd_vllm_kwargs_keeps_derived_hybrid_mamba_budget_on_sub_140gb_card(monkeypatch):
    from flash.engine.worker.opd_vllm import opd_vllm_kwargs

    _install_opd_kwargs_test_gpu(monkeypatch)

    out = opd_vllm_kwargs(
        "Qwen/Qwen3.6-35B-A3B",
        SimpleNamespace(prompts_per_step=8, group_size=1),
        1536,
    )

    assert out["max_num_seqs"] == 8
    assert out["max_num_batched_tokens"] is None


def test_opd_vllm_kwargs_leaves_non_mamba_budget_unset_on_sub_140gb_card(monkeypatch):
    from flash.engine.worker.opd_vllm import opd_vllm_kwargs

    _install_opd_kwargs_test_gpu(monkeypatch)

    out = opd_vllm_kwargs(
        "Qwen/Qwen3.5-4B",
        SimpleNamespace(prompts_per_step=4, group_size=1),
        256,
    )

    assert out["max_num_batched_tokens"] is None


def test_opd_vllm_kwargs_sizes_memory_for_full_prompt_batch(monkeypatch):
    from flash.engine import vram
    from flash.engine.worker import gpu_setup
    from flash.engine.worker.opd_vllm import opd_vllm_kwargs

    class _Cuda:
        @staticmethod
        def get_device_capability():
            return (8, 9)

        @staticmethod
        def get_device_properties(_idx):
            return SimpleNamespace(total_memory=80e9)

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = _Cuda
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setattr(gpu_setup, "force_vllm_backend_for_sm120", lambda: None)
    monkeypatch.setattr(gpu_setup, "force_vit_sdpa_on_blackwell", lambda: None)
    monkeypatch.setattr(vram, "resolve_params_b", lambda _model_id: 4.0)
    captured = {}

    def _capture_util(*_args, **kwargs):
        captured.update(kwargs)
        return 0.37

    monkeypatch.setattr(vram, "colocate_kv_util", _capture_util)
    knobs = SimpleNamespace(prompts_per_step=8, group_size=3)

    out = opd_vllm_kwargs("test/model", knobs, 4096)

    assert captured["num_generations"] == 24
    assert out["gpu_memory_utilization"] == 0.37


def test_opd_vllm_kwargs_reduces_rollout_batch_when_startup_memory_is_tight(monkeypatch):
    from flash.engine import vram
    from flash.engine.worker import gpu_setup
    from flash.engine.worker.opd_vllm import opd_vllm_kwargs

    class _Cuda:
        @staticmethod
        def get_device_capability():
            return (8, 0)

        @staticmethod
        def get_device_properties(_idx):
            return SimpleNamespace(total_memory=80 * 1024**3)

        @staticmethod
        def mem_get_info():
            return 23 * 1024**3, 80 * 1024**3

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = _Cuda
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setattr(gpu_setup, "force_vllm_backend_for_sm120", lambda: None)
    monkeypatch.setattr(gpu_setup, "force_vit_sdpa_on_blackwell", lambda: None)
    monkeypatch.setattr(vram, "resolve_params_b", lambda _model_id: 4.0)

    def _util_for(_params_b, _seq_cap, _card_gb, _sleep_mode, *, num_generations, **_kwargs):
        return {8: 0.40, 7: 0.36, 6: 0.32, 5: 0.28, 4: 0.22}[num_generations]

    monkeypatch.setattr(vram, "colocate_kv_util", _util_for)

    out = opd_vllm_kwargs("test/model", SimpleNamespace(prompts_per_step=8, group_size=1), 4096)

    assert out["gpu_memory_utilization"] == 0.22
    assert out["max_num_seqs"] == 4
    assert out["rollout_batch_size"] == 4


def test_opd_vllm_kwargs_raises_oom_when_single_rollout_cannot_fit(monkeypatch):
    from flash.engine import vram
    from flash.engine.worker import gpu_setup
    from flash.engine.worker.opd_vllm import opd_vllm_kwargs

    class _OutOfMemoryError(RuntimeError):
        pass

    class _Cuda:
        OutOfMemoryError = _OutOfMemoryError

        @staticmethod
        def get_device_capability():
            return (8, 0)

        @staticmethod
        def get_device_properties(_idx):
            return SimpleNamespace(total_memory=80 * 1024**3)

        @staticmethod
        def mem_get_info():
            return 10 * 1024**3, 80 * 1024**3

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = _Cuda
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setattr(gpu_setup, "force_vllm_backend_for_sm120", lambda: None)
    monkeypatch.setattr(gpu_setup, "force_vit_sdpa_on_blackwell", lambda: None)
    monkeypatch.setattr(vram, "resolve_params_b", lambda _model_id: 4.0)

    def _util_for(_params_b, _seq_cap, _card_gb, _sleep_mode, *, num_generations, **_kwargs):
        return {8: 0.40, 7: 0.36, 6: 0.32, 5: 0.28, 4: 0.22, 3: 0.20, 2: 0.18, 1: 0.16}[
            num_generations
        ]

    monkeypatch.setattr(vram, "colocate_kv_util", _util_for)

    with pytest.raises(_OutOfMemoryError, match="rollout_batch_size to 1"):
        opd_vllm_kwargs("test/model", SimpleNamespace(prompts_per_step=8, group_size=1), 4096)


@pytest.mark.parametrize("cc", [(8, 0), (9, 0)])
def test_opd_vllm_kwargs_keeps_cuda_graphs_on_datacenter_cards(monkeypatch, cc):
    from flash.engine import vram
    from flash.engine.worker import gpu_setup
    from flash.engine.worker.opd_vllm import opd_vllm_kwargs

    class _Cuda:
        @staticmethod
        def get_device_capability():
            return cc

        @staticmethod
        def get_device_properties(_idx):
            return SimpleNamespace(total_memory=80e9)

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = _Cuda
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__version__ = "0.19.1"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setattr(gpu_setup, "force_vllm_backend_for_sm120", lambda: None)
    monkeypatch.setattr(gpu_setup, "force_vit_sdpa_on_blackwell", lambda: None)
    monkeypatch.setattr(vram, "resolve_params_b", lambda _model_id: 4.0)
    monkeypatch.setattr(vram, "colocate_kv_util", lambda *a, **k: 0.37)

    out = opd_vllm_kwargs("test/model", SimpleNamespace(prompts_per_step=8, group_size=1), 4096)

    assert out["enforce_eager"] is None
    assert out["compilation_config"] is None


@pytest.mark.parametrize("cc", [(10, 0), (12, 0)])
def test_opd_vllm_kwargs_uses_decode_cuda_graphs_on_blackwell(monkeypatch, cc):
    from flash.engine import vram
    from flash.engine.worker import gpu_setup
    from flash.engine.worker.opd_vllm import opd_vllm_kwargs

    class _Cuda:
        @staticmethod
        def get_device_capability():
            return cc

        @staticmethod
        def get_device_properties(_idx):
            return SimpleNamespace(total_memory=32e9)

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = _Cuda
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__version__ = "0.19.1"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.setattr(gpu_setup, "force_vllm_backend_for_sm120", lambda: "FLASHINFER")
    monkeypatch.setattr(gpu_setup, "force_vit_sdpa_on_blackwell", lambda: None)
    monkeypatch.setattr(vram, "resolve_params_b", lambda _model_id: 4.0)
    monkeypatch.setattr(vram, "colocate_kv_util", lambda *a, **k: 0.37)

    out = opd_vllm_kwargs("test/model", SimpleNamespace(prompts_per_step=8, group_size=1), 4096)

    assert out["enforce_eager"] is False
    assert out["compilation_config"] == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_opd_vllm_kwargs_keeps_eager_workaround_on_unvalidated_non_blackwell(monkeypatch):
    from flash.engine import vram
    from flash.engine.worker import gpu_setup
    from flash.engine.worker.opd_vllm import opd_vllm_kwargs

    class _Cuda:
        @staticmethod
        def get_device_capability():
            return (8, 9)

        @staticmethod
        def get_device_properties(_idx):
            return SimpleNamespace(total_memory=32e9)

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = _Cuda
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__version__ = "0.19.1"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setattr(gpu_setup, "force_vllm_backend_for_sm120", lambda: None)
    monkeypatch.setattr(gpu_setup, "force_vit_sdpa_on_blackwell", lambda: None)
    monkeypatch.setattr(vram, "resolve_params_b", lambda _model_id: 4.0)
    monkeypatch.setattr(vram, "colocate_kv_util", lambda *a, **k: 0.37)

    out = opd_vllm_kwargs("test/model", SimpleNamespace(prompts_per_step=8, group_size=1), 4096)

    assert out["enforce_eager"] is True
    assert out["compilation_config"] is None


def test_opd_vllm_kwargs_forces_b200_v1_inprocess_on_vllm_0190(monkeypatch):
    from flash.engine import vram
    from flash.engine.worker import gpu_setup
    from flash.engine.worker.opd_vllm import opd_vllm_kwargs

    class _Cuda:
        @staticmethod
        def get_device_capability():
            return (10, 0)

        @staticmethod
        def get_device_properties(_idx):
            return SimpleNamespace(total_memory=180e9)

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = _Cuda
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__version__ = "0.19.0"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.setattr(gpu_setup, "force_vllm_backend_for_sm120", lambda: None)
    monkeypatch.setattr(gpu_setup, "force_vit_sdpa_on_blackwell", lambda: None)
    monkeypatch.setattr(vram, "resolve_params_b", lambda _model_id: 35.0)
    monkeypatch.setattr(vram, "colocate_kv_util", lambda *a, **k: 0.37)

    out = opd_vllm_kwargs("test/model", SimpleNamespace(prompts_per_step=1, group_size=1), 4096)

    assert out["max_num_batched_tokens"] == 8192
    assert out["enforce_eager"] is False
    assert out["compilation_config"] == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def _install_fake_vllm(monkeypatch, *, with_structured_outputs=True):
    """Minimal fake vllm (LLM + SamplingParams + lora + sampling_params) for engine tests."""

    class _SamplingParams:
        last_kwargs: ClassVar[dict] = {}
        instances: ClassVar[list] = []

        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            _SamplingParams.last_kwargs = self.kwargs
            _SamplingParams.instances.append(self)

    class _StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class _LoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    class _FakeLLM:
        last_kwargs: ClassVar[dict] = {}
        last_prompts: ClassVar[list] = []

        def __init__(self, **kwargs):
            _FakeLLM.last_kwargs = dict(kwargs)
            self.llm_engine = SimpleNamespace(remove_lora=lambda _id: None)

        def generate(self, prompts, *, sampling_params, lora_request, use_tqdm):
            _FakeLLM.last_prompts = list(prompts)
            comp = SimpleNamespace(token_ids=[3], text="x", finish_reason="stop", stop_reason=None)
            return [SimpleNamespace(outputs=[comp]) for _ in prompts]

    vllm = types.ModuleType("vllm")
    vllm.LLM = _FakeLLM
    vllm.SamplingParams = _SamplingParams
    lora_pkg = types.ModuleType("vllm.lora")
    req_mod = types.ModuleType("vllm.lora.request")
    req_mod.LoRARequest = _LoRARequest
    sp_mod = types.ModuleType("vllm.sampling_params")
    if with_structured_outputs:
        sp_mod.StructuredOutputsParams = _StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_pkg)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", req_mod)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp_mod)
    return _SamplingParams, _StructuredOutputsParams


def test_opd_vllm_sync_atomically_publishes_before_dropping_old_adapter(
    monkeypatch, tmp_path
):
    from flash.engine.worker import opd_vllm

    _install_fake_vllm(monkeypatch)
    events = []

    class _Model:
        def save_pretrained(self, path):
            events.append(("save", os.path.basename(path)))
            Path(path, "adapter_config.json").write_text("{}", encoding="utf-8")

    engine = opd_vllm.OpdVllmRolloutEngine(
        model_source="base",
        max_model_len=2048,
        temperature=0.7,
        top_p=0.9,
        adapter_root=str(tmp_path / "sync"),
    )
    engine.sync_from_model(_Model())
    events.clear()

    real_replace = opd_vllm.os.replace

    def _replace(source, destination):
        real_replace(source, destination)
        events.append(("replace", os.path.basename(destination)))

    def _remove(old_lora_id):
        events.append(
            (
                "remove",
                old_lora_id,
                engine._lora_request.lora_int_id,
                Path(engine._lora_request.lora_path, "adapter_config.json").is_file(),
            )
        )
        return True

    monkeypatch.setattr(opd_vllm.os, "replace", _replace)
    monkeypatch.setattr(engine, "_remove_lora", _remove)

    engine.sync_from_model(_Model())

    assert events[0][0] == "save"
    assert events[0][1].startswith(".adapter-000002-")
    assert events[1] == ("replace", "adapter-000002")
    assert events[2] == ("remove", 1, 2, True)
    assert engine.sync_count == 2
    assert engine._sync_dirs == [str(tmp_path / "sync" / "adapter-000002")]
    assert not any(path.name.startswith(".adapter-") for path in (tmp_path / "sync").iterdir())


def test_opd_vllm_adapter_store_prefers_tmpfs(monkeypatch, tmp_path):
    from flash.engine.worker import opd_vllm

    expected = tmp_path / "memory-backed"
    seen = []

    def _mkdtemp(*, prefix, dir=None):
        seen.append((prefix, dir))
        return str(expected)

    monkeypatch.setattr(opd_vllm.tempfile, "mkdtemp", _mkdtemp)

    assert opd_vllm._make_adapter_root() == (str(expected), True)
    assert seen == [("flash_opd_vllm_lora_", opd_vllm._ADAPTER_TMPFS_ROOT)]


def test_opd_vllm_adapter_store_falls_back_when_tmpfs_save_fails(monkeypatch, tmp_path):
    from flash.engine.worker import opd_vllm

    _install_fake_vllm(monkeypatch)
    tmpfs_root = tmp_path / "tmpfs"
    tmpfs_root.mkdir()
    fallback_parent = tmp_path / "fallback"
    fallback_parent.mkdir()
    real_mkdtemp = tempfile.mkdtemp

    def _mkdtemp(*, prefix, dir=None):
        if dir is None:
            dir = fallback_parent
        return real_mkdtemp(prefix=prefix, dir=dir)

    saved = []

    class _Model:
        def save_pretrained(self, path):
            saved.append(path)
            if Path(path).parent == tmpfs_root:
                raise OSError("tmpfs full")
            Path(path, "adapter_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(opd_vllm, "_make_adapter_root", lambda: (str(tmpfs_root), True))
    monkeypatch.setattr(opd_vllm.tempfile, "mkdtemp", _mkdtemp)
    engine = opd_vllm.OpdVllmRolloutEngine(
        model_source="base",
        max_model_len=2048,
        temperature=0.7,
        top_p=0.9,
    )
    engine.sync_from_model(_Model())

    assert len(saved) == 2
    assert Path(saved[0]).parent == tmpfs_root
    assert Path(engine.adapter_root).parent == fallback_parent
    assert Path(saved[1]).parent == Path(engine.adapter_root)
    assert Path(engine._lora_request.lora_path, "adapter_config.json").is_file()
    assert engine.sync_count == 1
    assert engine._adapter_root_is_tmpfs is False


def test_opd_vllm_image_requests_include_multimodal_data_and_tower_lora(monkeypatch, tmp_path):
    from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine

    sampling_params, _ = _install_fake_vllm(monkeypatch)
    fake_llm = sys.modules["vllm"].LLM

    class _Model:
        def save_pretrained(self, path):
            pass

    image = object()
    engine = OpdVllmRolloutEngine(
        model_source="base",
        max_model_len=2048,
        temperature=0.7,
        top_p=0.9,
        enable_tower_connector_lora=True,
        image_pad_token_id=99,
        adapter_root=str(tmp_path / "sync"),
    )
    engine.sync_from_model(_Model())
    engine.generate(
        [[1, 99, 2], [7, 8]],
        max_tokens=5,
        multi_modal_data_batch=[{"image": image}, None],
    )

    assert fake_llm.last_kwargs["enable_tower_connector_lora"] is True
    assert fake_llm.last_prompts == [
        {"prompt_token_ids": [1, 99, 2], "multi_modal_data": {"image": image}},
        {"prompt_token_ids": [7, 8]},
    ]
    assert sampling_params.instances[-2].kwargs["logit_bias"] == {99: -100.0}
    assert "logit_bias" not in sampling_params.instances[-1].kwargs
    with pytest.raises(ValueError, match="multimodal data count"):
        engine.generate([[1], [2]], max_tokens=1, multi_modal_data_batch=[None])


def test_opd_vllm_structured_outputs_reaches_sampling_params(monkeypatch, tmp_path):
    """[train] structured_outputs must constrain every OPD student rollout: the parsed spec is
    rebuilt into a StructuredOutputsParams and handed to SamplingParams alongside the stop knobs."""
    from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine

    _SamplingParams, _SOParams = _install_fake_vllm(monkeypatch)

    class _Model:
        def save_pretrained(self, path):
            pass

    spec = {"json": {"type": "object"}, "disable_any_whitespace": True}
    engine = OpdVllmRolloutEngine(
        model_source="base",
        max_model_len=2048,
        temperature=0.7,
        top_p=0.9,
        stop_sequences=("</answer>",),
        eos_token_ids=(2, 73, 151645),
        structured_outputs=spec,
        adapter_root=str(tmp_path / "sync"),
    )
    engine.sync_from_model(_Model())
    engine.generate([[1, 2]], max_tokens=5)

    so = _SamplingParams.last_kwargs["structured_outputs"]
    assert isinstance(so, _SOParams)
    assert so.kwargs == spec
    assert _SamplingParams.last_kwargs["stop"] == ["</answer>"]  # coexists with the stop knobs
    assert _SamplingParams.last_kwargs["stop_token_ids"] == [2, 73, 151645]


def test_opd_vllm_structured_outputs_never_silently_dropped(monkeypatch, tmp_path):
    """A configured constraint the sampler can't apply must FAIL the run — the include_stop retry
    must not swallow it and train on unconstrained text the reward believes is schema-bound."""
    from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine

    _SamplingParams, _ = _install_fake_vllm(monkeypatch)

    def _reject_structured(self, **kwargs):
        _SamplingParams.last_kwargs = dict(kwargs)
        if "structured_outputs" in kwargs:
            raise TypeError("unexpected keyword argument 'structured_outputs'")

    monkeypatch.setattr(_SamplingParams, "__init__", _reject_structured)

    class _Model:
        def save_pretrained(self, path):
            pass

    engine = OpdVllmRolloutEngine(
        model_source="base",
        max_model_len=2048,
        temperature=0.7,
        top_p=0.9,
        stop_sequences=("</answer>",),
        structured_outputs={"json_object": True},
        adapter_root=str(tmp_path / "sync"),
    )
    engine.sync_from_model(_Model())
    with pytest.raises(TypeError, match="structured_outputs"):
        engine.generate([[1, 2]], max_tokens=5)


def test_opd_vllm_reasoning_parser_reaches_engine_only_when_set(monkeypatch, tmp_path):
    """Under thinking + a constraint the caller passes reasoning_parser so vLLM's structured-output
    gate holds the guided grammar until </think>. It must reach EngineArgs via LLM(**kwargs) when set,
    and stay ABSENT when unset (the `if self.reasoning_parser` guard) — a stray or missing key would
    silently revert the student to token-0 constraining, the exact bug this fixes."""
    from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine

    _install_fake_vllm(monkeypatch)
    fake_llm = sys.modules["vllm"].LLM

    # set -> forwarded to LLM(**kwargs) -> EngineArgs.reasoning_parser
    OpdVllmRolloutEngine(
        model_source="base",
        max_model_len=2048,
        temperature=0.7,
        top_p=0.9,
        structured_outputs={"json_object": True},
        reasoning_parser="deepseek_r1",
        adapter_root=str(tmp_path / "on"),
    )
    assert fake_llm.last_kwargs["reasoning_parser"] == "deepseek_r1"

    # unset (default None) -> NOT forwarded, so vLLM's default decoding is unchanged
    OpdVllmRolloutEngine(
        model_source="base",
        max_model_len=2048,
        temperature=0.7,
        top_p=0.9,
        structured_outputs={"json_object": True},
        adapter_root=str(tmp_path / "off"),
    )
    assert "reasoning_parser" not in fake_llm.last_kwargs


def _lp(value):
    """A stand-in for vLLM's Logprob (only its .logprob float matters here)."""
    return SimpleNamespace(logprob=value)


def test_forced_from_logprobs_counts_finite_entries_not_dict_length():
    """The critical case: vLLM's top-k dict is fixed-size, so a forced position is a length-2 dict
    with one finite logprob and one -inf pad. Detection must count finite entries, not dict length."""
    from flash.engine.worker.opd_vllm import _forced_from_logprobs

    neg_inf = float("-inf")
    lps = [
        {1: _lp(0.0)},  # backend filtered the -inf -> single entry, forced
        {2: _lp(0.0), 7: _lp(neg_inf)},  # top-2 padded with -inf -> still one legal token, forced
        {3: _lp(-0.2), 8: _lp(-2.0)},  # two genuinely-legal tokens -> free
    ]
    assert _forced_from_logprobs(lps, 3) == (True, True, False)


def test_forced_from_logprobs_none_is_empty_short_masks_visible_prefix():
    from flash.engine.worker.opd_vllm import _forced_from_logprobs

    assert _forced_from_logprobs(None, 3) == ()  # unconstrained: no logprobs -> no mask
    # Fewer logprob rows than tokens (anomaly): mask the visible prefix, leave the tail unmasked
    # rather than abandoning the whole sample's mask.
    assert _forced_from_logprobs([{1: _lp(0.0)}], 3) == (True, False, False)


def test_forced_from_logprobs_empty_row_is_free_not_forced():
    """A logprob row with zero finite entries (empty dict, or every slot -inf) is a wiring anomaly,
    not a one-legal-token position -- it must read as FREE, else a genuine free choice's teacher
    signal is silently dropped from the loss."""
    from flash.engine.worker.opd_vllm import _forced_from_logprobs

    neg_inf = float("-inf")
    # row 0: empty -> 0 finite -> free; row 1: one finite -> forced; row 2: all -inf -> free.
    lps = [{}, {5: _lp(0.0)}, {6: _lp(neg_inf), 7: _lp(neg_inf)}]
    assert _forced_from_logprobs(lps, 3) == (False, True, False)


def test_opd_vllm_constrained_generate_uses_fresh_params_per_request(monkeypatch, tmp_path):
    """vLLM stamps per-request backend state onto a StructuredOutputsParams, so a constrained OPD
    batch must hand llm.generate a FRESH SamplingParams per prompt (each with its own
    StructuredOutputsParams), not one shared instance reused across the whole batch."""
    from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine

    seen = {}

    class _StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class _SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class _LoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_int_id = lora_int_id

    class _FakeLLM:
        def __init__(self, **kwargs):
            self.llm_engine = SimpleNamespace(remove_lora=lambda _id: None)

        def generate(self, prompts, *, sampling_params, lora_request, use_tqdm):
            seen["sampling_params"] = sampling_params
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=[3], text="x", finish_reason="stop", stop_reason=None
                        )
                    ]
                )
                for _ in prompts
            ]

    vllm = types.ModuleType("vllm")
    vllm.LLM = _FakeLLM
    vllm.SamplingParams = _SamplingParams
    lora_pkg = types.ModuleType("vllm.lora")
    req_mod = types.ModuleType("vllm.lora.request")
    req_mod.LoRARequest = _LoRARequest
    sp_mod = types.ModuleType("vllm.sampling_params")
    sp_mod.StructuredOutputsParams = _StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_pkg)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", req_mod)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp_mod)

    class _Model:
        def save_pretrained(self, path):
            pass

    engine = OpdVllmRolloutEngine(
        model_source="base",
        max_model_len=2048,
        temperature=0.7,
        top_p=0.9,
        structured_outputs={"json_object": True},
        adapter_root=str(tmp_path / "sync"),
    )
    engine.sync_from_model(_Model())
    out = engine.generate([[1, 2], [3, 4], [5, 6]], max_tokens=5)

    sp = seen["sampling_params"]
    # one fresh SamplingParams per prompt (a list), NOT a single shared instance...
    assert isinstance(sp, list)
    assert len(sp) == 3
    # ...each carrying its own distinct StructuredOutputsParams so vLLM can stamp per-request state.
    embedded = [p.kwargs["structured_outputs"] for p in sp]
    assert all(isinstance(e, _StructuredOutputsParams) for e in embedded)
    assert len({id(e) for e in embedded}) == 3
    assert len(out) == 3


def test_normalize_output_marks_grammar_forced_positions():
    """_normalize_output surfaces the per-token ``forced`` mask so the OPD loss can drop spans the
    student had no choice over. Positions 0 and 2 are grammar-forced (one finite logprob, the top-2
    slot padded with -inf); position 1 had two legal tokens."""
    from flash.engine.worker.opd_vllm import _normalize_output

    neg_inf = float("-inf")
    comp = SimpleNamespace(
        token_ids=[3, 4, 5],
        text="ok",
        finish_reason="stop",
        stop_reason=None,
        logprobs=[
            {3: _lp(0.0), 99: _lp(neg_inf)},
            {4: _lp(-0.3), 9: _lp(-1.4)},
            {5: _lp(0.0), 88: _lp(neg_inf)},
        ],
    )
    out = _normalize_output(SimpleNamespace(outputs=[comp]))
    assert out.forced == (True, False, True)


def test_normalize_output_without_logprobs_has_no_forced_mask():
    """Unconstrained rollouts request no logprobs; ``forced`` stays empty (a no-op mask downstream)."""
    from flash.engine.worker.opd_vllm import _normalize_output

    comp = SimpleNamespace(token_ids=[3, 4], text="ok", finish_reason="stop", stop_reason=None)
    out = _normalize_output(SimpleNamespace(outputs=[comp]))
    assert out.forced == ()
