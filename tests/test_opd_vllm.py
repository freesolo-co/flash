from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import pytest


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
    assert _FakeLLM.last_kwargs["enable_lora"] is True
    assert _FakeLLM.last_kwargs["max_lora_rank"] == 16
    assert _FakeLLM.last_kwargs["gpu_memory_utilization"] == 0.42
    assert _FakeLLM.last_kwargs["kv_cache_dtype"] == "fp8"
    assert _FakeLLM.last_kwargs["max_num_batched_tokens"] == 8192
    assert _FakeLLM.last_kwargs["attention_backend"] == "TRITON_ATTN"
    assert _FakeLLM.last_kwargs["mm_encoder_attn_backend"] == "TORCH_SDPA"
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

        def __init__(self, **_kwargs):
            pass

        def generate(self, prompts, *, sampling_params, lora_request, use_tqdm):
            _FakeLLM.batch_sizes.append(len(prompts))
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

    out = engine.generate([[1], [2], [3], [4], [5]], max_tokens=5)

    assert _FakeLLM.batch_sizes == [2, 2, 1]
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
    truncated = _gen_from_vllm_output(
        OpdVllmOutput([1], "ok", finish_reason="length"), _Tok(), knobs
    )
    assert truncated.truncated is True

    stopped = _gen_from_vllm_output(
        OpdVllmOutput([1, 2], "ok</answer>", finish_reason="stop"), _Tok(), knobs
    )
    assert stopped.truncated is False
    assert stopped.skip is False
    assert stopped.completion_ids == [1]
    assert stopped.completion_text == "ok"

    length_stopped = _gen_from_vllm_output(
        OpdVllmOutput([1, 2], "ok</answer>", finish_reason="length"), _Tok(), knobs
    )
    assert length_stopped.truncated is False
    assert length_stopped.skip is False
    assert length_stopped.completion_ids == [1]
    assert length_stopped.completion_text == "ok"

    length_eos = _gen_from_vllm_output(
        OpdVllmOutput([1, 99], "ok", finish_reason="length"), _Tok(), OpdKnobs()
    )
    assert length_eos.truncated is False
    assert length_eos.skip is False
    assert length_eos.completion_ids == [1, 99]
    assert length_eos.completion_text == "ok"


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

    assert out["enforce_eager"] is False
    assert out["compilation_config"] == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
