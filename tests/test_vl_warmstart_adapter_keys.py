"""offline tests for reading adapter tensor keys without loading model weights."""

from __future__ import annotations

import json
import struct

import pytest

# realistic qwen3.5 sft-adapter keys with the language model under the multimodal namespace
VL_KEYS = [
    "base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_A.default.weight",
    "base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_B.default.weight",
    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
    "base_model.model.model.language_model.layers.31.mlp.gate_proj.lora_B.default.weight",
]


def _write_torch_bin(path: str, names: list[str]) -> None:
    torch = pytest.importorskip("torch")
    torch.save({name: torch.zeros(1) for name in names}, path)


def test_read_adapter_keys_rejects_oversized_header(tmp_path):
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    st = adir / "adapter_model.safetensors"
    st.write_bytes(struct.pack("<Q", 4 * 1024**3))

    with pytest.raises(ValueError, match="implausible"):
        lora._read_adapter_tensor_keys(str(adir))


def test_read_adapter_keys_rejects_non_object_header(tmp_path):
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    st = adir / "adapter_model.safetensors"
    body = json.dumps([1, 2, 3]).encode("utf-8")
    st.write_bytes(struct.pack("<Q", len(body)) + body)

    with pytest.raises(ValueError, match="not a JSON object"):
        lora._read_adapter_tensor_keys(str(adir))


def test_read_adapter_keys_corrupt_json_reraises_with_path(tmp_path):
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    st = adir / "adapter_model.safetensors"
    body = b"this is not json{{{"
    st.write_bytes(struct.pack("<Q", len(body)) + body)

    with pytest.raises(
        ValueError, match=r"adapter_model\.safetensors: safetensors header is not valid JSON"
    ):
        lora._read_adapter_tensor_keys(str(adir))


def test_read_adapter_keys_reads_bin_state_dict(tmp_path):
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    _write_torch_bin(str(adir / "adapter_model.bin"), VL_KEYS)

    assert lora._read_adapter_tensor_keys(str(adir)) == VL_KEYS


def test_read_adapter_keys_rejects_malformed_bin_state_dict(tmp_path):
    torch = pytest.importorskip("torch")
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    torch.save({"metadata": "not a tensor"}, adir / "adapter_model.bin")

    with pytest.raises(ValueError, match="non-tensor entries"):
        lora._read_adapter_tensor_keys(str(adir))
