"""offline tests for reading adapter tensor metadata without loading model weights."""

from __future__ import annotations

import json
import struct
import sys
import types

import numpy as np
import pytest
from safetensors.numpy import save

# realistic qwen3.5 sft-adapter keys with the language model under the multimodal namespace
VL_KEYS = [
    "base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_A.default.weight",
    "base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_B.default.weight",
    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
    "base_model.model.model.language_model.layers.31.mlp.gate_proj.lora_B.default.weight",
]


def _write_torch_bin(path: str, names: list[str]) -> None:
    torch = pytest.importorskip("torch")
    torch.save({name: torch.zeros(2, 3) for name in names}, path)


def _raw_safetensors(header, payload=b"") -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def test_read_adapter_keys_rejects_oversized_header(tmp_path):
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    st = adir / "adapter_model.safetensors"
    st.write_bytes(struct.pack("<Q", 4 * 1024**3))

    with pytest.raises(ValueError, match="implausible"):
        lora._read_adapter_tensor_keys(str(adir))


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_raw_safetensors([1, 2, 3]), id="non-object"),
        pytest.param(struct.pack("<Q", 16) + b"this is not json", id="corrupt-json"),
    ],
)
def test_read_adapter_keys_rejects_invalid_header_with_path(tmp_path, body):
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    st = adir / "adapter_model.safetensors"
    st.write_bytes(body)

    with pytest.raises(
        ValueError, match=r"adapter_model\.safetensors: invalid safetensors structure"
    ):
        lora._read_adapter_tensor_keys(str(adir))


def test_read_adapter_metadata_reads_valid_safetensors_shape(tmp_path):
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    (adir / "adapter_model.safetensors").write_bytes(
        save({VL_KEYS[0]: np.zeros((2, 3), dtype=np.float16)})
    )

    assert lora._read_adapter_tensor_metadata(str(adir)) == {VL_KEYS[0]: (2, 3)}


@pytest.mark.parametrize(
    ("header", "payload"),
    [
        pytest.param(
            {"x": {"dtype": "NOPE", "shape": [1], "data_offsets": [0, 2]}},
            b"\x00\x00",
            id="invalid-dtype",
        ),
        pytest.param(
            {"x": {"dtype": "F16", "shape": [2], "data_offsets": [0, 2]}},
            b"\x00\x00",
            id="shape-payload-length",
        ),
        pytest.param(
            {"x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 4]}},
            b"\x00\x00",
            id="offset-outside-payload",
        ),
        pytest.param(
            {
                "x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                "y": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
            },
            b"\x00\x00",
            id="overlap",
        ),
        pytest.param(
            {"x": {"dtype": "F16", "shape": [2, -1], "data_offsets": [0, 2]}},
            b"\x00\x00",
            id="malformed-shape",
        ),
    ],
)
def test_read_adapter_metadata_rejects_invalid_safetensors_structure(tmp_path, header, payload):
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    (adir / "adapter_model.safetensors").write_bytes(_raw_safetensors(header, payload))

    with pytest.raises(ValueError, match="invalid safetensors structure"):
        lora._read_adapter_tensor_metadata(str(adir))


def test_read_adapter_metadata_reads_bin_shapes_without_gpu_dependencies(monkeypatch, tmp_path):
    import flash.engine.worker.model.lora as lora

    class FakeTensor:
        shape = (2, 3)

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTensor
    fake_torch.load = lambda *_args, **_kwargs: {VL_KEYS[0]: FakeTensor()}
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    adir = tmp_path / "adapter"
    adir.mkdir()
    (adir / "adapter_model.bin").write_bytes(b"stub")

    assert lora._read_adapter_tensor_metadata(str(adir)) == {VL_KEYS[0]: (2, 3)}
    assert lora._read_adapter_tensor_keys(str(adir)) == [VL_KEYS[0]]


def test_read_adapter_keys_reads_bin_state_dict(tmp_path):
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    _write_torch_bin(str(adir / "adapter_model.bin"), VL_KEYS)

    assert lora._read_adapter_tensor_keys(str(adir)) == VL_KEYS
    assert lora._read_adapter_tensor_metadata(str(adir)) == dict.fromkeys(VL_KEYS, (2, 3))


def test_read_adapter_keys_rejects_malformed_bin_state_dict(tmp_path):
    torch = pytest.importorskip("torch")
    import flash.engine.worker.model.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    torch.save({"metadata": "not a tensor"}, adir / "adapter_model.bin")

    with pytest.raises(ValueError, match="non-tensor entries"):
        lora._read_adapter_tensor_keys(str(adir))
