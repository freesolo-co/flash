"""offline tests for reading adapter tensor metadata without loading model weights."""

from __future__ import annotations

import json
import struct

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


def test_bf16_accessor_reads_only_the_requested_tensor_interval(monkeypatch, tmp_path):
    import builtins

    import flash.engine.worker.model.lora as lora

    path = tmp_path / "adapter_model.safetensors"
    prefix = b"p" * 4096
    encoded = np.array([0x3F80, 0x4000], dtype="<u2").tobytes()
    path.write_bytes(prefix + encoded + b"s" * 4096)
    descriptor = lora._SafetensorsTensorDescriptor(
        dtype="BF16",
        shape=(2,),
        data_start=0,
        start=len(prefix),
        end=len(prefix) + len(encoded),
    )

    class UnusedBackend:
        def get_tensor(self, key):
            pytest.fail(f"NumPy backend materialized BF16 tensor {key}")

    reads = []
    real_open = builtins.open

    class TrackingFile:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def seek(self, offset, whence=0):
            return self._handle.seek(offset, whence)

        def read(self, size=-1):
            reads.append((self._handle.tell(), size))
            return self._handle.read(size)

    def tracking_open(candidate, *args, **kwargs):
        handle = real_open(candidate, *args, **kwargs)
        return TrackingFile(handle) if candidate == str(path) else handle

    monkeypatch.setattr(builtins, "open", tracking_open)
    accessor = lora._SafetensorsNumpyAccessor(str(path), UnusedBackend(), {"tensor": descriptor})

    np.testing.assert_array_equal(accessor.get_tensor("tensor"), [1.0, 2.0])
    assert reads == [(len(prefix), len(encoded))]


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


def test_bin_only_adapter_has_no_selected_representation_and_is_not_loadable(tmp_path):
    from flash.adapters.artifacts import loadable_adapter_weight_files
    from flash.engine.worker.model.adapter import _warmstart_adapter_is_loadable
    from flash.engine.worker.model.lora import _read_adapter_tensor_metadata

    adir = tmp_path / "adapter"
    adir.mkdir()
    (adir / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (adir / "adapter_model.bin").write_bytes(b"unsupported")

    assert loadable_adapter_weight_files(path.name for path in adir.iterdir()) == []
    assert _read_adapter_tensor_metadata(str(adir)) is None
    assert not _warmstart_adapter_is_loadable(str(adir))
