"""Offline tests for the warm-start (init_from_adapter) VL SFT-adapter key remap.

Bug: SFT trains the FULL multimodal model, so the saved adapter keys are
``base_model.model.model.language_model.layers.*``. Warm-started GRPO loads the base via
``AutoModelForCausalLM`` (text-only tree, ``base_model.model.model.layers.*``), so
``PeftModel.from_pretrained`` can't match the SFT keys: peft only WARNS and silently keeps a fresh
zero-init LoRA, throwing the SFT away. The fix strips the ``.language_model.`` infix from the
downloaded adapter (on disk) BEFORE loading, but ONLY for VL checkpoints.

These tests exercise the pure key remap, the in-place safetensors/.bin rewrite, the VL gating, and
the loud post-load assertion — all without a GPU / transformers / peft / vllm.
"""

from __future__ import annotations

import json
import os
import struct
from collections import namedtuple

import pytest

from flash.engine.worker.lora import (
    assert_adapter_delta_nonzero,
    assert_adapter_load_clean,
    assert_lora_applied,
    remap_adapter_keys,
    remap_vl_adapter_dir,
    strip_language_model_infix,
)

# Realistic Qwen3.5 SFT-adapter keys (FULL multimodal model -> LM under language_model.).
VL_KEYS = [
    "base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_A.default.weight",
    "base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_B.default.weight",
    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
    "base_model.model.model.language_model.layers.31.mlp.gate_proj.lora_B.default.weight",
]
# The corresponding AutoModelForCausalLM-named LoRA params the GRPO trainer expects (no infix).
CAUSAL_LM_KEYS = [
    "base_model.model.model.layers.0.linear_attn.out_proj.lora_A.default.weight",
    "base_model.model.model.layers.0.linear_attn.out_proj.lora_B.default.weight",
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
    "base_model.model.model.layers.31.mlp.gate_proj.lora_B.default.weight",
]


# ---------------------------------------------------------------------------
# Pure key remap
# ---------------------------------------------------------------------------
def test_strip_infix_matches_causal_lm_param_set():
    # The remapped keys must EXACTLY equal the AutoModelForCausalLM-named LoRA param set.
    assert [strip_language_model_infix(k) for k in VL_KEYS] == CAUSAL_LM_KEYS


def test_strip_infix_leaves_non_language_model_keys_untouched():
    text_key = "base_model.model.model.layers.5.self_attn.k_proj.lora_A.default.weight"
    assert strip_language_model_infix(text_key) == text_key
    # And only the LM-vs-VL boundary infix is removed (a hypothetical second token is left alone).
    assert strip_language_model_infix("a.language_model.b.language_model.c") == "a.b.language_model.c"


def test_remap_adapter_keys_returns_only_changed():
    mapping = remap_adapter_keys(
        [*VL_KEYS, "base_model.model.model.layers.1.x.lora_A.default.weight"]
    )
    # the already-correct key is absent
    assert mapping == dict(zip(VL_KEYS, CAUSAL_LM_KEYS, strict=True))


# ---------------------------------------------------------------------------
# On-disk safetensors rewrite (pure stdlib, no torch/safetensors)
# ---------------------------------------------------------------------------
def _write_safetensors(path: str, names: list[str]) -> None:
    """Write a minimal valid safetensors file: 8-byte LE header len + JSON header + data.

    Each tensor is 4 bytes (one float32 scalar); data_offsets are relative to the data section.
    """
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    data = b""
    for i, n in enumerate(names):
        header[n] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        data += struct.pack("<f", float(i))
        offset += 4
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(data)


def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def _read_safetensors_tensor_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        f.read(n)
        return f.read()


def test_safetensors_remap_for_vl_strips_infix_and_preserves_data(monkeypatch, tmp_path):
    import flash.engine.worker.lora as lora

    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: True)
    adir = tmp_path / "adapter"
    adir.mkdir()
    st = str(adir / "adapter_model.safetensors")
    _write_safetensors(st, VL_KEYS)
    before_data = _read_safetensors_tensor_bytes(st)

    n = remap_vl_adapter_dir(str(adir), "Qwen/Qwen3.5-4B")
    assert n == len(VL_KEYS)

    hdr = _read_safetensors_header(st)
    keys = [k for k in hdr if k != "__metadata__"]
    assert sorted(keys) == sorted(CAUSAL_LM_KEYS)
    assert "__metadata__" in hdr  # metadata entry preserved
    # data_offsets unchanged for every (renamed) tensor, and the data section is byte-identical.
    for old, new in zip(VL_KEYS, CAUSAL_LM_KEYS, strict=True):
        assert hdr[new]["data_offsets"] == [VL_KEYS.index(old) * 4, VL_KEYS.index(old) * 4 + 4]
    assert _read_safetensors_tensor_bytes(st) == before_data
    # The rewrite streams via a temp file + atomic os.replace; it must not leave one behind.
    assert not os.path.exists(st + ".remap.tmp")
    assert [p for p in os.listdir(adir) if p.endswith(".tmp")] == []


def test_safetensors_remap_noop_for_non_vl(monkeypatch, tmp_path):
    import flash.engine.worker.lora as lora

    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: False)
    adir = tmp_path / "adapter"
    adir.mkdir()
    st = str(adir / "adapter_model.safetensors")
    _write_safetensors(st, VL_KEYS)
    with open(st, "rb") as f:
        before = f.read()

    n = remap_vl_adapter_dir(str(adir), "Qwen/Qwen3-4B-Instruct-2507")
    assert n == 0
    with open(st, "rb") as f:
        assert f.read() == before  # file untouched for a text model


def test_safetensors_remap_idempotent(monkeypatch, tmp_path):
    import flash.engine.worker.lora as lora

    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: True)
    adir = tmp_path / "adapter"
    adir.mkdir()
    st = str(adir / "adapter_model.safetensors")
    _write_safetensors(st, VL_KEYS)

    assert remap_vl_adapter_dir(str(adir), "Qwen/Qwen3.5-4B") == len(VL_KEYS)
    # Second pass: keys already stripped -> nothing to do, file remains valid.
    assert remap_vl_adapter_dir(str(adir), "Qwen/Qwen3.5-4B") == 0
    keys = [k for k in _read_safetensors_header(st) if k != "__metadata__"]
    assert sorted(keys) == sorted(CAUSAL_LM_KEYS)


def test_remap_missing_adapter_file_returns_zero(monkeypatch, tmp_path):
    import flash.engine.worker.lora as lora

    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: True)
    adir = tmp_path / "adapter"
    adir.mkdir()
    assert remap_vl_adapter_dir(str(adir), "Qwen/Qwen3.5-4B") == 0


def test_bin_remap_for_vl(monkeypatch, tmp_path):
    """The legacy ``.bin`` path (when no safetensors exists) needs torch; skip if absent."""
    torch = pytest.importorskip("torch")
    import flash.engine.worker.lora as lora

    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: True)
    adir = tmp_path / "adapter"
    adir.mkdir()
    bin_path = str(adir / "adapter_model.bin")
    torch.save({k: torch.zeros(1) for k in VL_KEYS}, bin_path)

    assert remap_vl_adapter_dir(str(adir), "Qwen/Qwen3.5-4B") == len(VL_KEYS)
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    assert sorted(sd.keys()) == sorted(CAUSAL_LM_KEYS)


# ---------------------------------------------------------------------------
# Loud post-load assertion
# ---------------------------------------------------------------------------
class _FakeModule:
    def __init__(self, names):
        self._names = names

    def named_modules(self):
        return [(n, object()) for n in self._names]


def test_assert_lora_applied_counts_modules():
    model = _FakeModule(
        [
            "",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default",
            "base_model.model.model.layers.0.self_attn.q_proj",  # parent, not counted
        ]
    )
    assert assert_lora_applied(model, "Qwen/Qwen3.5-4B") == 2


def test_assert_lora_applied_raises_when_zero():
    # No lora_A/lora_B submodules -> the silent-drop failure this fix prevents.
    model = _FakeModule(["", "base_model.model.model.layers.0.self_attn.q_proj"])
    with pytest.raises(RuntimeError, match="ZERO LoRA modules"):
        assert_lora_applied(model, "Qwen/Qwen3.5-4B")


def test_vl_keys_are_what_sft_actually_saves():
    # Guard: every VL fixture key must contain the infix the fix strips (so the fixtures stay
    # representative of the real SFT-saved adapter described in the bug report).
    assert all(".language_model." in k for k in VL_KEYS)
    assert all(".language_model." not in k for k in CAUSAL_LM_KEYS)
    # And the file path the rewriter targets exists in our fixture writer.
    assert os.path.basename("x/adapter_model.safetensors") == "adapter_model.safetensors"


# ---------------------------------------------------------------------------
# Fail-closed key-equality check (matched == saved) on the captured load result.
#
# peft injects the LoRA modules from target_modules BEFORE loading weights, so counting injected
# modules (assert_lora_applied) can't see a silent weight-discard: it passes even when ZERO saved
# weights matched. assert_adapter_load_clean inspects the load_result peft returns from load_adapter
# (load_state_dict(strict=False) -> _IncompatibleKeys) and fails closed when matched != saved. These
# run without torch/peft using a fake load result shaped like peft's namedtuple.
# ---------------------------------------------------------------------------
_LoadResult = namedtuple("_LoadResult", ["missing_keys", "unexpected_keys"])


def test_assert_adapter_load_clean_passes_on_full_match():
    # A correct load: every injected module got a weight, every saved key matched a module.
    assert assert_adapter_load_clean(_LoadResult([], []), "Qwen/Qwen3.5-4B") is None


def test_assert_adapter_load_clean_raises_on_vl_infix_mismatch():
    # The #67 silent-discard: the saved VL keys (infixed) match no injected text-only module ->
    # unexpected; the injected text-only modules get no saved weight -> missing. This is the exact
    # case the module-count check waves through and this check must catch.
    lr = _LoadResult(missing_keys=list(CAUSAL_LM_KEYS), unexpected_keys=list(VL_KEYS))
    with pytest.raises(RuntimeError, match="did NOT load cleanly"):
        assert_adapter_load_clean(lr, "Qwen/Qwen3.5-4B")


def test_assert_adapter_load_clean_raises_on_missing_only():
    lr = _LoadResult(missing_keys=list(CAUSAL_LM_KEYS), unexpected_keys=[])
    with pytest.raises(RuntimeError, match="did NOT load cleanly"):
        assert_adapter_load_clean(lr, "Qwen/Qwen3.5-4B")


def test_assert_adapter_load_clean_raises_on_unexpected_only():
    lr = _LoadResult(missing_keys=[], unexpected_keys=list(VL_KEYS))
    with pytest.raises(RuntimeError, match="did NOT load cleanly"):
        assert_adapter_load_clean(lr, "Qwen/Qwen3.5-4B")


# ---------------------------------------------------------------------------
# Non-zero-delta backstop: a silently-discarded adapter keeps zero-init lora_B, so its delta is
# identically zero. This is API-independent (no peft load_result needed). Needs real tensors for the
# zero check, so it is torch-gated like the .bin path above.
# ---------------------------------------------------------------------------
class _WeightModule:
    def __init__(self, weight):
        self.weight = weight


class _FakeModelWithWeights:
    def __init__(self, named_weights):
        self._items = [(n, _WeightModule(w)) for n, w in named_weights]

    def named_modules(self):
        return self._items


def test_assert_adapter_delta_nonzero_passes_when_a_b_is_trained():
    torch = pytest.importorskip("torch")
    model = _FakeModelWithWeights(
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.default", torch.zeros(2, 2)),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.default", torch.zeros(2, 2)),
            # a trained (non-zero) lora_B -> the adapter delta is real
            (
                "base_model.model.model.layers.1.self_attn.q_proj.lora_B.default",
                torch.tensor([[0.0, 0.7], [0.0, 0.0]]),
            ),
        ]
    )
    # only lora_B modules are inspected; exactly one is non-zero
    assert assert_adapter_delta_nonzero(model, "Qwen/Qwen3.5-4B") == 1


def test_assert_adapter_delta_nonzero_raises_when_all_b_zero():
    torch = pytest.importorskip("torch")
    # Every lora_B is zero -> identity adapter (the silent-discard signature) -> must raise.
    model = _FakeModelWithWeights(
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.default", torch.zeros(2, 2)),
            ("base_model.model.model.layers.0.self_attn.k_proj.lora_B.default", torch.zeros(3, 3)),
        ]
    )
    with pytest.raises(RuntimeError, match="ALL-ZERO lora_B"):
        assert_adapter_delta_nonzero(model, "Qwen/Qwen3.5-4B")
