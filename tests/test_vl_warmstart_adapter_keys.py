"""Offline tests for the warm-start (init_from_adapter) VL adapter key handling + load guards.

VL SFT trains the FULL multimodal model, so its adapter keys carry the ``.language_model.`` infix
(``base_model.model.model.language_model.layers.*``). ``strip_language_model_infix`` removes it (used
by the recombine path); ``adapter_is_vl_warmstart`` gates the #296 merge-into-base path off the
adapter's own keys (not just the config probe); and the three load guards (``assert_adapter_load_clean``
/ ``assert_lora_applied`` / ``assert_adapter_delta_nonzero``) fail closed on a silently-discarded
adapter. All exercised without a GPU / transformers / peft / vllm.
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


def _write_torch_bin(path: str, names: list[str]) -> None:
    torch = pytest.importorskip("torch")
    torch.save({name: torch.zeros(1) for name in names}, path)


# ---------------------------------------------------------------------------
# Pure key remap (strip_language_model_infix — used by the recombine path)
# ---------------------------------------------------------------------------
def test_strip_infix_matches_causal_lm_param_set():
    # The stripped keys must EXACTLY equal the AutoModelForCausalLM-named LoRA param set.
    assert [strip_language_model_infix(k) for k in VL_KEYS] == CAUSAL_LM_KEYS


def test_strip_infix_leaves_non_language_model_keys_untouched():
    text_key = "base_model.model.model.layers.5.self_attn.k_proj.lora_A.default.weight"
    assert strip_language_model_infix(text_key) == text_key
    # And only the LM-vs-VL boundary infix is removed (a hypothetical second token is left alone).
    assert (
        strip_language_model_infix("a.language_model.b.language_model.c") == "a.b.language_model.c"
    )


# ---------------------------------------------------------------------------
# _read_adapter_tensor_keys: safetensors header parsing (drives adapter_is_vl_warmstart)
# ---------------------------------------------------------------------------
def test_read_adapter_keys_rejects_oversized_header(tmp_path):
    # A corrupt / hostile safetensors file can declare a huge header length; we must reject it from
    # the declared length vs the real file size BEFORE allocating/reading the header payload.
    import flash.engine.worker.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    st = adir / "adapter_model.safetensors"
    # 8-byte length prefix claims a 4 GiB header, but the file is only 8 bytes.
    st.write_bytes(struct.pack("<Q", 4 * 1024**3))
    with pytest.raises(ValueError, match="implausible"):
        lora._read_adapter_tensor_keys(str(adir))


def test_read_adapter_keys_rejects_non_object_header(tmp_path):
    # The safetensors header must decode to a JSON object. A header that decodes to a list/int would
    # otherwise raise a confusing TypeError later in _is_lora_key (substring search on a non-str);
    # fail with a clear ValueError here instead. (JSON object keys are always str, so only the
    # container type is checked.)
    import flash.engine.worker.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    st = adir / "adapter_model.safetensors"
    # A well-formed length prefix + valid JSON that is a LIST, not an object.
    body = json.dumps([1, 2, 3]).encode("utf-8")
    st.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(ValueError, match="not a JSON object"):
        lora._read_adapter_tensor_keys(str(adir))


def test_read_adapter_keys_corrupt_json_reraises_with_path(tmp_path):
    # A header that isn't valid JSON at all must re-raise with the FILE PATH (a bare JSONDecodeError
    # names no file), so a bad adapter download is diagnosable (#198).
    import flash.engine.worker.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    st = adir / "adapter_model.safetensors"
    body = b"this is not json{{{"
    st.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(
        ValueError, match=r"adapter_model\.safetensors: safetensors header is not valid JSON"
    ):
        lora._read_adapter_tensor_keys(str(adir))


def test_read_adapter_keys_reads_legacy_bin_state_dict(tmp_path):
    import flash.engine.worker.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    _write_torch_bin(str(adir / "adapter_model.bin"), VL_KEYS)

    assert lora._read_adapter_tensor_keys(str(adir)) == VL_KEYS


def test_read_adapter_keys_rejects_malformed_bin_state_dict(tmp_path):
    torch = pytest.importorskip("torch")
    import flash.engine.worker.lora as lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    torch.save({"metadata": "not a tensor"}, adir / "adapter_model.bin")

    with pytest.raises(ValueError, match="non-tensor entries"):
        lora._read_adapter_tensor_keys(str(adir))


def test_vl_keys_are_what_sft_actually_saves():
    # Guard: every VL fixture key must contain the infix the strip removes (so the fixtures stay
    # representative of the real SFT-saved adapter described in the bug report).
    assert all(".language_model." in k for k in VL_KEYS)
    assert all(".language_model." not in k for k in CAUSAL_LM_KEYS)
    # And the file path the rewriter targets exists in our fixture writer.
    assert os.path.basename("x/adapter_model.safetensors") == "adapter_model.safetensors"


# ---------------------------------------------------------------------------
# Loud post-load assertion: module-count
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


def test_assert_adapter_load_clean_ignores_benign_base_misses():
    # An adapter-only checkpoint loaded with strict=False can report base-model params (no 'lora_'
    # prefix) as missing even when every LoRA weight matched. Those must NOT abort a correct
    # warm-start -- only LoRA-key mismatches are fatal.
    base_misses = [
        "base_model.model.model.layers.0.self_attn.q_proj.weight",
        "base_model.model.model.embed_tokens.weight",
    ]
    assert assert_adapter_load_clean(_LoadResult(base_misses, []), "Qwen/Qwen3.5-4B") is None


# ---------------------------------------------------------------------------
# Non-zero-delta backstop: a silently-discarded adapter keeps zero-init lora_B, so its delta is
# identically zero. This is API-independent (no peft load_result needed). Needs real tensors for the
# zero check, so it is torch-gated.
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


# ---------------------------------------------------------------------------
# adapter_is_vl_warmstart: the VL merge gate keys off adapter EVIDENCE, not only the config probe
# (Copilot MsAoq / Cursor MsATq — the #286 fragility applied to the #296 merge decision).
# ---------------------------------------------------------------------------
def test_adapter_is_vl_warmstart_trusts_adapter_evidence_over_failed_probe(monkeypatch, tmp_path):
    from flash.engine.worker import lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    _write_safetensors(str(adir / "adapter_model.safetensors"), VL_KEYS)
    # the config probe FAILED (returns False) for a genuine VL adapter — must still merge as VL
    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: False)
    assert lora.adapter_is_vl_warmstart(str(adir), "some/model") is True


def test_adapter_is_vl_warmstart_trusts_bin_adapter_evidence_over_failed_probe(
    monkeypatch, tmp_path
):
    from flash.engine.worker import lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    _write_torch_bin(str(adir / "adapter_model.bin"), VL_KEYS)
    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: False)
    assert lora.adapter_is_vl_warmstart(str(adir), "some/model") is True


def test_adapter_is_vl_warmstart_falls_back_to_probe_for_text_only(monkeypatch, tmp_path):
    from flash.engine.worker import lora

    adir = tmp_path / "adapter"
    adir.mkdir()
    _write_safetensors(str(adir / "adapter_model.safetensors"), CAUSAL_LM_KEYS)  # no language_model
    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: False)
    assert lora.adapter_is_vl_warmstart(str(adir), "some/model") is False
    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: True)
    assert (
        lora.adapter_is_vl_warmstart(str(adir), "some/model") is True
    )  # probe still authoritative


def test_adapter_is_vl_warmstart_missing_file_defers_to_probe(monkeypatch, tmp_path):
    from flash.engine.worker import lora

    adir = tmp_path / "empty"
    adir.mkdir()
    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: True)
    assert lora.adapter_is_vl_warmstart(str(adir), "m") is True
    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: False)
    assert lora.adapter_is_vl_warmstart(str(adir), "m") is False
