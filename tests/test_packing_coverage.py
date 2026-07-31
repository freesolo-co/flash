"""Pure CPU branch coverage for GatedDeltaNet packing capability probes."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from types import SimpleNamespace

import pytest

import flash.engine.worker.packing as packing


def test_gdn_hybrid_probe_failure_is_a_safe_false(monkeypatch, capsys) -> None:
    """Configuration probe failures must disable GDN packing and leave a useful diagnostic."""
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    assert packing.model_is_gdn_hybrid("missing/model") is False
    assert "gdn-hybrid probe failed" in capsys.readouterr().out


def test_gdn_forward_probe_resolves_model_type_from_config(monkeypatch) -> None:
    """The reset-argument probe must import the architecture selected by the model config."""
    import transformers

    class DemoGatedDeltaNet:
        def forward(self):
            return None

    seen = []
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="demo"),
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: seen.append(name) or SimpleNamespace(DemoGatedDeltaNet=DemoGatedDeltaNet),
    )
    monkeypatch.setattr(inspect, "getsource", lambda function: "cu_seq_lens_q seq_idx")

    assert packing._gdn_forward_threads_reset_kwargs("owner/model") is True
    assert seen == ["transformers.models.demo.modeling_demo"]


def test_gdn_forward_probe_rejects_module_without_gated_delta_net(monkeypatch) -> None:
    """An architecture module without a GatedDeltaNet class must not enable boundary resets."""
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace(OtherLayer=object))

    assert packing._gdn_forward_threads_reset_kwargs(None) is False


@pytest.mark.parametrize("source", ["cu_seq_lens_q", "seq_idx", "neither"])
def test_gdn_forward_probe_requires_both_reset_arguments(monkeypatch, source) -> None:
    """Packing must remain disabled unless the forward implementation threads both reset arguments."""

    class DemoGatedDeltaNet:
        def forward(self):
            return None

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(DemoGatedDeltaNet=DemoGatedDeltaNet),
    )
    monkeypatch.setattr(inspect, "getsource", lambda function: source)

    assert packing._gdn_forward_threads_reset_kwargs(None) is False


def test_gdn_forward_probe_swallows_import_and_inspection_failures(monkeypatch) -> None:
    """Import or source-inspection failures must degrade to unpacked execution instead of escaping."""
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("broken architecture")),
    )

    assert packing._gdn_forward_threads_reset_kwargs(None) is False


def _enable_kernel_probes(monkeypatch) -> None:
    import transformers.utils.import_utils as import_utils

    monkeypatch.setattr(import_utils, "is_flash_linear_attention_available", lambda: True)
    monkeypatch.setattr(import_utils, "is_causal_conv1d_available", lambda: True)


def test_gdn_packing_rejects_a_broken_causal_conv_import(monkeypatch) -> None:
    """A package that is discoverable but ABI-broken at import time must disable packing."""
    _enable_kernel_probes(monkeypatch)
    real_import = importlib.import_module

    def fake_import(name):
        if name == "causal_conv1d":
            raise ImportError("bad abi")
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert packing.gdn_packing_available() is False


def test_gdn_packing_rejects_a_forward_signature_without_resets(monkeypatch) -> None:
    """Installed kernels are insufficient when the model forward cannot reset sequence state."""
    _enable_kernel_probes(monkeypatch)
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace())
    monkeypatch.setattr(packing, "_gdn_forward_threads_reset_kwargs", lambda *a, **k: False)

    assert packing.gdn_packing_available() is False


def test_gdn_packing_succeeds_on_cpu_when_all_non_gpu_probes_pass(monkeypatch) -> None:
    """CPU-only control-plane probing must succeed without attempting a CUDA kernel smoke."""
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _enable_kernel_probes(monkeypatch)
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace())
    monkeypatch.setattr(packing, "_gdn_forward_threads_reset_kwargs", lambda *a, **k: True)

    assert packing.gdn_packing_available("owner/model") is True


def test_gdn_packing_catches_unexpected_probe_failures(monkeypatch) -> None:
    """Unexpected reset-probe errors must fail safely rather than aborting worker setup."""
    _enable_kernel_probes(monkeypatch)
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace())
    monkeypatch.setattr(
        packing,
        "_gdn_forward_threads_reset_kwargs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )

    assert packing.gdn_packing_available() is False


def test_pack_token_ids_rejects_nonpositive_capacity() -> None:
    """The pure bin packer must reject invalid capacities before inspecting token rows."""
    with pytest.raises(ValueError, match="max_length must be positive"):
        packing.pack_token_ids([[1, 2]], 0)
