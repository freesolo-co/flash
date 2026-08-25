"""Pure CPU branch coverage for GatedDeltaNet packing capability probes."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import pytest

import flash.engine.worker.model.packing as packing


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


def test_control_plane_packing_uses_the_fixed_worker_image_contract_without_gpu_modules(
    monkeypatch,
) -> None:
    """gdn estimates preserve the worker batch without importing model code or gpu modules."""
    monkeypatch.setattr(
        packing,
        "probe_is_pure_attention",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected config probe")),
    )
    monkeypatch.setattr(
        packing,
        "probe_is_gdn_hybrid",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected config probe")),
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"unexpected plane import: {name}")),
    )

    assert packing.worker_image_packing_support("Qwen/Qwen3.5-9B") == ("gdn-hybrid", True)


def test_the_packing_contract_gate_never_consults_the_device() -> None:
    """the control-plane gate must decide without a gpu.

    ``prepare_sft_workload`` runs for the static estimate and again on the gpu worker. a
    device-dependent gate would answer false on the control plane and true in training, producing
    different ``packing_mode`` or ``examples_per_update`` values. the availability helpers below
    are banned specifically because both open with ``is_torch_cuda_available()``.
    """
    import ast
    import inspect

    fn = ast.parse(inspect.getsource(packing.gdn_packing_contract_available)).body[0]
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]  # the docstring names these on purpose; only the CODE is the contract
    code = ast.unparse(fn)

    for banned in (
        "cuda",
        "is_flash_linear_attention_available",
        "is_causal_conv1d_available",
        "get_device_capability",
    ):
        assert banned not in code, (
            f"gdn_packing_contract_available consults {banned!r}, so the control plane and gpu "
            "worker can disagree on packing_mode"
        )


def test_the_strict_module_resolver_refuses_to_guess_the_arch(monkeypatch) -> None:
    """A packed run must abort on an unreadable config, not fall back to the dense module.

    `gdn_model_type` returns "qwen3_5" for a dense model AND for a config it could not read. Same
    string, different meaning. `Qwen/Qwen3.6-35B-A3B` is `qwen3_5_moe`, so on that model the
    fallback names a module the model does not use: the child clears it, the shim patches it, the
    real MoE layers stay unpatched, and packed examples bleed state while the log prints
    "gdn packed-boundary resets active".
    """
    import transformers

    from flash.engine.worker.train.entry.backend_common import strict_gdn_probe_module

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: (_ for _ in ()).throw(OSError("hub read failed")),
    )
    with pytest.raises(RuntimeError, match="refusing to guess"):
        strict_gdn_probe_module("Qwen/Qwen3.6-35B-A3B")

    # a config that loads but declares no model_type is the same ambiguity, not a usable answer.
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(model_type=None),
    )
    with pytest.raises(RuntimeError, match="no model_type"):
        strict_gdn_probe_module("Qwen/Qwen3.6-35B-A3B")

    # and it must return the arch the config actually declares, not the dense default.
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(model_type="qwen3_5_moe"),
    )
    assert (
        strict_gdn_probe_module("Qwen/Qwen3.6-35B-A3B")
        == "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    )
