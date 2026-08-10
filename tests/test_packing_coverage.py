"""pure cpu coverage for the live gated-deltanet architecture probe."""

from __future__ import annotations

import flash.engine.worker.model.packing as packing


def test_gdn_hybrid_probe_failure_is_a_safe_false(monkeypatch, capsys) -> None:
    """configuration probe failures must disable gdn packing and leave a useful diagnostic."""
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    assert packing.model_is_gdn_hybrid("missing/model") is False
    assert "gdn-hybrid probe failed" in capsys.readouterr().out
