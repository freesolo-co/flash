"""Shared ``raw_spec`` builder.

The minimal valid spec dict that ``spec_from_dict`` accepts, factored out of the
schema/thinking config tests that each carried a byte-identical local ``_raw``.
"""

from __future__ import annotations


def raw_spec(model: str = "Qwen/Qwen3.5-0.8B", **kw) -> dict:
    d = {
        "model": model,
        "algorithm": "sft",
        "train": {"epochs": 1, "hf_repo": "owner/runs"},
        "environment": {"id": "owner/env"},  # any published verifiers id (not loaded here)
    }
    d.update(kw)
    return d
