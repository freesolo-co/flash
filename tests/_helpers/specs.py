"""Shared ``raw_spec`` builder.

The minimal valid spec dict that ``spec_from_dict`` accepts, factored out of the
schema/thinking config tests that each carried a byte-identical local ``_raw``.

The environment id is a managed hub slug because that is the form valid on BOTH deployments:
the managed service accepts hub slugs only, while a self-hosted plane also takes the explicit
``github:`` forms. A ``github:`` default would make every caller implicitly self-hosted and
fail the moment a test parsed it through the hosted submit path.
"""

from __future__ import annotations


def raw_spec(model: str = "Qwen/Qwen3.5-9B", **kw) -> dict:
    d = {
        "model": model,
        "algorithm": "sft",
        "project": "11111111-1111-4111-8111-111111111111",
        "train": {"epochs": 1, "max_examples": 8},
        "environment": {"id": "owner/project/env"},
    }
    d.update(kw)
    return d
