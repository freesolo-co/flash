"""Keep model dispatch identities stable across hardware changes."""

from __future__ import annotations

import hashlib


def model_engine_name(base_model: str) -> str:
    """Hardware changes must update a function, never replace its dispatch name."""
    readable = "".join(ch if ch.isalnum() else "_" for ch in base_model)
    digest = hashlib.sha256(base_model.encode()).hexdigest()[:12]
    return f"LoraEngine_{readable}_{digest}"
