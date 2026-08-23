"""Fail-fast credential checks for Modal GPU training."""

from __future__ import annotations

from flash.providers.modal.auth import load_credentials


def missing_credentials(require_hf: bool = True) -> list[str]:
    """Return missing Modal credentials; shared Hugging Face checks stay central."""
    if load_credentials():
        return []
    return [
        (
            "  - MODAL_TOKEN_ID / MODAL_TOKEN_SECRET: the operator's Modal token pair "
            "(for the modal provider)"
        )
    ]
