"""tokenizer loading shared by control-plane profiling and training workers."""

from __future__ import annotations

from flash.engine.huggingface import model_revision_kwargs


def load_tokenizer(model_id: str, revision: str = "", *, trust_remote_code: bool = True):
    """load a tokenizer without importing worker runtime state."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        **model_revision_kwargs(revision),
    )


def load_control_plane_tokenizer(model_id: str, revision: str = ""):
    """load a tokenizer without executing model-repository python on the control plane."""
    return load_tokenizer(model_id, revision, trust_remote_code=False)
