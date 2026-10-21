"""tokenizer loading shared by control-plane profiling and training workers."""

from __future__ import annotations

import threading

from flash.engine.support.huggingface import model_revision_kwargs

_TRANSFORMERS_IMPORT_LOCK = threading.Lock()


def load_tokenizer(model_id: str, revision: str = "", *, trust_remote_code: bool = True):
    """load a tokenizer without importing worker runtime state."""
    # transformers resolves lazy exports outside python's module import lock. fastapi can prepare
    # multiple runs in worker threads, so serialize only the first symbol lookup rather than
    # serializing tokenizer downloads or retrying arbitrary import failures.
    with _TRANSFORMERS_IMPORT_LOCK:
        from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        **model_revision_kwargs(revision),
    )


def load_control_plane_tokenizer(model_id: str, revision: str = ""):
    """load a tokenizer without executing model-repository python on the control plane."""
    return load_tokenizer(model_id, revision, trust_remote_code=False)
