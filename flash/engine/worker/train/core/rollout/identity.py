"""Content identities for the tokenizer/template and immutable in-memory snapshots."""

import json
import re
from pathlib import Path

from .contract import PolicyIdentity, digest


def model_identity(model_config):
    options = model_config.runtime_options
    run_id = options.get("rollout_run_id", "")
    # the model path is resolved to a pinned HF snapshot before launching the child.
    path = Path(model_config.path)
    revision = path.name if path.parent.name == "snapshots" else ""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("rollout requires a pinned model snapshot path")
    tokenizer = model_config.tokenizer
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("rollout tokenizer must expose its serialized backend")
    backend_config = json.loads(backend.to_str())
    # HF tokenization calls mutate these per-call settings; they do not change
    # the vocabulary, normalizer, pre-tokenizer or decoder identity.
    backend_config.pop("padding", None)
    backend_config.pop("truncation", None)
    tokenizer_digest = digest([backend_config, tokenizer.special_tokens_map])
    processor = getattr(model_config, "processor", None)
    template_digest = digest(
        [
            tokenizer.chat_template,
            getattr(processor, "chat_template", None),
            dict(options.get("rollout_template_kwargs", {})),
        ]
    )
    # version is supplied only at the successful weight-sync boundary, not inferred
    # from the optimizer step on a response.
    return run_id, revision, tokenizer_digest, template_digest


def policy_identity(identity, version):
    run_id, revision, tokenizer_digest, template_digest = identity
    return PolicyIdentity(
        run_id,
        revision,
        tokenizer_digest,
        template_digest,
        version,
        # one immutable snapshot per version within this fresh process lineage;
        # a restart/resume gets a new run_id even when it restores the same step.
        f"{run_id}/policy/{version}",
    )
