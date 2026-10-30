"""Resolve generation termination IDs for prefetched training models."""

import contextlib


def _generation_eos_ids(model, tok) -> frozenset:
    """Collect valid EOS IDs from the tokenizer, generation config, and nested text config.

    Models may define several termination tokens. Preserve their union so a secondary EOS is
    recognized as natural termination rather than misclassified as a truncated completion.
    """
    ids: set[int] = set()

    def _add(v):
        # bool is an int subclass but never a token id; exclude it from both the scalar and list forms.
        if isinstance(v, bool):
            return
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple, set, frozenset)):
            ids.update(int(x) for x in v if isinstance(x, int) and not isinstance(x, bool))

    _add(getattr(tok, "eos_token_id", None))
    config = getattr(model, "config", None)
    for obj in (getattr(model, "generation_config", None), config):
        if obj is not None:
            _add(getattr(obj, "eos_token_id", None))
    if config is not None:
        text_configs = []
        get_text_config = None
        with contextlib.suppress(Exception):
            get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            with contextlib.suppress(Exception):
                text_configs.append(get_text_config(decoder=True))
        with contextlib.suppress(Exception):
            text_configs.append(getattr(config, "text_config", None))
        for text_config in text_configs:
            if text_config is not None:
                with contextlib.suppress(Exception):
                    _add(getattr(text_config, "eos_token_id", None))
    return frozenset(ids)


def generation_eos_from_cached_config(
    model_id: str, model_revision: str, tokenizer
) -> frozenset[int]:
    """``_generation_eos_ids`` for a model that is on disk but NOT loaded.

    The verl paths never hold the model in this process -- it is trained in a separate interpreter
    -- so the config and generation_config are read from the local hf cache and presented as the
    attributes ``_generation_eos_ids`` expects. ``local_files_only`` because the caller prefetched
    the model and the child runs with HF_HUB_OFFLINE.
    """
    from transformers import AutoConfig, GenerationConfig

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, revision=model_revision or None, local_files_only=True
    )
    generation_config = None
    with contextlib.suppress(OSError):
        generation_config = GenerationConfig.from_pretrained(
            model_id, revision=model_revision or None, local_files_only=True
        )
    model_like = type(
        "ModelGenerationMetadata",
        (),
        {"config": config, "generation_config": generation_config},
    )()
    return _generation_eos_ids(model_like, tokenizer)
