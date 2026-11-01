"""Normalize chat-template metadata from a checkpoint's own processor and tokenizer."""

from typing import Any


def inherit_tokenizer_chat_template(processor: Any) -> None:
    """Support checkpoints that publish their template only in tokenizer_config.json.

    Transformers loads that template onto the tokenizer but not AutoProcessor.
    Preserve explicit processor templates and never substitute another model's tokenizer.
    Missing templates remain missing so the normal request validation can reject them.
    """
    if getattr(processor, "chat_template", None) is None:
        tokenizer = getattr(processor, "tokenizer", None)
        template = getattr(tokenizer, "chat_template", None)
        if template is not None:
            processor.chat_template = template
