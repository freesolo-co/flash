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


def processor_template_kwargs(processor: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep rendering options out of Transformers' legacy processor-kwargs path.

    Serving renders text only: vLLM processes the decoded images separately. Options
    unused by the selected Jinja template (often preserve_thinking) must not be
    interpreted as image/tokenization options. Keep tools, an explicit HF argument.
    Use HF's cached introspection so the classification matches its warning check.
    """
    from transformers.utils.chat_template_utils import _get_template_variables

    template = getattr(processor, "chat_template", None)
    if isinstance(template, dict):
        template = template.get("default")
    if not isinstance(template, str):
        # Let the processor report missing/ambiguous template metadata as usual.
        return kwargs
    variables = _get_template_variables(template)
    return {key: value for key, value in kwargs.items() if key == "tools" or key in variables}
