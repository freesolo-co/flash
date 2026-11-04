"""Checkpoint templates embedded in tokenizer_config must work on processor paths."""

from types import SimpleNamespace

import pytest

from flash.content.processor import inherit_tokenizer_chat_template


@pytest.mark.parametrize("processor_template", [None, "processor-specific template"])
def test_processor_keeps_checkpoint_template_and_tokenizer_identity(processor_template):
    tokenizer = SimpleNamespace(chat_template="checkpoint template", eos_token="<|endoftext|>")
    processor = SimpleNamespace(tokenizer=tokenizer, chat_template=processor_template)
    inherit_tokenizer_chat_template(processor)
    assert processor.chat_template == (processor_template or "checkpoint template")
    assert processor.tokenizer is tokenizer
    assert processor.tokenizer.eos_token == "<|endoftext|>"


def test_absent_checkpoint_template_is_not_invented():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(), chat_template=None)
    inherit_tokenizer_chat_template(processor)
    assert processor.chat_template is None


@pytest.mark.parametrize("loader", ["hosted", "customer", "sft"])
def test_serving_and_training_loaders_accept_inline_tokenizer_template(monkeypatch, loader):
    import transformers

    model = "Qwen/Qwen3.5-9B-Base"
    revision = "68c46c4b3498877f3ef123c856ecfde50c39f404"
    tokenizer = SimpleNamespace(
        chat_template="checkpoint template",
        pad_token=None,
        eos_token="<|endoftext|>",
        eos_token_id=0,
    )
    processor = SimpleNamespace(tokenizer=tokenizer, chat_template=None)
    calls = []

    def load(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return processor

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load)
    if loader == "hosted":
        from flash.serving.src.engine.boot import load_tokenizer

        loaded_processor, loaded_tokenizer = load_tokenizer(
            model, SimpleNamespace(hf_api_key=None), SimpleNamespace(TRUST_REMOTE_CODE=True)
        )
    elif loader == "customer":
        from flash.serve.runtime import EngineConfig
        from flash.serve.runtime.engine import VllmLoraRuntime

        runtime = VllmLoraRuntime(
            EngineConfig(model=model, tokenizer_revision=revision, image_limit=32)
        )
        loaded_tokenizer, loaded_processor = runtime._load_tokenizer_processor()
    else:
        from flash.engine.profiling.sft_workload import _default_processor_loader

        loaded_processor = _default_processor_loader(model, revision)
        loaded_tokenizer = loaded_processor.tokenizer
    assert loaded_processor.chat_template == "checkpoint template"
    assert loaded_tokenizer is tokenizer
    assert loaded_tokenizer.eos_token == "<|endoftext|>"
    assert calls[0][0] == model
    assert calls[0][1]["revision"] == revision


def test_referenced_preserve_thinking_remains_a_template_option(caplog):
    from types import MethodType

    from transformers.processing_utils import ProcessorMixin

    from flash.content.processor import processor_template_kwargs

    processor = SimpleNamespace(
        chat_template="{{ enable_thinking }}|{{ preserve_thinking }}|{{ tools[0].function.name }}",
        tokenizer=SimpleNamespace(special_tokens_map={}),
    )
    processor.apply_chat_template = MethodType(ProcessorMixin.apply_chat_template, processor)
    kwargs = {
        "enable_thinking": True,
        "preserve_thinking": False,
        "unused": 1,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    original = processor.apply_chat_template(messages, tokenize=False, **kwargs)
    assert "Kwargs passed to" in caplog.text
    caplog.clear()
    rendered = processor.apply_chat_template(
        messages, tokenize=False, **processor_template_kwargs(processor, kwargs)
    )
    assert rendered == original == "True|False|lookup"
    assert "Kwargs passed to" not in caplog.text
