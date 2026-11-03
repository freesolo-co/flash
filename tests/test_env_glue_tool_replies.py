"""Multi-turn glue for tool-result replies under the real Qwen3.5 chat template.

The template raises "No user query found in messages" for a conversation without an ordinary
user turn, and tool results are deliberately not one, so a GRPO environment replying with a tool
message failed every rollout step. Glue must be exactly the text the template renders after that
assistant turn in the full conversation.
"""

from __future__ import annotations

import json
from pathlib import Path

import jinja2
import pytest

from flash.engine.worker.train.core.child.glue import EnvGlueTokenizer

TEMPLATE = (Path(__file__).parent / "fixtures/qwen3_5_9b_chat_template.jinja").read_text()


class _TemplateError(Exception):
    pass


def _raise(message):
    raise _TemplateError(message)


class _CharTokenizer:
    """the real chat template with a character-level tokenizer, so ids decode back to text."""

    def __init__(self):
        environment = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
        environment.globals["raise_exception"] = _raise
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False
        )
        self.template = environment.from_string(TEMPLATE)

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        assert tokenize is False
        return self.template.render(
            messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
        )

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


def _decode(ids):
    return "".join(map(chr, ids))


SYSTEM = {"role": "system", "content": "Complete the user's spreadsheet request."}
QUERY = {"role": "user", "content": "Fill E2:E9."}
CALL = "<tool_call>\n<function=list_sheets>\n</function>\n</tool_call>"


def _full_suffix(tokenizer, reply):
    conversation = [SYSTEM, QUERY, {"role": "assistant", "content": CALL}, *reply]
    text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    return text[text.index(CALL) + len(CALL) :]


@pytest.mark.parametrize(
    "reply",
    [
        [{"role": "tool", "content": '[{"name": "Sheet1", "rows": 9, "columns": 5}]'}],
        [{"role": "tool", "content": '{"saved": true}'}, {"role": "tool", "content": '{"x": 1}'}],
    ],
)
def test_tool_only_replies_glue_exactly_as_the_full_conversation_renders(reply):
    tokenizer = _CharTokenizer()
    glue = EnvGlueTokenizer(tokenizer, thinking=False)(reply)
    assert _decode(glue) == _full_suffix(tokenizer, reply)


def test_user_replies_keep_their_unanchored_glue():
    tokenizer = _CharTokenizer()
    reply = [{"role": "user", "content": "Observation: the page changed."}]
    glue = EnvGlueTokenizer(tokenizer, thinking=False)(reply)
    assert _decode(glue) == _full_suffix(tokenizer, reply)


def test_the_unanchored_probe_is_what_the_template_rejects():
    tokenizer = _CharTokenizer()
    with pytest.raises(_TemplateError, match="No user query"):
        tokenizer.apply_chat_template(
            [{"role": "assistant", "content": "probe"}, {"role": "tool", "content": "{}"}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
