"""tau-bench retail tools and transcripts, exactly as researchers hand them to Flash.

tau's native ``Tool.get_info()`` declarations omit ``additionalProperties``, and its OpenAI-format
transcripts carry ``tool_calls[].function.arguments`` as JSON strings. Both reach Flash through
Base chat (serving), SFT (preview quote and worker render) and a GRPO environment's
``tool_schemas``; every path must accept them and render the same closed declaration.
"""

from __future__ import annotations

import copy
import json
import os

import pytest
from pydantic import ValidationError

from flash.serve.request.openai import OpenAIRequestError, parse_chat_request
from flash.serve.request.tool_calls import detached_template_messages, normalize_tools, tools_wire
from flash.serving.src.io.openai_request import OpenAIGenerateRequest
from tests.test_sft_tools import ItemsTemplateTokenizer, _spec

# verbatim from tau_bench/envs/retail/tools/{find_user_id_by_email,exchange_delivered_order_items}.py
TAU_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_user_id_by_email",
            "description": "Find user id by email. If the user is not found, the function will "
            "return an error message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The email of the user, such as 'something@example.com'.",
                    },
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exchange_delivered_order_items",
            "description": "Exchange items in a delivered order to new items of the same product "
            "type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order id, such as '#W0'."},
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                    "new_item_ids": {"type": "array", "items": {"type": "string"}},
                    "payment_method_id": {"type": "string"},
                },
                "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
            },
        },
    },
]
EXCHANGE_ARGS = json.dumps(
    {
        "order_id": "#W2378156",
        "item_ids": ["1151293680"],
        "new_item_ids": ["7706410293"],
        "payment_method_id": "credit_card_9513926",
    }
)
TRANSCRIPT = [
    {"role": "system", "content": "# Retail agent policy"},
    {"role": "user", "content": "Hi, I'd like to exchange an item. I'm yusuf@example.com"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "find_user_id_by_email",
                    "arguments": '{"email": "yusuf@example.com"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "find_user_id_by_email", "content": "y_1"},
    {"role": "user", "content": "Yes, please go ahead."},
]
EXCHANGE_TURN = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "exchange_delivered_order_items", "arguments": EXCHANGE_ARGS},
        }
    ],
}


def _closed(tools):
    closed = copy.deepcopy(tools)
    for tool in closed:
        tool["function"]["parameters"]["additionalProperties"] = False
    return closed


def test_tau_declarations_normalize_to_the_closed_declaration() -> None:
    # an omitted additionalProperties is the closed object Flash already wires and parses
    assert tools_wire(normalize_tools(TAU_TOOLS)) == tools_wire(normalize_tools(_closed(TAU_TOOLS)))
    wired = tools_wire(normalize_tools(TAU_TOOLS))
    assert all(tool["function"]["parameters"]["additionalProperties"] is False for tool in wired)


@pytest.mark.parametrize("value", [True, {"type": "string"}])
def test_an_explicitly_open_object_is_still_refused(value) -> None:
    tools = copy.deepcopy(TAU_TOOLS)
    tools[0]["function"]["parameters"]["additionalProperties"] = value
    with pytest.raises(ValueError, match=r"tools\[0\].*additionalProperties must be false"):
        normalize_tools(tools)


def test_base_chat_accepts_tau_tools_and_string_argument_history() -> None:
    payload = {"messages": TRANSCRIPT, "tools": TAU_TOOLS, "tool_choice": "auto"}
    request = parse_chat_request(payload, require_model=False, allow_managed_selectors=True)
    assert [tool.name for tool in request.tools] == [
        "find_user_id_by_email",
        "exchange_delivered_order_items",
    ]
    hosted = OpenAIGenerateRequest(
        model_id="base",
        generation_id="generation",
        messages=TRANSCRIPT,
        tools=TAU_TOOLS,
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    assert hosted.tools == tools_wire(normalize_tools(_closed(TAU_TOOLS)))


def test_base_chat_still_refuses_an_explicitly_open_object() -> None:
    tools = copy.deepcopy(TAU_TOOLS)
    tools[1]["function"]["parameters"]["additionalProperties"] = True
    with pytest.raises(OpenAIRequestError, match="additionalProperties must be false"):
        parse_chat_request(
            {"messages": TRANSCRIPT, "tools": tools},
            require_model=False,
            allow_managed_selectors=True,
        )
    with pytest.raises(ValidationError, match="additionalProperties must be false"):
        OpenAIGenerateRequest(
            model_id="base", generation_id="g", messages=TRANSCRIPT, tools=tools, tool_choice="auto"
        )


def _tau_row():
    return {
        "input": TRANSCRIPT[1]["content"],
        "prompt_messages": TRANSCRIPT,
        "output": {"messages": [EXCHANGE_TURN]},
        "tools": TAU_TOOLS,
    }


class _TauSftEnvironment:
    multi_turn = False
    package_root = None

    def __init__(self, rows):
        self._rows = rows

    def dataset(self):
        return list(self._rows)

    def prompt_messages(self, row):
        raise AssertionError("stored prompt_messages must be used")

    def sft_completion(self, row):
        return row["output"]["messages"]


def _prepare_sft(tokenizer, *, max_context_tokens=256):
    from flash.engine.profiling.sft_workload import prepare_sft_workload

    return prepare_sft_workload(
        _spec(max_context_tokens),
        _TauSftEnvironment([_tau_row()]),
        tokenizer_loader=lambda *_: tokenizer,
        producer_version="1.2.3",
        packing_support=lambda *_: ("pure-attention", True),
    )


def test_sft_renders_a_tau_row_with_string_arguments() -> None:
    # the same workload the control plane quotes for POST /v1/runs and the worker trains on
    prepared = _prepare_sft(ItemsTemplateTokenizer())
    text = prepared.sampled_texts[0]
    assert "<find_user_id_by_email email=yusuf@example.com>" in text
    assert "<exchange_delivered_order_items order_id=#W2378156" in text
    assert "[tools:find_user_id_by_email,exchange_delivered_order_items]" in text


def test_grpo_environment_accepts_tau_tool_schemas() -> None:
    from tests.test_rl_native_tool_calls import _adapter

    assert _adapter(schemas=TAU_TOOLS).tools() == tools_wire(normalize_tools(_closed(TAU_TOOLS)))


@pytest.mark.skipif(
    os.environ.get("FLASH_SFT_TOOLS_TEMPLATE_LIVE") != "1",
    reason="real Qwen3.5 template parity disabled; set FLASH_SFT_TOOLS_TEMPLATE_LIVE=1",
)
def test_qwen35_renders_tau_rows_identically_across_sft_base_chat_and_grpo_glue() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    served_tools = tools_wire(normalize_tools(TAU_TOOLS))
    # serving (Base chat) and the GRPO glue both render through detached_template_messages
    served = tokenizer.apply_chat_template(
        detached_template_messages([*TRANSCRIPT, EXCHANGE_TURN]),
        tools=served_tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    assert "<parameter=email>\nyusuf@example.com\n</parameter>" in served
    assert '"additionalProperties": false' in served
    prepared = _prepare_sft(tokenizer, max_context_tokens=4096)
    assert prepared.sampled_texts == [served]
