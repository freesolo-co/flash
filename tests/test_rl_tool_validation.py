"""RL call turns share serving validation even before tool results arrive."""

from copy import deepcopy

import pytest

from flash.engine.worker.train.core.child.glue import (
    validate_structured_messages,
    validate_transcript_messages,
)
from flash.serve.request.tool_calls import validate_tool_history


def _call(*, arguments='{"order_id":"#W1"}', name="get_order_details", call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.mark.parametrize(
    "calls",
    [
        [_call(arguments='{"x":1,"x":2}')],
        [_call(arguments='{"x":NaN}')],
        [_call(arguments='{"x":Infinity}')],
        [_call(arguments='{"x":1e1000001}')],
        [_call(arguments='{"x":"\\ud800"}')],
        [_call(arguments='{"x":' + "[" * 10 + "0" + "]" * 10 + "}")],
        [_call(name="bad name")],
        [_call(call_id=" padded ")],
        [_call(call_id="\ud800")],
        [_call(), _call()],
    ],
)
def test_rl_rejects_invalid_calls_with_serving_diagnostics(calls):
    assistant = {"role": "assistant", "content": "", "tool_calls": calls}
    reply = {"role": "tool", "tool_call_id": calls[0]["id"], "content": "{}"}
    with pytest.raises(ValueError, match="message 0 tool call") as serving_error:
        validate_tool_history([assistant, reply])
    for validate in (validate_structured_messages, validate_transcript_messages):
        with pytest.raises(ValueError, match="environment reply message 0") as rl_error:
            validate([assistant], source="environment reply", tools=True)
        assert str(rl_error.value) == f"environment reply {serving_error.value}"


def test_rl_accepts_partial_call_turns_and_detaches_wire_arguments():
    arguments = '{"order_id": "#W1", "number": 9007199254740993}'
    assistant = {"role": "assistant", "content": None, "tool_calls": [_call(arguments=arguments)]}
    original = deepcopy(assistant)
    reply = {"role": "tool", "tool_call_id": "call_1", "content": "{}"}
    validate_tool_history([assistant, reply])
    # The rollout validates a call before its result exists. Serving alone requires full history.
    with pytest.raises(ValueError, match="before all tool calls were resolved"):
        validate_tool_history([assistant])
    for validate in (validate_structured_messages, validate_transcript_messages):
        (normalized,) = validate([assistant], source="environment reply", tools=True)
        assert normalized == dict(assistant, content="")
        normalized["tool_calls"][0]["function"]["arguments"] = "{}"
        assert assistant == original


def test_serving_still_rejects_reused_ids_across_call_turns():
    assistant = {"role": "assistant", "content": "", "tool_calls": [_call()]}
    reply = {"role": "tool", "tool_call_id": "call_1", "content": "{}"}
    with pytest.raises(ValueError, match="message 2 tool call 0 id is duplicated"):
        validate_tool_history([assistant, reply, deepcopy(assistant), deepcopy(reply)])
