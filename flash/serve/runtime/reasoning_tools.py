"""Parse executable tools only after the first reasoning close, preserving raw text."""

from collections.abc import Callable, Sequence

from flash.serve.request.tool_calls import FunctionTool
from flash.serve.request.tool_template import THINK_END

from .tool_calls import ToolCallStreamParser, ToolParseResult, parse_qwen3_coder_output


def parse_tool_output(
    text: str,
    tools: Sequence[FunctionTool],
    *,
    thinking: bool = False,
    id_factory: Callable[[], str] | None = None,
) -> ToolParseResult:
    prefix = ""
    if thinking:
        reasoning, close, answer = text.partition(THINK_END)
        if not close:
            return ToolParseResult(content=text, calls=())
        prefix, text = reasoning + close, answer
    result = parse_qwen3_coder_output(text, tools, id_factory=id_factory)
    return ToolParseResult(content=(prefix + (result.content or "")) or None, calls=result.calls)


class ReasoningToolStreamParser:
    """Forward reasoning untouched; give only answer deltas to the strict tool parser."""

    def __init__(
        self,
        tools: Sequence[FunctionTool],
        *,
        thinking: bool = False,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._closed = not thinking
        self._tail = ""
        self._tools = ToolCallStreamParser(tools, id_factory=id_factory)

    def feed(self, text: str) -> str:
        if self._closed:
            return self._tools.feed(text)
        pending = self._tail + text
        reasoning, close, answer = pending.partition(THINK_END)
        if close:
            self._tail = ""
            self._closed = True
            return reasoning + close + self._tools.feed(answer)
        # Retain only enough text to recognize a close split across deltas.
        retain = len(THINK_END) - 1
        self._tail = pending[-retain:]
        return pending[:-retain]

    def finish(self) -> ToolParseResult:
        if self._closed:
            return self._tools.finish()
        tail, self._tail = self._tail, ""
        return ToolParseResult(content=tail or None, calls=())
