"""Count the reasoning subset of native Qwen completion IDs, independently per choice."""

from collections.abc import Iterable
from typing import Any


def reasoning_token_ids(parser: str | None, tokenizer: Any) -> tuple[int, int]:
    """Resolve only the boundary IDs; never materialize the tokenizer vocabulary."""
    if parser != "qwen3":
        raise ValueError("thinking usage requires the qwen3 reasoning parser")
    start = tokenizer.convert_tokens_to_ids("<think>")
    end = tokenizer.convert_tokens_to_ids("</think>")
    unknown = getattr(tokenizer, "unk_token_id", None)
    if (
        type(start) is not int
        or type(end) is not int
        or start == end
        or min(start, end) < 0
        or unknown in (start, end)
    ):
        raise ValueError("thinking usage requires distinct native think token IDs")
    return start, end


class ReasoningTokenCounter:
    """Match Flash's first-close reasoning split without re-tokenizing rendered text.

    Qwen chat templates can put the opening think token in the prompt. Consequently
    each thinking choice starts in reasoning, even without a generated opening token.
    Generated boundary tokens remain completion tokens but are not reasoning content.
    A truncated or disconnected choice counts all reasoning generated so far. Once
    the first closing token arrives, subsequent tokens never reopen reasoning.
    """

    def __init__(self, token_ids: tuple[int, int] | None) -> None:
        self.total = 0
        self._thinking = token_ids is not None
        self._seen: set[int] = set()
        self._ended: set[int] = set()
        self._start, self._end = token_ids if token_ids is not None else (None, None)

    def consume(self, index: int, token_ids: Iterable[int]) -> None:
        if not self._thinking or index in self._ended:
            return
        for token in token_ids:
            initial = index not in self._seen
            self._seen.add(index)
            if token == self._end:
                self._ended.add(index)
                return
            if initial and token == self._start:
                continue
            self.total += 1
