"""Count the reasoning subset of native Qwen completion IDs, independently per choice."""

from collections.abc import Iterable
from typing import Any


class ReasoningTokenCounter:
    """Match Flash's first-close reasoning split without re-tokenizing rendered text.

    Qwen chat templates can put the opening think token in the prompt. Consequently
    each thinking choice starts in reasoning, even without a generated opening token.
    Generated boundary tokens remain completion tokens but are not reasoning content.
    A truncated or disconnected choice counts all reasoning generated so far. Once
    the first closing token arrives, subsequent tokens never reopen reasoning.
    """

    def __init__(self, *, thinking: bool, parser: str | None, tokenizer: Any) -> None:
        self.total = 0
        self._thinking = thinking
        self._seen: set[int] = set()
        self._ended: set[int] = set()
        self._start: int | None = None
        self._end: int | None = None
        if not thinking:
            return
        if parser != "qwen3":
            raise ValueError("thinking usage requires the qwen3 reasoning parser")
        vocab = tokenizer.get_vocab()
        self._start = vocab.get("<think>")
        self._end = vocab.get("</think>")
        if (
            type(self._start) is not int
            or type(self._end) is not int
            or self._start == self._end
            or min(self._start, self._end) < 0
        ):
            raise ValueError("thinking usage requires distinct native think token IDs")

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
