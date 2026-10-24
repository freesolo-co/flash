"""Incrementally decode native tokens without dropping reasoning or partial UTF-8."""


class Decoder:
    def __init__(self, tokenizer, *, thinking):
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.start = tokenizer.token_to_id("<think>")
        self.end = tokenizer.token_to_id("</think>")
        self.eos = tokenizer.token_to_id("<\uff5cend▁of▁sentence\uff5c>")
        if thinking and (self.start is None or self.end is None):
            raise RuntimeError("Pinned tokenizer is missing reasoning delimiters")
        self.pending = []

    def _flush(self, *, final=False):
        if not self.pending:
            return {}
        text = self.tokenizer.decode(self.pending, skip_special_tokens=False)
        if not final and text.endswith("\ufffd"):
            return {}
        self.pending.clear()
        field = "reasoning_content" if self.thinking else "content"
        return {field: text}

    def consume(self, event):
        deltas = []
        for token in event["tokens"]:
            self.completion_tokens += 1
            if self.thinking and token in {self.start, self.end}:
                if delta := self._flush(final=True):
                    deltas.append(delta)
                if token == self.end:
                    self.thinking = False
                continue
            if token == self.eos:
                continue
            if self.thinking:
                self.reasoning_tokens += 1
            self.pending.append(token)
        if delta := self._flush(final=bool(event.get("finish_reason"))):
            deltas.append(delta)
        return deltas
