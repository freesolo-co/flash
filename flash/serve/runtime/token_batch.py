"""Bounded token delivery: immediate first output, then size/time-limited batches."""

from dataclasses import dataclass, field


@dataclass
class TokenBatch:
    max_tokens: int = 16
    max_seconds: float = 0.05
    pending: list[int] = field(default_factory=list)
    last_emit: float | None = None
    pending_since: float | None = None

    def add(self, tokens, now):
        if tokens and not self.pending:
            self.pending_since = now
        self.pending.extend(tokens)

    def take(self, now, *, finished=False):
        if not self.pending:
            return []
        if (
            self.last_emit is not None
            and not finished
            and len(self.pending) < self.max_tokens
            and now - self.pending_since < self.max_seconds
        ):
            return []
        tokens, self.pending = self.pending, []
        self.last_emit = now
        self.pending_since = None
        return tokens
