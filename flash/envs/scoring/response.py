"""String-compatible scorer input with optional generation metadata."""

from flash.envs.scoring.context import RolloutContext


class ScoredResponseText(str):
    """Answer text for existing scorers; structured views for scorers that opt in."""

    __slots__ = ("completion", "raw", "rollout_context", "thinking")

    completion: str
    thinking: str | None
    raw: str
    rollout_context: RolloutContext | None

    def __new__(
        cls,
        completion: str,
        *,
        raw: str,
        thinking: str | None,
        rollout_context: RolloutContext | None = None,
    ):
        obj = str.__new__(cls, completion)
        obj.completion = completion
        obj.thinking = thinking
        obj.raw = raw
        obj.rollout_context = rollout_context
        return obj

    def __getnewargs_ex__(self):
        return (str(self),), {
            "raw": self.raw,
            "thinking": self.thinking,
            "rollout_context": self.rollout_context,
        }
