"""Policy-load acknowledgement and at-most-once request admission.

Owned by one server event loop; mutations contain no awaits. Weight loading must
invalidate the acknowledgement before touching weights, then acknowledge only
once transfer and cache reset have succeeded. In-flight results fail closed.
"""

from uuid import uuid4

from .contract import PolicyIdentity, RolloutRequest


class PolicyLease:
    def __init__(self):
        self.policy: PolicyIdentity | None = None
        self.ack: str | None = None
        self._last_version = -1
        self._attempts: set[str] = set()

    def invalidate(self):
        self.ack = None

    def acknowledge(self, policy: PolicyIdentity):
        if policy.policy_version <= self._last_version:
            raise ValueError("rollout policy versions must increase within a run")
        self.policy = policy
        self._last_version = policy.policy_version
        self.ack = uuid4().hex
        self._attempts.clear()
        return self.ack

    def admit(self, request: RolloutRequest):
        if self.ack is None or request.policy != self.policy:
            raise ValueError("rollout requested policy has not been acknowledged as loaded")
        if request.request_id in self._attempts:
            raise ValueError("duplicate rollout request; retry with a new attempt identity")
        self._attempts.add(request.request_id)
        return self.ack

    def finish(self, acknowledgement):
        if self.ack is None or acknowledgement != self.ack:
            raise ValueError("rollout policy changed during generation")
