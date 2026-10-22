"""Collect model versions within one episode, including concurrent agent reuse."""

from contextlib import contextmanager
from contextvars import ContextVar


class ObservedRolloutClient:
    def __init__(self, client):
        self.client = client
        self._versions = ContextVar("rollout_versions", default=None)

    @contextmanager
    def episode(self):
        token = self._versions.set([])
        try:
            yield
        finally:
            self._versions.reset(token)

    def _episode_versions(self):
        versions = self._versions.get()
        if versions is None:
            raise RuntimeError("rollout observations require an episode scope")
        return versions

    async def generate(self, *args, **kwargs):
        versions = self._episode_versions()
        output = await self.client.generate(*args, **kwargs)
        for key in ("min_global_steps", "max_global_steps", "global_steps"):
            value = output.extra_fields.get(key)
            if value is not None:
                versions.append(int(value))
        return output

    def annotate(self, output):
        versions = self._episode_versions()
        if versions:
            output.extra_fields["min_global_steps"] = min(versions)
            output.extra_fields["max_global_steps"] = max(versions)
        return output
