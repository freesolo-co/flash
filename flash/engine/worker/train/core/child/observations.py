"""Collect model versions across the turns of one owned rollout client."""


class ObservedRolloutClient:
    def __init__(self, client):
        self.client = client
        self.versions = []

    async def generate(self, *args, **kwargs):
        output = await self.client.generate(*args, **kwargs)
        for key in ("min_global_steps", "max_global_steps", "global_steps"):
            value = output.extra_fields.get(key)
            if value is not None:
                self.versions.append(int(value))
        return output

    def annotate(self, output):
        if self.versions:
            output.extra_fields["min_global_steps"] = min(self.versions)
            output.extra_fields["max_global_steps"] = max(self.versions)
        return output
