"""shared managed chat api test helpers."""

from tests.test_server_api import SPEC, _bearer, _login


class _RawManagedChatResponse:
    def __init__(self, chunks, *, status_code=200, headers=None):
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/event-stream"}
        self.chunks = chunks
        self.closed = False

    def iter_bytes(self):
        try:
            yield from self.chunks
        finally:
            self.closed = True

    def close(self):
        self.closed = True


def _deployed_chat_run(api):
    """A deployed run ready to chat, returned as (key, run_id)."""
    import flash.runner as runner

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    revision = f"{run_id}@final." + "a" * 40
    runner.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )
    return key, run_id
