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
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    checkpoint_id = f"{run_id}/final"
    runner_transitions.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "checkpoint_id": checkpoint_id,
            "openai_model": checkpoint_id,
        },
        verification_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
    )
    return key, run_id
