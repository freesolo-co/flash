"""Only pre-admission failures can prove that retrying will not duplicate a job."""

import pytest
from fastapi import HTTPException

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.submit as runner_submit
from tests.test_server_api import SPEC, _bearer, _new_user_token

pytest_plugins = ("tests._helpers.server_api_plugin",)


@pytest.mark.parametrize("preparation_failure", [True, False])
@pytest.mark.parametrize("http_error", [False, True])
def test_submission_outcome_distinguishes_rejection_from_retained_run(
    api, monkeypatch, preparation_failure, http_error
):
    key = _new_user_token()

    def fail(spec, **kwargs):
        if not preparation_failure:
            runner_state._save_status(
                runner_state.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
            )
        if http_error:
            raise HTTPException(503, "preparation unavailable", headers={"Retry-After": "5"})
        raise ValueError("could not resolve model_revision")

    monkeypatch.setattr(runner_submit, "prepare_job" if preparation_failure else "submit_job", fail)
    response = api.post("/v1/runs", headers=_bearer(key), json={"spec": SPEC})

    assert response.status_code == (503 if http_error else 400)
    assert response.headers.get("X-Flash-Submission-Outcome") == (
        "not-created" if preparation_failure else None
    )
    if http_error:
        assert response.headers["Retry-After"] == "5"
    runs = api.get("/v1/runs", headers=_bearer(key)).json()["runs"]
    assert len(runs) == (0 if preparation_failure else 1)
    if not preparation_failure:
        assert runs[0]["state"] == "queued"
