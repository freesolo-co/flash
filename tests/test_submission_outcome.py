"""Only pre-admission failures can prove that retrying will not duplicate a job."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.submit as runner_submit
from tests.test_server_api import SPEC, _bearer, _new_user_token

pytest_plugins = ("tests._helpers.server_api_plugin",)


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"spec": SPEC, "dry_run": "false"}, "dry_run must be a boolean"),
        ({"spec": []}, "spec must be a JSON object"),
        ({"spec": SPEC, "runtime_secrets": []}, "runtime_secrets must be a JSON object"),
        (
            {"spec": {**SPEC, "train": {"save_at_steps": [46]}}},
            "train.save_at_steps requires positive train.max_steps",
        ),
    ],
)
def test_invalid_submission_is_definitively_not_created(api, monkeypatch, payload, detail):
    key = _new_user_token()
    monkeypatch.setattr(
        runner_submit, "prepare_job", lambda *_a, **_k: pytest.fail("invalid request prepared")
    )

    response = api.post("/v1/runs", headers=_bearer(key), json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == detail
    assert response.headers.get("X-Flash-Submission-Outcome") == "not-created"
    assert api.get("/v1/runs", headers=_bearer(key)).json()["runs"] == []


@pytest.mark.parametrize(
    "gate",
    [
        "flash.server.domain.registry.projects.require_project_access",
        "flash.server.routes.runs._resolve_managed_environment",
        "flash.server.routes.runs._submission_context",
    ],
)
def test_early_submission_gate_preserves_rejection_and_headers(api, monkeypatch, gate):
    key = _new_user_token()

    def reject(*_args, **_kwargs):
        raise HTTPException(503, "gate unavailable", headers={"Retry-After": "5"})

    monkeypatch.setattr(gate, reject)
    monkeypatch.setattr(
        runner_submit, "prepare_job", lambda *_a, **_k: pytest.fail("rejected request prepared")
    )

    response = api.post("/v1/runs", headers=_bearer(key), json={"spec": SPEC})

    assert response.status_code == 503
    assert response.json()["detail"] == "gate unavailable"
    assert response.headers.get("X-Flash-Submission-Outcome") == "not-created"
    assert response.headers["Retry-After"] == "5"
    assert api.get("/v1/runs", headers=_bearer(key)).json()["runs"] == []


def test_corrected_checkpoint_schedule_can_submit_after_rejection(api, monkeypatch):
    key = _new_user_token()
    submissions = []

    def submit(spec, **kwargs):
        submissions.append(spec.run_id)
        status = runner_state.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
        runner_state._save_status(status)
        return status

    monkeypatch.setattr(runner_submit, "submit_job", submit)
    monkeypatch.setattr(
        runner_submit,
        "prepare_job",
        lambda spec, **_kwargs: SimpleNamespace(public_spec=spec, estimated_cost_usd=0),
    )
    spec = {**SPEC, "algorithm": "sft", "train": {"epochs": 2, "save_at_steps": [46]}}
    rejected = api.post("/v1/runs", headers=_bearer(key), json={"spec": spec})
    assert rejected.status_code == 400
    assert rejected.headers.get("X-Flash-Submission-Outcome") == "not-created"
    assert submissions == []

    spec["train"]["max_steps"] = 92
    accepted = api.post("/v1/runs", headers=_bearer(key), json={"spec": spec})
    assert accepted.status_code == 200, accepted.text
    assert accepted.headers.get("X-Flash-Submission-Outcome") is None
    assert submissions == [accepted.json()["run_id"]]
    runs = api.get("/v1/runs", headers=_bearer(key)).json()["runs"]
    assert [(run["run_id"], run["state"]) for run in runs] == [(submissions[0], "queued")]


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
