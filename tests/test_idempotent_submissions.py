"""Fault-injected retries must converge on one paid run, including across processes."""

import hashlib
import json
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from flash.runner.lifecycle import state, status
from flash.runner.lifecycle import submit as runner_submit
from flash.server.platform import db
from flash.server.routes import submissions
from tests.test_server_api import SPEC, _bearer, _new_user_token

pytest_plugins = ("tests._helpers.server_api_plugin",)
ID = "a" * 32
URL = f"/v1/run-submissions/{ID}"


@pytest.fixture
def submitting(api, monkeypatch):
    calls = []

    def submit(spec, **kwargs):
        calls.append(spec.run_id)
        saved = state.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
        state._save_status(saved)
        if accepted := kwargs.get("on_persisted"):
            accepted(spec.run_id)
        return saved

    monkeypatch.setattr(runner_submit, "submit_job", submit)
    monkeypatch.setattr(
        runner_submit,
        "prepare_job",
        lambda spec, **kwargs: SimpleNamespace(public_spec=spec, estimated_cost_usd=0),
    )
    return api, _bearer(_new_user_token()), calls


def test_lost_response_replays_existing_run_and_latest_status(submitting):
    api, headers, calls = submitting
    first = api.put(URL, headers=headers, json={"spec": SPEC})
    assert first.status_code == 200, first.text
    run_id = first.json()["run_id"]
    status._update(run_id, "running")
    retry = api.put(URL, headers=headers, json={"spec": SPEC})
    assert retry.status_code == 200
    assert retry.json()["run_id"] == run_id
    assert retry.json()["state"] == "running"
    assert retry.json()["submission_id"] == ID
    assert calls == [run_id]
    assert api.get(URL, headers=headers).json()["run_id"] == run_id
    assert len(api.get("/v1/runs", headers=headers).json()["runs"]) == 1


@pytest.mark.parametrize("after_persistence", [False, True])
def test_interrupted_server_can_retry_same_id(submitting, monkeypatch, after_persistence):
    api, headers, calls = submitting
    original = runner_submit.submit_job

    def broken(spec, **kwargs):
        if after_persistence:
            original(spec, **kwargs)
        raise RuntimeError("lost response")

    monkeypatch.setattr(runner_submit, "submit_job", broken)
    first = api.put(URL, headers=headers, json={"spec": SPEC})
    assert first.status_code == 400
    monkeypatch.setattr(runner_submit, "submit_job", original)
    retry = api.put(URL, headers=headers, json={"spec": SPEC})
    assert retry.status_code == 200, retry.text
    assert calls == [retry.json()["run_id"]]


def test_payload_conflict_and_owner_isolation(submitting):
    api, headers, calls = submitting
    original = api.put(URL, headers=headers, json={"spec": SPEC}).json()
    changed = {**SPEC, "train": {"epochs": 2, "max_examples": 1}}
    assert api.put(URL, headers=headers, json={"spec": changed}).status_code == 422
    other = _bearer(_new_user_token())
    assert api.get(URL, headers=other).status_code == 404
    second = api.put(URL, headers=other, json={"spec": SPEC})
    assert second.status_code == 200
    assert second.json()["run_id"] != original["run_id"]
    assert len(calls) == 2


def test_cancel_before_delayed_submission_never_creates_job(submitting):
    api, headers, calls = submitting
    cancelled = api.delete(URL, headers=headers)
    assert cancelled.status_code == 200
    assert cancelled.json() == {"submission_id": ID, "run_id": None, "state": "cancelled"}
    assert api.put(URL, headers=headers, json={"spec": SPEC}).status_code == 410
    assert api.get(URL, headers=headers).json()["state"] == "cancelled"
    assert calls == []


def test_cancel_after_lost_response_cancels_original(submitting, monkeypatch):
    api, headers, calls = submitting
    run_id = api.put(URL, headers=headers, json={"spec": SPEC}).json()["run_id"]
    cancelled = []

    def cancel(run_id):
        cancelled.append(run_id)
        status._update(run_id, "cancelled")
        return status.get_status(run_id)

    monkeypatch.setattr("flash.server.routes.runs.cancel_run", cancel)
    assert api.delete(URL, headers=headers).json()["state"] == "cancelled"
    assert api.put(URL, headers=headers, json={"spec": SPEC}).status_code == 410
    assert calls == cancelled == [run_id]


def test_simultaneous_retries_do_not_dispatch_twice(submitting, monkeypatch):
    api, headers, calls = submitting
    original = runner_submit.submit_job
    entered, release = threading.Event(), threading.Event()

    def slow(spec, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(spec, **kwargs)

    monkeypatch.setattr(runner_submit, "submit_job", slow)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(api.put, URL, headers=headers, json={"spec": SPEC})
        try:
            assert entered.wait(10)
            retry = api.put(URL, headers=headers, json={"spec": SPEC})
            assert retry.status_code == 409
            assert retry.headers["Retry-After"] == "2"
        finally:
            release.set()
        assert pending.result(timeout=10).status_code == 200
    assert api.put(URL, headers=headers, json={"spec": SPEC}).status_code == 200
    assert len(calls) == 1


def test_cancel_racing_preparation_prevents_dispatch(submitting, monkeypatch):
    api, headers, calls = submitting
    original = runner_submit.prepare_job
    entered, release = threading.Event(), threading.Event()

    def slow(spec, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(spec, **kwargs)

    monkeypatch.setattr(runner_submit, "prepare_job", slow)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(api.put, URL, headers=headers, json={"spec": SPEC})
        try:
            assert entered.wait(10)
            assert api.delete(URL, headers=headers).status_code == 409
        finally:
            release.set()
        assert pending.result(timeout=10).status_code == 410
    assert status.get_status(calls[0]).state == "cancelled"
    assert api.put(URL, headers=headers, json={"spec": SPEC}).status_code == 410


def _reserve_and_crash(database, runs_dir, key_id, digest, ready):
    db.DB_PATH = database
    state.RUNS_DIR = runs_dir
    row = db.reserve_submission(key_id, ID, digest, "crash-reserved-run")
    with submissions._submission_lock(row["run_id"]):
        ready.set()
        threading.Event().wait(30)
        os._exit(1)


def test_process_crash_releases_lock_and_keeps_durable_identity(submitting):
    api, headers, calls = submitting
    token = headers["Authorization"].removeprefix("Bearer ")
    # Authenticate/provision its owner row without creating a run.
    api.get("/v1/runs", headers=headers)
    key_id = db.lookup_key(token)["id"]
    identity = {"spec": SPEC, "dry_run": False, "org_id": "org-" + token.removeprefix("fslo-user-")}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=_reserve_and_crash,
        args=(db.DB_PATH, state.RUNS_DIR, key_id, digest, ready),
    )
    process.start()
    try:
        assert ready.wait(15)
        assert api.put(URL, headers=headers, json={"spec": SPEC}).status_code == 409
        process.kill()
        process.join(timeout=15)
        assert process.exitcode != 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()
    row = db.get_submission(key_id, ID)
    assert row["run_id"] == "crash-reserved-run"
    response = api.put(URL, headers=headers, json={"spec": SPEC})
    assert response.status_code == 200, response.text
    assert calls == ["crash-reserved-run"]


def test_missing_accepted_status_is_never_recreated(submitting):
    api, headers, calls = submitting
    first = api.put(URL, headers=headers, json={"spec": SPEC}).json()
    os.unlink(state.runs_file_path(first["run_id"], ".json"))
    assert api.put(URL, headers=headers, json={"spec": SPEC}).status_code == 503
    assert len(calls) == 1


def test_runtime_secret_values_are_not_stored_or_part_of_request_identity(submitting):
    api, headers, calls = submitting
    first = api.put(URL, headers=headers, json={"spec": SPEC}).json()
    # Replay reads the existing job without resolving current (possibly rotated) credentials.
    response = api.put(
        URL, headers=headers, json={"spec": SPEC, "runtime_secrets": {"KEY": "rotated-secret"}}
    )
    assert response.status_code == 200
    assert response.json()["run_id"] == first["run_id"]
    assert len(calls) == 1
    with db._connect() as connection:
        row = dict(connection.execute("SELECT * FROM run_submissions").fetchone())
    assert "rotated-secret" not in str(row)


def test_acceptance_hook_runs_after_durable_status_and_before_dispatch(api, monkeypatch):
    from flash.core.spec import JobSpec
    from flash.providers._lifecycle.net import worker
    from tests._helpers.source_snapshot import valid_source_snapshot

    spec = JobSpec(run_id="acceptance-hook", model="Qwen/Qwen3.5-9B", algorithm="grpo")
    prepared = runner_submit.PreparedJob(spec, spec, estimated_cost_usd=0)
    monkeypatch.setattr(worker, "publish_source_snapshot", lambda _repo: valid_source_snapshot())
    dispatched = []

    def interrupted(run_id):
        assert status.get_status(run_id).state == "queued"
        raise RuntimeError("crash at acceptance boundary")

    with pytest.raises(RuntimeError, match="acceptance boundary"):
        runner_submit._submit_prepared_job(
            prepared,
            dry_run=False,
            runtime_secrets=None,
            billing_context=None,
            platform_context=None,
            dispatch=dispatched.append,
            on_persisted=interrupted,
        )
    assert dispatched == []
