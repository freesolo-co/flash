"""Unit tests for the immutable serving registry client."""

from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from uuid import uuid4

import httpx
import pytest

from flash.serve import deploy as d

MODEL = "Qwen/Qwen3.5-0.8B"
SHA = "a" * 40
ORG = "00000000-0000-0000-0000-000000000001"


class Response:
    def __init__(self, status_code=200, payload=None, etag=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = {} if etag is None else {"ETag": f'"{etag}"'}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://serve.example")
            raise httpx.HTTPStatusError(
                "failed", request=request, response=httpx.Response(self.status_code)
            )


def _stub_artifact(monkeypatch, tmp_path, *, sha=SHA, rank=32):
    config = tmp_path / "adapter_config.json"
    config.write_text(json.dumps({"r": rank}), encoding="utf-8")
    seen = {"downloads": [], "trees": [], "repo_info": []}

    class HfApi:
        def repo_info(self, **kwargs):
            seen["repo_info"].append(kwargs)
            return types.SimpleNamespace(sha=sha)

        def list_repo_tree(self, **kwargs):
            seen["trees"].append(kwargs)
            return [types.SimpleNamespace(path="adapter_model.safetensors", size=10)]

    def download(**kwargs):
        seen["downloads"].append(kwargs)
        return str(config)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=HfApi, hf_hub_download=download),
    )
    return seen


def _deploy(monkeypatch, tmp_path, *, prior=None, thinking=False, structured_outputs=""):
    _stub_artifact(monkeypatch, tmp_path)
    state = {"record": prior, "posted": [], "headers": []}

    def read(_run_id):
        return state["record"]

    def request(method, _url, *, json=None, headers=None, **_kwargs):
        assert method == "POST"
        state["posted"].append(dict(json))
        state["headers"].append(headers)
        revision = 1 if prior is None else int(prior["registry_revision"]) + 1
        state["record"] = {**json, "registry_revision": revision}
        return Response(etag=revision)

    monkeypatch.setattr(d, "read_adapter_record", read)
    monkeypatch.setattr(d, "_serving_request", request)
    deployment = d.deploy_adapter(
        "flash-1-abc12345",
        MODEL,
        "org/repo",
        "sft/flash-1-abc12345",
        mutation_id=str(uuid4()),
        org_id=ORG,
        thinking=thinking,
        structured_outputs=structured_outputs,
    )
    return deployment, state


def test_deploy_dry_run_does_not_touch_hub_or_serving(monkeypatch):
    monkeypatch.setattr(d, "resolve_repo_revision", lambda *_: pytest.fail("hub access"))
    monkeypatch.setattr(d, "read_adapter_record", lambda *_: pytest.fail("serving access"))
    deployment = d.deploy_adapter(
        "r1", MODEL, "org/repo", "sft/r1", mutation_id=str(uuid4()), dry_run=True
    )
    assert deployment.state == "dry_run"
    assert deployment.to_dict()["openai_base_url"].endswith("/v1")


def test_artifact_validation_uses_one_exact_repo_revision(monkeypatch, tmp_path):
    seen = _stub_artifact(monkeypatch, tmp_path)
    assert d.resolve_repo_revision("org/repo") == SHA
    assert d.adapter_artifact_lora_rank("org/repo", "sft/r1/adapter", SHA) == 32
    assert seen["downloads"][0]["revision"] == SHA
    assert seen["trees"][0]["revision"] == SHA
    assert len(seen["repo_info"]) == 1


@pytest.mark.parametrize("sha", ["main", "A" * 40, "a" * 39, "g" * 40])
def test_repo_revision_must_be_full_lowercase_commit(monkeypatch, tmp_path, sha):
    _stub_artifact(monkeypatch, tmp_path, sha=sha)
    with pytest.raises(d.ServingError, match="full lowercase 40-character"):
        d.resolve_repo_revision("org/repo")


def test_canonical_thinking_record_uses_structured_outputs(monkeypatch, tmp_path):
    deployment, state = _deploy(
        monkeypatch,
        tmp_path,
        thinking=True,
        structured_outputs=json.dumps({"choice": ["4"]}),
    )
    body = state["posted"][0]
    assert body == deployment.desired_record
    assert body["repo_revision"] == SHA
    assert body["repo_type"] == "dataset"
    assert body["thinking"] is True
    assert body["structured_outputs"] == {"choice": ["4"]}


def test_nonthinking_record_uses_same_structured_outputs_field(monkeypatch, tmp_path):
    deployment, _ = _deploy(
        monkeypatch,
        tmp_path,
        structured_outputs=json.dumps({"json_object": True}),
    )
    assert deployment.desired_record["thinking"] is False
    assert deployment.desired_record["structured_outputs"] == {"json_object": True}


def test_registry_mutation_guard_covers_post_and_exact_readback(monkeypatch, tmp_path):
    _stub_artifact(monkeypatch, tmp_path)
    state = {"record": None, "guarded": False}
    events = []

    def read(_run_id):
        if state["record"] is not None:
            assert state["guarded"] is True
            events.append("readback")
        return state["record"]

    def request(_method, _url, *, json=None, **_kwargs):
        assert state["guarded"] is True
        events.append("post")
        state["record"] = {**json, "registry_revision": 1}
        return Response(etag=1)

    @contextmanager
    def guard():
        state["guarded"] = True
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")
            state["guarded"] = False

    monkeypatch.setattr(d, "read_adapter_record", read)
    monkeypatch.setattr(d, "_serving_request", request)
    d.deploy_adapter(
        "r1",
        MODEL,
        "org/repo",
        "sft/r1",
        mutation_id=str(uuid4()),
        org_id=ORG,
        registry_mutation_guard=guard,
    )

    assert events == ["enter", "post", "readback", "exit"]


def test_initial_intent_callback_keeps_five_argument_contract(monkeypatch, tmp_path):
    _stub_artifact(monkeypatch, tmp_path)
    state = {"record": None}
    captured = []

    def persist(*args):
        captured.append(args)

    def request(_method, _url, *, json=None, **_kwargs):
        state["record"] = {**json, "registry_revision": 1}
        return Response(etag=1)

    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: state["record"])
    monkeypatch.setattr(d, "_serving_request", request)
    d.deploy_adapter(
        "r1",
        MODEL,
        "org/repo",
        "sft/r1",
        mutation_id="m1",
        org_id=ORG,
        before_registry_mutation=persist,
    )

    assert len(captured) == 1
    assert len(captured[0]) == 5
    assert captured[0][0] is None


def test_redeploy_intent_callback_keeps_five_argument_contract(monkeypatch, tmp_path):
    _stub_artifact(monkeypatch, tmp_path)
    prior = {
        "adapter_id": "r1",
        "registry_revision": 7,
        "mutation_id": "m7",
        "org_id": ORG,
        "status": "ready",
    }
    state = {"record": prior}
    captured = []

    def persist(prior_revision, desired, target_revision, mutation_id, repo_revision):
        captured.append(
            (prior_revision, desired, target_revision, mutation_id, repo_revision)
        )

    def request(_method, _url, *, json=None, **_kwargs):
        assert d._PRIOR_MUTATION_ID_CONTEXT_KEY not in json
        state["record"] = {**json, "registry_revision": 8}
        return Response(etag=8)

    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: state["record"])
    monkeypatch.setattr(d, "_serving_request", request)
    d.deploy_adapter(
        "r1",
        MODEL,
        "org/repo",
        "sft/r1",
        mutation_id="m8",
        org_id=ORG,
        before_registry_mutation=persist,
    )

    assert len(captured) == 1
    prior_revision, desired, target_revision, mutation_id, _repo_revision = captured[0]
    assert prior_revision == 7
    assert target_revision == 8
    assert mutation_id == "m8"
    assert desired[d._PRIOR_MUTATION_ID_CONTEXT_KEY] == "m7"


def test_lost_post_response_is_committed_by_exact_readback(monkeypatch, tmp_path):
    _stub_artifact(monkeypatch, tmp_path)
    state = {"record": None}

    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: state["record"])

    def request(_method, _url, *, json=None, **_kwargs):
        state["record"] = {**json, "registry_revision": 1}
        raise d.ServingError("connection lost")

    monkeypatch.setattr(d, "_serving_request", request)
    deployment = d.deploy_adapter(
        "r1", MODEL, "org/repo", "sft/r1", mutation_id=str(uuid4()), org_id=ORG
    )
    assert deployment.target_revision == 1
    assert state["record"]["mutation_id"] == deployment.mutation_id


def test_identical_visible_payloads_get_distinct_mutations(monkeypatch, tmp_path):
    first, state = _deploy(monkeypatch, tmp_path)
    prior = dict(state["record"])
    second, state2 = _deploy(monkeypatch, tmp_path, prior=prior)
    assert first.mutation_id != second.mutation_id
    assert state2["headers"] == [{"If-Match": "1"}]
    visible = {"mutation_id", "registry_revision"}
    assert {k: v for k, v in prior.items() if k not in visible} == {
        k: v for k, v in state2["record"].items() if k not in visible
    }


def test_different_concurrent_submission_is_superseded(monkeypatch):
    desired = {"adapter_id": "r1", "mutation_id": "mine", "status": "ready"}
    other = {
        "adapter_id": "r1",
        "mutation_id": "other",
        "status": "ready",
        "registry_revision": 2,
    }
    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: other)
    with pytest.raises(d.DeploymentSuperseded):
        d._readback_target("r1", desired, 2, {"registry_revision": 1, "mutation_id": "old"})


@pytest.mark.parametrize("prior", [None, {"registry_revision": 4, "mutation_id": "old"}])
def test_unchanged_prior_or_absent_create_is_not_committed(monkeypatch, prior):
    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: prior)
    with pytest.raises(d.ServingError, match="did not commit"):
        d._readback_target("r1", {"mutation_id": "new"}, 1 if prior is None else 5, prior)


def test_readback_outage_is_not_misclassified(monkeypatch):
    monkeypatch.setattr(
        d,
        "read_adapter_record",
        lambda _run_id: (_ for _ in ()).throw(d.ServingError("readback unavailable")),
    )
    with pytest.raises(d.ServingError, match="readback unavailable"):
        d._readback_target("r1", {"mutation_id": "new"}, 1, None)


def test_readback_outage_after_unchanged_observation_stays_inconclusive(monkeypatch):
    prior = {"registry_revision": 4, "mutation_id": "old"}
    reads = iter([prior, d.ServingError("readback unavailable")])

    def read(_run_id):
        result = next(reads)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(d, "READBACK_ATTEMPTS", 2)
    monkeypatch.setattr(d, "READBACK_DELAY_SECONDS", 0)
    monkeypatch.setattr(d, "read_adapter_record", read)
    with pytest.raises(d.ServingError, match="readback unavailable"):
        d._readback_target("r1", {"mutation_id": "new"}, 5, prior)


@pytest.mark.parametrize("etag", [None, "1", '"1', '1"', '"0"', 'W/"1"', '"01"'])
def test_malformed_response_etag_is_rejected(etag):
    response = Response(etag=None)
    if etag is not None:
        response.headers["ETag"] = etag
    with pytest.raises(d.ServingError, match="valid quoted ETag"):
        d._etag_revision(response)


def test_lost_delete_response_is_resolved_by_readback(monkeypatch):
    ready = {"adapter_id": "r1", "registry_revision": 3, "mutation_id": "m1", "status": "ready"}
    disabled = {**ready, "registry_revision": 4, "status": "disabled"}
    reads = iter([ready, disabled])
    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: next(reads))
    monkeypatch.setattr(
        d,
        "_serving_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(d.ServingError("lost response")),
    )
    assert d.disable_owned_adapter("r1", 3, "m1") is True


def test_disable_readback_outage_after_unchanged_observation_stays_inconclusive(monkeypatch):
    ready = {"adapter_id": "r1", "registry_revision": 3, "mutation_id": "m1", "status": "ready"}
    reads = iter([ready, ready, d.ServingError("readback unavailable")])

    def read(_run_id):
        result = next(reads)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(d, "READBACK_ATTEMPTS", 2)
    monkeypatch.setattr(d, "READBACK_DELAY_SECONDS", 0)
    monkeypatch.setattr(d, "read_adapter_record", read)
    monkeypatch.setattr(d, "_serving_request", lambda *_args, **_kwargs: Response(etag=4))
    with pytest.raises(d.ServingError, match="readback unavailable"):
        d.disable_owned_adapter("r1", 3, "m1")


def test_disable_never_touches_superseding_mutation(monkeypatch):
    current = {"adapter_id": "r1", "registry_revision": 4, "mutation_id": "new", "status": "ready"}
    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: current)
    monkeypatch.setattr(d, "_serving_request", lambda *_a, **_k: pytest.fail("must not delete"))
    with pytest.raises(d.DeploymentSuperseded):
        d.disable_owned_adapter("r1", 3, "old")


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (
            {"registry_revision": 8, "mutation_id": "m8", "status": "ready"},
            ("r1", 8, "m8"),
        ),
        (
            {"registry_revision": 7, "mutation_id": "m7", "status": "ready"},
            ("r1", 7, "m7"),
        ),
    ],
    ids=["target", "prior"],
)
def test_cleanup_reconciliation_disables_exact_owned_row(monkeypatch, current, expected):
    cleanup = {
        "target": {"revision": 8, "mutation_id": "m8"},
        "prior": {"revision": 7, "mutation_id": "m7"},
    }
    disabled = []
    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: current)
    monkeypatch.setattr(
        d,
        "disable_owned_adapter",
        lambda *args: disabled.append(args) or True,
    )

    assert d.reconcile_owned_adapter_cleanup("r1", cleanup) is True
    assert disabled == [expected]


@pytest.mark.parametrize(
    "current",
    [
        {"registry_revision": 8, "mutation_id": "m8", "status": "ready"},
        {"registry_revision": 7, "mutation_id": "m7", "status": "ready"},
    ],
    ids=["target", "prior"],
)
def test_cleanup_reconciliation_retains_ownership_after_transient_disable_failure(
    monkeypatch, current
):
    cleanup = {
        "target": {"revision": 8, "mutation_id": "m8"},
        "prior": {"revision": 7, "mutation_id": "m7"},
    }
    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: current)
    monkeypatch.setattr(
        d,
        "disable_owned_adapter",
        lambda *_args: (_ for _ in ()).throw(d.ServingError("delete readback unavailable")),
    )

    with pytest.raises(d.ServingError, match="readback unavailable"):
        d.reconcile_owned_adapter_cleanup("r1", cleanup)


@pytest.mark.parametrize(
    "current",
    [
        {"registry_revision": 9, "mutation_id": "m8", "status": "disabled"},
        {"registry_revision": 8, "mutation_id": "m7", "status": "disabled"},
    ],
    ids=["disabled-target", "disabled-prior"],
)
def test_cleanup_reconciliation_accepts_confirmed_owned_disable(monkeypatch, current):
    cleanup = {
        "target": {"revision": 8, "mutation_id": "m8"},
        "prior": {"revision": 7, "mutation_id": "m7"},
    }
    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: current)
    monkeypatch.setattr(
        d,
        "disable_owned_adapter",
        lambda *_args: pytest.fail("confirmed disable must not issue another delete"),
    )

    assert d.reconcile_owned_adapter_cleanup("r1", cleanup) is True


def test_cleanup_reconciliation_protects_true_forward_supersession(monkeypatch):
    cleanup = {
        "target": {"revision": 8, "mutation_id": "m8"},
        "prior": {"revision": 7, "mutation_id": "m7"},
    }
    newer = {"registry_revision": 9, "mutation_id": "m9", "status": "ready"}
    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: newer)
    monkeypatch.setattr(
        d,
        "disable_owned_adapter",
        lambda *_args: pytest.fail("newer deployment must not be disabled"),
    )

    assert d.reconcile_owned_adapter_cleanup("r1", cleanup) is True


def test_cleanup_reconciliation_rejects_unexpected_older_identity(monkeypatch):
    cleanup = {
        "target": {"revision": 8, "mutation_id": "m8"},
        "prior": {"revision": 7, "mutation_id": "m7"},
    }
    unexpected = {"registry_revision": 6, "mutation_id": "m6", "status": "ready"}
    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: unexpected)
    monkeypatch.setattr(
        d,
        "disable_owned_adapter",
        lambda *_args: pytest.fail("unexpected older deployment must not be disabled"),
    )

    with pytest.raises(d.ServingError, match="unexpected older"):
        d.reconcile_owned_adapter_cleanup("r1", cleanup)


def test_cleanup_reconciliation_rejects_nonpredecessor_prior(monkeypatch):
    cleanup = {
        "target": {"revision": 8, "mutation_id": "m8"},
        "prior": {"revision": 6, "mutation_id": "m6"},
    }
    monkeypatch.setattr(
        d,
        "read_adapter_record",
        lambda _run_id: pytest.fail("malformed cleanup must fail before registry access"),
    )

    with pytest.raises(d.ServingError, match="target predecessor"):
        d.reconcile_owned_adapter_cleanup("r1", cleanup)


def test_cleanup_reconciliation_rejects_malformed_current_identity(monkeypatch):
    cleanup = {
        "target": {"revision": 8, "mutation_id": "m8"},
        "prior": {"revision": 7, "mutation_id": "m7"},
    }
    monkeypatch.setattr(
        d,
        "read_adapter_record",
        lambda _run_id: {"registry_revision": 9, "status": "ready"},
    )
    monkeypatch.setattr(
        d,
        "disable_owned_adapter",
        lambda *_args: pytest.fail("malformed registry row must not be disabled"),
    )

    with pytest.raises(d.ServingError, match="malformed registry identity"):
        d.reconcile_owned_adapter_cleanup("r1", cleanup)


def test_redeploy_rejects_prior_without_exact_mutation_identity(monkeypatch, tmp_path):
    prior = {"registry_revision": 7, "org_id": ORG, "status": "ready"}
    _stub_artifact(monkeypatch, tmp_path)
    monkeypatch.setattr(d, "read_adapter_record", lambda _run_id: prior)
    monkeypatch.setattr(d, "_serving_request", lambda *_a, **_k: pytest.fail("must not post"))

    with pytest.raises(d.ServingError, match="prior registry mutation identity"):
        d.deploy_adapter(
            "r1",
            MODEL,
            "org/repo",
            "sft/r1",
            mutation_id=str(uuid4()),
            org_id=ORG,
        )


def test_smoke_chat_sends_complete_deployment_fence(monkeypatch):
    seen = {}

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, json=None, headers=None):
            seen.update(url=url, json=json, headers=headers)
            return Response(payload={"choices": []})

    monkeypatch.setattr(d.httpx, "Client", Client)
    d.chat(
        "r1",
        [{"role": "user", "content": "hi"}],
        expected_checkpoint="r1/step-2",
        expected_registry_revision=7,
        expected_mutation_id="m7",
    )
    assert seen["headers"]["X-Freesolo-Expected-Checkpoint"] == "r1/step-2"
    assert seen["headers"]["X-Freesolo-Expected-Registry-Revision"] == "7"
    assert seen["headers"]["X-Freesolo-Expected-Mutation-ID"] == "m7"


def test_serving_url_normalization(monkeypatch):
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
    assert d.serving_base_url() == "https://serve.example"
    assert d.serving_openai_base_url() == "https://serve.example/v1"
