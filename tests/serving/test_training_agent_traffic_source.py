"""The training agent's own model traffic is recorded as ``training_agent`` and settles at $0.

Only two paths may mark it: the backend's authorization of a live training job's base-model
sampling (``trafficSource`` in the authorize response) and the Flash control plane forwarding an
internal-key chat that carries a training-job scope. Everything else, including an unknown or
spoofed value, stays ordinary billable ``freesolo`` traffic.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from flash.serve.contract.protocol import TRAFFIC_SOURCE_HEADER, TRAINING_AGENT_TRAFFIC_SOURCE
from flash.serve.request import transport
from flash.server.routes import serving_chat
from flash.serving.app.modal_app import _authorize_response
from flash.serving.src.accounting.usage import (
    AuthorizedTraffic,
    TrustedInternalAuthorization,
    principal_for_external_org,
)
from flash.serving.src.accounting.usage_outbox import (
    FREESOLO_TRAFFIC_SOURCE,
    traffic_source_or_default,
)
from flash.serving.src.http.context import require_attributed_traffic
from tests.serving.test_rate_limit import BASE, ORG
from tests.serving.test_usage_outbox import _usage_event


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("training_agent", "training_agent"),
        (" Training_Agent ", "training_agent"),
        ("freesolo", "freesolo"),
        ("", "freesolo"),
        (None, "freesolo"),
        ("free", "freesolo"),
        (7, "freesolo"),
    ],
)
def test_unknown_traffic_sources_bill_as_usual(value, expected):
    assert traffic_source_or_default(value) == expected


def test_authorize_response_carries_the_training_agent_marker():
    marked = _authorize_response(
        httpx.Response(200, json={"orgId": ORG, "trafficSource": "training_agent"}), BASE
    )
    assert marked.traffic_source == TRAINING_AGENT_TRAFFIC_SOURCE
    plain = _authorize_response(httpx.Response(200, json={"orgId": ORG}), BASE)
    assert plain.traffic_source == FREESOLO_TRAFFIC_SOURCE
    spoofed = _authorize_response(
        httpx.Response(200, json={"orgId": ORG, "trafficSource": "free"}), BASE
    )
    assert spoofed.traffic_source == FREESOLO_TRAFFIC_SOURCE


def test_trusted_internal_attribution_keeps_its_traffic_source():
    target = SimpleNamespace(org_id=ORG, is_checkpoint=False, deployment_generation=None)
    traffic = require_attributed_traffic(
        TrustedInternalAuthorization(org_id=ORG, traffic_source=TRAINING_AGENT_TRAFFIC_SOURCE),
        target,
    )
    assert traffic.traffic_source == TRAINING_AGENT_TRAFFIC_SOURCE
    assert traffic.principal.orgId == ORG


def test_outbox_payload_records_the_traffic_source():
    event = _usage_event()
    assert event.rpc_payload()["traffic_source"] == "freesolo"
    agent = replace(event, traffic_source=TRAINING_AGENT_TRAFFIC_SOURCE)
    assert agent.rpc_payload()["traffic_source"] == "training_agent"


def test_internal_key_header_names_the_traffic_source(monkeypatch):
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal")
    assert TRAFFIC_SOURCE_HEADER not in transport._internal_key_header(org_id=ORG)
    headers = transport._internal_key_header(org_id=ORG, traffic_source="training_agent")
    assert headers[TRAFFIC_SOURCE_HEADER] == "training_agent"


def _request(headers: dict[str, str]):
    return SimpleNamespace(headers=httpx.Headers(headers))


def test_control_plane_marks_only_the_training_agents_internal_chats():
    job = {"X-Freesolo-Training-Job-Id": "job-1"}
    internal = {"auth_kind": "internal"}
    assert serving_chat._traffic_source(_request(job), internal) == "training_agent"
    # the platform web proxy acting for a signed-in user has no job scope: billable.
    assert serving_chat._traffic_source(_request({}), internal) is None
    # a customer key can never mark its own traffic free, job header or not.
    assert serving_chat._traffic_source(_request(job), {"auth_kind": "user"}) is None
    assert serving_chat._traffic_source(_request(job), {}) is None


def test_authorized_traffic_defaults_to_billable():
    assert AuthorizedTraffic(principal_for_external_org(ORG)).traffic_source == "freesolo"
