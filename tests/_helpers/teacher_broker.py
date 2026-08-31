"""Shared ledger fixture and request builders for the managed-teacher broker suites.

The broker contracts are split across two modules -- capability, quota, and worker behaviour in
``test_teacher_broker``, startup and turnover ownership in ``test_teacher_broker_ownership`` -- and
both seed the same ledger, mint the same capability, and build the same completion payloads. They
live here so one definition serves both rather than drifting as two near-identical copies.

They stay out of ``_helpers.teacher``: that module is imported by suites which never touch the
control plane, and it must not pull the server package in behind them.

Names keep their original leading underscore so neither suite needs a call-site rewrite to adopt
them; this whole package is private to the test tree.
"""

from __future__ import annotations

import json
import time

import pytest

from flash.server.domain.teacher import broker as teacher_broker
from flash.server.platform import db

# not a credential: the autouse scrub in conftest guarantees no operator key is present, and the
# broker only checks that the name is set and non-blank.
PLANE_API_KEY = "control-plane-only-canary"


@pytest.fixture
def broker_db(monkeypatch, tmp_path):
    """Point the plane at an empty ledger owning a single run."""
    path = tmp_path / "server.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    owner = db.ensure_internal_key("test-owner-key")
    db.record_run("run-1", owner["id"])
    return path


def _limits(**updates):
    values = {
        "max_requests": 4,
        "max_score_items": 8,
        "max_request_bytes": teacher_broker.MAX_REQUEST_BODY_BYTES,
        "max_response_bytes": teacher_broker.MAX_RESPONSE_BODY_BYTES,
        "max_concurrency": 2,
        "max_upstream_attempts": 1,
        "max_request_tokens": 128,
        "max_total_tokens": 512,
    }
    values.update(updates)
    return values


def _issue(*, now=None, expires_at=None, limits=None):
    current = time.time() if now is None else float(now)
    return db.issue_teacher_capability(
        run_id="run-1",
        attempt=2,
        teacher_alias="glm-5.2",
        provider=teacher_broker.PARASAIL_PROVIDER,
        model="parasail-glm-52",
        scoring_mode=teacher_broker.PARASAIL_SCORING_MODE,
        expires_at=expires_at if expires_at is not None else current + 600,
        limits=limits or _limits(),
        now=current,
    )


def _body(*, prompt="questionanswer", model="parasail-glm-52", **extra):
    value = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "echo": True,
        "logprobs": 1,
        "prompt_logprobs": 1,
        "return_token_ids": True,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
    }
    value.update(extra)
    return json.dumps(value, separators=(",", ":")).encode()


def _response(*, token="questionanswer"):
    return json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "prompt_token_ids": [7],
                    "token_ids": [8],
                    "prompt_logprobs": [{"7": {"logprob": -0.1, "decoded_token": token}}],
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()
