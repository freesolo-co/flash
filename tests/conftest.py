"""Offline-by-default test harness.

There is no "skip the network" env flag; instead this autouse fixture stubs the network
boundaries the production code would otherwise reach, so the whole suite stays hermetic (no
real RunPod / Hugging Face calls) without any env switch. A test that exercises one of
these boundaries monkeypatches it itself — applied after this fixture, so the test's patch
wins. (The freesolo auth verify isn't stubbed here because tests that touch it either patch
``_freesolo_verify`` or ``urllib.request.urlopen`` directly; a global ``urlopen`` stub would
also break the client tests, which talk to a real loopback server.)
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # HF param probe -> offline: sizing for an unlisted model falls back to the heuristic
    # (24 GB tier) instead of hitting the Hugging Face API.
    import flash.engine.vram as vram

    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda model_id: None, raising=False)

    # RunPod endpoint listing -> offline: the idle-endpoint sweep (deploy-time quota reclaim
    # and the startup/post-run orphan sweep) lists account endpoints. Default to "no endpoints"
    # so a sweep never reaches the real API; sweep tests monkeypatch it after this fixture.
    import flash.providers.runpod.api as runpod_api

    monkeypatch.setattr(runpod_api, "list_endpoints", lambda: [], raising=False)

    # Lambda and Vast are OPT-IN instance-based complements (keyed by LAMBDA_API_KEY / VAST_API_KEY).
    # On an operator box whose shell sources a .env, those keys are present in the process env — which
    # would make the provider "available" and pull live capacity/pricing into offline tests (allocation
    # candidates, registry). Delete them by default so the suite stays hermetic and RunPod-only; a
    # provider test opts back in with ``monkeypatch.setenv(...)``.
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)

    # The RunPod key pool caches the parsed RUNPOD_API_KEY at module level (so collapsing
    # it to a single active key never loses the rest of the pool). Reset it around every
    # test so a key set/collapsed by one test can't leak into the next.
    import flash.providers.runpod.keys as rp_keys

    rp_keys.reset()

    # Always-on artifact GC: the control-plane lifespan sweeps ONCE on startup (when an operator
    # HF_TOKEN is set). Stub it to a no-op so offline TestClient startups never reach HF/serving;
    # tests/test_repo_cleanup.py restores the real function to exercise the genuine sweep.
    import flash.server.repo_cleanup as _rc

    monkeypatch.setattr(_rc, "run_scheduled_cleanup", lambda *a, **k: 0, raising=False)

    yield
    rp_keys.reset()


@pytest.fixture(autouse=True)
def _fast_serving_readback(monkeypatch):
    """Zero the deploy read-back backoff so verification polls don't slow the suite."""
    import flash.serve.deploy as _deploy

    monkeypatch.setattr(_deploy, "READBACK_DELAY_SECONDS", 0.0)


@pytest.fixture
def stub_serving_registry(monkeypatch):
    """Model the revisioned privileged registry used by deployment verification.

    Each positional record is the expected post-mutation adapter state, in mutation
    order. The stub starts empty (GET 404); every _etag_revision call commits the
    next mutation, so snapshot reads return the latest committed record at its
    monotonic revision. Repeated deploys of the same adapter therefore see the
    prior revision on the pre-mutation read and the next revision on readback,
    matching the serving compare-and-swap contract.
    """

    def _stub(*records: dict):
        import flash.serve.deploy as _deploy

        normalized = [
            {
                "repo_id": "org/repo",
                "base_model": "Qwen/Qwen3.5-0.8B",
                "repo_type": "dataset",
                "status": "ready",
                "thinking": False,
                **record,
            }
            for record in records
        ]
        state = {"committed": 0}

        class _RegistryResp:
            def __init__(self, status_code: int, payload: dict | None = None):
                self.status_code = status_code
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                assert self._payload is not None
                return self._payload

        def fake_get(url, **_kwargs):
            if state["committed"] == 0 or not normalized:
                return _RegistryResp(404)
            record = normalized[min(state["committed"], len(normalized)) - 1]
            return _RegistryResp(
                200,
                {
                    "adapter": record,
                    "org_id": record.get("org_id"),
                    "revision": state["committed"],
                },
            )

        def fake_etag_revision(_response):
            state["committed"] += 1
            return state["committed"]

        monkeypatch.setattr(_deploy.httpx, "get", fake_get)
        monkeypatch.setattr(_deploy, "_etag_revision", fake_etag_revision)

    return _stub
