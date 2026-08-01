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


@pytest.fixture(scope="session", autouse=True)
def _close_status_reporter_after_suite():
    yield
    import flash.runner as runner

    runner._shutdown_status_reporter(close=True)


@pytest.fixture(autouse=True)
def _reset_status_reporter_before_test():
    import flash.runner as runner

    runner._shutdown_status_reporter(close=True)
    runner._open_status_reporter()


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

    monkeypatch.setattr(runpod_api, "list_endpoints", list, raising=False)

    # Lambda and Vast are OPT-IN instance-based complements (keyed by LAMBDA_API_KEY / VAST_API_KEY).
    # On an operator box whose shell sources a .env, those keys are present in the process env — which
    # would make the provider "available" and pull live capacity/pricing into offline tests (allocation
    # candidates, registry). Delete them by default so the suite stays hermetic and RunPod-only; a
    # provider test opts back in with ``monkeypatch.setenv(...)``.
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)

    # Same hazard, one layer up: the client discovers runtime secrets from the process env at submit
    # time, so an operator shell that exports WANDB_API_KEY makes every dry-run test assert against
    # that real value instead of the fixture's. CI has no such key and passes; the box that does have
    # one fails, which reads as a broken test rather than a leaked env. Drive the scrub off the real
    # constant so a new default key cannot reintroduce the gap. A test that wants one sets it back.
    from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS

    for _key in DEFAULT_RUNTIME_SECRET_KEYS:
        monkeypatch.delenv(_key, raising=False)

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
    monkeypatch.setattr(_deploy, "ACTIVATION_READBACK_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(_deploy, "SMOKE_RETRY_FALLBACK_DELAY_SECONDS", 0.0)


@pytest.fixture
def stub_serving_registry(monkeypatch):
    """Patch GET /adapters to return the given records (deploy read-back verification)."""

    def _stub(*records: dict):
        import flash.serve.deploy as _deploy

        class _RegistryResp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"ok": True, "adapters": list(records)}

        monkeypatch.setattr(_deploy.httpx, "get", lambda *a, **k: _RegistryResp())

    return _stub
