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

    # Lambda is an OPT-IN instance-based complement (keyed by LAMBDA_API_KEY). On an operator box
    # whose shell sources a .env, that key is present in the process env — which would make the
    # provider "available" and pull live capacity/pricing into offline tests (allocation candidates,
    # registry). Delete it by default so the suite stays hermetic and RunPod-only; a provider test
    # opts back in with ``monkeypatch.setenv(...)``.
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)

    # The RunPod key pool caches the parsed RUNPOD_API_KEY at module level (so collapsing
    # it to a single active key never loses the rest of the pool). Reset it around every
    # test so a key set/collapsed by one test can't leak into the next.
    import flash.providers.runpod.keys as rp_keys

    rp_keys.reset()
    yield
    rp_keys.reset()
