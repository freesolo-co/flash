"""Offline-by-default test harness.

There is no ``FLASH_SKIP_NET`` flag anymore; instead this autouse fixture stubs the network
boundaries the production code would otherwise reach, so the whole suite stays hermetic (no
real RunPod / Vast / Hugging Face calls) without any env switch. A test that exercises one of
these boundaries monkeypatches it itself — applied after this fixture, so the test's patch
wins. (The freesolo auth verify isn't stubbed here because tests that touch it either patch
``_freesolo_verify`` or ``urllib.request.urlopen`` directly; a global ``urlopen`` stub would
also break the client tests, which talk to a real loopback server.)
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    # No live provider key in the suite: Vast stays unconfigured (RunPod-only allocation),
    # and the Vast pricing/offer paths short-circuit to their static snapshot.
    monkeypatch.delenv("VAST_API_KEY", raising=False)

    # RunPod live pricing -> static snapshot (no HTTP): stub the fetch, reset the in-memory
    # cache, and point the disk cache at a non-existent tmp file so a dev machine's cached
    # live rates can't leak into a test.
    import flash.providers.runpod.pricing as runpod_pricing

    monkeypatch.setattr(runpod_pricing, "_fetch_live_rates", lambda: {}, raising=False)
    monkeypatch.setattr(runpod_pricing, "_CACHE_PATH", tmp_path / "runpod_rates.json", raising=False)
    runpod_pricing._MEM.update(ts=0.0, rates={})

    # HF param probe -> offline: sizing for an unlisted model falls back to the heuristic
    # (24 GB tier) instead of hitting the Hugging Face API.
    import flash.engine.vram as vram

    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda model_id: None, raising=False)
    return
