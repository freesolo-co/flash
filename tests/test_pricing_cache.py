"""Live-pricing cache behavior: TTL reuse, disk cache, and offline fallbacks."""

from __future__ import annotations

import importlib
import json


def _fresh_pricing(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOSLM_SKIP_NET", raising=False)
    import autoslm.providers.runpod.pricing as pricing

    pricing = importlib.reload(pricing)
    monkeypatch.setattr(pricing, "_CACHE_PATH", tmp_path / "rates.json")
    pricing._MEM.update(ts=0.0, rates={})
    return pricing


def test_skip_net_returns_static_rates(monkeypatch, tmp_path) -> None:
    pricing = _fresh_pricing(monkeypatch, tmp_path)
    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    rates = pricing.live_rates()
    assert rates["RTX 4090"] > 0
    assert pricing.hourly_rate("RTX 4090") == rates["RTX 4090"]


def test_fetch_failure_falls_back_to_static(monkeypatch, tmp_path) -> None:
    pricing = _fresh_pricing(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pricing, "_fetch_live_rates", lambda: (_ for _ in ()).throw(RuntimeError("net down"))
    )
    rates = pricing.live_rates(refresh=True)
    assert rates["RTX 4090"] > 0  # static snapshot still answers


def test_fetched_rates_are_cached_in_memory_and_disk(monkeypatch, tmp_path) -> None:
    pricing = _fresh_pricing(monkeypatch, tmp_path)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return {"RTX 4090": 0.123}

    monkeypatch.setattr(pricing, "_fetch_live_rates", fetch)
    assert pricing.live_rates(refresh=True)["RTX 4090"] == 0.123
    # Second call inside the TTL must not refetch.
    assert pricing.live_rates()["RTX 4090"] == 0.123
    assert calls["n"] == 1
    # The disk cache holds the fetched rates for fresh processes.
    disk = json.loads((tmp_path / "rates.json").read_text())
    assert disk["rates"]["RTX 4090"] == 0.123


def test_disk_cache_warms_a_cold_process(monkeypatch, tmp_path) -> None:
    pricing = _fresh_pricing(monkeypatch, tmp_path)
    import time

    (tmp_path / "rates.json").write_text(
        json.dumps({"ts": time.time(), "rates": {"RTX 4090": 0.456}})
    )
    monkeypatch.setattr(
        pricing, "_fetch_live_rates", lambda: (_ for _ in ()).throw(AssertionError("no fetch"))
    )
    assert pricing.live_rates()["RTX 4090"] == 0.456
