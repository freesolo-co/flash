"""Cost estimator: the ``CostEstimate`` result type (breakdown, provider). No network."""

from __future__ import annotations

import dataclasses

import pytest

from flash.cost import RunConfig, estimate_cost
from flash.cost.types import CostEstimate


@pytest.fixture
def est() -> CostEstimate:
    return estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 150))


def test_is_frozen(est):
    with pytest.raises(dataclasses.FrozenInstanceError):
        est.total_usd = 0.0  # type: ignore[misc]


def test_wall_clock_hours_derivation(est):
    assert est.wall_clock_hours == pytest.approx(est.wall_clock_seconds / 3600.0)


def test_billable_hours_derivation(est):
    assert est.billable_hours == pytest.approx(est.train_seconds / 3600.0)


def test_breakdown_lists_every_term(est):
    b = est.breakdown()
    for needle in ("GPU", "Setup", "Per step", "Train", "Wall clock", "Billable", "TOTAL"):
        assert needle in b
    assert "not billed" in b
    # GRPO estimate carries explanatory notes.
    assert "Notes" in b


def test_capped_estimate_flags_in_breakdown():
    capped = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 100_000))
    assert capped.wall_capped
    assert "CAPPED" in capped.breakdown()


def test_subhour_cap_note_renders_minutes_not_zero_hours():
    # A sub-hour wall cap (floored to 60s) must render the CAPPED duration as "1m", never a
    # confusing "0h". (The note also reports the uncapped duration, which is many hours -- so we
    # assert the cap SLOT specifically rather than scanning the whole note for "0h".)
    capped = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 100_000, max_wall_seconds=60))
    assert capped.wall_capped
    cap_note = next(n for n in capped.notes if "wall cap" in n)
    assert "fit the 1m " in cap_note  # 60s -> "1m", not "0h"


def test_fmt_duration_units():
    from flash.cost.analytical import _fmt_duration

    assert _fmt_duration(20) == "20s"  # sub-minute -> seconds, never "0m"
    assert _fmt_duration(59) == "59s"
    assert _fmt_duration(60) == "1m"  # sub-hour -> minutes, never "0h"
    assert _fmt_duration(1800) == "30m"
    assert _fmt_duration(24 * 3600) == "24h"  # whole hours stay clean
    assert _fmt_duration(int(1.5 * 3600)) == "1.5h"  # fractional multi-hour -> one decimal


def test_provider_is_normalized_and_validated():
    # Case/whitespace variants normalize to the canonical substrate; empty -> "auto".
    assert RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="RunPod").provider == "runpod"
    assert RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="").provider == "auto"
    assert RunConfig("Qwen/Qwen3.5-4B", "grpo", 10).provider == "auto"
    # An unknown substrate fails fast here (clear error) instead of as "no GPU class fits".
    with pytest.raises(ValueError, match="unknown provider"):
        RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="aws")


def test_estimate_reports_the_runs_provider():
    # Provider is reported as configured: the default is "auto", and an explicit substrate is
    # passed through unchanged.
    assert estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 10)).provider == "auto"
    assert (
        estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="runpod")).provider
        == "runpod"
    )


def test_estimate_cost_vast_market_duration_mirrors_launch(monkeypatch):
    # Cursor MuXiS: the Vast market duration filter used for pricing/GPU-selection must mirror
    # usable_offers at LAUNCH, NOT the 60s-floored billing cap. Launch passes
    # `spec.gpu.max_wall_seconds or 0.0` and treats a NON-POSITIVE wall as "no duration filter"; a
    # positive one is floored at 60s inside usable_offers. So an explicit 0 must price with NO filter
    # (0.0), not 60. Capture what reaches usable_offers for explicit-0 vs a positive wall.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast import pricing

    seen: list[float] = []

    def fake_usable(min_vram_gb, disk_gb, *a, max_wall_seconds=0, **k):
        seen.append(max_wall_seconds)
        return []

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)

    def market_walls(cfg) -> set[float]:
        seen.clear()
        monkeypatch.setattr(pricing, "_rates_cache", {"ts": 0.0, "data": None})  # isolate cache
        estimate_cost(cfg)
        return set(seen)

    # explicit 0 -> usable_offers gets 0.0 (NO duration filter), NOT the 60s-floored cap
    assert market_walls(
        RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="vast", max_wall_seconds=0)
    ) == {0.0}
    # a positive wall -> that wall reaches the market query (usable_offers floors at 60s itself)
    assert market_walls(
        RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="vast", max_wall_seconds=7200)
    ) == {7200.0}


def test_pick_gpu_vast_duration_bound_fetches_market_once(monkeypatch):
    # Copilot: pick_gpu ranks every fitting class by $/hr. A duration-bound Vast query bypasses the
    # per-call rate cache (Codex MtzrI), so pricing each candidate individually inside min(key=...)
    # would fire one identical full market fetch PER fitting class. pick_gpu must fetch the live rate
    # map ONCE and rank from it -> exactly one usable_offers call no matter how many classes fit.
    from flash.cost.facts import pick_gpu
    from flash.providers.vast import jobs as vast
    from flash.providers.vast import pricing

    calls = {"n": 0}

    def fake_usable(min_vram_gb, disk_gb, *a, max_wall_seconds=0, **k):
        calls["n"] += 1
        return []

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    monkeypatch.setattr(pricing, "_rates_cache", {"ts": 0.0, "data": None})  # isolate cache

    # A tiny VRAM floor leaves MANY fitting Vast classes -> per-candidate pricing would fetch many times.
    gpu = pick_gpu(8, provider="vast", max_wall_seconds=7200.0)
    assert gpu  # a class was chosen
    assert calls["n"] == 1  # ONE market fetch despite multiple fitting candidates (was N)


def test_pick_gpu_vast_skips_classes_without_a_live_offer(monkeypatch):
    # Codex: ranking via the static-merged map could SELECT and quote a cheaper class (e.g. RTX 4090)
    # that has NO surviving offer under the wall cap — one the launch-time usable_offers path would
    # never rent. pick_gpu(provider="vast") must restrict selection to classes that ACTUALLY have a
    # rentable offer, even when a cheaper class fits and is cheaper on its static (RunPod) rate.
    from types import SimpleNamespace

    from flash.cost.facts import pick_gpu
    from flash.providers.vast import jobs as vast

    # The live market has ONLY an A100 SXM offer (a larger/pricier class); the cheaper 24/48 GB classes
    # fit the 8 GB requirement and are cheaper statically, but have NO surviving offer.
    def fake_usable(min_vram_gb, disk_gb, *a, max_wall_seconds=0, **k):
        return [SimpleNamespace(gpu="A100 SXM", dph_total=1.20)]

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    gpu = pick_gpu(8, provider="vast", max_wall_seconds=7200.0)
    assert gpu == "A100 SXM"  # the only class with a rentable offer, NOT the cheaper-static 4090


def test_pick_gpu_vast_floors_market_search_at_required_vram(monkeypatch):
    # Codex: the live Vast offer map for a HIGH-VRAM job must be searched at the job's required VRAM, not
    # the smallest managed class. The market page is price-sorted + LIMITED, so a small-class floor lets
    # cheap 24-40 GB offers crowd the 80 GB classes off it -> they'd be omitted and the quote would fall
    # back to static pricing. pick_gpu must thread required_vram_gb into the search floor (allocator parity).
    from types import SimpleNamespace

    from flash.cost.facts import pick_gpu
    from flash.providers.vast import jobs as vast

    seen = {}

    def fake_usable(min_vram_gb, disk_gb, *a, max_wall_seconds=0, **k):
        seen["floor"] = min_vram_gb
        return [SimpleNamespace(gpu="A100 SXM", dph_total=1.20)]

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    pick_gpu(80, provider="vast", max_wall_seconds=7200.0)
    # Old behavior floored at the smallest managed class (~24 GB); the fix floors at the required 80 GB.
    assert seen["floor"] == 80


def test_a100_sxm_40gb_has_real_tflops_not_default():
    # Codex: the 40 GB A100 SXM4 class (selectable for Lambda/Vast) must carry a real TFLOPS entry, or
    # gpu_tflops() falls to the 100-default and inflates its seconds_per_step / quoted cost ~3x. It has
    # the same SMs as the 80 GB SXM, so its compute equals the other A100 entries.
    from flash.cost.facts import _DEFAULT_TFLOPS, gpu_tflops

    assert gpu_tflops("A100 SXM 40GB") == gpu_tflops("A100 SXM")
    assert gpu_tflops("A100 SXM 40GB") != _DEFAULT_TFLOPS


def test_pick_gpu_vast_offline_falls_back_to_static(monkeypatch):
    # When the market is unreachable (no VAST_API_KEY -> live_offer_rates returns {}), selection must
    # stay offline-safe: rank ALL fitting classes by their static rate rather than crash or pick nothing.
    from flash.cost.facts import pick_gpu

    monkeypatch.delenv("VAST_API_KEY", raising=False)
    gpu = pick_gpu(8, provider="vast")
    assert gpu  # a fitting class is still chosen from the static fallback
