"""Dynamic per-region health quarantine for the instance providers (flash.providers._health):
the TTL state machine, the usable_instances() region filter that consumes it, and that a host-fault
poll result quarantines the region so the next allocation/launch avoids it."""

from __future__ import annotations

import flash.providers._health as health


def test_mark_and_expire():
    health.clear()
    # Not sick until marked; sick within the window; self-heals (popped) once the TTL lapses.
    assert not health.region_is_sick("lambda", "us-east-1", now=1000.0)
    health.mark_region_sick("lambda", "us-east-1", ttl_s=100.0, now=1000.0)
    assert health.region_is_sick("lambda", "us-east-1", now=1050.0)
    assert health.region_is_sick("lambda", "us-east-1", now=1099.9)
    assert not health.region_is_sick("lambda", "us-east-1", now=1100.0)  # expired -> healed
    # and the expired entry was popped (not lingering)
    assert ("lambda", "US-EAST-1") not in health.sick_regions(now=1100.0)


def test_provider_and_case_scoping():
    health.clear()
    health.mark_region_sick("lambda", "us-east-1", ttl_s=100.0, now=0.0)
    # case-insensitive region match
    assert health.region_is_sick("lambda", "US-EAST-1", now=10.0)
    # a different provider with the same region name is NOT affected
    assert not health.region_is_sick("hyperstack", "us-east-1", now=10.0)
    # blank region is never sick / never marks
    assert not health.region_is_sick("lambda", "", now=10.0)
    health.mark_region_sick("lambda", None, ttl_s=100.0, now=0.0)  # no-op, no crash


def test_healthy_regions_filters_in_order():
    health.clear()
    health.mark_region_sick("hyperstack", "CANADA-1", ttl_s=100.0, now=0.0)
    regions = ["NORWAY-1", "CANADA-1", "US-1"]
    assert health.healthy_regions("hyperstack", regions, now=10.0) == ["NORWAY-1", "US-1"]


def test_healthy_regions_ignore_sick_keeps_quarantined():
    """ignore_sick=True is the allocator's last-resort pass: it returns every region unfiltered (order
    preserved) even when quarantined, so the quarantine can only DEMOTE a region, never zero out the
    last candidate and hard-fail a run (the module's bounded-demotion contract)."""
    health.clear()
    health.mark_region_sick("hyperstack", "CANADA-1", ttl_s=100.0, now=0.0)
    regions = ["NORWAY-1", "CANADA-1", "US-1"]
    # normal filtering drops the sick region; the last-resort pass keeps it (all regions, in order).
    assert health.healthy_regions("hyperstack", regions, now=10.0) == ["NORWAY-1", "US-1"]
    assert health.healthy_regions("hyperstack", regions, now=10.0, ignore_sick=True) == regions


def test_ttl_zero_disables_quarantine(monkeypatch):
    health.clear()
    monkeypatch.setenv("FLASH_REGION_SICK_TTL_S", "0")
    health.mark_region_sick("lambda", "us-east-1")  # ttl resolves to 0 -> no-op
    assert not health.region_is_sick("lambda", "us-east-1")
    assert health.sick_regions() == {}


def test_ttl_env_override(monkeypatch):
    monkeypatch.setenv("FLASH_REGION_SICK_TTL_S", "42")
    assert health.sick_ttl_s() == 42.0
    monkeypatch.setenv("FLASH_REGION_SICK_TTL_S", "garbage")
    assert health.sick_ttl_s() == health._DEFAULT_SICK_TTL_S


def test_lambda_usable_instances_drops_sick_region(monkeypatch):
    """The Lambda allocator/launch capacity view excludes a quarantined region."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs, pricing

    health.clear()
    monkeypatch.setattr(lambda_api, "regions_with_capacity", lambda itype, force=False: ["us-east-1", "us-west-2"])
    monkeypatch.setattr(pricing, "hourly_rate", lambda gpu: 1.29)

    assert {i.region for i in jobs.usable_instances("A10")} == {"us-east-1", "us-west-2"}
    health.mark_region_sick("lambda", "us-east-1")
    assert {i.region for i in jobs.usable_instances("A10")} == {"us-west-2"}


def test_hyperstack_usable_instances_drops_sick_region(monkeypatch):
    """The Hyperstack capacity view excludes a quarantined region (on top of the static denylist)."""
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    health.clear()
    monkeypatch.setattr(hs_api, "regions_with_stock", lambda flavor, force=False: ["NORWAY-1", "US-1"])
    monkeypatch.setattr(hs_api, "environment_for_region", lambda r: f"default-{r}")

    assert {i.region for i in jobs.usable_instances("RTX A6000")} == {"NORWAY-1", "US-1"}
    health.mark_region_sick("hyperstack", "NORWAY-1")
    assert {i.region for i in jobs.usable_instances("RTX A6000")} == {"US-1"}
