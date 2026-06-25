"""The control-plane idle-endpoint reaper: it must protect every live run's endpoint and pass the
run-aware protected set + idle grace down to the provider sweep. (Server-side; no fastapi needed.)
"""

from __future__ import annotations

import flash.providers.runpod.jobs as jobs
import flash.server.app as app_mod
from flash.providers.base import canonical_gpu
from flash.providers.runpod.train import _run_suffix, endpoint_name
from flash.runner import RunStatus


def _derived(gpu: str, run_id: str) -> str:
    return endpoint_name(canonical_gpu(gpu), _run_suffix(run_id))


def test_protected_names_cover_live_runs_only(monkeypatch):
    rows = [{"run_id": r} for r in ("flash-active", "flash-prov", "flash-done")]
    statuses = {
        # running run with a persisted handle + a spec gpu
        "flash-active": RunStatus(
            run_id="flash-active",
            state="running",
            spec={"gpu": {"type": "RTX 5090"}},
            remote={"endpoint_name": "flash-5090-handle", "endpoint_id": "e", "job_id": "j"},
        ),
        # provisioning run with no handle yet (submit -> handle-persist window)
        "flash-prov": RunStatus(
            run_id="flash-prov", state="provisioning", spec={"gpu": {"type": "A100"}}
        ),
        # terminal run: must NOT be protected
        "flash-done": RunStatus(
            run_id="flash-done", state="done", spec={"gpu": {"type": "RTX 5090"}}
        ),
    }
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: rows)
    monkeypatch.setattr(app_mod, "get_status", lambda rid: statuses[rid])

    names = app_mod._protected_train_endpoint_names()

    # active run: the persisted handle name AND the spec-derived name, both registered forms.
    assert {"flash-5090-handle", "live-flash-5090-handle"} <= names
    active_derived = _derived("RTX 5090", "flash-active")
    assert {active_derived, f"live-{active_derived}"} <= names
    # provisioning run: protected by its spec-derived name even with no handle.
    assert _derived("A100", "flash-prov") in names
    # terminal run: not protected (its endpoint is reapable once idle).
    assert _derived("RTX 5090", "flash-done") not in names


def test_reap_once_passes_protected_set_and_grace(monkeypatch):
    monkeypatch.setattr(app_mod, "_protected_train_endpoint_names", lambda: {"flash-live"})
    captured: dict = {}

    def fake_sweep(protected, min_idle_s=0.0):
        captured["protected"] = protected
        captured["grace"] = min_idle_s
        return 3

    monkeypatch.setattr(jobs, "_sweep_idle_flash_endpoints", fake_sweep)
    assert app_mod._reap_idle_endpoints_once(900.0) == 3
    assert captured == {"protected": {"flash-live"}, "grace": 900.0}


def test_protected_names_skip_unreadable_run(monkeypatch):
    # A run row whose status file vanished (FileNotFoundError) is skipped, not fatal.
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "gone"}, {"run_id": "live"}])

    def get_status(rid):
        if rid == "gone":
            raise FileNotFoundError(rid)
        return RunStatus(run_id="live", state="running", spec={"gpu": {"type": "A100"}})

    monkeypatch.setattr(app_mod, "get_status", get_status)
    names = app_mod._protected_train_endpoint_names()
    assert _derived("A100", "live") in names


# --- the in-lifetime instance orphan sweep (Lambda/Hyperstack) ---------------------------------


class _FakeProvider:
    """A configured provider whose ``sweep_orphans`` records the active set it was handed and
    returns a fixed list of torn-down ids (or raises, to model an API blip)."""

    def __init__(self, name, torn=(), raises=False):
        self.name = name
        self._torn = list(torn)
        self._raises = raises
        self.seen_active = None

    def sweep_orphans(self, active_labels=None):
        if self._raises:
            raise RuntimeError(f"{self.name} api blip")
        self.seen_active = active_labels
        return self._torn


def test_active_run_ids_covers_live_runs_only(monkeypatch):
    rows = [{"run_id": r} for r in ("flash-run", "flash-prov", "flash-done", "flash-failed")]
    statuses = {
        "flash-run": RunStatus(run_id="flash-run", state="running", spec={}),
        # provisioning: no handle yet, but its instance may already be launching -> must be protected
        "flash-prov": RunStatus(run_id="flash-prov", state="provisioning", spec={}),
        "flash-done": RunStatus(run_id="flash-done", state="done", spec={}),
        "flash-failed": RunStatus(run_id="flash-failed", state="failed", spec={}),
    }
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: rows)
    monkeypatch.setattr(app_mod, "get_status", lambda rid: statuses[rid])

    assert app_mod._active_run_ids() == {"flash-run", "flash-prov"}


def test_active_run_ids_skips_unreadable_run(monkeypatch):
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "gone"}, {"run_id": "live"}])

    def get_status(rid):
        if rid == "gone":
            raise FileNotFoundError(rid)
        return RunStatus(run_id="live", state="running", spec={})

    monkeypatch.setattr(app_mod, "get_status", get_status)
    assert app_mod._active_run_ids() == {"live"}


def test_sweep_instances_dispatches_active_set_and_sums(monkeypatch):
    monkeypatch.setattr(app_mod, "_active_run_ids", lambda: {"flash-live"})
    lam = _FakeProvider("lambda", torn=["i-1", "i-2"])
    hyp = _FakeProvider("hyperstack", torn=["vm-9"])
    rp = _FakeProvider("runpod", torn=[])  # no-op for RunPod, still dispatched
    monkeypatch.setattr(
        "flash.providers.configured_providers", lambda: [rp, lam, hyp], raising=False
    )

    # 2 lambda + 1 hyperstack + 0 runpod torn down.
    assert app_mod._sweep_orphan_instances_once() == 3
    # Every provider got the SAME live-run protection set.
    assert lam.seen_active == {"flash-live"}
    assert hyp.seen_active == {"flash-live"}
    assert rp.seen_active == {"flash-live"}


def test_sweep_instances_one_provider_blip_does_not_skip_others(monkeypatch):
    monkeypatch.setattr(app_mod, "_active_run_ids", lambda: set())
    boom = _FakeProvider("lambda", raises=True)
    ok = _FakeProvider("hyperstack", torn=["vm-1", "vm-2"])
    monkeypatch.setattr(
        "flash.providers.configured_providers", lambda: [boom, ok], raising=False
    )

    # The raising provider is swallowed; the other still reaps.
    assert app_mod._sweep_orphan_instances_once() == 2


def test_instance_providers_configured_gating(monkeypatch):
    monkeypatch.setattr(
        "flash.providers.available_providers", lambda: ("runpod",), raising=False
    )
    assert app_mod._instance_providers_configured() is False

    monkeypatch.setattr(
        "flash.providers.available_providers", lambda: ("runpod", "lambda"), raising=False
    )
    assert app_mod._instance_providers_configured() is True

    monkeypatch.setattr(
        "flash.providers.available_providers", lambda: ("hyperstack",), raising=False
    )
    assert app_mod._instance_providers_configured() is True
