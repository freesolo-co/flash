"""LIVE E2E (rents real GPUs — costs money): tiny SFT+eval on Vast verified datacenters.

For each target class with available offers: rent the cheapest verified-DC instance,
run Qwen3-0.6B SFT for a handful of steps + a 20-example eval through the REAL
worker path (HF code delivery, two-phase bootstrap, artifact polling), then verify
metrics, stamped cost, and that the instance is destroyed.

Gated hard: needs AUTOSLM_LIVE_VAST=1 + VAST_API_KEY + HUGGINGFACE_TOKEN. Each class
costs roughly $0.05-0.30 of instance time. HF_REPO defaults to a private
<whoami>/autoslm-vast-smoke dataset repo when unset.

    AUTOSLM_LIVE_VAST=1 uv run pytest tests/live/test_vast_live_smoke.py -q -s
"""

from __future__ import annotations

import os
import sys
import time

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("AUTOSLM_LIVE_VAST") != "1"
        or not os.environ.get("VAST_API_KEY")
        or not os.environ.get("HUGGINGFACE_TOKEN")
        or bool(os.environ.get("AUTOSLM_SKIP_NET")),
        reason="live e2e needs AUTOSLM_LIVE_VAST=1 + VAST_API_KEY + HUGGINGFACE_TOKEN + network",
    ),
]

# Cheap, sub-8B-relevant classes spanning Ampere consumer/workstation, Ada
# datacenter, big-VRAM Ampere, and Blackwell consumer/workstation (the Pro 4000
# also exercises the CUDA-13 image fork). Classes without live offers skip (the
# market moves).
SMOKE_CLASSES = ("RTX 3090", "L4", "A40", "RTX 5090", "RTX A5000", "RTX Pro 4000")
MAX_SMOKE_DPH = 1.5  # never rent above this for a smoke


@pytest.fixture(scope="session", autouse=True)
def _hf_repo_and_code():
    if not os.environ.get("HF_REPO"):
        from huggingface_hub import HfApi

        me = HfApi(token=os.environ["HUGGINGFACE_TOKEN"]).whoami()["name"]
        os.environ["HF_REPO"] = f"{me}/autoslm-vast-smoke"
    from autoslm.flash.train import upload_code

    upload_code()  # ship the package once for every smoke in the session
    return


@pytest.fixture(autouse=True)
def _tiny_run_env(monkeypatch):
    monkeypatch.setenv("SFT_MAX_STEPS", "5")  # forwarded by build_worker_env
    monkeypatch.setenv("SFT_PER_DEVICE_BS", "2")


def _smoke_spec(gpu: str):
    from autoslm.flash.gpus import gpu_short
    from autoslm.worker_spec import JobSpec

    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3-0.6B",
            "algorithm": "sft",
            "run_id": f"autoslm-{int(time.time())}-smk{gpu_short(gpu)}",
            "train": {"epochs": 1, "seeds": [0], "eval_examples": 20},
            "gpu": {
                "type": gpu,
                "provider": "vast",
                "requested": gpu,
                "max_wall_seconds": 2700,
                "max_retries": 0,
            },
        }
    )


@pytest.mark.parametrize("gpu", SMOKE_CLASSES)
def test_vast_smoke_train_eval(gpu):
    from autoslm.flash.gpus import GPU_INFO
    from autoslm.providers import vast_api
    from autoslm.providers.vast import submit_train_durable_vast, usable_offers

    # AUTOSLM_SMOKE_EXCLUDE_MACHINES: comma-separated machine ids observed killing
    # containers mid-run (production handles these via the per-run blacklist+retry;
    # the smoke runs with max_retries=0 so it sidesteps them up front).
    exclude = {
        int(m) for m in os.environ.get("AUTOSLM_SMOKE_EXCLUDE_MACHINES", "").split(",") if m.strip()
    }
    offers = [
        o
        for o in usable_offers(GPU_INFO[gpu].vram_gb, disk_gb=60, exclude_machine_ids=exclude)
        if o.gpu == gpu and o.dph_total <= MAX_SMOKE_DPH
    ]
    if not offers:
        pytest.skip(f"no verified-datacenter offers for {gpu} right now")
    spec = _smoke_spec(gpu)
    handles: list[dict] = []
    print(
        f"\n[smoke {gpu}] {len(offers)} offers, cheapest ${offers[0].dph_total:.2f}/hr "
        f"({offers[0].geolocation}); run {spec.run_id}",
        flush=True,
    )
    try:
        res = submit_train_durable_vast(
            spec, seed=0, log=sys.stderr, on_handle=handles.append, offers=offers[:5]
        )
    finally:
        # The runner already destroys in its own finally; this is the test's
        # belt-and-suspenders so an assertion bug can never leak a paid instance.
        for h in handles:
            vast_api.destroy_instance(int(h["instance_id"]))

    assert res.ok, f"{gpu} smoke failed: {res.failure}: {res.detail}"
    metrics = res.metrics
    assert "trained_eval_acc" in metrics
    assert "base_eval_acc" in metrics
    assert 0 < metrics["cost_usd"] < 1.0, f"smoke cost out of bounds: {metrics['cost_usd']}"
    assert metrics["notes"]["provider"] == "vast"
    assert metrics["notes"]["vast_gpu"] == gpu
    # the paid instance is gone
    assert handles, "runner never reported a handle"
    deadline = time.time() + 120
    inst = vast_api.get_instance(int(handles[-1]["instance_id"]))
    while inst is not None and time.time() < deadline:
        time.sleep(10)
        inst = vast_api.get_instance(int(handles[-1]["instance_id"]))
    assert inst is None, f"instance {handles[-1]['instance_id']} still exists after the run"
    print(
        f"[smoke {gpu}] OK: acc {metrics['base_eval_acc']}->{metrics['trained_eval_acc']}, "
        f"cost ${metrics['cost_usd']:.4f}",
        flush=True,
    )


def test_no_orphans_after_smokes():
    from autoslm.providers.vast import sweep_orphans

    assert sweep_orphans() == [], "orphaned vast instances found (and destroyed) after smokes"
