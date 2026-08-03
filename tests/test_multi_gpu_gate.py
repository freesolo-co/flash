"""CPU tests for multi-gpu parity: every provider rents n cards on one machine, none is special.

There is no submit-time multi-gpu gate any more. Every phase shards (sft, grpo and opd all delegate
to verl, whose workers launch nproc-per-node == gpu.count ranks) and all three providers can reach a
real n-card box, so a rejection would have nothing left to reject. What replaces the gate is a
contract each provider must hold up, and that is what these tests pin:

  * a provider only ever OFFERS a count it can actually rent (``live_candidates``), and
  * the submit path actually RENTS the count that was allocated.

The second half is the one that costs money if it breaks: an allocated 4-card shape that rents one
card still bills for four ranks' worth of wall time while oversubscribing a single card.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# the keys each phase used to be selected by. all dead: no worker reads any of them now.
_STALE_BACKEND_ENV = {
    "sft": "FLASH_SFT_BACKEND",
    "grpo": "FLASH_RL_BACKEND",
    "opd": "FLASH_OPD_BACKEND",
}

_PROVIDERS = ("runpod", "lambda", "vast")
# the only managed class all three providers stock. a parity test needs one class every provider can
# actually provision, or the spec is rejected on catalog grounds before parity is ever exercised.
_TRI_PROVIDER_GPU = "H100"


# a fake Lambda catalog covering the counts the tests ask for. Lambda names the count in the type,
# so 1x/2x/4x are three separate entries and an absent one means "Lambda does not sell that shape".
def _fake_lambda_types() -> dict:
    """A Lambda catalog stocking 1/2/4-card boxes of the tri-provider class.

    Keys are derived through ``instance_type_for`` rather than spelled out, so the fixture cannot
    drift from the real naming (the class is ``gpu_1x_h100_pcie``, not the ``_sxm5`` one might
    guess) and quietly stock a catalog nothing ever looks up.
    """
    from flash.providers.lambdalabs.gpus import instance_type_for

    return {
        instance_type_for(_TRI_PROVIDER_GPU, n): {
            # per INSTANCE, so a 4-card box lists at 4x the 1-card price
            "instance_type": {"price_cents_per_hour": 300 * n},
            "regions_with_capacity_available": [{"name": "us-west-1"}],
        }
        for n in (1, 2, 4)
    }


def _fake_vast_row(cards: int) -> dict:
    """One verified-datacenter H100 offer carrying ``cards`` GPUs, priced for the WHOLE box."""
    return {
        "id": 100 + cards,
        "gpu_name": "H100 SXM",
        "gpu_ram": 81920,
        "num_gpus": cards,
        "dph_total": 2.0 * cards,
        "cuda_max_good": 99.0,
        "hosting_type": 1,
        "verification": "verified",
        "reliability2": 0.999,
        "disk_space": 2000.0,
        "inet_down": 5000.0,
        "machine_id": 900 + cards,
    }


@pytest.fixture
def all_providers_configured(monkeypatch):
    """Enable lambda/vast AND stub their markets so a parity assertion really covers all three.

    Two separate problems, both of which would otherwise turn a three-way parity test into a
    runpod-only one. ``is_configured`` is keyed on an operator API key and this box carries only
    RunPod's, so lambda/vast would fail on credentials. But a key alone sends the real client at the
    network, which a CPU test must never do (and which fails here as a 404). So the catalog/market
    calls are stubbed at their single entry points -- ``list_instance_types`` and ``search_offers``
    -- leaving every line of the count-aware provider code under test and only the transport faked.
    """
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("LAMBDA_API_KEY", "test-key-not-used")
    monkeypatch.setenv("VAST_API_KEY", "test-key-not-used")
    monkeypatch.setattr(lambda_api, "list_instance_types", lambda *a, **k: _fake_lambda_types())
    monkeypatch.setattr(
        vast_api,
        "search_offers",
        lambda *a, num_gpus=1, **k: [_fake_vast_row(int(num_gpus))],
    )


def _spec(count: int, algorithm: str = "grpo", backend: str = "", provider: str = "runpod"):
    from flash.spec import JobSpec

    gpu: dict = {"type": "RTX 5090", "count": count}
    if provider:
        gpu["provider"] = provider
    body: dict = {"model": "test/gate", "algorithm": algorithm, "gpu": gpu}
    if backend:
        key = _STALE_BACKEND_ENV[algorithm]
        body["worker_env"] = {key: backend}
    return JobSpec.from_dict(body)


def test_no_multi_gpu_gate_survives_anywhere():
    """The runpod-only gate is gone from every layer it used to fire in.

    Asserted by NAME across the source tree rather than by calling it: the gate had five call sites
    (the two submit_job specs, the server route's two, and the recovery path), and a test that only
    exercised one of them would pass while a survivor still rejected lambda/vast in production.
    """
    import pathlib

    import flash

    root = pathlib.Path(flash.__file__).parent
    survivors = [
        f"{path.relative_to(root)}:{n}"
        for path in root.rglob("*.py")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if "_require_supported_gpu_count" in line or "_MULTI_GPU_PROVIDERS" in line
    ]
    assert survivors == [], f"multi-gpu gate survives at {survivors}"


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_submit_accepts_multi_gpu_on_every_provider(
    monkeypatch, all_providers_configured, provider, algorithm
):
    """No provider is special: a multi-gpu spec reaches preparation on all three.

    dry_run stops before provisioning, so this asserts only that nothing rejects the shape up front.
    """
    from flash import runner

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        status = runner.submit_job(
            _submittable(algorithm, count=4, provider=provider), dry_run=True
        )
        assert status is not None


@pytest.mark.parametrize("stale", ["verl", "trl", "bogus"])
@pytest.mark.parametrize("provider", _PROVIDERS)
def test_a_stale_backend_key_changes_no_submit_outcome(
    monkeypatch, all_providers_configured, stale, provider
):
    """A [worker_env] selector carried over from the trl era must not move the outcome either way.

    Asserted as "same outcome as the identical spec WITHOUT the key" rather than as a fixed verdict,
    so it fails the moment anything starts reading worker_env again, in either direction.
    """
    from flash import runner

    def outcome(backend: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
            try:
                runner.submit_job(
                    _submittable("grpo", count=4, provider=provider, backend=backend),
                    dry_run=True,
                )
            except ValueError as exc:
                return f"rejected: {exc}"
            return "allowed"

    assert outcome(stale) == outcome("")


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_provider_only_offers_counts_it_can_rent(all_providers_configured, provider):
    """``live_candidates`` never advertises a shape the provider cannot actually rent.

    Each provider reaches a count differently (runpod passes it at launch, lambda names it in the
    instance type, vast has it baked into the offer), so the allocator cannot synthesise n-card
    combinations on its own -- it must take what the provider reports. Counts are powers of two
    because verl asserts ``num_attention_heads % sp_size == 0``, and every managed model's head
    count is a power of two: a 3-card box would abort at step 0.
    """
    from flash.providers import get_provider
    from flash.providers.base import AllocationConstraints, rentable_gpu_counts

    # spelled out rather than taken from rentable_gpu_counts: the providers call that same helper, so
    # comparing candidates against it would compare the code to itself and could never disagree.
    assert rentable_gpu_counts(4) == (4, 2, 1), "powers of two, largest first"

    prov = get_provider(provider)
    constraints = AllocationConstraints(disk_gb=100.0, max_wall_seconds=3600.0, max_gpu_count=4)
    candidates = prov.live_candidates(24, constraints)
    counts = {c.gpu_count for c in candidates}

    assert counts <= {1, 2, 4}, (
        f"{provider} offered {sorted(counts - {1, 2, 4})} cards, a shape verl cannot shard over "
        f"(num_attention_heads % sp_size != 0 aborts at step 0)"
    )
    # the load-bearing half: a provider that quietly ignored the constraint and returned only
    # single-card shapes would satisfy the subset check above without supporting multi-gpu at all.
    assert max(counts) > 1, f"{provider} offered no multi-card shape at max_gpu_count=4"


def test_single_card_constraint_yields_only_single_card_offers(all_providers_configured):
    """max_gpu_count=1 must produce no multi-card candidate on any provider.

    The pairing that makes the count-aware path failable: with the cap at 1 a provider that ignored
    the constraint entirely would still look correct in the max_gpu_count=4 test above.
    """
    from flash.providers import available_providers, get_provider
    from flash.providers.base import AllocationConstraints

    checked = []
    for name in available_providers():
        prov = get_provider(name)
        candidates = prov.live_candidates(24, AllocationConstraints(max_gpu_count=1))
        assert candidates, f"{name} returned nothing, so it asserts nothing about the cap"
        assert all(c.gpu_count == 1 for c in candidates), f"{name} ignored max_gpu_count=1"
        checked.append(name)
    # without this the loop would pass on a box where no provider is enabled, reporting coverage of
    # three providers while having checked none.
    assert checked == list(_PROVIDERS), f"expected all three providers enabled, checked {checked}"


def test_lambda_names_the_card_count_in_the_instance_type():
    """Lambda reaches an n-card box only by rewriting the count segment of the type name."""
    from flash.providers.lambdalabs.gpus import instance_type_for

    one = instance_type_for("H100")
    assert one.startswith("gpu_1x_")
    assert instance_type_for("H100", 4) == one.replace("gpu_1x_", "gpu_4x_", 1)
    # count 1 must be byte-identical to the no-count call, or single-gpu runs change shape.
    assert instance_type_for("H100", 1) == one


def test_lambda_price_is_per_card_not_per_instance():
    """``price_cents_per_hour`` prices the whole instance; ``Candidate.hourly_usd`` is per card.

    Without the division an n-card box prices n^2 (the allocator re-multiplies by gpu_count via
    total_hourly_usd) and would never be chosen -- a silent, permanent multi-gpu outage.
    """
    import flash.providers.lambdalabs.jobs as lj

    seen: dict = {}

    def fake_rate(name, *, gpu_count=1, **kwargs):
        seen["count"] = gpu_count
        return 4.0 * gpu_count  # a real 4-card instance costs 4x a 1-card instance

    def fake_regions(itype, **kwargs):
        return ["us-west-1"]

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("flash.providers.lambdalabs.pricing.hourly_rate", fake_rate)
        monkey.setattr(lj.lambda_api, "regions_with_capacity", fake_regions)
        four = lj.usable_instances("H100", gpu_count=4)
    finally:
        monkey.undo()
    assert seen["count"] == 4, "the n-card instance type must be the one priced"
    assert four, "no instance came back, so the rate assertion below would never run"
    assert four[0].gpu_count == 4
    assert four[0].price_usd_hr == pytest.approx(4.0), "per-card rate, not the instance total"


def test_vast_price_is_per_card_not_per_offer():
    """``dph_total`` prices the WHOLE offer; the per-card rate is what Candidate carries.

    Same failure mode as lambda's: an undivided 4-card offer prices 4x too high and loses every
    ranking, so multi-gpu on vast would look supported and never actually be selected.
    """
    import flash.providers.vast.jobs as vj

    row = {
        "id": 1,
        "gpu_name": "H100 SXM",
        "gpu_ram": 81920,
        "num_gpus": 4,
        "dph_total": 8.0,  # whole 4-card box
        "cuda_max_good": 99.0,
        "hosting_type": 1,
        "verification": "verified",
        "reliability2": 0.999,
        "disk_space": 500.0,
        "inet_down": 5000.0,
        "machine_id": 7,
    }
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(vj.vast_api, "search_offers", lambda *a, **k: [row])
        offers = vj.usable_offers(80, 100.0, num_gpus=4)
    finally:
        monkey.undo()
    assert offers, "a fitting verified 4-card offer must survive the filter"
    assert offers[0].gpu_count == 4
    assert offers[0].dph_total == pytest.approx(2.0), "per-card rate, not the offer total"


def test_vast_rejects_an_offer_whose_card_count_disagrees():
    """A row that does not carry the requested count is dropped client-side.

    The server-side num_gpus filter is an {"eq": n} query, but the count divides dph_total into the
    per-card rate -- trusting it would mis-price a mismatched row by exactly the ratio of counts.
    """
    import flash.providers.vast.jobs as vj

    row = {
        "id": 1,
        "gpu_name": "H100 SXM",
        "gpu_ram": 81920,
        "num_gpus": 1,  # server ignored the filter
        "dph_total": 8.0,
        "cuda_max_good": 99.0,
        "hosting_type": 1,
        "verification": "verified",
        "reliability2": 0.999,
        "disk_space": 500.0,
        "inet_down": 5000.0,
        "machine_id": 7,
    }
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(vj.vast_api, "search_offers", lambda *a, **k: [row])
        assert vj.usable_offers(80, 100.0, num_gpus=4) == []
        # the SAME row is accepted when the count matches, so the rejection is the count and not
        # some other filter in the chain quietly failing.
        assert vj.usable_offers(80, 100.0, num_gpus=1)
    finally:
        monkey.undo()


def test_lambda_submit_requests_the_allocated_card_count():
    """``submit_run_lambda`` must ask for gpu.count cards, not one.

    This is the expensive half of the contract: the worker spawns gpu.count ranks regardless, so a
    single-card rental oversubscribes one card while the run bills for the full wall time.
    """
    import flash.providers.lambdalabs.jobs as lj

    seen: dict = {}

    def fake_usable(gpu_class, force=False, *, gpu_count=1, **kwargs):
        seen["count"] = gpu_count
        raise lj.lambda_api.LambdaApiError("stop before launching")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(lj, "usable_instances", fake_usable)
        with pytest.raises(lj.lambda_api.LambdaApiError):
            lj.submit_run_lambda(
                _submittable("grpo", count=4, provider="lambda"),
                42,
                deadline_at=9_999_999_999.0,
            )
    finally:
        monkey.undo()
    assert seen["count"] == 4, "submit rented a shape other than the one allocated"


def test_vast_submit_requests_the_allocated_card_count():
    """``submit_run_vast`` must search for gpu.count cards, not one. Same billing exposure."""
    import flash.providers.vast.jobs as vj

    seen: dict = {}

    def fake_offers(*args, num_gpus=1, **kwargs):
        seen["count"] = num_gpus
        raise vj.vast_api.VastApiError("stop before renting")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(vj, "usable_offers", fake_offers)
        with pytest.raises(vj.vast_api.VastApiError):
            vj.submit_run_vast(
                _submittable("grpo", count=4, provider="vast"),
                42,
                deadline_at=9_999_999_999.0,
            )
    finally:
        monkey.undo()
    assert seen["count"] == 4, "submit rented a shape other than the one allocated"


def _submittable(
    algorithm: str,
    backend: str = "",
    *,
    count: int = 1,
    provider: str = "runpod",
    gpu: str = _TRI_PROVIDER_GPU,
):
    """a spec that survives a full submit, unlike a bare gate fixture."""
    from flash.spec import JobSpec

    train: dict = {"max_examples": 4}
    if algorithm == "opd":
        train["teacher_model"] = "kimi-k2.6"
    body: dict = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": algorithm,
        "gpu": {"type": gpu, "count": count, "provider": provider},
        "train": train,
    }
    if backend:
        body["worker_env"] = {_STALE_BACKEND_ENV[algorithm]: backend}
    return JobSpec.from_dict(body)


@pytest.mark.parametrize(
    ("algorithm", "backend"),
    [("grpo", "verl"), ("grpo", ""), ("grpo", "trl"), ("sft", ""), ("opd", "")],
)
def test_submit_records_the_resolved_backend(monkeypatch, algorithm, backend):
    from flash import runner

    # the run records which trainer actually ran, so it stays auditable from the run itself rather
    # than from what the config appeared to ask for. a stale key must not change the record.
    expected = "verl"
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        status = runner.submit_job(_submittable(algorithm, backend), dry_run=True)
        assert (status.effective_preparation or {}).get("backend") == expected
