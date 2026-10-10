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
from types import SimpleNamespace

import pytest

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.submit as runner_submit
from tests._helpers.profile import satisfy_sft_profile
from tests._helpers.source_snapshot import valid_source_snapshot
from tests._helpers.teacher import configure_managed_teacher

_PROVIDERS = ("runpod", "lambda", "vast")
_SOURCE_SNAPSHOT = valid_source_snapshot()
# the only managed class all three providers stock. a parity test needs one class every provider can
# actually provision, or the spec is rejected on catalog grounds before parity is ever exercised.
_TRI_PROVIDER_GPU = "H100"


# a fake Lambda catalog covering the counts the tests ask for. Lambda names the count in the type,
# so 1x/2x/4x/8x are separate entries and an absent one means "Lambda does not sell that shape".
def _fake_lambda_types() -> dict:
    """A Lambda catalog stocking 1/2/4/8-card boxes of the tri-provider class.

    Keys are derived through ``instance_type_for`` rather than spelled out, so the fixture cannot
    drift from the real naming (the class is ``gpu_1x_h100_pcie``, not the ``_sxm5`` one might
    guess) and quietly stock a catalog nothing ever looks up.
    """
    from flash.providers.lambda_.client.gpus import instance_type_for

    return {
        instance_type_for(_TRI_PROVIDER_GPU, n): {
            # per INSTANCE, so a 4-card box lists at 4x the 1-card price
            "instance_type": {"price_cents_per_hour": 300 * n},
            "regions_with_capacity_available": [{"name": "us-west-1"}],
        }
        for n in (1, 2, 4, 8)
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
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("LAMBDA_API_KEY", "test-key-not-used")
    monkeypatch.setenv("VAST_API_KEY", "test-key-not-used")
    monkeypatch.setattr(lambda_api, "list_instance_types", lambda *a, **k: _fake_lambda_types())
    monkeypatch.setattr(
        vast_api,
        "search_offers",
        lambda *a, num_gpus=1, **k: [_fake_vast_row(int(num_gpus))],
    )


def _spec(count: int, algorithm: str = "grpo", provider: str = "runpod"):
    from flash.core.spec import JobSpec

    gpu: dict = {"type": "RTX 5090", "count": count}
    if provider:
        gpu["provider"] = provider
    body: dict = {"model": "test/gate", "algorithm": algorithm, "gpu": gpu}
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

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(tmp, "runs"))
        spec = _submittable(algorithm, count=4, provider=provider)
        satisfy_sft_profile(monkeypatch, spec)
        configure_managed_teacher(monkeypatch, spec)
        status = runner_submit.submit_job(spec, dry_run=True)
        assert status is not None


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_provider_only_offers_counts_it_can_rent(all_providers_configured, provider):
    """``live_candidates`` never advertises a shape the provider cannot actually rent.

    Each provider reaches a count differently (runpod passes it at launch, lambda names it in the
    instance type, vast has it baked into the offer), so the allocator cannot synthesise n-card
    combinations on its own -- it must take what the provider reports. Counts are powers of two
    because vllm asserts ``num_attention_heads % tp_size == 0`` for the rollout engine, and every
    managed model's head count is a power of two: a 3-card box would abort at engine init.
    """
    from flash.providers.core.base import AllocationConstraints, rentable_gpu_counts
    from flash.providers.core.registry import get_provider

    # spelled out rather than taken from rentable_gpu_counts: the providers call that same helper, so
    # comparing candidates against it would compare the code to itself and could never disagree.
    assert rentable_gpu_counts(8) == (8, 4, 2, 1), "powers of two, largest first"

    prov = get_provider(provider)
    constraints = AllocationConstraints(disk_gb=100.0, max_wall_seconds=3600.0, max_gpu_count=8)
    candidates = prov.live_candidates(24, constraints)
    counts = {c.gpu_count for c in candidates}

    assert counts <= {1, 2, 4, 8}, (
        f"{provider} offered {sorted(counts - {1, 2, 4, 8})} cards, a shape the rollout engine "
        f"cannot shard heads over (num_attention_heads % tp_size != 0 aborts at engine init)"
    )
    # the load-bearing half: the public maximum must reach every provider. A provider that quietly
    # capped itself at 4 would satisfy the subset check while leaving its live 8-card SKUs unreachable.
    assert 8 in counts, f"{provider} offered no 8-card shape at max_gpu_count=8"


def test_single_card_constraint_yields_only_single_card_offers(all_providers_configured):
    """max_gpu_count=1 must produce no multi-card candidate on any provider.

    The pairing that makes the count-aware path failable: with the cap at 1 a provider that ignored
    the constraint entirely would still look correct in the max_gpu_count=8 test above.
    """
    from flash.providers.core.base import AllocationConstraints
    from flash.providers.core.registry import available_providers, get_provider

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
    from flash.providers.lambda_.client.gpus import instance_type_for

    one = instance_type_for("H100")
    assert one.startswith("gpu_1x_")
    assert instance_type_for("H100", 4) == one.replace("gpu_1x_", "gpu_4x_", 1)
    # count 1 must be byte-identical to the no-count call, or single-gpu runs change shape.
    assert instance_type_for("H100", 1) == one


def test_lambda_resolves_a_multi_card_sku_that_renames_its_suffix():
    """The count rewrite alone can name a type Lambda does not sell; the catalog is authoritative.

    H100 is registered as ``gpu_1x_h100_pcie`` but Lambda's multi-card family is ``_sxm5``, so the
    derived ``gpu_8x_h100_pcie`` does not exist. ``regions_with_capacity`` answers [] for an unknown
    type, which reads as "sold out" rather than "wrong name" -- real capacity disappears silently.
    """
    from flash.providers.lambda_.client.gpus import instance_type_for

    derived = instance_type_for("H100", 4)
    real = derived.replace("_pcie", "_sxm5")
    assert real != derived, "fixture must model a renamed suffix or it proves nothing"

    # only the renamed spelling is stocked, so a rewrite-only answer cannot pass by accident.
    assert instance_type_for("H100", 4, {real: {}}) == real
    # a catalog that DOES stock the derived name must not be second-guessed.
    assert instance_type_for("H100", 4, {derived: {}, real: {}}) == derived
    # a count the catalog stocks at neither spelling still yields the derived name: naming is this
    # function's job, and rentability is ``usable_instances``' check.
    assert instance_type_for("H100", 2, {real: {}}) == instance_type_for("H100", 2)


def test_lambda_catalog_suffix_fallback_preserves_the_managed_memory_class():
    """Family matching must not turn A100 40 GB into the costlier A100 80 GB SKU.

    Lambda stocks both as 8-card boxes. Matching only the ``a100`` family makes catalog order choose
    arbitrarily; in the live catalog that selected the 80 GB box at $22.32/hr instead of the fitting
    40 GB box at $15.92/hr while still labelling the candidate ``A100 SXM 40GB``.
    """
    from flash.providers.lambda_.client.gpus import instance_type_for

    forty = "gpu_8x_a100"
    eighty = "gpu_8x_a100_80gb_sxm4"
    catalog = {
        eighty: {
            "instance_type": {
                "gpu_description": "A100 (80 GB SXM4)",
                "price_cents_per_hour": 2232,
            }
        },
        forty: {
            "instance_type": {
                "gpu_description": "A100 (40 GB SXM4)",
                "price_cents_per_hour": 1592,
            }
        },
    }
    assert instance_type_for("A100 SXM 40GB", 8, catalog) == forty
    # dictionary order is not a contract; reversing it must not change the selected memory class.
    assert instance_type_for("A100 SXM 40GB", 8, dict(reversed(catalog.items()))) == forty
    # A sole explicit 80 GB entry is still the WRONG class, not a renamed 40 GB spelling.
    assert instance_type_for("A100 SXM 40GB", 8, {eighty: catalog[eighty]}) == instance_type_for(
        "A100 SXM 40GB", 8
    )


def test_lambda_missing_required_count_sku_is_terminal(monkeypatch):
    """An absent 8-card SKU is structural, while an existing sold-out SKU remains retryable.

    Without the catalog check both cases return no live candidates and the allocator retries them as
    capacity failures. A shape Lambda does not sell can never recover by retrying.
    """
    from flash.providers.core.base import AllocationConstraints, UnsupportedGpuError
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.lambda_.client.gpus import instance_type_for
    from flash.providers.lambda_.execution.provider import LambdaProvider

    gpu = "A100 SXM 40GB"

    def _catalog(counts):
        return {
            instance_type_for(gpu, count): {
                "instance_type": {"gpu_description": "A100 (40 GB SXM4)"},
                "regions_with_capacity_available": [],
            }
            for count in counts
        }

    provider = LambdaProvider()
    constraints = AllocationConstraints(gpu_type=gpu, required_vram_gb=129, max_gpu_count=8)
    monkeypatch.setattr(lambda_api, "list_instance_types", lambda *a, **k: _catalog((1, 2, 4)))
    with pytest.raises(UnsupportedGpuError, match=r"does not offer a rentable.*8 cards"):
        provider.live_candidates(19, constraints)

    # The required SKU exists but has no live regions: return no candidates so allocate() classifies
    # it as sold out/retryable rather than structurally impossible.
    monkeypatch.setattr(lambda_api, "list_instance_types", lambda *a, **k: _catalog((1, 2, 4, 8)))
    assert provider.live_candidates(19, constraints) == []

    # Unpinned searches need the same distinction. Otherwise a Lambda-only run retries forever for
    # a count-specific SKU Lambda does not sell.
    unpinned = AllocationConstraints(required_vram_gb=129, max_gpu_count=8)
    monkeypatch.setattr(lambda_api, "list_instance_types", lambda *a, **k: _catalog((1, 2, 4)))
    with pytest.raises(UnsupportedGpuError, match=r"does not offer a rentable.*8 cards"):
        provider.live_candidates(19, unpinned)


def test_lambda_sku_miss_is_provider_local_during_auto_allocation(monkeypatch):
    """A missing Lambda count SKU must not discard a valid shape from another provider."""
    import flash.providers.core.allocator as allocator
    from flash.providers.core.base import Candidate, UnsupportedGpuError, gpu_classes_for

    class _LambdaMiss:
        live_capacity = True

        def live_candidates(self, _need, _constraints):
            raise UnsupportedGpuError("lambda does not sell this count-specific SKU")

        def gpu_classes(self):
            return gpu_classes_for("lambda_name")

    class _RunPodHit:
        live_capacity = False

        def live_candidates(self, _need, _constraints):
            return [Candidate("runpod", "H100", 3.29, 80, 2)]

        def gpu_classes(self):
            return gpu_classes_for("enum_member")

    providers = {"lambda": _LambdaMiss(), "runpod": _RunPodHit()}
    monkeypatch.setattr(allocator, "available_providers", lambda: ("lambda", "runpod"))
    monkeypatch.setattr(allocator, "get_provider", providers.__getitem__)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 100)

    chosen = allocator.allocate("Qwen/Qwen3.5-9B", "sft", gpu_type="H100", max_gpu_count=2)
    assert (chosen.provider, chosen.gpu, chosen.gpu_count) == ("runpod", "H100", 2)


def test_lambda_instance_type_never_reaches_the_network():
    """``instance_type_for`` must stay offline: it is called from pricing, which is called from it.

    An in-function catalog fetch turns a pure name lookup into live I/O on every sizing call and
    deadlocks the offline test path. Callers holding a catalog pass it; nobody else pays.
    """
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.lambda_.client.gpus import instance_type_for

    def explode(*_a, **_k):
        raise AssertionError("instance_type_for fetched the catalog")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(lambda_api, "list_instance_types", explode)
        monkey.setattr(lambda_api, "request_with_retries", explode)
        assert instance_type_for("H100", 4)
    finally:
        monkey.undo()


def test_lambda_price_is_per_card_not_per_instance():
    """``price_cents_per_hour`` prices the whole instance; ``Candidate.hourly_usd`` is per card.

    Without the division an n-card box prices n^2 (the allocator re-multiplies by gpu_count via
    total_hourly_usd) and would never be chosen -- a silent, permanent multi-gpu outage.
    """
    import flash.providers.lambda_.jobs as lj

    seen: dict = {}

    def fake_rate(name, *, gpu_count=1, **kwargs):
        seen["count"] = gpu_count
        return 4.0 * gpu_count  # a real 4-card instance costs 4x a 1-card instance

    def fake_regions(itype, **kwargs):
        return ["us-west-1"]

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("flash.providers.lambda_.client.pricing.hourly_rate", fake_rate)
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
    """``submit_attempt_lambda`` must ask for gpu.count cards, not one.

    This is the expensive half of the contract: the worker spawns gpu.count ranks regardless, so a
    single-card rental oversubscribes one card while the run bills for the full wall time.
    """
    import flash.providers.lambda_.jobs as lj

    seen: dict = {}

    def fake_usable(gpu_class, force=False, *, gpu_count=1, **kwargs):
        seen["count"] = gpu_count
        raise lj.lambda_api.LambdaApiError("stop before launching")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(lj, "usable_instances", fake_usable)
        with pytest.raises(lj.lambda_api.LambdaApiError):
            lj.submit_attempt_lambda(
                _submittable("grpo", count=4, provider="lambda"),
                42,
                deadline_at=9_999_999_999.0,
            )
    finally:
        monkey.undo()
    assert seen["count"] == 4, "submit rented a shape other than the one allocated"


def test_vast_submit_requests_the_allocated_card_count():
    """``submit_attempt_vast`` must search for gpu.count cards, not one. Same billing exposure."""
    import flash.providers.vast.jobs as vj

    seen: dict = {}

    def fake_offers(*args, num_gpus=1, **kwargs):
        seen["count"] = num_gpus
        raise vj.vast_api.VastApiError("stop before renting")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(vj, "usable_offers", fake_offers)
        with pytest.raises(vj.vast_api.VastApiError):
            vj.submit_attempt_vast(
                _submittable("grpo", count=4, provider="vast"),
                42,
                deadline_at=9_999_999_999.0,
            )
    finally:
        monkey.undo()
    assert seen["count"] == 4, "submit rented a shape other than the one allocated"


def test_lambda_capacity_refresh_keeps_the_allocated_card_count():
    """The mid-walk capacity refresh must re-search the SAME shape, not fall back to one card.

    ``usable_instances`` defaults ``gpu_count`` to 1, so a refresh that omits it silently rents a
    1-card box while the worker still starts n ranks -- the exact billing exposure the submit-path
    test guards, reached through the other door.
    """
    from flash.providers.lambda_ import jobs as lj
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.lambda_.jobs.builders import LambdaInstance

    def _inst(region: str) -> LambdaInstance:
        return LambdaInstance(
            gpu=_TRI_PROVIDER_GPU,
            instance_type="gpu_4x_h100_sxm5",
            region=region,
            vram_gb=80,
            price_usd_hr=3.0,
            gpu_count=4,
        )

    refreshed_with: dict = {}

    def fake_launch(*, region_name, **_kw):
        if region_name != "us-fresh-1":
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: insufficient-capacity")
        return "i-42"

    def fake_usable(gpu_class, force=False, *, gpu_count=1, **_kw):
        refreshed_with["count"] = gpu_count
        return [_inst("us-fresh-1")]

    spec = _submittable("grpo", count=4, provider="lambda")
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(lj, "resolve_ssh_key_names", lambda **_k: ["jk"])
        monkey.setattr(lambda_api, "launch_instance", fake_launch)
        monkey.setattr(lj, "usable_instances", fake_usable)
        handle = lj.launch_and_submit(
            spec,
            instances=[_inst("us-east-1")],
            source_snapshot=_SOURCE_SNAPSHOT,
            deadline_at=9_999_999_999.0,
        )
    finally:
        monkey.undo()

    assert refreshed_with.get("count") == 4, "refresh dropped the count and would rent one card"
    # the refresh is what produced the launch, so the assertion above cannot pass vacuously.
    assert handle.instance_id == "i-42"


def test_vast_capacity_refresh_keeps_the_allocated_card_count():
    """Same contract on Vast: ``usable_offers`` defaults ``num_gpus`` to 1 on an omitted refresh."""
    from flash.providers.vast import jobs as vj
    from flash.providers.vast.client import api as vast_api
    from flash.providers.vast.jobs.builders import VastOffer

    def _offer(offer_id: int) -> VastOffer:
        return VastOffer(
            offer_id=offer_id,
            machine_id=900 + offer_id,
            gpu=_TRI_PROVIDER_GPU,
            vram_gb=80,
            dph_total=2.0,
            cuda_max_good=99.0,
            disk_space=2000.0,
            reliability=0.999,
            inet_down=5000.0,
            geolocation="US",
            gpu_count=4,
        )

    refreshed_with: dict = {}

    def fake_create(offer_id, **_kw):
        if offer_id != 7:
            raise vast_api.VastCreateRejected(f"offer {offer_id} taken")
        return 4242

    def fake_usable(*_a, num_gpus=1, **_kw):
        refreshed_with["count"] = num_gpus
        return [_offer(7)]

    spec = _submittable("grpo", count=4, provider="vast")
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(vast_api, "create_instance", fake_create)
        monkey.setattr(vj, "usable_offers", fake_usable)
        handle = vj.deploy_and_submit(
            spec,
            offers=[_offer(1)],
            source_snapshot=_SOURCE_SNAPSHOT,
            deadline_at=9_999_999_999.0,
        )
    finally:
        monkey.undo()

    assert refreshed_with.get("count") == 4, "refresh dropped the count and would rent one card"
    assert handle.instance_id == 4242


@pytest.mark.parametrize("provider", ["lambda", "vast"])
def test_handle_rate_prices_the_whole_instance_not_one_card(provider):
    """The handle's ``hourly_usd`` is billed against wall-clock ONCE, so it must cover every card.

    Both the metrics cost stamp and ``_instance_realized_cost`` compute ``wall_h * hourly_usd`` with
    no card-count factor anywhere. Storing the per-card rate that ranking needs would under-report
    an n-card box's COGS by exactly n -- invisible, because the number still looks like a price.
    """
    per_card = 3.0
    cards = 4

    if provider == "lambda":
        from flash.providers.lambda_ import jobs as lj
        from flash.providers.lambda_.client import api as lambda_api
        from flash.providers.lambda_.jobs.builders import LambdaInstance

        inst = LambdaInstance(
            gpu=_TRI_PROVIDER_GPU,
            instance_type="gpu_4x_h100_sxm5",
            region="us-east-1",
            vram_gb=80,
            price_usd_hr=per_card,
            gpu_count=cards,
        )
        spec = _submittable("grpo", count=cards, provider="lambda")
        monkey = pytest.MonkeyPatch()
        try:
            monkey.setattr(lj, "resolve_ssh_key_names", lambda **_k: ["jk"])
            monkey.setattr(lambda_api, "launch_instance", lambda **_kw: "i-1")
            handle = lj.launch_and_submit(
                spec,
                instances=[inst],
                source_snapshot=_SOURCE_SNAPSHOT,
                deadline_at=9_999_999_999.0,
            )
        finally:
            monkey.undo()
    else:
        from flash.providers.vast import jobs as vj
        from flash.providers.vast.client import api as vast_api
        from flash.providers.vast.jobs.builders import VastOffer

        offer = VastOffer(
            offer_id=1,
            machine_id=901,
            gpu=_TRI_PROVIDER_GPU,
            vram_gb=80,
            dph_total=per_card,
            cuda_max_good=99.0,
            disk_space=2000.0,
            reliability=0.999,
            inet_down=5000.0,
            geolocation="US",
            gpu_count=cards,
        )
        spec = _submittable("grpo", count=cards, provider="vast")
        monkey = pytest.MonkeyPatch()
        try:
            monkey.setattr(vast_api, "create_instance", lambda *_a, **_kw: 4242)
            handle = vj.deploy_and_submit(
                spec,
                offers=[offer],
                source_snapshot=_SOURCE_SNAPSHOT,
                deadline_at=9_999_999_999.0,
            )
        finally:
            monkey.undo()

    assert handle.hourly_usd == pytest.approx(per_card * cards)
    # and the realized-COGS reader must agree, since that is the consumer that would under-bill.
    from flash.providers.core.realized import realized_cost_for_remote

    realized = realized_cost_for_remote(
        {
            "provider": provider,
            "hourly_usd": handle.hourly_usd,
            "instance_id": handle.instance_id,
            "started_ts": 1.0,
        },
        start=1.0,
        end=3601.0,
    )
    assert realized is not None
    assert realized.realized_usd == pytest.approx(per_card * cards), (
        "one hour on an n-card box must bill the whole box"
    )


def _submittable(
    algorithm: str,
    *,
    count: int = 1,
    provider: str = "runpod",
    gpu: str = _TRI_PROVIDER_GPU,
):
    """a spec that survives a full submit, unlike a bare gate fixture."""
    from flash.core.spec import JobSpec

    train: dict = {"max_examples": 4}
    if algorithm == "opd":
        train["teacher_model"] = "kimi-k3"
    body: dict = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": algorithm,
        "gpu": {"type": gpu, "count": count, "provider": provider},
        "train": train,
    }
    if algorithm == "sft":
        # sft is workload-profiled before it is quoted, and the profile is keyed on an immutable
        # environment revision. Pair this with satisfy_sft_profile at the submit site.
        body["environment"] = {"id": "github:owner/repo@main:env/environment.py"}
    return JobSpec.from_dict(body)


@pytest.mark.parametrize("algorithm", ["grpo", "sft", "opd"])
def test_submit_records_the_resolved_backend(monkeypatch, algorithm):

    # the run records which trainer actually ran, so it stays auditable from the run itself.
    expected = "verl"
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(tmp, "runs"))
        spec = _submittable(algorithm)
        satisfy_sft_profile(monkeypatch, spec)
        configure_managed_teacher(monkeypatch, spec)
        status = runner_submit.submit_job(spec, dry_run=True)
        assert (status.effective_preparation or {}).get("backend") == expected


def test_gpu_count_is_honoured_by_parse_time_sizing():
    """`--gpus N` must reach parse-time sizing, or a large run is rejected before sharding.

    This is the half of "multi-card works" that lives before the allocator. `provisional_gpu` sized
    every run against ONE card, so a 27B run needing 234 GB was rejected at parse time with "no
    validated GPU class has >= 234 GB VRAM" no matter what the user passed to `--gpus`: the flag
    was inert for exactly the runs that need it. The allocator would happily have rented 2 x H200.

    Pinned to a need no single validated class holds (180 GB max) but four cards do, so the
    assertion cannot pass by accident on a run that fits one card anyway.
    """
    from flash.providers.core.base import (
        UnsupportedGpuError,
        cheapest_gpu,
    )
    from flash.providers.core.sharding import combined_vram_gb

    need = 234
    assert all(info.vram_gb < need for info in _validated_infos()), (
        "need must exceed every single card, or the multi-card path is not what is being tested"
    )

    with pytest.raises(UnsupportedGpuError):
        cheapest_gpu(need)  # one card cannot, and must still say so

    for count in (2, 4):
        chosen = cheapest_gpu(need, gpu_count=count)
        # the shape it names must genuinely hold the run under the SAME fit model the allocator
        # applies at submit time -- naming a class that does not fit would just move the failure.
        from flash.providers.core.base import get_gpu_info

        assert combined_vram_gb(get_gpu_info(chosen).vram_gb, count) >= need


def test_public_max_gpu_count_is_rentable_not_silently_clamped():
    """The public 8-card maximum must buy an 8-card shape, not quietly behave like 4.

    Lambda can have live 8x inventory while every 2x/4x SKU is sold out. The schema already accepts
    ``gpu.count = 8``; clamping it to 4 makes that provider capacity unreachable and contradicts the
    authored ceiling without an error.
    """
    from flash.providers.core.base import (
        UnsupportedGpuError,
        cheapest_gpu,
    )
    from flash.providers.core.sharding import combined_vram_gb

    # above the widest 4-card shape but below 8x B200, so restoring the old cap to 4 kills this test.
    need = 700
    assert combined_vram_gb(180, 4) < need <= combined_vram_gb(180, 8)
    with pytest.raises(UnsupportedGpuError):
        cheapest_gpu(need, gpu_count=4)
    chosen = cheapest_gpu(need, gpu_count=8)
    from flash.providers.core.base import get_gpu_info

    assert combined_vram_gb(get_gpu_info(chosen).vram_gb, 8) >= need


def test_eight_cards_require_validated_head_geometry(monkeypatch):
    """A run must not rent 8 cards before its attention-head count is known to divide.

    verl requires ``num_attention_heads % sp_size == 0``, so a width nobody certified can pay for a
    box and abort during sequence-parallel initialization. A pin whose geometry cannot be read
    certifies nothing and keeps the four-card ceiling; a pin that CAN be read is capped on the head
    count in that commit's own config -- see ``test_a_certified_pin_reaches_eight_cards``.
    """
    import flash.engine.plan.model_config_probe as model_config_probe
    import flash.engine.plan.vram as vram
    from flash.providers.core.allocator import geometry_safe_gpu_cap

    monkeypatch.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})

    def _unreadable(*_a, **_k):
        raise RuntimeError("transient hub error")

    monkeypatch.setattr(vram, "fetch_hf_model_geometry", _unreadable)

    assert geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8) == 8
    assert geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8, model_revision="a" * 40, certify=True) == 4
    # odd ceilings still normalize through the shared rentable-count helper.
    assert geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 3, model_revision="a" * 40, certify=True) == 2


def test_geometry_cap_follows_each_models_own_head_count():
    """The cap must divide each row's RECORDED head count, never a derived stand-in.

    ``hidden_size // head_dim`` is not the head count: these checkpoints decouple ``head_dim`` from
    that ratio, and the quotient is wrong for two of the three surviving rows. Every row is checked
    against ``num_attention_heads`` so a future model with, say, 20 heads is capped at 4 instead of
    rented at 8 and failed in Ulysses init.
    """
    from flash.core.catalog import MODELS
    from flash.providers.core.allocator import geometry_safe_gpu_cap

    for model_id, info in MODELS.items():
        heads = info.num_attention_heads
        assert heads > 0, f"{model_id}: catalog row records no num_attention_heads"
        # the derived quotient must never be used as the head count; pin the divergence so a future
        # refactor cannot quietly reintroduce it.
        assert heads % geometry_safe_gpu_cap(model_id, 8) == 0

    # the derivation still disagrees with the recorded geometry for these surviving rows.
    derived_wrong = {
        m for m, i in MODELS.items() if i.hidden_size // i.head_dim != i.num_attention_heads
    }
    assert derived_wrong == {"Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B"}

    # every CURRENT row divides by 8, so no row is narrowed today. that is a property of this
    # catalog, not an invariant -- the loop above is what enforces it for whatever is added next.
    assert geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8) == 8
    assert geometry_safe_gpu_cap("Qwen/Qwen3.8-27B", 8) == 8
    # an authored ceiling below the geometric limit still wins; the cap only ever narrows.
    assert geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 2) == 2
    # a model outside the catalog has no readable geometry, so it keeps the unvalidated ceiling.
    assert geometry_safe_gpu_cap("some-org/not-in-catalog", 8) == 4


def _stub_model_config_probe(monkey, info, *, heads=None):
    """Answer the pinned-config fetch from a catalog row, optionally with drifted head geometry."""
    import flash.engine.plan.model_config_probe as model_config_probe
    import flash.engine.plan.vram as vram

    def _pinned(_model_id, revision="", strict=False):
        return (
            info.params_b,
            info.vocab_size,
            info.hidden_size,
            info.num_layers,
            info.num_attention_heads if heads is None else heads,
        )

    monkey.setattr(vram, "fetch_hf_model_geometry", _pinned)
    monkey.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})


def test_a_certified_pin_reaches_eight_cards():
    """A pinned SFT run must be offered the width its own commit's geometry certifies.

    Treating "pinned" as "unknown geometry" made 8 cards unreachable for SFT specifically: every
    SFT run reaches allocation with a revision already resolved to a sha (`prepare_job` ->
    `_resolve_model_revision(required=True)`), so the pin branch fired on all six catalog models
    and `--gpus 8` silently became `--gpus 4` -- including for a run that only FITS at eight. The
    pinned commit's own config is readable, so it is what decides the width.
    """
    from flash.core.catalog import MODELS
    from flash.providers.core.allocator import geometry_safe_gpu_cap

    monkey = pytest.MonkeyPatch()
    try:
        for model_id, info in MODELS.items():
            _stub_model_config_probe(monkey, info)
            heads = info.num_attention_heads
            capped = geometry_safe_gpu_cap(model_id, 8, model_revision="a" * 40, certify=True)
            # the real invariant is divisibility; every current row happens to divide 8.
            assert heads % capped == 0, model_id
            assert capped == 8, model_id
            # the pin must never WIDEN past the authored ceiling either.
            assert geometry_safe_gpu_cap(model_id, 2, model_revision="a" * 40) == 2, model_id
    finally:
        monkey.undo()


def test_a_pin_that_contradicts_the_catalog_certifies_nothing():
    """A pin whose own config disagrees with its catalog row must not widen the run.

    The width follows the PINNED commit, so a commit contradicting the row it matches is exactly
    the case that must not be trusted for 8 -- the catalog row divides 8 cleanly and the pin is
    what the worker actually loads. Such a pin is rejected outright by `_validated_revision_geometry`
    when sizing runs, so the conservative ceiling here is belt-and-braces rather than the only gate.
    """
    from flash.core.catalog import MODELS
    from flash.providers.core.allocator import geometry_safe_gpu_cap

    info = MODELS["Qwen/Qwen3.5-9B"]
    assert info.num_attention_heads == 16

    monkey = pytest.MonkeyPatch()
    try:
        # a drifted head count is REJECTED by the catalog cross-check, so it certifies nothing and
        # keeps the conservative ceiling rather than widening on a config nothing else agrees with.
        _stub_model_config_probe(monkey, info, heads=20)
        assert (
            geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8, model_revision="a" * 40, certify=True) == 4
        )
        # a pin that omits the field entirely certifies nothing either.
        _stub_model_config_probe(monkey, info, heads=0)
        assert (
            geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8, model_revision="a" * 40, certify=True) == 4
        )
    finally:
        monkey.undo()


def test_a_pin_without_parameter_metadata_certifies_nothing():
    """A commit exposing no parameter count must not widen a run.

    Sizing rejects that pin outright, but the schema preflight asks for the CAP before it sizes
    anything, so the cap cannot lean on sizing having run first -- it has to fail closed itself.
    """
    import flash.engine.plan.model_config_probe as model_config_probe
    import flash.engine.plan.vram as vram
    from flash.core.catalog import MODELS
    from flash.providers.core.allocator import geometry_safe_gpu_cap

    info = MODELS["Qwen/Qwen3.5-9B"]

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})
        monkey.setattr(
            vram,
            "fetch_hf_model_geometry",
            # no safetensors.total, but a perfectly readable config with matching heads.
            lambda *_a, **_k: (
                None,
                info.vocab_size,
                info.hidden_size,
                info.num_layers,
                info.num_attention_heads,
            ),
        )
        assert (
            geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8, model_revision="a" * 40, certify=True) == 4
        )
    finally:
        monkey.undo()


def test_a_drifted_pin_head_count_is_rejected_by_sizing():
    """Head count joins the fail-closed geometry cross-check, not just the width decision.

    The pin's config is already validated against the catalog for params/vocab/hidden/layers. Heads
    now decide how wide the run is rented, so a commit that disagrees on heads has to be rejected by
    the same gate rather than silently sizing one shape and renting another.
    """
    import flash.engine.plan.vram as vram
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]

    monkey = pytest.MonkeyPatch()
    try:
        _stub_model_config_probe(monkey, info, heads=20)
        with pytest.raises(ValueError, match="attention head count"):
            vram.model_required_vram_gb("Qwen/Qwen3.5-9B", "sft", model_revision="a" * 40)
        # a config that simply omits the field is not a CONFLICT, so it must not be rejected here.
        _stub_model_config_probe(monkey, info, heads=0)
        vram.model_required_vram_gb("Qwen/Qwen3.5-9B", "sft", model_revision="a" * 40)
    finally:
        monkey.undo()


def test_a_pinned_head_lookup_is_not_repeated_or_cached_on_failure():
    """The pin is read once per quote, and a hub blip must not become permanent.

    `geometry_safe_gpu_cap` is called several times per quote (schema preflight, offline shape,
    allocation). An uncached lookup turns one quote into repeated hub round trips; a CACHED failure
    would pin the run to four cards until the plane restarted.
    """
    import flash.engine.plan.model_config_probe as model_config_probe
    import flash.engine.plan.vram as vram
    from flash.core.catalog import MODELS
    from flash.providers.core.allocator import geometry_safe_gpu_cap

    info = MODELS["Qwen/Qwen3.5-9B"]
    rev = "a" * 40
    calls: list[str] = []

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})

        def _blip(*_a, **_k):
            calls.append("fail")
            raise RuntimeError("transient hub error")

        monkey.setattr(vram, "fetch_hf_model_geometry", _blip)
        assert geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8, model_revision=rev, certify=True) == 4
        assert geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8, model_revision=rev, certify=True) == 4
        assert model_config_probe._CONFIG_PROBE_MEMO == {}, "a hub failure must not be cached"
        assert len(calls) == 2, "a failed lookup must stay retryable"

        calls.clear()

        def _ok(_model_id, revision="", strict=False):
            calls.append(revision)
            return (
                info.params_b,
                info.vocab_size,
                info.hidden_size,
                info.num_layers,
                info.num_attention_heads,
            )

        monkey.setattr(vram, "fetch_hf_model_geometry", _ok)
        assert geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8, model_revision=rev, certify=True) == 8
        before = len(calls)
        assert geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8, model_revision=rev, certify=True) == 8
        assert len(calls) == before, f"pinned head lookup hit the hub {len(calls)} times"
    finally:
        monkey.undo()


def test_only_immutable_complete_pin_reads_are_memoized():
    """The memo may only hold what is genuinely immutable AND complete.

    Two things that look immutable are not, and caching either one strands a valid pin until the
    plane restarts:

    - A REF is not a commit. `spec_from_dict` passes the AUTHORED `model_revision` through, so
      `main` or a tag reaches the memo before `_resolve_model_revision` resolves it to a sha. Cached
      under a moving name, a ref that later advances keeps serving the old geometry forever.
    - A COMPLETE read is not any read. `params_b` is None when the hub omits `safetensors.total` --
      hub metadata, not commit-immutable `config.json` geometry. Cached, `_validated_revision_
      geometry` raises from the memo on every later call and the pin stays rejected.
    """
    import flash.engine.plan.model_config_probe as model_config_probe
    import flash.engine.plan.vram as vram

    monkey = pytest.MonkeyPatch()
    try:
        calls: list[str] = []

        def _answer(params_b):
            def _fetch(_model_id, revision="", strict=False):
                calls.append(revision)
                return (params_b, 248320, 4096, 32, 16)

            return _fetch

        # an incomplete read stays retryable rather than becoming a permanent rejection.
        monkey.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})
        monkey.setattr(vram, "fetch_hf_model_geometry", _answer(None))
        vram._memoized_config_probe("M", "a" * 40)
        vram._memoized_config_probe("M", "a" * 40)
        assert model_config_probe._CONFIG_PROBE_MEMO == {}, "an incomplete read must not be cached"
        assert len(calls) == 2, "an incomplete read must stay retryable"

        # a moving ref is re-read every time, because it can point somewhere else tomorrow.
        calls.clear()
        monkey.setattr(vram, "fetch_hf_model_geometry", _answer(9.0))
        vram._memoized_config_probe("M", "main")
        vram._memoized_config_probe("M", "main")
        assert model_config_probe._CONFIG_PROBE_MEMO == {}, "a moving ref must not be cached"
        assert len(calls) == 2, "a moving ref must be re-read"

        # a complete sha read IS shared -- that is the whole point of the memo.
        calls.clear()
        vram._memoized_config_probe("M", "b" * 40)
        vram._memoized_config_probe("M", "b" * 40)
        assert len(calls) == 1, "a complete sha read must be shared, not repeated"
        assert list(model_config_probe._CONFIG_PROBE_MEMO) == [("M", "b" * 40)]
    finally:
        monkey.undo()


def test_only_the_submission_path_certifies_a_pin():
    """Every OFFLINE caller of the cap must leave `certify` off, so a hub blip cannot narrow a run.

    Certification is network i/o whose failure degrades to the four-card ceiling. That is a safe
    answer at submission, which already does i/o and can retry. It is not safe on the offline paths:
    `spec_from_dict` feeds this cap to `provisional_gpu`, which RAISES to reject an unplaceable run,
    and `_offline_gpu_shape` is contractually structural. On either, a transient hub error would turn
    a 35B that genuinely needs eight cards into a terminal "does not fit" rejection.

    Asserted by reading the call sites, because the defect is a keyword argument at a call site: a
    behavioral test of the submit path passes just as well when an offline path also certifies.
    """
    import ast
    import inspect

    import flash.cost.analytical as analytical
    import flash.providers.core.allocator as allocator
    import flash.schema as schema

    def _certifying_calls(module) -> list[bool]:
        tree = ast.parse(inspect.getsource(module))
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name != "geometry_safe_gpu_cap":
                continue
            found.append(
                any(
                    kw.arg == "certify" and getattr(kw.value, "value", False) is True
                    for kw in node.keywords
                )
            )
        return found

    offline = _certifying_calls(schema) + _certifying_calls(analytical)
    assert offline, "expected to find the offline geometry-cap call sites"
    assert not any(offline), (
        "an offline path opted into pin certification: a hub outage there rejects a run that fits"
    )

    submit = _certifying_calls(allocator)
    assert submit, "expected to find the allocator's geometry-cap call sites"
    assert all(submit), "the submission path must certify the pin; that is where widening happens"


def test_a_low_ceiling_pin_does_not_reach_the_hub():
    """Parse-time validation stays offline at EVERY ceiling, not just where widening is impossible.

    `spec_from_dict` calls the cap while parsing a config, so certification is opt-in and parse time
    does not opt in. An earlier version skipped the hub only at `cap <= 4` (where a certified head
    count could not raise the ceiling anyway); that left the widenable case reaching the network from
    an otherwise-offline path, where a hub timeout blocks config validation and a transient failure
    narrows the run to four and gets it rejected as unplaceable.

    Certification still happens where it belongs, under `certify=True`, which is asserted at the end
    here and covered in ``test_only_the_submission_path_certifies_a_pin``.
    """
    import flash.engine.plan.model_config_probe as model_config_probe
    import flash.engine.plan.vram as vram
    from flash.providers.core.allocator import geometry_safe_gpu_cap

    monkey = pytest.MonkeyPatch()
    try:
        calls: list[str] = []

        def _fetch(_model_id, revision="", strict=False):
            calls.append(revision)
            raise RuntimeError("hub must not be reached")

        monkey.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})
        monkey.setattr(vram, "fetch_hf_model_geometry", _fetch)

        # every ceiling, including the widenable ones -- an offline caller never reaches the hub.
        for ceiling in (1, 2, 3, 4, 8):
            geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", ceiling, model_revision="a" * 40)
        assert calls == [], f"parse-time cap reached the hub {len(calls)} times"

        # opting in is what permits the round trip, and only above the uncertified ceiling.
        geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 4, model_revision="a" * 40, certify=True)
        assert calls == [], "certification cannot widen at or below the uncertified cap"
        geometry_safe_gpu_cap("Qwen/Qwen3.5-9B", 8, model_revision="a" * 40, certify=True)
        assert len(calls) == 1, "a widenable ceiling must certify the pin when asked to"
    finally:
        monkey.undo()


def test_a_blip_after_sizing_cannot_narrow_an_already_validated_pin():
    """Sizing and the cap must share one geometry read, or an eight-card run dies mid-allocation.

    `allocate()` sizes the run first (`required_vram_gb` -> `_validated_revision_geometry`, which
    fetches and validates this exact pin) and only then asks for the head cap. With two independent
    lookups, a hub blip landing between them narrowed a just-validated pin to four cards -- and for a
    run that only FITS at eight, `_structurally_fits` then reports it as terminally unplaceable
    rather than retryable. A pinned commit's geometry is immutable, so one success settles it.
    """
    import flash.engine.plan.model_config_probe as model_config_probe
    import flash.engine.plan.vram as vram
    from flash.core.catalog import MODELS
    from flash.providers.core.allocator import geometry_safe_gpu_cap

    model = "Qwen/Qwen3.5-9B"
    info = MODELS[model]
    rev = "a" * 40

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})
        _stub_model_config_probe(monkey, info)
        monkey.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})

        # 1. sizing succeeds and validates the pin, exactly as `allocate()` does first.
        assert vram.model_required_vram_gb(model, "sft", model_revision=rev) > 0

        # 2. the hub dies immediately afterwards.
        def _blip(*_a, **_k):
            raise RuntimeError("transient hub error")

        monkey.setattr(vram, "fetch_hf_model_geometry", _blip)

        # 3. the cap must still certify 8 from the geometry step 1 already read and validated.
        # `certify=True` is what makes this test mean anything: without it the cap never consults
        # the memo, so the blip stub above is dead and a lost memo-reuse regression still passes.
        assert geometry_safe_gpu_cap(model, 8, model_revision=rev, certify=True) == 8
    finally:
        monkey.undo()


def test_geometry_cap_narrows_a_row_whose_heads_do_not_divide_eight():
    """A row with awkward geometry must be capped, not rented and failed at Ulysses init.

    No current catalog row exercises this -- all six divide by 8 -- so the guard is proven against a
    synthetic 20-head row rather than left as untested code that only runs on a future model.
    """
    from dataclasses import replace

    from flash.core.catalog import MODELS
    from flash.providers.core.allocator import geometry_safe_gpu_cap

    awkward = replace(MODELS["Qwen/Qwen3.5-9B"], id="Qwen/Fake-20-Head", num_attention_heads=20)
    patched = dict(MODELS)
    patched["Qwen/Fake-20-Head"] = awkward

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("flash.core.catalog.MODELS", patched)
        # 20 % 8 != 0 and 20 % 4 == 0, so the widest legal rentable shape is 4.
        assert geometry_safe_gpu_cap("Qwen/Fake-20-Head", 8) == 4
        # 24 heads divide 8, so a 24-head row is NOT narrowed.
        patched["Qwen/Fake-24-Head"] = replace(awkward, num_attention_heads=24)
        assert geometry_safe_gpu_cap("Qwen/Fake-24-Head", 8) == 8
        # a row with no recorded head count cannot be certified, so it keeps the safe ceiling.
        patched["Qwen/Fake-Unknown"] = replace(awkward, num_attention_heads=0)
        assert geometry_safe_gpu_cap("Qwen/Fake-Unknown", 8) == 4
    finally:
        monkey.undo()


def test_an_uncertified_pin_still_narrows_below_the_four_card_ceiling():
    """The four-card ceiling for an uncertified pin must NARROW the divisor search, not replace it.

    A ceiling is a bound, not a divisibility proof: 4 divides 24 but not 6. Returning the ceiling
    directly once a pin fails to certify skips the row's own head check, so a row whose heads do not
    divide 4 gets rented at 4 and aborts in Ulysses init -- the exact failure the cap exists to
    prevent, reintroduced on the uncertified path.

    No current row exercises this (all six are 8/8/16/16/24/16), so it is proven against a synthetic
    6-head row rather than left as an invariant that only a future model would discover.
    """
    from dataclasses import replace

    from flash.core.catalog import MODELS
    from flash.providers.core.allocator import geometry_safe_gpu_cap

    six = replace(MODELS["Qwen/Qwen3.5-9B"], id="Qwen/Fake-6-Head", num_attention_heads=6)
    patched = dict(MODELS)
    patched["Qwen/Fake-6-Head"] = six

    monkey = pytest.MonkeyPatch()
    try:
        import flash.engine.plan.model_config_probe as model_config_probe
        import flash.engine.plan.vram as vram

        monkey.setattr("flash.core.catalog.MODELS", patched)
        monkey.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})

        def _unreadable(*_a, **_k):
            raise RuntimeError("transient hub error")

        monkey.setattr(vram, "fetch_hf_model_geometry", _unreadable)

        # `certify=True` is required for the stub above to fire at all: without it the cap never
        # reaches the hub, the pin is never uncertified, and this asserts the unpinned path twice.
        # 6 % 4 != 0 and 6 % 2 == 0, so an uncertified pin must land on 2, not on the bare ceiling.
        assert (
            geometry_safe_gpu_cap("Qwen/Fake-6-Head", 8, model_revision="a" * 40, certify=True) == 2
        )
        # the unpinned path already checked the row and must be unchanged by this.
        assert geometry_safe_gpu_cap("Qwen/Fake-6-Head", 8) == 2
    finally:
        monkey.undo()


def test_unpinned_sold_out_live_market_stays_retryable():
    """An unpinned run on a live-market provider must not die terminally when stock runs out.

    `live_capacity` was consulted only for an EXACT pin, so an auto-allocated run whose fitting
    class was merely sold out fell through to `UnsupportedGpuError` -- which the lifecycle treats
    as terminal, killing a run a retry would have placed. A genuinely oversized run must still fail
    terminally, so the distinction is gated on whether any offered shape could hold the run at all.
    """
    import flash.providers.core.allocator as alloc
    from flash.providers.core.base import CapacityLookupError, UnsupportedGpuError

    class _SoldOutLiveMarket:
        name = "lambda"
        live_capacity = True

        def live_candidates(self, need, constraints):
            return []  # structurally offered, nothing free right now

        def gpu_classes(self):
            from flash.providers.core import base

            return base.gpu_classes_for("lambda_name")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(alloc, "get_provider", lambda _n: _SoldOutLiveMarket())
        monkey.setattr(alloc, "available_providers", lambda: ("lambda",))
        # a run that fits an offered shape: sold out now -> retryable.
        with pytest.raises(CapacityLookupError):
            alloc.allocate("Qwen/Qwen3-4B", "sft")
        # a run no offered shape can ever hold -> still terminal, not an infinite retry.
        monkey.setattr(alloc, "required_vram_gb", lambda *a, **k: 100_000)
        with pytest.raises(UnsupportedGpuError):
            alloc.allocate("Qwen/Qwen3-4B", "sft")
    finally:
        monkey.undo()


def _validated_infos():
    from flash.providers.core.base import GPU_INFO

    return [info for info in GPU_INFO.values() if info.validated]


def test_effective_spec_validation_accepts_an_allocator_narrowed_count():
    """A ceiling satisfied with FEWER cards must survive the pre-provision persistence check.

    gpu.count is a ceiling, and `_spec_with_gpu` writes the SELECTED count onto the worker spec (the
    worker sizes its ranks from it, the provider payload rents it). `_validate_effective_spec`
    compared that against the authored ceiling, so every narrowed run raised
    "persisted effective preparation does not match the public run" and died before reaching any
    provider -- newly reachable here because this PR removed the submit gate that used to reject
    unpinned `count > 1` outright. Narrowing only: claiming MORE cards than authorized is still an
    integrity failure.
    """
    from flash.core.spec import JobSpec, gpu_count_of
    from flash.runner.lifecycle.preparation import _validate_effective_spec
    from flash.runner.supervise.lifecycle import _spec_with_gpu

    public = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3-4B",
            "algorithm": "sft",
            "seed": 42,
            "environment": {"id": "will/gsm8k"},
            "gpu": {"count": 4},
        }
    )
    # the fixture must actually narrow, or it proves nothing about the comparison.
    narrowed = _spec_with_gpu(public, _TRI_PROVIDER_GPU, 2)
    assert gpu_count_of(narrowed) == 2 < gpu_count_of(public)
    _validate_effective_spec(public, narrowed)  # must not raise
    # the full ceiling is still fine, and so is a single card.
    _validate_effective_spec(public, _spec_with_gpu(public, _TRI_PROVIDER_GPU, 4))
    _validate_effective_spec(public, _spec_with_gpu(public, _TRI_PROVIDER_GPU, 1))
    # widening past the authored ceiling is NOT an allocator narrowing -> still rejected.
    with pytest.raises(ValueError, match="does not match the public run"):
        _validate_effective_spec(public, _spec_with_gpu(public, _TRI_PROVIDER_GPU, 8))


def test_unpinned_quote_bills_the_allocator_selected_count():
    """The exact lifecycle quote charges selected count and timing, never the authored ceiling."""
    from flash.cost.analytical import estimate_cost
    from flash.cost.types import RunConfig
    from flash.providers.core.base import Candidate

    config = RunConfig(model_id="Qwen/Qwen3.5-9B", method="sft", steps=100, gpu_count=8)
    one = estimate_cost(config, allocation=Candidate("runpod", "H100", 3.29, 80, 1))
    two = estimate_cost(config, allocation=Candidate("runpod", "H100", 3.29, 80, 2))

    assert one.gpu_count == 1
    assert two.gpu_count == 2
    assert two.train_seconds < one.train_seconds
    assert two.total_usd < 2 * one.total_usd


def test_vast_keeps_confirmed_shapes_when_another_count_query_fails():
    """One count's market blip must not discard shapes another count already confirmed rentable.

    Each card count is its own Vast market search. Raising on the first failure threw away
    candidates from earlier successful queries, so a Vast-only run holding a live 4-card offer still
    failed allocation when the 2-card query blipped -- unrecoverable at `max_retries=0`. Only a
    total lookup failure may raise.
    """
    from flash.providers.core.base import AllocationConstraints, CapacityLookupError
    from flash.providers.vast.execution.provider import VastProvider

    provider = VastProvider()
    fitting_vram = min(g.vram_gb for g in provider.gpu_classes())
    tri = {g.name for g in provider.gpu_classes()}
    name = sorted(tri)[0]

    calls: list[int] = []

    def _flaky(_vram, _disk, _wall, *, num_gpus, **_kw):
        calls.append(num_gpus)
        if num_gpus == 4:
            return {name: 1.5}
        raise RuntimeError("market blip")

    def _all_dead(_vram, _disk, _wall, *, num_gpus, **_kw):
        calls.append(num_gpus)
        raise RuntimeError("market blip")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("flash.providers.vast.client.pricing.live_candidate_rates", _flaky)
        got = provider.live_candidates(
            fitting_vram, AllocationConstraints(max_gpu_count=4, disk_gb=100.0)
        )
        # the confirmed 4-card offer survives the 2-card and 1-card failures.
        assert [c.gpu_count for c in got] == [4]
        assert calls == [4, 2, 1]  # every count still attempted, none short-circuited

        calls.clear()
        monkey.setattr("flash.providers.vast.client.pricing.live_candidate_rates", _all_dead)
        # nothing confirmed anywhere -> still retryable, not a terminal "no GPU fits".
        with pytest.raises(CapacityLookupError):
            provider.live_candidates(
                fitting_vram, AllocationConstraints(max_gpu_count=4, disk_gb=100.0)
            )
    finally:
        monkey.undo()


def test_lambda_single_card_pricing_survives_a_catalog_outage():
    """A catalog blip must not downgrade a 1x Lambda quote to the static list price.

    ``instance_type_for`` returns the registry name unconditionally at count 1 -- the catalog is
    only consulted to resolve a MULTI-card suffix (gpu_1x_h100_pcie vs gpu_8x_h100_sxm5). Fetching
    it on every call put a second network round-trip on the single-card path and, worse, let one
    ``/instance-types`` failure escape into the shared ``except`` and fall back to the static rate
    for EVERY Lambda quote -- even though the per-type price lookup underneath would have answered
    fine. Live pricing that silently reverts to a stale snapshot misprices allocation and billing.
    """
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.lambda_.client.pricing import _STATIC_RATES, hourly_rate

    live_rate, calls = 2.49, []

    def _dead_catalog(*_a, **_k):
        calls.append("catalog")
        raise RuntimeError("instance-types outage")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(lambda_api, "list_instance_types", _dead_catalog)
        monkey.setattr(lambda_api, "instance_type_price_usd_hr", lambda *_a, **_k: live_rate)
        # the static fallback must actually differ, or this asserts nothing.
        assert _STATIC_RATES["H100"] != live_rate
        assert hourly_rate("H100") == live_rate
        assert calls == []  # the 1x path never asked for the catalog at all
        # multi-card still resolves through the catalog, so its outage still falls back.
        assert hourly_rate("H100", gpu_count=4) == _STATIC_RATES["H100"] * 4
        assert calls == ["catalog"]
    finally:
        monkey.undo()


def test_structured_opd_compiler_vocab_is_card_independent():
    """The worker's structured-OPD check reads vocabulary size, which no card shape can change.

    This used to thread the allocated gpu/gpu_count down so an open model's fit check would judge the
    shape it was actually placed on -- resolving a shardable run as one card re-raised "does not fit"
    on the worker, after the box was rented. With only curated models trainable there is no fit check
    inside resolution, so the shape cannot reach it and cannot reject anything: the resolved
    vocabulary is a function of the model and its revision alone.
    """
    from flash.engine.worker.train.opd.orchestration.validation import _resolve_compiler_vocab_size

    seen: list[tuple] = []

    def _fake_resolve(model, algo, model_revision="", **kwargs):
        seen.append((model, algo, model_revision, tuple(sorted(kwargs))))
        return SimpleNamespace(vocab_size=151936)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("flash.core.catalog.resolve_model", _fake_resolve)
        assert (
            _resolve_compiler_vocab_size(model_id="Qwen/Qwen3.5-9B", model_revision="a" * 40)
            == 151936
        )
        # the pin reaches resolution, and no card shape is passed along with it.
        assert seen == [("Qwen/Qwen3.5-9B", "opd", "a" * 40, ())]
    finally:
        monkey.undo()


def _vllm_tp_axis_failures(info, tp: int) -> list[str]:
    """Names of the vllm tensor-parallel axes a catalog row violates at width ``tp``.

    Rebuilt from the row's OWN recorded geometry, never hardcoded, so a future row is judged on
    what it actually declares. Empty list == the rollout engine can shard this row at this width.
    """
    failures = []
    if info.num_attention_heads % tp:
        failures.append("query_heads")
    # kv heads PARTITION when there are at least as many as ranks, and REPLICATE otherwise. which
    # branch applies flips with the width, so both are expressed rather than assuming one.
    kv = info.num_key_value_heads
    if (kv % tp) if kv >= tp else (tp % kv):
        failures.append("kv_heads")
    if info.linear_num_value_heads % tp:
        failures.append("gdn_value_heads")
    conv_dim = (
        info.linear_key_head_dim * info.linear_num_key_heads * 2
        + info.linear_value_head_dim * info.linear_num_value_heads
    )
    if conv_dim % tp:
        failures.append("gdn_conv_dim")
    return failures


def test_every_catalog_row_shards_on_every_vllm_tensor_parallel_axis():
    """A rentable width must satisfy EVERY vllm tp axis, not just the query heads we gate on.

    grpo and opd hand the rented card count straight to the rollout engine as
    ``tensor_model_parallel_size`` (``train/rl/verl_config.py``, ``train/opd/overrides.py``), and
    vllm shards four INDEPENDENT axes when the engine initializes -- which happens after the box is
    rented and billing has started. ``geometry_safe_gpu_cap`` certifies only the first of them and
    documents that limit; the other three hold today purely as a property of this catalog, and a
    new row is exactly what would break that silently.

    The axes, read out of vllm 0.19.1 rather than inferred from docs:

      * query heads -- ``config/model.py`` raises when ``total_num_attention_heads % tp != 0``.
      * kv heads -- ``models/qwen3_next.py`` PARTITIONS them when ``kv >= tp`` (needs
        ``kv % tp == 0``) and REPLICATES otherwise (needs ``tp % kv == 0``).
      * GDN value heads and conv width -- ``layers/mamba/mamba_utils.py`` sizes the recurrent state
        with ``divide(num_v_heads, tp)`` and ``divide(conv_dim, tp)``, and ``divide`` asserts exact
        divisibility.

    This pins the whole surface at the widths a run can actually be rented at, so a row that only
    divides on query heads is caught here rather than on a paid box.
    """
    from flash.core.catalog import MODELS
    from flash.providers.core.allocator import geometry_safe_gpu_cap
    from flash.providers.core.base import rentable_gpu_counts

    for model_id, info in MODELS.items():
        cap = geometry_safe_gpu_cap(model_id, 8)
        admitted = [width for width in rentable_gpu_counts(8) if width <= cap]
        assert admitted, f"{model_id}: no rentable width survives the cap"
        for width in admitted:
            failures = _vllm_tp_axis_failures(info, width)
            assert not failures, (
                f"{model_id} is admitted at {width} cards but vllm cannot shard it: {failures}"
            )


def test_the_tensor_parallel_axis_check_fails_a_row_vllm_would_reject():
    """The axis check must have teeth: a row that only divides on query heads has to fail.

    Without this, ``test_every_catalog_row_shards_on_every_vllm_tensor_parallel_axis`` would pass
    just as happily against a helper that never returns a failure, and would go on passing after a
    refactor quietly broke it. The counter-example is the shape the query-head gate cannot see: 16
    query heads (divides 8 cleanly) over kv and GDN widths that do not.
    """
    from dataclasses import replace

    from flash.core.catalog import MODELS

    row = replace(
        MODELS["Qwen/Qwen3.5-9B"],
        num_attention_heads=16,
        num_key_value_heads=3,
        linear_num_value_heads=12,
    )
    # the axis the allocator certifies is clean, so a query-head-only check would admit this.
    assert row.num_attention_heads % 8 == 0
    # the axes it does NOT certify reject the row, and each violation is named.
    assert _vllm_tp_axis_failures(row, 8) == ["kv_heads", "gdn_value_heads"]
    # genuinely width-dependent rather than always-failing: 12 value heads divide 4, so only the
    # kv axis (3 can neither partition across nor replicate into 4 ranks) still objects.
    assert _vllm_tp_axis_failures(row, 4) == ["kv_heads"]
    # a real row stays clean at every width, so the helper is not merely rejecting everything.
    assert _vllm_tp_axis_failures(MODELS["Qwen/Qwen3.8-27B"], 8) == []
    # the conv axis is reported, and ISOLATED: conv_dim is `head_k*num_k*2 + head_v*num_v`, which
    # the catalog's 128-wide head dims keep a multiple of 8 for any head count -- so no value-head
    # count alone can break it. Narrow the dims instead: 1*3*2 + 1*16 = 22 fails while the value
    # heads (16) still divide 8, leaving conv_dim as the only objection.
    narrow_conv = replace(
        MODELS["Qwen/Qwen3.5-9B"],
        linear_key_head_dim=1,
        linear_num_key_heads=3,
        linear_value_head_dim=1,
        linear_num_value_heads=16,
    )
    assert _vllm_tp_axis_failures(narrow_conv, 8) == ["gdn_conv_dim"]
