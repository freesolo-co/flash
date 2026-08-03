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


def test_lambda_resolves_a_multi_card_sku_that_renames_its_suffix():
    """The count rewrite alone can name a type Lambda does not sell; the catalog is authoritative.

    H100 is registered as ``gpu_1x_h100_pcie`` but Lambda's multi-card family is ``_sxm5``, so the
    derived ``gpu_8x_h100_pcie`` does not exist. ``regions_with_capacity`` answers [] for an unknown
    type, which reads as "sold out" rather than "wrong name" -- real capacity disappears silently.
    """
    from flash.providers.lambdalabs.gpus import instance_type_for

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


def test_lambda_instance_type_never_reaches_the_network():
    """``instance_type_for`` must stay offline: it is called from pricing, which is called from it.

    An in-function catalog fetch turns a pure name lookup into live I/O on every sizing call and
    deadlocks the offline test path. Callers holding a catalog pass it; nobody else pays.
    """
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs.gpus import instance_type_for

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


def test_lambda_capacity_refresh_keeps_the_allocated_card_count():
    """The mid-walk capacity refresh must re-search the SAME shape, not fall back to one card.

    ``usable_instances`` defaults ``gpu_count`` to 1, so a refresh that omits it silently rents a
    1-card box while the worker still starts n ranks -- the exact billing exposure the submit-path
    test guards, reached through the other door.
    """
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.lambdalabs.jobs.builders import LambdaInstance

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
            seed=spec.seed,
            instances=[_inst("us-east-1")],
            deadline_at=9_999_999_999.0,
        )
    finally:
        monkey.undo()

    assert refreshed_with.get("count") == 4, "refresh dropped the count and would rent one card"
    # the refresh is what produced the launch, so the assertion above cannot pass vacuously.
    assert handle.instance_id == "i-42"


def test_vast_capacity_refresh_keeps_the_allocated_card_count():
    """Same contract on Vast: ``usable_offers`` defaults ``num_gpus`` to 1 on an omitted refresh."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vj
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
            seed=spec.seed,
            offers=[_offer(1)],
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
        from flash.providers.lambdalabs import api as lambda_api
        from flash.providers.lambdalabs import jobs as lj
        from flash.providers.lambdalabs.jobs.builders import LambdaInstance

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
                seed=spec.seed,
                instances=[inst],
                deadline_at=9_999_999_999.0,
            )
        finally:
            monkey.undo()
    else:
        from flash.providers.vast import api as vast_api
        from flash.providers.vast import jobs as vj
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
                seed=spec.seed,
                offers=[offer],
                deadline_at=9_999_999_999.0,
            )
        finally:
            monkey.undo()

    assert handle.hourly_usd == pytest.approx(per_card * cards)
    # and the realized-COGS reader must agree, since that is the consumer that would under-bill.
    from flash.providers.realized import realized_cost_for_remote

    realized = realized_cost_for_remote(
        {
            "provider": provider,
            "hourly_usd": handle.hourly_usd,
            "instance_id": handle.instance_id,
            "started_ts": 0.0,
        },
        start=0.0,
        end=3600.0,
    )
    assert realized is not None
    assert realized.realized_usd == pytest.approx(per_card * cards), (
        "one hour on an n-card box must bill the whole box"
    )


def test_retry_bookkeeping_distinguishes_card_counts_of_one_class():
    """A class at 2 cards and the same class at 4 are different rentable shapes.

    This PR made multiple counts per class reachable (the fit filter keeps every shape that fits),
    so a 2-tuple ``(provider, gpu)`` retry key marks EVERY count tried the moment one fails and the
    walk skips a wider shape that would have fit.
    """
    from flash.providers.base import Candidate
    from flash.runner.lifecycle import _select_candidate, _shape_key

    two = Candidate(
        provider="runpod", gpu=_TRI_PROVIDER_GPU, hourly_usd=3.0, vram_gb=80, gpu_count=2
    )
    four = Candidate(
        provider="runpod", gpu=_TRI_PROVIDER_GPU, hourly_usd=3.0, vram_gb=80, gpu_count=4
    )

    assert _shape_key(two) != _shape_key(four), "count must be part of the retry identity"

    # having burned the 2-card shape, the walk must still reach the 4-card one rather than treat
    # the whole class as exhausted.
    chosen = _select_candidate([two, four], set(), {_shape_key(two)})
    assert chosen is four


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


def test_gpu_count_is_honoured_by_parse_time_sizing():
    """`--gpus N` must reach parse-time sizing, or a large run is rejected before sharding.

    This is the half of "multi-card works" that lives before the allocator. `provisional_gpu` sized
    every run against ONE card, so a 27B run needing 234 GB was rejected at parse time with "no
    validated GPU class has >= 234 GB VRAM" no matter what the user passed to `--gpus`: the flag
    was inert for exactly the runs that need it. The allocator would happily have rented 2 x H200.

    Pinned to a need no single validated class holds (180 GB max) but four cards do, so the
    assertion cannot pass by accident on a run that fits one card anyway.
    """
    from flash.providers.base import UnsupportedGpuError, cheapest_gpu, combined_vram_gb

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
        from flash.providers.base import get_gpu_info

        assert combined_vram_gb(get_gpu_info(chosen).vram_gb, count) >= need


def test_gpu_count_above_the_combination_cap_does_not_oversell():
    """A count the allocator will never combine must not be honoured by sizing either.

    gpu.count accepts up to 8 but the allocator caps combinations at MAX_COMBINATION_CARDS. Sizing
    against 8 would admit a spec at parse time only for submit to reject it -- the same parse/submit
    divergence this fix removes, pointing the other way.
    """
    from flash.providers.base import MAX_COMBINATION_CARDS, cheapest_gpu, combined_vram_gb

    # a need the cap can hold but a smaller shape cannot, so "sized as 8" and "sized as the cap"
    # would pick different classes and the assertion is failable.
    need = 500
    assert combined_vram_gb(180, MAX_COMBINATION_CARDS) >= need > combined_vram_gb(180, 2)
    assert cheapest_gpu(need, gpu_count=8) == cheapest_gpu(need, gpu_count=MAX_COMBINATION_CARDS)


def test_open_model_validation_is_judged_on_the_allocated_shape():
    """`model_policy = "allow"` must size an open model on its card COUNT, not on one card.

    Threading the count into parse-time sizing deliberately picks a SMALLER per-card class as the
    ceiling widens (a 32B run resolves to RTX Pro 6000 at 1 card, RTX 5090 at 4). If the open-model
    fit check still evaluates one card, that smaller class is then rejected as `too_big` -- so
    `--gpus 4` refuses a model `--gpus 1` accepts, and multi-card stays unusable for exactly the
    large open models it exists to serve.
    """
    from flash.catalog import resolve_model
    from flash.engine.vram import check_fit

    model, card = "acme/open-32b", "RTX 5090"
    monkey = pytest.MonkeyPatch()
    try:
        # size is stubbed so the assertion is about the COUNT, not about HF reachability.
        monkey.setattr("flash.engine.vram.fetch_hf_params_b", lambda _m, **_k: 32.0)
        # one card genuinely cannot hold it -> the count is what makes the wide shape legal.
        assert check_fit(model, "sft", card, gpu_count=1).verdict == "too_big"
        assert check_fit(model, "sft", card, gpu_count=4).verdict != "too_big"
        # the real resolution path must follow: reject on one card, accept on four.
        with pytest.raises(ValueError, match="does not fit"):
            resolve_model(model, "sft", policy="allow", gpu=card, gpu_count=1)
        assert resolve_model(model, "sft", policy="allow", gpu=card, gpu_count=4).params_b == 32.0
    finally:
        monkey.undo()


def test_unpinned_sold_out_live_market_stays_retryable():
    """An unpinned run on a live-market provider must not die terminally when stock runs out.

    `live_capacity` was consulted only for an EXACT pin, so an auto-allocated run whose fitting
    class was merely sold out fell through to `UnsupportedGpuError` -- which the lifecycle treats
    as terminal, killing a run a retry would have placed. A genuinely oversized run must still fail
    terminally, so the distinction is gated on whether any offered shape could hold the run at all.
    """
    import flash.providers.allocator as alloc
    from flash.providers.base import CapacityLookupError, UnsupportedGpuError

    class _SoldOutLiveMarket:
        name = "lambda"
        live_capacity = True

        def live_candidates(self, need, constraints):
            return []  # structurally offered, nothing free right now

        def gpu_classes(self):
            from flash.providers import base

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
    from flash.providers.base import GPU_INFO

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
    from flash.runner import _validate_effective_spec
    from flash.runner.lifecycle import _spec_with_gpu
    from flash.spec import JobSpec, gpu_count_of

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


def test_open_model_fit_sizes_on_the_rentable_count_not_the_raw_ceiling():
    """`check_fit` must clamp its count the same way sizing and submit do.

    Only powers of two up to MAX_COMBINATION_CARDS are ever rented, so a ceiling of 3 buys 2 cards
    and a ceiling of 8 buys 4. Sizing the open-model gate on the raw ceiling would accept a shape
    allocation never provisions -- the parse/submit divergence this parameter exists to close,
    pointing the other way.
    """
    from flash.engine.vram import check_fit
    from flash.providers.base import MAX_COMBINATION_CARDS

    model, card = "acme/open-32b", "RTX 5090"
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("flash.engine.vram.fetch_hf_params_b", lambda _m, **_k: 32.0)
        # a ceiling of 3 must be judged as 2 cards, not 3.
        assert check_fit(model, "sft", card, gpu_count=3).verdict == (
            check_fit(model, "sft", card, gpu_count=2).verdict
        )
        # and above the combination cap, as the cap.
        assert check_fit(model, "sft", card, gpu_count=8).verdict == (
            check_fit(model, "sft", card, gpu_count=MAX_COMBINATION_CARDS).verdict
        )
        # the estimate reports the shape it JUDGED, so the message cannot contradict the verdict.
        assert check_fit(model, "sft", card, gpu_count=3).gpu_count == 2
        assert "2x" in check_fit(model, "sft", card, gpu_count=3).describe()
    finally:
        monkey.undo()


def test_unpinned_quote_bills_the_rentable_count_not_the_ceiling():
    """An unpinned multi-card quote must not charge for cards the ceiling never buys.

    The unpinned branch skips `allocate()` and billed `config.gpu_count` verbatim, so a `--gpus 3`
    run was quoted for 3 cards while submit rents 2. That quote is persisted at submit and charged
    verbatim by `_status_estimated_charge`, so the over-count is a real overbill, and this path is
    newly reachable because the PR removed the gate that rejected unpinned `count > 1`.
    """
    from flash.cost.analytical import estimate_cost
    from flash.cost.types import RunConfig

    def _quote(count: int):
        # no gpu_type -> the unpinned branch, which is the one that billed the raw ceiling.
        # cost estimation is catalog-only, so the model has to be a catalog row.
        return estimate_cost(
            RunConfig(model_id="Qwen/Qwen3.5-4B", method="sft", steps=100, gpu_count=count)
        )

    three, two = _quote(3), _quote(2)
    # a ceiling of 3 buys 2 cards, so it must be quoted as 2 -- not as 3.
    assert three.gpu_count == 2 == two.gpu_count
    assert three.total_usd == pytest.approx(two.total_usd)
    # and the quote still scales with a count that IS rentable, or the assertion above is vacuous.
    assert _quote(4).gpu_count == 4
    assert _quote(4).total_usd > two.total_usd


def test_vast_keeps_confirmed_shapes_when_another_count_query_fails():
    """One count's market blip must not discard shapes another count already confirmed rentable.

    Each card count is its own Vast market search. Raising on the first failure threw away
    candidates from earlier successful queries, so a Vast-only run holding a live 4-card offer still
    failed allocation when the 2-card query blipped -- unrecoverable at `max_retries=0`. Only a
    total lookup failure may raise.
    """
    from flash.providers.base import AllocationConstraints, CapacityLookupError
    from flash.providers.vast import VastProvider

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
        monkey.setattr("flash.providers.vast.pricing.live_candidate_rates", _flaky)
        got = provider.live_candidates(
            fitting_vram, AllocationConstraints(max_gpu_count=4, disk_gb=100.0)
        )
        # the confirmed 4-card offer survives the 2-card and 1-card failures.
        assert [c.gpu_count for c in got] == [4]
        assert calls == [4, 2, 1]  # every count still attempted, none short-circuited

        calls.clear()
        monkey.setattr("flash.providers.vast.pricing.live_candidate_rates", _all_dead)
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
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs.pricing import _STATIC_RATES, hourly_rate

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


def test_structured_opd_validation_is_fit_checked_on_the_allocated_card_count():
    """The worker's structured-OPD check must judge the model on the shape it was ALLOCATED.

    Under ``model_policy="allow"`` an open model is fit-checked during ``resolve_model``. This path
    resolved without ``gpu_count``, so it took the single-card default and re-raised "does not fit"
    for a shardable run that submit had already placed -- on the worker, after the GPU instance was
    rented and billed. Every other gate on this branch is count-aware; this one was not.
    """
    from flash.opd_validation import _resolve_compiler_vocab_size

    seen: list[int] = []

    def _fake_resolve(_model, _algo, **kwargs):
        seen.append(int(kwargs.get("gpu_count", 1)))
        return SimpleNamespace(vocab_size=151936)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("flash.catalog.resolve_model", _fake_resolve)
        _resolve_compiler_vocab_size(
            model_id="acme/open-32b",
            model_revision="",
            model_policy="allow",
            gpu="RTX 5090",
            gpu_count=4,
        )
        # the allocated count reaches the fit check, rather than silently defaulting to one card.
        assert seen == [4]
        # and the default is still one card when nothing is threaded, so the assert above is real.
        _resolve_compiler_vocab_size(
            model_id="acme/open-32b", model_revision="", model_policy="allow", gpu="RTX 5090"
        )
        assert seen == [4, 1]
    finally:
        monkey.undo()
