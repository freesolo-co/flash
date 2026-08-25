"""Cost estimator: GPU compute table, pricing/VRAM lookups, cheapest-fit selection.

No network. The compute table and the selection rule must stay consistent with the
RunPod GPU registry in ``flash.providers.base``.
"""

from __future__ import annotations

import pytest

from flash.cost.facts import GPU_COMPUTE_TFLOPS, gpu_tflops, gpu_vram_gb
from flash.providers.core.base import GPU_INFO


def test_compute_table_only_lists_real_classes():
    # Every GPU we assign a TFLOPS figure to must be a real managed class (no drift).
    for name in GPU_COMPUTE_TFLOPS:
        assert name in GPU_INFO, f"{name} is not a managed GPU class"


def test_gpu_tflops_known_and_default():
    assert gpu_tflops("RTX 5090") == GPU_COMPUTE_TFLOPS["RTX 5090"]
    assert gpu_tflops("RTX 5090") > gpu_tflops("RTX 4090")  # newer/faster
    assert gpu_tflops("totally-unknown-gpu") == 100.0  # documented default


def test_lambda_a100_40gb_band_has_real_tflops():
    assert gpu_tflops("A100 SXM 40GB") == GPU_COMPUTE_TFLOPS["A100 SXM 40GB"]
    assert gpu_tflops("A100 SXM 40GB") > 100.0


def test_vram_tracks_the_registry():
    for name, gpu in GPU_INFO.items():
        assert gpu_vram_gb(name) == gpu.vram_gb


def test_unknown_gpu_vram_lookup_raises():
    with pytest.raises(KeyError):
        gpu_vram_gb("Tesla T4")


def test_effective_train_tflops_caps_b200_at_h200_class():
    # b200/sm100 training falls back to portable kernels, so realized training throughput is
    # h200-class, not the 2.25 pflops peak. the cost model must not treat b200 as faster than h200.
    from flash.cost.facts import effective_train_tflops

    assert gpu_tflops("B200") == 2250.0  # raw peak unchanged (vram/serving still use gpu_tflops)
    # assert the RELATIONSHIP, not a literal: the cap tracks whatever the H200 entry is, and that
    # entry is anchored to a measured rate that recalibration can move.
    assert effective_train_tflops("B200") == effective_train_tflops("H200")
    assert effective_train_tflops("B200") < gpu_tflops("B200")


def test_effective_train_tflops_is_peak_for_uncapped_classes():
    from flash.cost.facts import effective_train_tflops

    for name in ("H100", "H200", "A100 SXM", "RTX 4090", "B200"):
        # only b200 is capped; every other class keeps its peak.
        expected = effective_train_tflops("H200") if name == "B200" else gpu_tflops(name)
        assert effective_train_tflops(name) == expected


def test_nvlink_classification_is_by_form_factor():
    from flash.cost.facts import has_nvlink

    # sxm datacenter parts carry nvlink.
    assert has_nvlink("A100 SXM")
    assert has_nvlink("A100 SXM 40GB")
    assert has_nvlink("H100")
    # geforce parts do not; the 4090 dropped the nvlink connector entirely. l40s is a pcie board.
    assert not has_nvlink("RTX 4090")
    assert not has_nvlink("L40S")
    # an unclassified class must fall to the conservative side rather than raise.
    assert not has_nvlink("some-unlisted-gpu")


def test_nvlink_classification_tracks_the_provisioned_board():
    """Classification must follow the pin a MULTI-CARD run actually lands on.

    Multi-card provisioning is runpod-only, and runpod pins H100 to the HBM3 sxm part while
    negating the pcie/NVL boards in the same pool. Assert against those pins rather than restating
    the classification, so re-pinning a class to a different board fails here instead of silently
    pricing it on an interconnect it no longer has.
    """
    from flash.cost.facts import has_nvlink
    from flash.providers.core.base import GPU_INFO
    from flash.providers.runpod.client.gpus import _POOL_MEMBERS_MISSING_FROM_SDK

    assert GPU_INFO["H100"].enum_member == "NVIDIA_H100_80GB_HBM3"  # sxm, not the pcie board
    assert has_nvlink("H100")
    # the non-sxm members of the same runpod pool are negated, so a pin cannot land on them.
    assert _POOL_MEMBERS_MISSING_FROM_SDK["ADA_80_PRO"] == ("NVIDIA H100 PCIe", "NVIDIA H100 NVL")


def test_multi_card_speedup_is_interconnect_aware():
    from flash.cost.analytical import multi_card_speedup

    # MEASURED on runpod with one identical 2-card fsdp benchmark per interconnect:
    #   nvlink 2x A100-SXM4-80GB 1.7675x, pcie 2x L40S 1.4212x.
    # the model must land near each measurement, not split the difference with one constant.
    assert multi_card_speedup(2, "A100 SXM") == pytest.approx(1.7675, abs=0.05)
    assert multi_card_speedup(2, "RTX 4090") == pytest.approx(1.4212, abs=0.05)
    # one card is exactly one card on any fabric.
    for name in ("A100 SXM", "RTX 4090", "unknown"):
        assert multi_card_speedup(1, name) == 1.0


def test_pcie_scaling_is_never_credited_nvlink_bandwidth():
    """The invariant the measurement exists to protect.

    A pcie pair delivered 1.42x where the old global 0.85 constant claimed 1.70x. Crediting that
    difference lets a 2-card pcie combination win a ranking on scaling it does not have, and then
    bills both cards for the longer wall time. Assert the ORDERING, so recalibrating either
    constant cannot silently reintroduce the inversion.
    """
    from flash.cost.analytical import multi_card_speedup

    for n in (2, 3, 4):
        assert multi_card_speedup(n, "RTX 4090") < multi_card_speedup(n, "A100 SXM")
        # and neither may ever claim linear scaling, which no fabric delivers.
        assert multi_card_speedup(n, "A100 SXM") < n


def test_multi_card_speedup_never_decreases_with_card_count():
    """Adding a card must never model as slower.

    The raw geometric curve turns back down below ~0.72 scaling (at the measured pcie 0.71: 3 cards
    1.512x, 4 cards 1.432x). Left unclamped the allocator would price a 4-card pcie combination as
    slower than a 3-card one and reject cards that do add throughput. No real fabric loses aggregate
    throughput when a card is added; the extrapolation flattens, it does not reverse.
    """
    from flash.cost.analytical import multi_card_speedup

    for name in ("A100 SXM", "RTX 4090", "H100", "unlisted-class"):
        vals = [multi_card_speedup(n, name) for n in range(1, 9)]
        assert vals == sorted(vals), f"{name} speedup decreases: {vals}"


def test_sft_shards_by_data_like_grpo_and_opd(monkeypatch):
    """sft must read the fsdp data-parallel constant, the same one grpo/opd read.

    sft used to pin ulysses_sp_size to the card count (dp_size == 1: pure sequence parallelism) and
    carried its own scaling pair for it. Sequence parallelism is wrong for the catalog's GDN
    hybrids, so `sft_train_runner` pins it off and fsdp splits the batch instead. All three
    algorithms now pay the same collective, so one constant must drive all three -- a surviving
    sft-only constant would be a quote nothing executes.
    """
    from flash.cost import analytical
    from flash.cost.types import RunConfig

    assert not hasattr(analytical, "MULTI_CARD_SCALING_SP_PCIE"), (
        "the sequence-parallel constant outlived the strategy it priced"
    )
    assert not hasattr(analytical, "sequence_parallel_speedup"), (
        "the sequence-parallel curve outlived the strategy it priced"
    )

    def cfg(method: str) -> RunConfig:
        return RunConfig(model_id="Qwen/Qwen3.5-9B", method=method, steps=8)

    # 0.95 is well clear of the non-decreasing clamp (which pins anything at or below 0.5 to 1.0x
    # at 2 cards), so a wrong reading cannot coincidentally land on the right number.
    monkeypatch.setattr(analytical, "MULTI_CARD_SCALING_PCIE", 0.95)
    monkeypatch.setattr(analytical, "MULTI_CARD_SCALING_NVLINK", 0.95)

    for card in ("RTX 4090", "A100 SXM"):
        for method in ("sft", "grpo", "opd"):
            speedup = analytical.method_card_speedup(cfg(method), 2, card)
            assert speedup == pytest.approx(2 * 0.95), (
                f"{method} on {card} ignored the data-parallel constant: {speedup}"
            )


def test_multi_card_sft_quote_moves_with_the_data_parallel_constant(monkeypatch):
    """The seam has to reach the shipped dollar figure, not just the helper.

    A 9B sft run pinned to 2x RTX 4090 is the reachable multi-card sft cell: one 4090 cannot hold
    it (needs 32 GB), two can, and sft has no step floor so the whole gpu-bound half is the token
    compute term. That makes the scaling constant map straight to the price -- if the quote does
    not move when it changes, the fix never reached estimate_cost().
    """
    from flash.cost import analytical
    from flash.cost.types import RunConfig

    spec = RunConfig(
        model_id="Qwen/Qwen3.5-9B",
        method="sft",
        steps=64,
        seq_len=2048,
        batch_size=8,
        train_tokens=1_000_000,
        gpu_type="RTX 4090",
        gpu_count=2,
    )
    before = analytical.estimate_cost(spec)
    assert before.gpu_count == 2, "the reachable multi-card sft cell collapsed to one card"

    # degrade the realized scaling: the same work must take longer and cost more. 0.6 stays clear
    # of the non-decreasing clamp, so this exercises the constant rather than the floor under it.
    monkeypatch.setattr(analytical, "MULTI_CARD_SCALING_PCIE", 0.6)
    after = analytical.estimate_cost(spec)
    assert after.train_seconds > before.train_seconds
    assert after.total_usd > before.total_usd


def test_a_vast_multi_card_run_is_not_credited_with_nvlink_scaling():
    """Vast sells a canonical class as a market, so its nvlink membership cannot be trusted.

    `providers/base.py` normalizes the explicit pcie aliases into nvlink-classed entries -- `H100
    PCIE` -> H100 and `A100 PCIE` -> A100 SXM 40GB -- and a Vast offer carries no interconnect
    field, so a multi-card combination can land on pcie boards while the class says nvlink. Vast
    does search multi-card (`rentable_gpu_counts` with `num_gpus=count`), so these combinations are
    real candidates, and crediting them with the measured nvlink curve prices them on bandwidth
    they may not have.
    """
    from flash.cost.analytical import multi_card_speedup
    from flash.cost.facts import has_nvlink

    for ambiguous in ("H100", "A100 SXM 40GB"):
        assert has_nvlink(ambiguous, "runpod"), (
            "runpod pins an exact gpu id per class, so its nvlink classification still holds"
        )
        assert not has_nvlink(ambiguous, "vast"), (
            f"a vast {ambiguous} combination was credited with nvlink scaling, but vast lists the "
            "pcie board under this same class and its offers carry no topology"
        )
        # and the credit is large enough to change a ranking: ~24% at 2 cards, ~80% at 4.
        assert multi_card_speedup(2, ambiguous, "vast") < multi_card_speedup(2, ambiguous, "runpod")
        assert multi_card_speedup(4, ambiguous, "vast") < multi_card_speedup(4, ambiguous, "runpod")

    # an unpinned/unknown provider keeps the class answer: this narrows a known-ambiguous market,
    # it does not downgrade everything that fails to name a provider.
    assert has_nvlink("H100", "")
    assert has_nvlink("H100", "lambda"), "lambda has no multi-card path, so nothing to narrow"
    # and a pcie class is unaffected in either direction.
    assert not has_nvlink("RTX 4090", "runpod")
    assert not has_nvlink("RTX 4090", "vast")


def test_a_vast_sharded_quote_reads_the_provider_off_the_run_config():
    """The narrowing is only real if it reaches the quote, not just the classifier.

    `method_card_speedup` is the one point where the card count and the run's provider are both in
    hand, and every sharded quote goes through it.
    """
    from flash.cost.analytical import method_card_speedup
    from flash.cost.types import RunConfig

    def cfg(provider, method="grpo"):
        return RunConfig(
            model_id="Qwen/Qwen3.5-9B", method=method, steps=10, provider=provider, gpu_count=2
        )

    for method in ("grpo", "sft"):
        vast = method_card_speedup(cfg("vast", method), 2, "H100")
        runpod = method_card_speedup(cfg("runpod", method), 2, "H100")
        assert vast < runpod, (
            f"a {method} quote on vast still divided by the nvlink multiplier, so a pcie box is "
            "quoted on scaling it may not deliver and can win the ranking on it"
        )
        # `auto` is not a provider; it must not be read as one and must keep the class answer.
        assert method_card_speedup(cfg("auto", method), 2, "H100") == runpod
        # an explicit provider argument outranks the config, which is what the ranking and
        # re-quote paths rely on -- their config carries no provider at all.
        assert method_card_speedup(cfg("auto", method), 2, "H100", "vast") == vast


def test_ranking_prices_the_authored_rollout_batch_not_the_recipe_default():
    """Regression: hardware ranking read the optimizer batch from the sft-only key.

    `RunConfig.batch_size` means "examples per optimizer update", but grpo/opd author that as
    `prompts_per_step` -- the schema rejects `batch_size` for them outright. Reading only the sft
    name left the ranker at None for every authored rollout batch, so it priced the recipe default
    (64) against an authored 32 and could select a costlier shape than the run needs. The persisted
    quote already reads the new key, so the two silently disagreed.
    """
    from flash.engine.plan.recipe import RECIPE
    from flash.providers.core.base import run_config_for_ranking

    authored = int(RECIPE.rl.prompts_per_step) // 2
    assert authored >= 1
    # must differ from the default, or the assertions below pass on the broken read too.
    assert authored != RECIPE.rl.prompts_per_step

    for algorithm in ("grpo", "opd"):
        train = {"epochs": 1, "group_size": 4, "prompts_per_step": authored}
        config = run_config_for_ranking("Qwen/Qwen3.5-9B", algorithm, train=train)
        assert config.batch_size == authored, algorithm
        assert config.normalized().batch_size == authored, algorithm

    # sft still authors the batch under its own name, and it is a different quantity.
    sft = run_config_for_ranking("Qwen/Qwen3.5-9B", "sft", train={"epochs": 1, "batch_size": 4})
    assert sft.batch_size == 4
    # an unauthored rollout batch still falls through to the recipe default.
    bare = run_config_for_ranking("Qwen/Qwen3.5-9B", "grpo", train={"epochs": 1})
    assert bare.batch_size is None
    assert bare.normalized().batch_size == RECIPE.rl.prompts_per_step


def test_ranking_caps_the_rollout_batch_at_the_retained_prompt_count():
    """Ranking must price the batch a step can actually reach, not the authored ceiling.

    Both rollout workers retain at most ``max_examples`` rows and then clamp the batch to what is
    left, so `prompts_per_step = 128` with `max_examples = 2` trains on 2. Ranking the raw 128 sizes
    hardware for a step that cannot happen, and disagrees with the persisted quote, which already
    takes this minimum -- so the run is billed against one number and provisioned against another.

    Only the validated ``[train] max_examples`` is a cap here. ``[environment.params] max_examples``
    is an opaque kwarg flash never enforces, and `_on_policy_example_count` deliberately does not
    take a min() with it; a run whose environment ignores the key would otherwise be underprovisioned.
    """
    from types import SimpleNamespace

    from flash.cost.spec import _on_policy_prompts_per_step, _on_policy_requested_prompts_per_step
    from flash.engine.plan.recipe import RECIPE
    from flash.providers.core.base import run_config_for_ranking

    for algorithm in ("grpo", "opd"):
        default = int((RECIPE.opd if algorithm == "opd" else RECIPE.rl).prompts_per_step)
        # (authored prompts_per_step, max_examples, expected ranked batch)
        for authored, retained, expected in (
            (128, 2, 2),  # the reported shape: pool far below the authored batch
            (2, 128, 2),  # pool above the batch changes nothing
            (8, 8, 8),  # equal
            (8, 0, 8),  # 0 means uncapped, not "a pool of zero"
            (8, None, 8),  # unset
            # UNAUTHORED batch: the recipe default must be capped too, because the workers clamp
            # whatever the batch resolved to. Leaving this to `normalized()` ranked 64/8 uncapped.
            (None, 2, 2),
            (None, None, None),  # nothing to cap; `normalized()` fills the default
            (None, 0, None),  # 0 is uncapped, so there is still nothing to resolve
            (None, 10**6, default),  # pool far above the default
        ):
            train = {"epochs": 1, "group_size": 4}
            if authored is not None:
                train["prompts_per_step"] = authored
            if retained is not None:
                train["max_examples"] = retained
            config = run_config_for_ranking("Qwen/Qwen3.5-9B", algorithm, train=train)
            assert config.batch_size == expected, (algorithm, authored, retained)
            # what actually gets priced, after the recipe fills any remaining None.
            assert config.normalized().batch_size == (
                expected if expected is not None else default
            ), (algorithm, authored, retained)

    # sft is untouched: its `batch_size` is examples per update on a dataset it may revisit across
    # epochs, so a small `max_examples` does not bound it the way a rollout pool bounds a step.
    sft = run_config_for_ranking(
        "Qwen/Qwen3.5-9B", "sft", train={"epochs": 1, "batch_size": 8, "max_examples": 2}
    )
    assert sft.batch_size == 8

    # and it agrees with the quote that bills the run, which is the disagreement being fixed.
    spec = SimpleNamespace(
        algorithm="grpo", train=SimpleNamespace(prompts_per_step=128, max_examples=2)
    )
    assert _on_policy_requested_prompts_per_step(spec) == 128
    assert _on_policy_prompts_per_step(spec, spec.train.max_examples) == 2


def test_the_persisted_quote_prices_the_same_batch_ranking_selects_hardware_for():
    """The quote a run is BILLED from must price the batch the ranker sized the card for.

    `runconfig_from_spec` fed `RunConfig.batch_size` the raw authored `prompts_per_step`, so a run
    with `prompts_per_step = 128, max_examples = 2` was quoted for a batch of 128 while training on
    2. The persisted quote is what a completed or cancelled run is charged against
    (`flash/runner/accounting/costs.py`), and it also gates the pre-submit affordability check, so the gap both
    overcharges and can reject an affordable run.

    Internally inconsistent too: `spec_steps` already counted steps against the CAPPED batch, so one
    quote mixed a capped step count with an uncapped per-step price.
    """
    from flash.core.spec import JobSpec
    from flash.cost.spec import runconfig_from_spec
    from flash.providers.core.base import run_config_for_ranking

    base = {
        "model": "Qwen/Qwen3.5-9B",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {"type": "H100", "count": 1},
    }
    for algorithm in ("grpo", "opd"):
        for label, train in (
            (
                "authored above the pool",
                {"epochs": 1, "group_size": 4, "prompts_per_step": 128, "max_examples": 2},
            ),
            ("unauthored, small pool", {"epochs": 1, "group_size": 4, "max_examples": 2}),
            (
                "authored below the pool",
                {"epochs": 1, "group_size": 4, "prompts_per_step": 2, "max_examples": 128},
            ),
            # no pool size, horizon stated instead -- quotable, and nothing to cap against.
            (
                "authored, max_steps horizon",
                {"epochs": 1, "group_size": 4, "prompts_per_step": 32, "max_steps": 10},
            ),
            (
                "uncapped pool, max_steps horizon",
                {
                    "epochs": 1,
                    "group_size": 4,
                    "prompts_per_step": 8,
                    "max_examples": 0,
                    "max_steps": 10,
                },
            ),
        ):
            spec = JobSpec.from_dict(
                {**base, "algorithm": algorithm, "run_id": "q", "train": train}
            )
            quoted = runconfig_from_spec(spec).normalized().batch_size
            ranked = (
                run_config_for_ranking("Qwen/Qwen3.5-9B", algorithm, train=train)
                .normalized()
                .batch_size
            )
            # compared after normalization: the two reach the recipe default by different routes,
            # and it is the effective number that has to match, not how each got there.
            assert quoted == ranked, (algorithm, label, quoted, ranked)

    # the headline shape, pinned to its literal value so a change to BOTH sides cannot pass silently.
    capped = JobSpec.from_dict(
        {
            **base,
            "algorithm": "grpo",
            "run_id": "q",
            "train": {"epochs": 1, "group_size": 4, "prompts_per_step": 128, "max_examples": 2},
        }
    )
    assert runconfig_from_spec(capped).normalized().batch_size == 2

    # a stated horizon needs no pool size, so capping must not turn `max_steps` into a refusal.
    # `spec_steps` returns before asking for a row count; asking anyway here would reject a fully
    # specified run at submit time.
    stated = JobSpec.from_dict(
        {
            **base,
            "algorithm": "grpo",
            "run_id": "q",
            "train": {"epochs": 1, "group_size": 4, "prompts_per_step": 32, "max_steps": 10},
        }
    )
    assert runconfig_from_spec(stated).normalized().batch_size == 32

    # and a pool that genuinely cannot be known is still dev's explicit refusal, not a cheap quote.
    from flash.cost.spec import UnknownPromptPoolSize

    unbounded = JobSpec.from_dict(
        {
            **base,
            "algorithm": "grpo",
            "run_id": "q",
            "train": {"epochs": 1, "group_size": 4, "prompts_per_step": 32},
        }
    )
    with pytest.raises(UnknownPromptPoolSize):
        runconfig_from_spec(unbounded)

    # [environment.params] max_examples is not horizon evidence at all, so a spec stating only that
    # is refused rather than priced from it. it is an opaque kwarg forwarded to the user's
    # environment factory that neither worker applies, which makes it wrong in BOTH directions
    # against `prompts_per_step = 128`: an environment that honours it trains 2 prompts, so pricing
    # an uncapped 128-prompt step overcharges 64x, while one that ignores it yields every row, so
    # deriving a 1-step horizon from the 2 underquotes a 1153-row pool 10x. Neither branch is a
    # quote, and the run is billed from this estimate.
    env_only = JobSpec.from_dict(
        {
            **base,
            "algorithm": "grpo",
            "run_id": "q",
            "environment": {
                "id": "github:owner/repo@main:env/environment.py",
                "params": {"max_examples": 2},
            },
            "train": {"epochs": 1, "group_size": 4, "prompts_per_step": 128},
        }
    )
    with pytest.raises(UnknownPromptPoolSize):
        runconfig_from_spec(env_only)

    # and when both are stated, the enforced [train] cap is the one that binds.
    both = JobSpec.from_dict(
        {
            **base,
            "algorithm": "grpo",
            "run_id": "q",
            "environment": {
                "id": "github:owner/repo@main:env/environment.py",
                "params": {"max_examples": 2},
            },
            "train": {"epochs": 1, "group_size": 4, "prompts_per_step": 128, "max_examples": 4},
        }
    )
    assert runconfig_from_spec(both).normalized().batch_size == 4


def test_allocator_ranking_narrows_a_vast_combination_it_is_pricing():
    """The ranking config carries NO provider, so reading it off the config alone was inert.

    `run_config_for_ranking` builds a bare one-step config -- provider defaults to `auto` -- and the
    allocator then prices every candidate against it. Before this fix a vast pair was ranked on the
    nvlink curve, which is precisely the direction the module warns about: it lets a pcie
    combination win on scaling it does not deliver, then bills both cards for the longer wall.
    """
    from flash.providers.core.allocator import _step_cost_ranker
    from flash.providers.core.base import Candidate, run_config_for_ranking

    # the defect's root: the config the allocator ranks with never names a provider.
    assert run_config_for_ranking("Qwen/Qwen3.5-9B", "grpo").normalized().provider == "auto"

    key = _step_cost_ranker("Qwen/Qwen3.5-9B", "grpo", None, False, "")
    assert key is not None, "this model must be priceable, or the assertions below prove nothing"

    def at(provider, hourly=2.0):
        return key(
            Candidate(provider=provider, gpu="H100", hourly_usd=hourly, vram_gb=80, gpu_count=2)
        )

    assert at("vast") > at("runpod"), (
        "a vast pair was ranked on nvlink scaling it may not deliver, so it can win a ranking on "
        "throughput it cannot reach and then bill both cards for the longer wall"
    )
    # a single card has no interconnect, so the narrowing must not touch it: that path is shared
    # with the preview and estimate, and they must agree exactly whenever one card is enough.
    single = {
        p: key(Candidate(provider=p, gpu="H100", hourly_usd=2.0, vram_gb=80, gpu_count=1))
        for p in ("vast", "runpod")
    }
    assert single["vast"] == single["runpod"]
    # not a blanket penalty: a genuinely cheaper vast pair still wins, it is just priced honestly.
    #
    # the discount has to clear the interconnect penalty, and how big that is depends on how much of
    # the step is gpu work. it used to be swamped: a phantom `completions x 1.0s` reward wall sat
    # beside a step floor already fitted with grading included, so most of a modelled grpo step was
    # an off-gpu wait no interconnect touches and a 12.8% discount was enough. with the double-count
    # removed the step is gpu-bound, pcie costs ~17.4% more wall, and the discount must beat that.
    # 1.70 (12.8% off) no longer does; 1.55 (20.5% off) does.
    assert at("vast", hourly=1.55) < at("runpod", hourly=1.95)


def test_a_live_vast_allocation_is_requoted_without_nvlink_credit():
    """The re-quote before provisioning is what the run is actually billed against.

    `estimate_cost` takes the exact selected candidate, so this is the last point where the wrong
    interconnect assumption can reach a persisted quote -- and its config is the user's, which for
    an auto run names no provider.
    """
    from types import SimpleNamespace

    from flash.cost.analytical import estimate_cost
    from flash.cost.types import RunConfig

    config = RunConfig(model_id="Qwen/Qwen3.5-9B", method="grpo", steps=20)
    assert config.normalized().provider == "auto", "an auto run is the case that was mispriced"

    def quote(provider, gpu_count=2):
        return estimate_cost(
            config,
            allocation=SimpleNamespace(
                gpu="H100",
                provider=provider,
                hourly_usd=2.0,
                min_vram_gb=80,
                gpu_count=gpu_count,
            ),
        )

    assert quote("vast").train_seconds > quote("runpod").train_seconds, (
        "a vast allocation was re-quoted on the nvlink curve, so the persisted quote understates "
        "the wall the run is billed for"
    )
    assert quote("vast", gpu_count=1).train_seconds == quote("runpod", gpu_count=1).train_seconds


def test_sft_fit_credits_only_the_ranks_that_will_launch():
    """Regression: the fit gate credited cards the sft run never puts in the fsdp group.

    Multi-card fit comes from SHARDING, so a card that does not join contributes no memory. sft
    shards by data and bounds its width by the batch and the row count, and an unpacked profile pins
    the batch to 1 -- so the run launches ONE rank no matter how many cards are rented.

    The decisive shape is a run that one card correctly rejects: a 4B at 32k needs 28 GB and does not
    fit a 24 GB card. Before this, renting two credited 35 GB and ADMITTED it, then launched a single
    rank with 24 GB and OOM'd on paid hardware. Allocating more cards must not be a way to pass a
    gate the executed run cannot pass.
    """
    from flash.providers.core.allocator import _executed_gpu_count, _fits
    from flash.providers.core.base import Candidate

    need = 28.0

    def fits(cards, algorithm="sft", train=None, overrides=None):
        candidate = Candidate(
            provider="runpod", gpu="A10", hourly_usd=1.0, vram_gb=24, gpu_count=cards
        )
        width = _executed_gpu_count(algorithm, train, overrides, candidate.gpu_count)
        return _fits(candidate, need, width)

    unpacked = {"batch_size": 1}
    # one card cannot hold it -- that is the baseline the wider shapes must not be able to dodge.
    assert not fits(1, train=unpacked)
    for cards in (2, 4, 8):
        assert not fits(cards, train=unpacked), (
            f"{cards} cards were credited as if all joined the run, but an unpacked sft run "
            "launches one rank, so the run is admitted on memory it never has and OOMs when paid"
        )

    # a batch that DOES divide the allocation launches every card, so sharding is real and credited.
    packed = {"batch_size": 8}
    assert fits(2, train=packed, overrides={"sft_retained_examples": 64}), (
        "a width the batch and rows both divide must keep its shard credit"
    )

    # ROWS bind too, and they arrive by a different route: `sft_retained_examples` is a profile
    # measurement, not a TrainSpec knob, so `_overridden_train` never copies it onto `train`.
    # reading it from there would find nothing and miss this narrowing entirely. sized against a
    # need BETWEEN the 2- and 4-card credits, so narrowing 4 -> 2 actually crosses the gate rather
    # than landing on the same side of it.
    def fits_rows(cards, rows, need_gb):
        candidate = Candidate(
            provider="runpod", gpu="A10", hourly_usd=1.0, vram_gb=24, gpu_count=cards
        )
        width = _executed_gpu_count(
            "sft", packed, {"sft_retained_examples": rows}, candidate.gpu_count
        )
        return _fits(candidate, need_gb, width)

    between = 50.0  # 2 cards credit 35.2, 4 credit 62.4
    assert fits_rows(4, 64, between), (
        "64 rows split 4 ways, so all four cards join and are credited"
    )
    assert not fits_rows(4, 10, between), (
        "10 rows cannot be split 4 ways, so the run launches 2 ranks -- crediting 4 admits it on "
        "memory only the wider world would have had"
    )

    # grpo and opd bound their width too, but on their OWN unit of work: they never read
    # `batch_size` (their batch is `prompts_per_step`, repeated `group_size` times), so an sft-shaped
    # knob must leave them at full width rather than silently narrowing them through the wrong field.
    assert fits(2, algorithm="grpo", train=unpacked)
    assert fits(2, algorithm="opd", train=unpacked)

    # their real narrowing: one prompt in a group of two is 2 sequences, which cannot fill 4 dp
    # ranks. verl raises on the uneven chunk instead of degrading, so crediting 4 would admit a run
    # that dies at step 0 on paid hardware -- the same over-credit as the sft case above.
    tiny_rl = {"prompts_per_step": 1, "group_size": 2}
    assert _executed_gpu_count("grpo", tiny_rl, None, 4) == 2
    assert _executed_gpu_count("opd", {"prompts_per_step": 1, "group_size": 1}, None, 2) == 1
    # and a real step fills every card, so the fix costs no capacity where it matters.
    assert _executed_gpu_count("grpo", {"prompts_per_step": 8, "group_size": 4}, None, 8) == 8

    # an OMITTED group takes the recipe default for that algorithm, which is what the worker resolves
    # it to (`train/rl/inputs.py`) and what the quote fills in (`RunConfig.normalized`). the defaults
    # differ -- grpo groups 8 completions per prompt, opd distills 1 -- so hardcoding either number
    # here breaks the other. reading 1 for grpo under-credited it eightfold and REJECTED runs that fit.
    assert _executed_gpu_count("grpo", {"prompts_per_step": 1}, None, 8) == 8
    assert _executed_gpu_count("opd", {"prompts_per_step": 1}, None, 8) == 1
    # an explicit supported group still wins over the default.
    assert _executed_gpu_count("grpo", {"prompts_per_step": 1, "group_size": 2}, None, 8) == 2

    # the quote must agree on every supported shape, or a shape it prices gets refused at submit.
    from flash.cost.analytical import executed_gpu_count
    from flash.cost.types import RunConfig

    for method, prompts, group in (("grpo", 1, None), ("opd", 1, None), ("grpo", 1, 2)):
        train = {"prompts_per_step": prompts} | ({"group_size": group} if group else {})
        quoted = executed_gpu_count(
            RunConfig(
                model_id="Qwen/Qwen3.5-9B",
                method=method,
                steps=10,
                batch_size=prompts,
                group_size=group,
            ),
            8,
        )
        assert quoted == _executed_gpu_count(method, train, None, 8), (method, prompts, group)

    # an UNKNOWN batch must not be read as a batch of 1, and unmeasured rows must not invent a limit.
    # callers that rank without knobs would otherwise have every multi-card sft shape rejected -- a
    # worse failure than the over-credit this clamp exists to stop, and one no test above catches.
    assert fits(2, train=None)
    assert fits(2, train={})
    assert fits(4, train=packed, overrides={})
