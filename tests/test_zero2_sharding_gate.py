"""CPU tests for the ZeRO-2 sharding gate: one policy, read identically by both sides.

verl's ``reshard_after_forward=false`` selects ZeRO-2, which keeps each rank's parameter copy
resident after the forward instead of re-gathering it in the backward. That removes one all-gather
per step (measured 1.377x on pcie, 1.150x on nvlink under flash's own config) and costs a CONSTANT
per-card slice of the weights, flat in card count.

The failure this file exists to prevent is a SPLIT policy: the allocator admitting and pricing a
shape as ZeRO-3 while the worker trains it under ZeRO-2 and overspends the per-card budget the
shape was admitted with. So the tests pin the gate as one shared predicate rather than checking the
allocator and the worker separately and hoping the two agree.
"""

from __future__ import annotations

import pytest

from flash.core.catalog import MODELS
from flash.engine.worker.train.opd.orchestration.overrides import build_opd_overrides
from flash.engine.worker.verl.parallelism import resolve_reshard_after_forward
from flash.providers.core.sharding import (
    REPLICATED_PER_CARD_GB,
    SHARD_VRAM_EFFICIENCY,
    ZERO2_CHARGED_RESIDENCY,
    ZERO2_WEIGHT_RESIDENCY,
    combined_vram_gb,
    zero2_enabled,
    zero2_replicated_floor_gb,
)

# the models this gate can ever apply to, sized from the catalog rather than restated here.
_CATALOG_PARAMS_B = tuple(sorted({float(info.params_b) for info in MODELS.values()}))
_CARD_VRAM_GB = (24, 32, 40, 80, 96, 141, 180)
_WIDTHS = (2, 4, 8)


def test_zero3_capacity_is_unchanged_by_the_new_parameter():
    """The default path must be byte-identical to the pre-gate fit model.

    ``combined_vram_gb`` is the shared fit model for every sizing boundary in flash, so a change in
    its ZeRO-3 answer would silently re-shape every existing quote and admission.
    """
    for vram in _CARD_VRAM_GB:
        for n in (1, *_WIDTHS):
            expected = (
                float(vram)
                if n <= 1
                else n * (vram - REPLICATED_PER_CARD_GB) * 0.85 + REPLICATED_PER_CARD_GB
            )
            assert combined_vram_gb(vram, n) == expected
            assert combined_vram_gb(vram, n, zero2_params_b=0.0) == expected


def test_zero2_floor_is_the_zero3_floor_plus_a_retained_weight_copy():
    """The penalty is a per-card FLOOR, not a poolable term: it does not shrink with card count."""
    assert zero2_replicated_floor_gb(0.0) == REPLICATED_PER_CARD_GB
    for params_b in _CATALOG_PARAMS_B:
        floor = zero2_replicated_floor_gb(params_b)
        assert floor > REPLICATED_PER_CARD_GB
        # the CHARGED constant, deliberately above the measured 0.883: the measurement comes from
        # one 1.59B model and the catalog runs to 35B, so the gate pays for that extrapolation.
        assert floor == pytest.approx(
            REPLICATED_PER_CARD_GB + ZERO2_CHARGED_RESIDENCY * params_b * 2.0
        )
        assert ZERO2_CHARGED_RESIDENCY > ZERO2_WEIGHT_RESIDENCY


def test_the_gate_charges_more_than_it_measured():
    """A full retained bf16 copy, plus margin, is the floor the gate is allowed to assume.

    The residency was measured once, at 1.59B. Charging it bare left 35B-A3B opd on 2x B200 clearing
    its requirement by 0.4%, which an extrapolation error of 2.6% would turn into an OOM on a paid
    multi-card run. Anything at or below 1.0 would be assuming the retained copy is FREE in part.
    """
    assert ZERO2_CHARGED_RESIDENCY >= 1.0


@pytest.mark.parametrize("params_b", _CATALOG_PARAMS_B)
def test_the_zero2_penalty_does_not_amortize_across_cards(params_b):
    """Adding cards must never make the retained copy cheaper per card.

    A model that let the penalty shrink with width would re-derive the very over-credit the gate
    exists to prevent: the run would be admitted on pooled capacity no rank actually has.
    """
    for vram in _CARD_VRAM_GB:
        per_card = [
            combined_vram_gb(vram, n) - combined_vram_gb(vram, n, zero2_params_b=params_b)
            for n in _WIDTHS
            if combined_vram_gb(vram, n, zero2_params_b=params_b) > 0
        ]
        # the gap between the two models grows with n (each card pays it once) and never inverts.
        assert per_card == sorted(per_card)


def test_zero2_never_widens_what_the_allocator_admits():
    """THE safety invariant: ZeRO-2 is only ever selected on a shape that already fits ZeRO-3.

    Admission therefore keeps using the stricter model, and the gate can only ever DOWNGRADE a run
    that already fits to a faster strategy -- it can never let a run in that ZeRO-3 would reject.
    """
    for vram in _CARD_VRAM_GB:
        for n in _WIDTHS:
            for params_b in _CATALOG_PARAMS_B:
                for need in range(5, 400, 5):
                    if zero2_enabled(vram, n, params_b, need):
                        assert combined_vram_gb(vram, n) >= need


def test_the_gate_is_off_on_a_single_rank():
    """PyTorch clamps every FSDP strategy to NO_SHARD at world size one.

    There is no all-gather to remove there, so enabling ZeRO-2 would buy nothing and still charge
    the memory. The worker must ask for verl's default instead.
    """
    for params_b in _CATALOG_PARAMS_B:
        assert not zero2_enabled(180, 1, params_b, 10)
    assert resolve_reshard_after_forward(
        model_id="Qwen/Qwen3.5-9B", algorithm="opd", gpu_type="RTX 5090", n_gpus=1
    )


@pytest.mark.parametrize(
    ("gpu_type", "n_gpus"),
    [("", 2), ("Not A Real Card", 2)],
)
def test_an_unknown_card_falls_closed_to_zero3(gpu_type, n_gpus):
    """ZeRO-3 is the strictly lower-memory strategy, so every unknown must resolve to it.

    A wrong ZeRO-2 answer OOMs a paid run; a wrong ZeRO-3 answer only forgoes a speedup.
    """
    assert resolve_reshard_after_forward(
        model_id="Qwen/Qwen3.5-9B", algorithm="opd", gpu_type=gpu_type, n_gpus=n_gpus
    )


def test_an_unsizable_model_falls_closed_to_zero3():
    """An uncataloged id cannot be priced, so it cannot be granted the memory trade."""
    assert resolve_reshard_after_forward(
        model_id="nobody/not-a-model", algorithm="opd", gpu_type="RTX 5090", n_gpus=2
    )
    assert not zero2_enabled(80, 2, 0.0, 41)


def test_a_shape_without_room_for_the_retained_copy_stays_on_zero3():
    """The speedup must never be bought with headroom the run needs.

    4B GRPO on 2x24 GB fits only because the pooled model credits the shard; charging the retained
    copy takes it back under the requirement, so the gate has to refuse.
    """
    assert not zero2_enabled(24, 2, 4.7, 35)
    assert resolve_reshard_after_forward(
        model_id="Qwen/Qwen3.5-9B", algorithm="grpo", gpu_type="RTX 4090", n_gpus=2
    )


def test_a_shape_with_room_takes_zero2():
    """The gate has to actually fire somewhere, or it is dead code shipped as an optimization.

    A small model on big cards is where the retained copy is cheapest relative to the card, so this
    is the shape class the gate exists to serve.
    """
    assert zero2_enabled(80, 2, 0.9, 84)
    assert not resolve_reshard_after_forward(
        model_id="Qwen/Qwen3.5-9B", algorithm="opd", gpu_type="A100 SXM", n_gpus=2
    )


def test_the_gate_never_admits_more_than_a_card_physically_holds():
    """Every enabled shape must fit CARD-BY-CARD, not just in the pooled model.

    The pooled comparison is not sufficient on its own: `combined_vram_gb` ends in a trailing
    addend, and adding the ZeRO-2 floor there instead of `REPLICATED_PER_CARD_GB` refunds exactly
    the retained copy the floor just charged. That refund is invisible to a pooled check (the gate
    and the capacity move together) but shows up immediately here, because `need` never included a
    retained copy. It admitted 10 catalog shapes whose per-card demand exceeded the card, including
    35B-A3B opd on 2x B200 asking 210.79 GB of a 180 GB card -- paid runs ZeRO-3 would have
    completed.
    """
    from flash.engine.plan.vram import model_required_vram_gb
    from flash.providers.core.base import get_gpu_info

    impossible = []
    fired = 0
    for model_id in MODELS:
        params_b = float(MODELS[model_id].params_b)
        for algorithm in ("grpo", "opd"):
            need = float(model_required_vram_gb(model_id, algorithm))
            for gpu_type in ("RTX 4090", "RTX 5090", "A100 PCIe", "H200", "B200"):
                vram = int(get_gpu_info(gpu_type).vram_gb)
                for n_gpus in _WIDTHS:
                    if not zero2_enabled(vram, n_gpus, params_b, need):
                        continue
                    fired += 1
                    # what one card actually has to hold: the ZeRO-2 replicated floor plus this
                    # card's share of the shardable remainder, at the same efficiency the pooled
                    # model credits.
                    per_card = zero2_replicated_floor_gb(params_b) + (
                        need - REPLICATED_PER_CARD_GB
                    ) / (n_gpus * SHARD_VRAM_EFFICIENCY)
                    if per_card > vram + 1e-9:
                        impossible.append(
                            f"{model_id} {algorithm} {n_gpus}x{gpu_type}: "
                            f"needs {per_card:.2f} GB per card, card holds {vram}"
                        )
    assert fired, "the gate never fired, so this proves nothing"
    assert not impossible, "gate admitted shapes no card can hold:\n" + "\n".join(impossible)


def test_the_worker_and_the_allocator_cannot_disagree():
    """The worker's flag must be exactly the allocator's predicate, for every catalog shape.

    Pinned as a cross-check of the two entry points rather than of one implementation: this is the
    drift that makes a worker-only flip unsafe, and it is invisible to a test that exercises either
    side alone.
    """
    from flash.engine.plan.vram import model_required_vram_gb
    from flash.providers.core.base import get_gpu_info

    checked = 0
    fired = 0
    # grpo and opd only: they are the two callers that render the key. sft never calls the resolver,
    # so sweeping it here would assert agreement about a run that stays on zero-3 either way and
    # read as coverage the gate does not have.
    for model_id in MODELS:
        params_b = float(MODELS[model_id].params_b)
        for algorithm in ("grpo", "opd"):
            need = float(model_required_vram_gb(model_id, algorithm))
            for gpu_type in ("RTX 4090", "RTX 5090", "A100 PCIe", "H200", "B200"):
                vram = int(get_gpu_info(gpu_type).vram_gb)
                for n_gpus in _WIDTHS:
                    allocator_says = zero2_enabled(vram, n_gpus, params_b, need)
                    worker_says = not resolve_reshard_after_forward(
                        model_id=model_id,
                        algorithm=algorithm,
                        gpu_type=gpu_type,
                        n_gpus=n_gpus,
                    )
                    assert allocator_says == worker_says, (
                        f"{model_id} {algorithm} {n_gpus}x{gpu_type}: "
                        f"allocator={allocator_says} worker={worker_says}"
                    )
                    checked += 1
                    fired += int(allocator_says)
    assert checked, "the cross-check swept nothing"
    assert fired, "the gate never fired across the whole catalog: it would be dead code"


def test_opd_renders_the_verl_key_and_defaults_to_zero3_when_absent(monkeypatch):
    """The rendered hydra key is the one verl reads, and omission must mean ZeRO-3.

    A config assembled without the gate has to render verl's own default rather than fall into the
    memory-hungry strategy by omission.
    """
    from tests.test_opd_train import _config

    config = _config()
    config.pop("reshard_after_forward", None)
    rendered = build_opd_overrides(config)
    # hydra bools render lowercase, which is what verl's config parser reads.
    assert "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true" in rendered, (
        "an absent gate must render verl's zero-3 default"
    )

    config["reshard_after_forward"] = False
    rendered = build_opd_overrides(config)
    assert "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false" in rendered
