"""The shared checkpoint lifecycle vocabulary the three trainers publish through."""

from __future__ import annotations

from flash.engine.worker.train.core.lifecycle.ledger import CheckpointLedger


def test_resume_and_deployable_are_recorded_independently_in_either_order():
    """both publication orders are legitimate, so neither may imply or overwrite the other.

    sft and opd publish the deployable from `before_upload`, reaching deployable_published FIRST.
    grpo stages a required adapter while its gradient gate is shut, uploads resume state meanwhile,
    and publishes on a later sweep -- the opposite order. a linear enum would have to call one of
    these a regression, which is exactly why the facts are independent.
    """
    sft_order = CheckpointLedger()
    sft_order.mark_deployable_published(4)
    sft_order.mark_resume_uploaded(4)

    grpo_order = CheckpointLedger()
    grpo_order.mark_resume_uploaded(4)
    grpo_order.mark_deployable_published(4)

    assert sft_order.facts(4) == grpo_order.facts(4)
    assert sft_order.facts(4).resume_uploaded
    assert sft_order.facts(4).deployable_published


def test_one_milestone_does_not_imply_the_other():
    resume_only = CheckpointLedger()
    resume_only.mark_resume_uploaded(2)
    assert resume_only.facts(2).resume_uploaded
    assert not resume_only.facts(2).deployable_published
    assert resume_only.missing_deployables(frozenset({2})) == [2]

    deployable_only = CheckpointLedger()
    deployable_only.mark_deployable_published(2)
    assert deployable_only.facts(2).deployable_published
    assert not deployable_only.facts(2).resume_uploaded


def test_failed_preserves_the_milestones_already_reached():
    """`staged + deployable_published + failed` is a real state, not a contradiction.

    an sft required step can publish its adapter through `before_upload` and then exhaust the
    full-state upload. the adapter is servable and must keep saying so; clearing the earlier facts
    would report a published checkpoint as absent.
    """
    ledger = CheckpointLedger()
    ledger.mark_discovered(7)
    ledger.mark_staged(7)
    ledger.mark_deployable_published(7)
    ledger.mark_failed(7)

    facts = ledger.facts(7)
    assert facts.staged
    assert facts.deployable_published
    assert facts.failed
    assert ledger.deployable_published_steps == {7}
    assert ledger.missing_deployables(frozenset({7})) == []


def test_marks_are_idempotent_and_monotonic():
    ledger = CheckpointLedger()
    ledger.mark_discovered(1)
    ledger.mark_resume_uploaded(1)
    ledger.mark_discovered(1)
    ledger.mark_resume_uploaded(1)

    assert ledger.facts(1).discovered
    assert ledger.facts(1).resume_uploaded
    assert ledger.discovered_steps == {1}


def test_an_unseen_step_knows_nothing_rather_than_raising():
    ledger = CheckpointLedger()
    facts = ledger.facts(99)
    assert not any(
        (facts.discovered, facts.staged, facts.resume_uploaded, facts.deployable_published)
    )
    assert ledger.discovered_steps == set()


def test_crediting_a_remote_deployable_does_not_claim_the_step():
    """grpo credits a required step's adapter from hf without claiming the step itself.

    discovery is what suppresses the next sweep. a required step whose resume state is durable but
    whose adapter is not must stay discoverable so this worker still stages and publishes it, so
    crediting one fact must never silently set the other.
    """
    ledger = CheckpointLedger()
    ledger.mark_deployable_published(5)

    assert ledger.deployable_published_steps == {5}
    assert ledger.discovered_steps == set()
    assert not ledger.facts(5).discovered


def test_seeding_a_resumed_required_step_leaves_it_discoverable():
    """the rule all three publishers share, asserted once where it now lives.

    a required step must NOT be claimed by seeding: the previous attempt may have uploaded resume
    state while a gradient gate withheld its deployable adapter, and this worker still has to stage
    and publish that adapter. claiming it would filter it out of the first sweep and the run would
    end owing a servable artifact it never wrote.
    """
    required = CheckpointLedger()
    required.seed_resumed_step(40, frozenset({40}))

    assert required.facts(40).resume_uploaded, "hf holds the state this attempt restored from"
    assert not required.facts(40).discovered, "a required step must stay visible to the first sweep"
    assert not required.facts(40).deployable_published, "seeding never infers a published adapter"

    # an optional step has nothing further owed, so claiming it stops a pointless multi-GB re-upload.
    optional = CheckpointLedger()
    optional.seed_resumed_step(40, frozenset({80}))
    assert optional.facts(40).discovered

    # no resume step means nothing was restored, so nothing may be asserted about step 0.
    fresh = CheckpointLedger()
    fresh.seed_resumed_step(0, frozenset())
    assert fresh.discovered_steps == set()
    assert not fresh.facts(0).resume_uploaded


def test_missing_deployables_reports_ascending_and_ignores_extra_published_steps():
    ledger = CheckpointLedger()
    ledger.mark_deployable_published(9)
    ledger.mark_deployable_published(2)

    assert ledger.missing_deployables(frozenset({2, 5, 9, 11})) == [5, 11]
    assert ledger.missing_deployables(frozenset()) == []
