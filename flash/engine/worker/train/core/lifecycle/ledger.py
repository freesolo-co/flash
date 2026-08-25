"""One checkpoint lifecycle vocabulary shared by the sft, grpo and opd publishers.

The three trainers each grew their own bookkeeping and reused the same words for different facts:
grpo's "uploaded" meant *attempted*, sft's "staged" was a transient hardlink of the full checkpoint
while grpo's was a retained exported adapter, and sft's "processed" covered six distinct outcomes
including "coalesced away, never published". Durability was being inferred from sets that did not
mean durability.

The five facts below are recorded where each one actually becomes true, and are deliberately
INDEPENDENT rather than a linear enum, because both publication orders occur legitimately:

- sft and opd publish the deployable from ``before_upload``, so a required step reaches
  ``deployable_published`` BEFORE ``resume_uploaded``.
- grpo stages a required adapter while its terminal publication latch is shut, uploads resume state meanwhile,
  and publishes the deployable on a later sweep -- the opposite order.

A single ordered enum would have to call one of those orders a regression. Independent monotonic
facts describe both without ranking them.

This module holds no filesystem, artifact-store, threading or policy logic on purpose: it replaces
misleading state, it does not absorb publication behaviour. Which steps are eligible, when a
terminal publication latch opens, whether a backlog may be coalesced and what a retry contract requires all stay
with the algorithm that owns them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CheckpointFacts:
    """What is durably true about one checkpoint step.

    ``failed`` never erases an earlier fact. ``staged + deployable_published + failed`` is a real
    state: the adapter landed and is servable, then the required full-state upload was exhausted.
    Clearing the earlier facts there would report a published checkpoint as absent.
    """

    discovered: bool = False
    staged: bool = False
    resume_uploaded: bool = False
    deployable_published: bool = False
    failed: bool = False


_NOTHING = CheckpointFacts()


@dataclass
class CheckpointLedger:
    """Per-step lifecycle facts for one training run's checkpoints.

    Every mark is idempotent and monotonic, so a publisher that re-enters a step after a retry
    cannot walk a fact backwards. Only the thread running the watcher loop mutates a ledger; the
    parent reads it after ``join_while_draining`` returns.
    """

    _facts: dict[int, CheckpointFacts] = field(default_factory=dict)

    def facts(self, step: int) -> CheckpointFacts:
        """Everything known about ``step``; all-false for a step never seen."""
        return self._facts.get(int(step), _NOTHING)

    def _mark(self, step: int, **updates: bool) -> None:
        step = int(step)
        self._facts[step] = CheckpointFacts(
            **{**vars(self.facts(step)), **updates},
        )

    def mark_discovered(self, step: int) -> None:
        """Claim ``step``: this watcher has taken responsibility for it.

        Discovery suppression only. It says nothing about durability -- an intentionally skipped or
        coalesced step is discovered and never published, which is exactly the conflation this
        ledger exists to remove.
        """
        self._mark(step, discovered=True)

    def mark_staged(self, step: int) -> None:
        """Algorithm-specific local preparation for ``step`` finished.

        Historical: sft deletes its staged copy once the upload returns, so this records that the
        preparation happened, not that the directory still exists.
        """
        self._mark(step, staged=True)

    def mark_resume_uploaded(self, step: int) -> None:
        """``step``'s full trainer state is committed under the singular ``checkpoint/`` tree.

        Mark this from ``upload_resume_checkpoint``'s ``after_upload`` callback, which runs only
        once the folder commit and its prune have both succeeded. The function's boolean return
        cannot carry this fact: it is also True when there is no artifact repository at all and
        when a caller's ``skip_upload`` short-circuits, and it is False when the commit landed but
        the closing heartbeat failed.
        """
        self._mark(step, resume_uploaded=True)

    def mark_deployable_published(self, step: int) -> None:
        """``step``'s adapter is servable under the plural ``checkpoints/step-N/`` tree.

        This is the fact a required save is owed, so it is what required-save completeness is
        checked against.
        """
        self._mark(step, deployable_published=True)

    def mark_failed(self, step: int) -> None:
        """This step's policy gave up on further progress; earlier facts stay true."""
        self._mark(step, failed=True)

    def seed_resumed_step(
        self, resume_step: int, required_steps: frozenset[int] | set[int]
    ) -> None:
        """Record what the attempt being resumed from already made durable.

        The restored checkpoint lands in ``local_dir`` as ``global_step_N`` with the tracker
        pointing at it, so an unseeded watcher treats it as pending on its first sweep and re-runs
        the merge and the multi-GB resume upload for state the artifact store already holds.

        A required step is deliberately NOT claimed as discovered: the previous attempt may have
        uploaded resume state while a terminal publication latch withheld its deployable adapter, and this
        watcher must still stage and publish that adapter. The deployable is a separate artifact
        and is credited only by a caller that confirmed it, never inferred from the step counter.
        """
        if not resume_step:
            return
        self.mark_resume_uploaded(resume_step)
        if resume_step not in required_steps:
            self.mark_discovered(resume_step)

    @property
    def discovered_steps(self) -> set[int]:
        """Steps already claimed, for filtering the next sweep."""
        return {step for step, facts in self._facts.items() if facts.discovered}

    @property
    def deployable_published_steps(self) -> set[int]:
        """Steps with a durable servable adapter."""
        return {step for step, facts in self._facts.items() if facts.deployable_published}

    def missing_deployables(self, required_steps: frozenset[int] | set[int]) -> list[int]:
        """Required steps that never became servable, ascending."""
        return sorted(set(required_steps) - self.deployable_published_steps)
