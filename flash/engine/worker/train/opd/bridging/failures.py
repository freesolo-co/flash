"""the bridge's failure bookkeeping, split out of bridge.py.

one cohesive concern: every field here is failure state, every method takes the same
``_stats_lock``, and none of it touches sessions, routing, scoring, or the http server. it reads as
a unit and it is the part of the bridge a reader can understand without holding the rest, which is
why this is the seam rather than an arbitrary line-count cut.

the classification vocabulary is the one the rest of the opd path already speaks: "transient" means
retry is meaningful, anything else is permanent and terminal for the run.
"""

from __future__ import annotations


class _RecordedMutationCallbackFailure(RuntimeError):
    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


class TeacherFailureRecording:
    """failure recording and promotion for the teacher-alignment bridge.

    a mixin rather than a collaborator object: every method needs ``self._stats_lock`` and the
    counters that live beside it on the bridge, so handing it its own state would mean either
    duplicating the lock or threading the bridge back in. mixing in keeps one lock and one owner.
    """

    def _init_failure_state(self) -> None:
        self._teacher_failure: tuple[str, str] | None = None
        self._mutation_failure: tuple[str, str] | None = None
        self._mutation_callback_failure: tuple[str, str] | None = None
        self._mutation_callback_succeeded = False
        self._pending_teacher_transient: tuple[str, str] | None = None
        self._pending_teacher_success = False

    def _record_teacher_failure(
        self,
        classification: str,
        message: str,
        *,
        terminal: bool = False,
    ) -> None:
        with self._stats_lock:
            if classification == "transient":
                self.teacher_transient += 1
                if terminal and self._teacher_failure is None:
                    self._teacher_failure = (classification, message)
                elif self._pending_teacher_transient is None:
                    self._pending_teacher_transient = (classification, message)
            else:
                self.teacher_error += 1
                self._teacher_failure = (classification, message)

    @property
    def teacher_failure(self) -> tuple[str, str] | None:
        with self._stats_lock:
            return self._teacher_failure

    def _promote_recovered_teacher_failure(self, failure: tuple[str, str]) -> None:
        with self._stats_lock:
            if self._teacher_failure is None:
                self._teacher_failure = failure

    def _record_teacher_delivery_failure(self, error: Exception) -> None:
        with self._stats_lock:
            if self._teacher_failure is None:
                self._teacher_failure = (
                    "transient",
                    f"teacher bridge response delivery failed: {type(error).__name__}",
                )

    def _record_mutation_failure(self, classification: str, message: str) -> None:
        with self._stats_lock:
            if self._mutation_callback_failure is not None:
                return
            if self._mutation_callback_succeeded:
                return
            if classification == "permanent" or self._mutation_failure is None:
                self._mutation_failure = (classification, message)

    def _record_mutation_callback_failure(
        self,
        classification: str,
        message: str,
    ) -> tuple[str, str]:
        with self._stats_lock:
            if self._mutation_callback_failure is None:
                self._mutation_callback_failure = (classification, message)
            return self._mutation_callback_failure

    @staticmethod
    def _raise_recorded_mutation_failure(failure: tuple[str, str]) -> None:
        classification, message = failure
        raise _RecordedMutationCallbackFailure(classification, message)

    @property
    def mutation_failure(self) -> tuple[str, str] | None:
        with self._stats_lock:
            if self._mutation_callback_failure is not None:
                return self._mutation_callback_failure
            if self._mutation_callback_succeeded:
                return None
            return self._mutation_failure

    def _promote_pending_teacher_failure(self) -> bool:
        with self._stats_lock:
            if (
                self._teacher_failure is None
                and self._pending_teacher_transient is not None
                and not self._pending_teacher_success
            ):
                self._teacher_failure = self._pending_teacher_transient
                self._pending_teacher_transient = None
                self._pending_teacher_success = False
                return True
            return False
