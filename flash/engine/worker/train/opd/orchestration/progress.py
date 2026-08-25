"""Thread-safe OPD optimizer progress and resume accounting."""

from __future__ import annotations

import threading
import time

from flash.engine.worker.train.opd.orchestration.failures import _TruncationWindow


class _OpdProgressState:
    def __init__(self, resume_state: dict | None = None) -> None:
        state = resume_state or {}
        self._condition = threading.Condition()
        self.loss_curve = [float(value) for value in state.get("loss_curve", [])]
        self.coverage_curve = [float(value) for value in state.get("coverage_curve", [])]
        self.base_train_wall_seconds = float(state.get("train_wall_seconds", 0.0))
        self._prev_aligned = int(state.get("aligned_sequences", 0))
        self._prev_cov_sum = float(state.get("coverage_sum", 0.0))
        self._prev_truncated = int(state.get("truncated_rollouts", 0))
        self._prev_samples_seen = int(state.get("samples_seen", 0))
        self._prev_no_signal_skipped_steps = int(state.get("no_signal_skipped_steps", 0))
        self._train_started_at: float | None = None
        self._step_states: dict[int, dict] = {}
        self._terminal_error: str = ""
        if resume_state is not None:
            self._step_states[int(state["opt_steps"])] = dict(state)

    def start_training(self) -> None:
        self._train_started_at = time.time()

    def _train_wall_seconds(self) -> float:
        elapsed = 0.0
        if self._train_started_at is not None:
            elapsed = max(0.0, time.time() - self._train_started_at)
        return self.base_train_wall_seconds + elapsed

    def record_step(self, step: int, loss: float, bridge: object) -> tuple[float, int]:
        with self._condition:
            expected_step = len(self.loss_curve) + 1
            if step != expected_step:
                raise RuntimeError(
                    f"verl OPD metric step {step} does not follow accumulated step {expected_step - 1}"
                )
            snapshot = bridge.accounting_snapshot()
            self.loss_curve.append(float(loss))
            aligned = int(snapshot["aligned_sequences"])
            cov_sum = float(snapshot["coverage_sum"])
            # per-step coverage: delta over the previous snapshot, so the curve shows each step's
            # own alignment quality instead of a cumulative average that flattens regressions.
            d_aligned = aligned - self._prev_aligned
            d_cov = cov_sum - self._prev_cov_sum
            self._prev_aligned, self._prev_cov_sum = aligned, cov_sum
            coverage = (
                (d_cov / d_aligned) if d_aligned > 0 else (cov_sum / aligned if aligned else 0.0)
            )
            self.coverage_curve.append(coverage)
            truncated = int(snapshot["truncated_rollouts"])
            samples_seen = int(snapshot["samples_seen"])
            # truncation can change abruptly with prompt mix, so report this step's rollout share
            # rather than the cumulative average. a zero-delta snapshot reports 0.0 because no
            # rollouts belong to this step, and reusing history would misattribute old truncations.
            d_truncated = truncated - self._prev_truncated
            d_samples_seen = samples_seen - self._prev_samples_seen
            self._prev_truncated, self._prev_samples_seen = truncated, samples_seen
            self._prev_no_signal_skipped_steps = int(snapshot.get("no_signal_skipped_steps", 0))
            truncation_rate = (
                min(1.0, max(0.0, d_truncated / d_samples_seen)) if d_samples_seen > 0 else 0.0
            )
            # bound by this step's own sample delta for the same reason the rate clamps to 1.0: an
            # in-flight snapshot can read truncated_rollouts and samples_seen from different steps,
            # and an unbounded count would report more discards than the step drew.
            #
            # the clamp bounds the magnitude; it does NOT establish ownership. these are deltas of
            # counters the bridge advances asynchronously, so when stdout parsing lags a wave, a
            # truncation raised by the next step can still be attributed here (and the next step
            # then reads zero). the same asynchrony has always applied to truncation_rate. read
            # both as a rolling indicator of truncation pressure, not an exact per-step ledger.
            discarded_rollouts = max(0, min(d_truncated, d_samples_seen))
            # the per-step values are returned for the heartbeat and deliberately kept OUT of the
            # snapshot: this dict is spread verbatim into the persisted opd_state.json resume
            # contract, whose schema is fail-closed and holds cumulative counters. per-step display
            # values have no meaning on resume and nothing reads them back.
            snapshot.update(
                {
                    "train_wall_seconds": self._train_wall_seconds(),
                    "loss_curve": list(self.loss_curve),
                    "coverage_curve": list(self.coverage_curve),
                }
            )
            self._step_states[step] = snapshot
            self._condition.notify_all()
            return truncation_rate, discarded_rollouts

    def truncation_window(
        self,
        bridge: object,
        max_completion: int,
    ) -> _TruncationWindow:
        snapshot = bridge.accounting_snapshot()
        with self._condition:
            # the fatal no-signal diagnosis must describe the attempts after the last completed step.
            # cumulative rollout history can otherwise blame an old truncation spike for a current
            # empty-alignment failure.
            return _TruncationWindow(
                truncated_rollouts=max(
                    0, int(snapshot["truncated_rollouts"]) - self._prev_truncated
                ),
                samples_seen=max(0, int(snapshot["samples_seen"]) - self._prev_samples_seen),
                no_signal_skipped_steps=max(
                    0,
                    int(snapshot.get("no_signal_skipped_steps", 0))
                    - self._prev_no_signal_skipped_steps,
                ),
                max_completion=max_completion,
            )

    def fail(self, reason: str) -> None:
        """Record that the verl child died, and wake anything waiting on its accounting.

        Without this a crash is indistinguishable from slowness: `checkpoint_state` blocks on a
        condition only `record_step` notifies, so a child that exits before printing the step's
        metrics leaves the waiter to burn its full timeout and then blame accounting for a failure
        that happened elsewhere. Observed on a 27B image OPD run whose real cause was a vLLM
        `wake_up` CUDA OOM two frames deeper.
        """
        with self._condition:
            # keep the FIRST reason: it is the cause, and later ones are usually its fallout.
            self._terminal_error = self._terminal_error or str(reason).strip()
            self._condition.notify_all()

    def checkpoint_state(self, step: int, *, timeout_s: float = 300.0) -> dict:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while step not in self._step_states:
                # a dead child will never record this step, so report why it died rather than
                # waiting out a timeout and attributing the failure to accounting.
                if self._terminal_error:
                    raise RuntimeError(
                        f"OPD child exited before accounting for checkpoint step {step}: "
                        f"{self._terminal_error}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"timed out waiting for honest OPD accounting through checkpoint step {step}"
                    )
                self._condition.wait(remaining)
            return dict(self._step_states[step])

    def final_state(self, bridge: object) -> dict:
        snapshot = bridge.accounting_snapshot()
        snapshot.update(
            {
                "train_wall_seconds": self._train_wall_seconds(),
                "loss_curve": list(self.loss_curve),
                "coverage_curve": list(self.coverage_curve),
            }
        )
        return snapshot
