# Changelog

Notable changes to `freesolo-flash`. The dev-channel package `freesolo-flash-dev` ships in
lockstep at the same version.

Releases happen by merging a version bump from `dev` into `main`, which publishes to PyPI
(`.github/workflows/publish.yml`). Versions are `MAJOR.MINOR.PATCH`.

This file starts at 1.1.35. Earlier releases are not reconstructed here; use
`git log` for anything before that.

## Unreleased

### Added

- A `flash-cli` console script, the same entry point as `flash` under a name nothing else
  claims. The `server` and `dev` extras install `runpod-flash`, which declares its own `flash`
  script; whichever distribution is installed last wins, so on a control-plane host
  `pip install 'freesolo-flash[server]'` could leave `flash` running RunPod's CLI, which exits 0
  and does nothing. `flash` is unchanged and remains correct on a base (client-only) install.

### Changed

- CI runs the offline test suite on both supported interpreters (3.11 and 3.12) rather than
  3.11 alone, and sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` so a test that reaches the
  network fails deterministically instead of depending on runner connectivity.
- Every GitHub Action is pinned to a commit SHA rather than a mutable tag.
- `mypy` runs in CI as an advisory (non-blocking) job.

### Fixed

- A training run whose child process stops making progress now fails instead of billing a rented
  GPU indefinitely at 0% utilisation. The worker already counted consecutive heartbeat ticks with
  no new child output and published them as `child_tail_silent_ticks`, but nothing consumed the
  signal: the run stayed `running` with `error: null` until a human noticed and cancelled it. Two
  changes make it terminate. The counter now ignores a line identical to the one before it, so a
  wedged child re-printing one frozen status line reads as no progress rather than as activity
  (previously it reset the counter and capped it around 9, which is why the published value never
  climbed). After 30 minutes of elapsed time with no new content and no completed optimizer step,
  the child's process group is torn down and the run fails with a message naming what was observed.
  The deadline is measured in elapsed time rather than in loop iterations, because each iteration
  costs its sleep plus however long the tick's `nvidia-smi` probes and heartbeat upload took: on a
  degraded host a 60-tick count stretched to roughly 47 minutes, past the provider's own 3000s
  grace, which reports the retriable `stalled` and bills the dead run through its retries. Only
  optimizer steps recognised by the shared verl step parser count as progress, so a pre-training
  diagnostic such as `global_step: 1` no longer switches the detector off for the rest of the run,
  and a resumed attempt reports steps taken by the current child rather than the restored step it
  started from. That same validated parser now also gates the real per-step heartbeat, not just the
  worker's own detector: a heartbeat carrying a step tells the provider cold start is over and swaps
  its 3000s setup grace for a 1500s window, 300s tighter than the watchdog's deadline, so one
  pre-training diagnostic line handed the provider a retriable `stalled` verdict before the watchdog
  could fire. Genuinely new output resets the window, so the 600s teacher bridge request and the
  warmup-dominated first step are unaffected. A staged heartbeat is emitted as the watchdog arms, so
  the provider's own stall clock restarts alongside the child's: it advances only on staged pings,
  and the whole setup span before the child is held open by liveness pings it does not credit, so
  ~20 minutes of ordinary setup previously left less grace than the watchdog needs and the provider
  reported the retriable `stalled` first. That staged heartbeat is bounded and abandoned rather than
  awaited, because it runs synchronously before the child (and therefore the watchdog) exists and
  the underlying upload has no timeout: a hung connection there would strand the run in the very
  state this change exists to end. Detection runs entirely on a thread that never uploads --
  it samples the counter and decides on it -- because the heartbeat upload has no timeout, and a
  detector that only advances between uploads can see neither a child going quiet nor one coming
  back. It takes its first sample before its first sleep, so the earliest heartbeat already carries
  the tail and the silent-tick count rather than racing the emitting loop for them; a child that
  dies during import is judged by that first payload and a torn-down run publishes no later one to
  correct it. The window is cleared as soon as the child stops reporting the counter at all, which
  is how a completed first optimizer step announces that silence is no longer evidence of anything.

- Commands printed for the operator to run (the resume/cancel hand-off after `flash train`,
  usage strings, `next:` hints) now name the executable actually invoked rather than always
  `flash`. On a host where `runpod-flash` owns the `flash` script, the printed
  `flash runs cancel <run-id>` exited 0 without cancelling, leaving a run billing while the
  operator believed they had stopped it. Reached through the `python -m flash.cli` escape hatch,
  the printed command now names the interpreter that is actually running, rather than a bare
  `python` that is absent on a python3-only host and may resolve to a different environment
  inside a virtualenv.

- `flash env setup` scaffolded hosted-only environment instructions on every plane. A
  self-hosted plane cannot publish to Freesolo's managed environment hub, so the generated
  configs left `[environment] id` empty (which fails validation) beside a `flash env push`
  command that could not work. Against a self-hosted plane the scaffold now emits the
  `github:owner/repo@ref:path` id form, which such a plane resolves directly. Output against
  the managed plane is unchanged.

- `flash-server` reported missing operator configuration as an unhandled ASGI startup
  exception, so the actionable message arrived under ~20 frames of starlette/contextlib. The
  refusing half of the preflight now runs before uvicorn starts, and `flash-server` prints
  `error: ...` and exits 3 (uvicorn's `STARTUP_FAILURE`, which this path already used, so
  supervision keying on it is unaffected). The advisory warnings still come from the lifespan
  alone, so a booted plane logs them once.
- `flash env test` blamed the reward function for a gold answer that scored zero, when the
  gold completion is equally likely to be at fault: with no `sft_completion` hook the replayed
  answer is the dataset row's raw `output`, so a dataset whose `output` is a bare value fails a
  grader that (correctly) requires a wrapper. The warning now names both candidates and prints
  the exact text it scored, verbatim rather than through the whitespace-collapsing, 200-character
  preview that hid the formatting faults it exists to reveal. The scaffolded `environment.py`
  documents that `output` is also the gold assistant turn SFT trains on, and that per-row scorer
  state belongs in `metadata`.
- `flash env test` labelled the replayed dataset turns as the text that scored zero, but a
  multi-turn environment whose `step_episode` returns a `final_response_text` has that override
  replace the episode's response before scoring, so the diagnostic named text the grader never
  saw. It now reports what the scorer actually received, captured at the scoring call.
- An environment that overrode its answer to the empty string had `flash env test` report the
  replayed turns instead: the captured text was tested for truth, so a graded `""` was
  indistinguishable from an episode that never reached the scorer. The override propagates on
  `is not None`, so `""` really is what the grader saw -- and an empty answer explains a zero
  reward on its own, making it the reading a reader most needs named rather than hidden.
- A crashed scorer went unreported on an `echo` episode -- a row with no gold answer to replay,
  which is also the case that leaves the GRPO gate with nothing to count. `flash env test` could
  exit `overall: PASS` with every reward coming from a scorer that never ran. The scorer's error is
  now surfaced for any policy.
- A scorer that crashed behind the SDK's guard reached `flash env test` as a bare `0.000000`,
  indistinguishable from a judged-wrong answer, because the adapter kept only
  `RewardResult.score`. The new `FreesoloEnvironment.reward_with_error` returns the reward and the
  discarded error from a single scoring call, so a missing runtime dependency names itself without
  re-scoring the episode -- which would bill a paid judge twice and, for a non-deterministic
  scorer, could report an error that did not produce the printed reward.
- Two tests resolved Hugging Face model metadata over the network and passed only on a
  connected runner. They now stub the resolver like their siblings already did.
