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

- An attempt's terminal artifacts (its strict marker plus `metrics.json`) are now interpreted by
  one shared protocol instead of being decoded separately by live instance polling and by each
  recovery path. The two decoders disagreed about identical bytes. A marker that could not be tied
  to the attempt was a terminal failure to the live poller but indistinguishable from "no marker
  yet" to recovery, so the same corrupt artifact produced a different story depending on which
  layer happened to observe it; recovery now names it rather than logging silence. Neither path
  ever adopted such a marker as completed work, and neither does now. Recovery also computed a
  fresh observation window for the marker and then another for `metrics.json`, so the real ceiling
  was their sum and moved with however long the first read took; both reads now share one window,
  which makes the bound the cutoff rather than the sum.
- CI runs the offline test suite on both supported interpreters (3.11 and 3.12) rather than
  3.11 alone, and sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` so a test that reaches the
  network fails deterministically instead of depending on runner connectivity.
- Every GitHub Action is pinned to a commit SHA rather than a mutable tag.
- `mypy` runs in CI as an advisory (non-blocking) job.

### Fixed

- `flash models deploy` now warns before it moves the shared `<run-id>` model id onto a
  different checkpoint. Every checkpoint of a run is served under that one id, so deploying
  `<run-id>/step-50` while `step-100` was live silently changed what `<run-id>` served for
  everyone using it — which made "deploy a second checkpoint to compare" a destructive
  operation on the first, and read as a serving regression rather than the deploy that caused
  it. The warning names both checkpoints and points at `flash models chat <run-id>/step-N`,
  which addresses a checkpoint directly and does not depend on where the id points. It is
  advisory: the deploy still proceeds, the pre-deploy read is abandoned outright once it exceeds
  a few seconds so no shape of slow plane can hold a deploy behind it, and a control plane that
  cannot answer (or answers with a checkpoint step this client cannot read) does not fail the
  command. A deployment whose activation outcome was never settled still warns, but without
  naming a checkpoint: that record's step describes the incoming attempt rather than what the id
  currently serves, and the authoritative target is server-side only, so the warning says the
  live checkpoint cannot be determined and points at `flash models deployments`.

- A response deadline did not actually bound a response that arrived slowly. The deadline is
  checked between reads, but each read asked for a fixed number of bytes and blocked until all of
  them arrived, so a peer trickling a short body kept one read inside the socket timeout
  indefinitely and the check never ran: a 2s deadline took 12s on a 42-byte body, and 61s when
  the socket timeout was the client default. Reading whatever has already arrived lets the check
  run, which bounds every stall shape. This affected `flash env list`, whose deadline exists to
  stop a slow environment hub hanging the command, as well as the new pre-deploy read.

- A response deadline could still be overrun by a read that began near its end. The socket timeout
  is installed once, when the request opens, and time spent connecting and waiting for headers is
  charged to the deadline but not to that timeout, so a peer that delayed its headers and then
  stalled mid-body held the call open past the deadline: 3.5s against a 2s budget. Each read now
  re-caps the socket to the remaining budget, so the two bounds agree instead of stacking. On a
  body still arriving this surfaces as a timeout rather than the "stalled" message; both bound the
  call, and which one wins is a race not worth depending on.

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
