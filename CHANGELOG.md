# Changelog

Notable changes to `freesolo-flash`. The dev-channel package `freesolo-flash-dev` ships in
lockstep at the same version.

Releases happen by merging a version bump from `dev` into `main`, which publishes to PyPI
(`.github/workflows/publish.yml`). Versions are `MAJOR.MINOR.PATCH`.

This file starts at 1.1.35. Earlier releases are not reconstructed here; use
`git log` for anything before that.

## Unreleased

### Added

- A `flash-cli` console script, the same entry point as `flash`, retained as a stable alias for
  existing operator automation and self-hosting instructions.

### Changed

- Managed RunPod training, cache preload, and kernel-cache baking now use Secure Cloud,
  on-demand, non-interruptible Pods instead of Serverless endpoints and queue jobs. Flash uses
  account-scoped REST APIs for Pods, payload secrets, network volumes, and catalog discovery;
  records the complete non-secret owning-key fingerprint; keeps the RunPod API key out of Pods;
  and confirms exact Pod plus payload-secret absence before retiring cleanup state. Shared cache
  volumes are preserved, while malformed or handleless recovery is reconciled by the shared
  run-label orphan sweep. The legacy endpoint reaper, endpoint billing, queue-job APIs,
  `runpod-flash` dependency, and generated Serverless handler were removed. Operators upgrading
  an existing plane must use the maintenance-window cutover documented in `SELF_HOSTING.md`.
- Completed training continues to charge exactly the accepted submit-time estimate. Partial work
  remains prorated from estimated completed work; measured RunPod Pod wall time and provider cost
  are internal COGS and telemetry only and never reprice the customer quote.

- `[train] init_from_adapter` now works for every source/target algorithm pair. SFT was rejected as
  a warm-start target, so an adapter could be continued only by GRPO or OPD; that restriction
  described the retired trl SFT backend, and the verl backend that replaced it loads a warm-start
  adapter exactly like the other two. All nine combinations of `sft`, `grpo`, and `opd` are
  supported, including same-algorithm continuation. A warm-started SFT run also inherits its
  source's base-model pin instead of resolving its own, so it stays deployable and no longer breaks
  when the base model's hub tip moves.
- Managed attempts now pin, verify, and report one immutable Flash source archive across retries and recovery.
- Multi-turn GRPO now clones each sibling rollout from the task state that produced its prepared
  prompt. Environments that record randomized episode choices on the task during `start_episode`
  are stepped and scored against the same episode the model saw, without sharing mutable state
  between sibling rollouts.
- A RunPod attempt's stall window is now measured from evidence timestamps rather than from when
  the control plane happened to look at it. The window was seeded from the moment polling started
  and was reset again by the first status read, and a heartbeat was credited at the time it was
  read rather than the time it was written. Every one of those is a wall-clock reading, so each
  reattach handed a worker that was already wedged a complete fresh setup window, and a supervisor
  that reattached repeatedly could keep a dead worker on a paid GPU until the run's wall deadline.
  The window is now anchored to the persisted launch, a heartbeat is credited at its own timestamp
  clamped to that launch, and only a genuine status transition counts as progress. Instance
  polling (Vast/Lambda) already worked this way; this brings RunPod onto the same rule. A job that
  is still queued is unaffected: the capacity wait keeps its own exemption, so a worker granted
  late still gets its full cold-start budget.
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

- Completed verl children now finish each captured W&B run before the W&B service shuts down,
  avoiding teardown tracebacks, and `flash env setup` now gives every starter config explicit W&B
  project and folder-derived run metadata.

- An ordered `[gpu] type` list is now priced over every class it declares acceptable, not just the
  first. Allocation cost-ranks the whole set, so quoting the head alone priced a shape the run may
  never be given — an authored `["B200", "H100"]` quoted roughly 3x the H100 the allocator would
  really rent, and the submit-time affordability check runs on that estimate, so a run the
  organization could afford could be refused before allocation ever got to choose. A single
  authored class is still a hard pin, and a no-fit rejection now names every acceptable class
  rather than whichever happened to be written first.

- A provider preference that exhausts every configured provider no longer falls back to quoting
  hardware this plane cannot rent. The final fallback dropped the restriction and ranked the
  registered RunPod pool regardless of credentials, so a vast-only plane could quote a B200: the
  estimate passed affordability and the run was recorded, only for live allocation to fail. The
  quote now raises against the eligible set, which names the real constraint.

- A GitHub token that is invalid, expired, or missing a scope now fails during preparation instead
  of after a GPU is rented. Non-rate-limit 401 and 403 responses were left as untyped errors, which
  the submit-time preflight treats as a transient blip worth deferring, so the run proceeded
  through affordability and allocation and hit the same permanent credential error on the worker.
  They are now classified permanent, like 404 and 422. The rate-limit 403 is still transient.

- A slow heartbeat upload no longer stalls every other heartbeat behind it. The throttle clock only
  advances once a commit lands, so during a slow HuggingFace commit other callers still saw an
  upload as due and blocked on the upload lock for up to its acquire timeout — 30s, or 120s on a
  critical stage — which stalled the liveness daemon and delayed leaving the wrapped stage by the
  same window. A throttled heartbeat now skips while a commit is in flight, since that commit
  already carries the stage's state. Terminal, error, initial, and forced step commits still queue
  and land: they carry state no later heartbeat would repair.

- A GRPO or OPD run whose rollout actor exited hard no longer wedges on a paid GPU until the
  wall-time limit. The actor's `os._exit` bypassed verl's own prompt handler, so the prompt
  stayed `running` forever and verl's unbounded replay poll waited on it at ~0% utilization. The
  child now writes durable failure evidence, marks the prompt failed, cleans up, and only then
  exits, and replay sampling fails the step on an explicit failure tag or a positively dead
  actor instead of returning a survivor-only batch that would silently train on fewer rollouts
  than were collected. A child that reaches the fit loop and then stops producing distinct
  output, while the parent has no teacher or environment work in flight, is torn down with a
  named cause rather than running silently to the limit; repeated identical lines count as
  silence, and authoritative OOM, host-RAM, and infrastructure evidence still classifies the
  exit first.
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
  usage strings, and `next:` hints) now name the executable actually invoked rather than always
  `flash`. Reached through `python -m flash.cli`, the printed command names the interpreter that
  is actually running rather than a bare `python` that may be absent or resolve to another
  environment.

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
