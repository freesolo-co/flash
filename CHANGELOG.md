# Changelog

Notable changes to `freesolo-flash`. The dev-channel package `freesolo-flash-dev` ships in
lockstep at the same version.

Releases happen by merging a version bump from `dev` into `main`, which publishes to PyPI
(`.github/workflows/publish.yml`). Versions are `MAJOR.MINOR.PATCH`.

This file starts at 1.1.35. Earlier releases are not reconstructed here; use
`git log` for anything before that.

## Unreleased

### Added

- OpenAI chat serving now preserves mirror sampling controls across packaged, hosted, and managed
  entry points: up to four indexed choices, signed seeds, frequency and presence penalties, and
  OpenAI token logprobs. Buffered and raw SSE responses retain independent choice terminals and
  aggregate request usage; the decoded text-only stream remains single-choice.
- A `flash-cli` console script, the same entry point as `flash` under a name nothing else
  claims. The `server` and `dev` extras install `runpod-flash`, which declares its own `flash`
  script; whichever distribution is installed last wins, so on a control-plane host
  `pip install 'freesolo-flash[server]'` could leave `flash` running RunPod's CLI, which exits 0
  and does nothing. `flash` is unchanged and remains correct on a base (client-only) install.

### Changed

- Each training run now owns one authoritative top-level `seed` and may execute multiple durable
  fenced attempts on replacement hosts or GPU classes. Retry remains an action and budget policy,
  not an identity. Provider handles and CLI status expose attempt identity without persisting a
  duplicate seed. RunPod endpoints use an explicit `aN` suffix for every attempt, including attempt
  zero, while Lambda and Vast instances use `<run-prefix>-aN`. Resource cleanup recognizes only the
  exact run prefix and the new attempt boundary.
- Active training now exposes exactly `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.8-27B`, and
  `Qwen/Qwen3.6-35B-A3B`. Hosted serving remains active for 9B and 35B-A3B; hosted Qwen3.8 27B
  remains inactive. Qwen3.8 uses the immutable BF16 revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and official native block-FP8 serving revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, with the logical base and tokenizer provenance kept
  distinct from the served checkpoint. The Qwen3.5 0.8B, 2B, and 4B tiers and
  `Qwen/Qwen3.6-27B` cannot activate, while historical status, accepted billing retry,
  cancellation, cleanup, hosted revocation, and exact identity-based undeploy remain available.
  Qwen3.6 27B adapters are not compatible with Qwen3.8.
- Customer-owned serving is now Modal-only and requires explicit `--provider modal`. Modal remains
  live-qualified for all three active models. The 27B and 35B-A3B qualifications stay bound to the
  exact certified serving image digest. Each deployment still serves one exact base model. The 27B
  engine keeps its `H100!` placement and distinct served checkpoint and tokenizer provenance. The
  35B-A3B engine keeps its H200, BF16, FP8 KV, 32K context, eight-sequence, 4096 batched-token, and
  six rank-64 LoRA-slot contract. Customer-serving RunPod commands, profiles, identities, and
  lifecycle code are removed without a compatibility or teardown shim. Managed RunPod training,
  serverless workers, dependencies, provider routing, artifacts, and caching are unchanged.
- `[train] init_from_adapter` now works for every source/target algorithm pair. SFT was rejected as
  a warm-start target, so an adapter could be continued only by GRPO or OPD; that restriction
  described the retired trl SFT backend, and the verl backend that replaced it loads a warm-start
  adapter exactly like the other two. All nine combinations of `sft`, `grpo`, and `opd` are
  supported, including same-algorithm continuation. A warm-started SFT run also inherits its
  source's base-model pin instead of resolving its own, so it stays deployable and no longer breaks
  when the base model's hub tip moves.
- Managed training now uses durable fenced attempts with immutable grant, work, result, and run
  deadlines. Workers publish cumulative attempt-scoped progress and one digest-addressed terminal
  result manifest through the private Hugging Face dataset repository. Provider status describes
  resource state only, and the result manifest is the sole terminal worker authority for RunPod,
  Lambda, and Vast.
- `flash runs status` now exposes attempt, progress, resource, and result projections. Sparse or old
  progress is observational only: it cannot terminate work, trigger retry or teardown, extend a
  deadline, or authorize success. Provider observations keep the CLI visibly active while progress
  is quiet, without inferring failure from age.
- Managed attempts pin, verify, and report one immutable Flash source archive across retries and
  recovery.
- Multi-turn GRPO now clones each sibling rollout from the task state that produced its prepared
  prompt. Environments that record randomized episode choices on the task during `start_episode`
  are stepped and scored against the same episode the model saw, without sharing mutable state
  between sibling rollouts.
- The mutable training heartbeat, stage registry, progress-carry state, heartbeat-derived provider
  timers, and `stalled` retry outcome have been removed. Fixed deadlines and explicit provider
  capacity, transport, preemption, resource-loss, OOM, cancellation, and worker-result evidence now
  determine lifecycle transitions.
- The Flash-to-Freesolo status report no longer sends `lastHeartbeat` or `gpuStatus`. All existing
  lifecycle, project, cost, deployment, model, algorithm, artifact, and timestamp fields remain.
  Freesolo live progress is intentionally unavailable until its separate consumer update; Flash
  does not send new progress under compatibility names.
- CI runs the offline test suite on both supported interpreters (3.11 and 3.12) rather than
  3.11 alone, and sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` so a test that reaches the
  network fails deterministically instead of depending on runner connectivity.
- Every GitHub Action is pinned to a commit SHA rather than a mutable tag.
- `mypy` runs in CI as an advisory (non-blocking) job.

### Deployment cutover

This release requires a maintenance-window cutover because old mutable lifecycle records and old
provider resource names are not translated or read by the new protocol:

1. Stop new training submissions.
2. Drain or explicitly cancel every old-source queued, provisioning, running, finalizing, or
   reconciling attempt.
3. Confirm that no old RunPod job or endpoint and no Lambda or Vast instance remains. Old RunPod
   `rN` or implicit-attempt names and old Lambda or Vast `-s<seed>-a<attempt>` names are deliberately
   not matched after cutover.
4. Confirm that cleanup ledgers contain no unconfirmed resources.
5. Deploy the Flash control plane, worker images, and CLI package together at version `1.2.121`.
6. Accept that Freesolo live progress is unavailable until its separate consumer update; lifecycle
   mirroring continues without `lastHeartbeat` or `gpuStatus`.
7. Run the smallest safe SFT, GRPO, and OPD lifecycle smokes before reopening submissions.

### Fixed

- Qwen3.6 FP8 MoE serving now preserves unquantized activations for LoRA A while keeping the base
  MoE path quantized, backporting the exact vLLM PR #42120 behavior into the pinned 0.23.0 image.
- GRPO publication now validates complete finite optimizer-step actor gradient norms instead of
  masked advantage-bound spread, so valid truncated groups are not falsely rejected and deployable
  checkpoints remain withheld until terminal evidence validation succeeds.
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

- Verl child silence now emits bounded diagnostics only. It no longer terminates the child,
  classifies lifecycle failure, tears down provider resources, or starts a retry. Top-level worker
  cancellation and fixed work-deadline enforcement use exact process-group supervision with TERM,
  bounded grace, KILL, and group-disappearance verification.
- Cancellation billing reads only current-attempt, current-fence cumulative progress. Positive
  completed steps are prorated against the accepted quote, entering training carries a one-step
  floor, and stale attempt progress is ignored.
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
