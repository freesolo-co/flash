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

- `flash env test` reported `overall: PASS` on environments that could not have trained anything,
  because a well-formed hook return was the only thing it checked. Three silent-pass classes now
  speak up. It still exits 0 for all three - they are warnings, not new blocking gates - but a run
  no longer passes without a word about what it found:

  - **Every episode scored 0.0.** The existing blocking gate is GRPO-only and abstains for a
    reasoning-markup reference, a replay that ran out mid-episode, and a junk probe that raised, so
    an all-zero run routinely printed `PASS` and nothing else. A constant reward is never a
    measurement, and for GRPO it is actively silent: rewards are mean-centred within each group, so
    a constant reward is a zero advantage and a zero gradient - training completes, the loss curve
    looks unremarkable, W&B looks healthy, and the adapter comes out identical to its warm start.
    The end-of-run advantage-spread guard catches that only after the GPUs are paid for. An
    all-`echo` run is reported differently and more weakly, because zero is the _correct_ score for
    the deliberate junk echo replays - what is wrong there is that no gold answer was ever scored.
  - **A gold replay that never terminates.** An environment applying every move twice can solve no
    board, yet gold scored 0.60-0.65, beat a junk answer, and burned the full turn cap - clearing
    every existing check. What exposed it in the field was reading `turns=12` on a board with a
    five-move solution, a comparison the verdict never made. An episode stopped by the hard turn cap
    rather than the environment's own done signal now says so.
  - **A gold completion whose rendered role sequence collapses.** SFT does not replay a completion
    turn by turn; it renders one training string from `prompt_messages + sft_completion`, and
    nothing validated that concatenation. A multi-turn gold answer returned as assistant turns alone
    rendered as one user question followed by every answer back to back, training the model to dump
    the whole episode into a single reply - the opposite of the behaviour being taught. Because this
    gate scored each turn separately, it passed either way. The rendered role sequence is now checked
    for consecutive same-role turns and printed, and a collision at the prompt/completion seam is
    named separately from one inside the completion, since they are different edits.

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
