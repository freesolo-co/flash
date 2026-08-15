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
    non-GRPO algorithm and for a junk probe that raised, so an all-zero run routinely printed `PASS`
    and nothing else. The warning is independent of those abstentions - but not of the two that are
    about this command's own fidelity rather than the environment's health. A reference written in
    reasoning markup is replayed verbatim here (there is no run config, so `thinking` defaults off)
    and scores zero against a correct strict grader that would have seen the span stripped; a replay
    that ran out mid-episode is graded part reference and part junk. Neither zero is evidence about
    the environment, so both are excluded from the warning exactly as they are from the gate.
    Counting them said "this run measured nothing" about a working reasoning environment - the very
    conclusion the gate deliberately withholds. A constant reward that IS interpretable is never a
    measurement, and for GRPO it carries the risk of being actively silent: rewards are mean-centred
    within each group, so if sampled rollouts score alike the advantage and gradient are zero -
    training completes, the loss curve looks unremarkable, W&B looks healthy, and the adapter comes
    out identical to its warm start. That is stated as a risk rather than a prediction: this command
    scores at most three offline gold replays, never a group of policy samples, so it cannot know
    what a real rollout group would pay.
    The end-of-run advantage-spread guard catches that only after the GPUs are paid for. An
    For a non-GRPO run the warning says what is actually true of it: SFT and OPD never read
    `env.reward`, so an all-zero scorer costs that run nothing and a placeholder `reward()` on an
    SFT-only environment is a legitimate choice. It reports that the run is no evidence the grader
    works - so the PASS is not clearance to train GRPO on the same environment later - instead of
    sending the author to debug a function this algorithm never calls.
    An all-`echo` run is reported differently and more weakly, because zero is the _correct_ score for
    the deliberate junk echo replays - what is wrong there is that no gold text was ever scored,
    which includes a real gold completion whose payload sits outside `content`: a native tool call
    keeps its arguments in `tool_calls` and leaves `content` null. That row is a correct SFT target,
    so it gets the opposite remedy - the warning says explicitly not to add assistant text to
    satisfy the check, and to exercise the reward function against a sampled rollout instead, rather
    than sending the author to corrupt a working target. It stays silent when the grader was proven
    to separate - a centered scale paying gold 0.0 and junk -1.0, or a per-turn vector that
    separates while the scalar is only a placeholder - since both have a real gradient and calling
    them unmeasured would be false. A per-turn vector only counts as separation when the paid
    worker would actually train from it: `_validated_reward` discards a vector holding a non-number,
    a non-finite value, or a count that disagrees with the assistant turns the episode emitted, and
    falls back to the flat episode reward. Reading such a vector as separation here was therefore
    backwards - it suppressed both the warning and the blocking gate on precisely the runs that do
    train on all-zero rewards, so two rewards for a three-turn episode bought a silent `PASS`. The
    same three checks now apply before a vector is believed. Neither probe behind that judgement is free: the junk probe
    drives a whole extra episode through user code, and the per-turn probe runs the environment's
    own `score_episodes`, which may be the same paid judge the episodes just used. Both are spent
    only where their answer can change what the command prints _and_ the algorithm trains from the
    reward, since for SFT and OPD the answer could only soften a warning about a number that is
    never read. Billing is therefore unchanged from the previous release in every case measured
    against a counting grader: a healthy environment costs one call under both SFT and GRPO, and an
    all-zero environment costs one under SFT and OPD and three under GRPO, exactly as before.
  - **A gold replay that never terminates.** An environment applying every move twice can solve no
    board, yet gold scored 0.60-0.65, beat a junk answer, and burned the full turn cap - clearing
    every existing check. What exposed it in the field was reading `turns=12` on a board with a
    five-move solution, a comparison the verdict never made. An episode stopped by the hard turn cap
    rather than the environment's own done signal now says so. The verdict is drawn after the
    deferred final `env_reply`, so an environment that solves its task on the last allowed turn is
    not misreported. A gold answer shorter than the episode is judged at the moment it ran out
    rather than at the end: the junk the driver pads with cannot advance the environment, so the
    final state says nothing about the reference either way. Both halves stay reportable - a
    five-move solution against a twelve-turn cap on an unsolvable board (the field case, since real
    datasets carry minimal solutions) is still reported, and so is a short gold answer on a working
    environment, because at that sampling point the two are indistinguishable: both are not-done,
    one because it never can be and one because a real rollout would simply have continued. Rather
    than resolve that ambiguity silently in either direction, the warning states it - a reference
    that ran out before the cap names both explanations and puts the cheaper one first, since
    extending a dataset row costs less than auditing termination logic that works. Only a reference
    that covered every turn and still never finished leaves the environment as the sole
    explanation. A gold trajectory _longer_ than the cap is separated out before either reading,
    since its cause is fully determined rather than guessed: only a prefix could be replayed, so
    the remaining gold turns were never scored, and the report names both numbers and the exact
    remedy (raise the cap, or shorten the row) instead of sending the author to audit it. That
    mismatch is reported independently of whether the episode finished, because the two are not the
    same question: an environment that sets `done` while applying the capped prefix ends the episode
    tidily, so the not-done signal never fires, and the dropped gold turns went unmentioned on an
    otherwise clean `overall: PASS`. The count it reports is the turns actually replayed, not the
    cap - an environment that terminates before reaching the cap sends fewer, and quoting the cap
    claimed a turn had been replayed that never was, contradicting the episode line just below it.
    Otherwise
    the wording stops short of a verdict, because `rollout_done`
    returns True at `turn >= cap` regardless of `state["done"]`, so a fixed-length episode that ends
    purely by using its whole budget is a supported shape that never sets `done` and is
    indistinguishable from a dead one here. Both are reported, the reward that separates them in
    practice is quoted on the line, and the reader is told which reading applies to which. The
    ceiling compared against is the
    _effective_ one: a row setting `max_episode_turns` below the dataset-wide `max_turns` is stopped
    by its own budget, which `rollout_done` gives precedence, so measuring exhaustion against
    `max_turns` alone silenced this warning for exactly the rows whose budget is tightest.

    Measuring against the effective cap also made an existing hole reachable: for an episode
    stopped at the ceiling, the final `env_reply` is issued by a deferred call after the loop, and
    that call never checked its envelope. An environment whose last reply is malformed - scalar
    `content`, which breaks the chat template on the paid run - therefore reached `overall: PASS`.
    It is now validated exactly like the in-loop replies, with an empty reply still allowed, since
    an environment with nothing further to observe legitimately returns none.

  - **A gold completion whose rendered role sequence collapses.** SFT does not replay a completion
    turn by turn; it renders one training string from `prompt_messages + sft_completion`, and
    nothing validated that concatenation. On a single-turn environment a gold answer returned as
    assistant turns alone rendered as one user question followed by every answer back to back,
    training the model to dump the whole episode into a single reply - the opposite of the
    behaviour being taught. Because this gate scored each turn separately, it passed either way.
    The rendered role sequence is now checked for consecutive same-role turns and printed.

    The check is scoped to where the defect is real. It is skipped entirely for multi-turn
    environments, where consecutive assistant turns are the contract rather than a fault - the
    intervening user turns come from `env_reply` at rollout time and are deliberately absent from
    the dataset. `tool` is exempt, since parallel tool calls are answered by one `tool` message per
    call and chat templates render each as its own block; `user` and `system` are not, because a
    doubled user turn is how an off-by-one trajectory capture shows up and it duplicates that text
    in the trained string just as a doubled assistant turn does. Only repeats the completion owns
    are reported, with `sft_completion`'s own indices: a repeat wholly inside `prompt_messages` is
    an ordinary prompt shape (a retrieved document in its own user message, a task and its
    reply-format instruction as separate turns) that no edit to the completion could change, and an
    `assistant` turn at the prompt/completion seam is a prefill the completion continues, the one
    adjacency meant to merge. A doubled `user` or `system` turn at that same seam has no such
    innocent form and is still reported, since it is the completion restating a turn the prompt
    already contains.

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
