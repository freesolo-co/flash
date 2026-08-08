"""Offline contract checks for local Freesolo environments."""

from __future__ import annotations

import json
import math
import sys
import tomllib
from pathlib import Path

from . import render
from .envpush import _err, _resolve_local_env_entrypoint

_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_ECHO_RESPONSE = "test"
# the one algorithm that trains from the environment reward. sft optimizes a supervised loss and opd
# a teacher token loss; neither reads `env.reward` anywhere (flash/engine/worker/sft.py, opd.py), so
# a scorer they never call cannot be evidence of anything for them.
_REWARD_DRIVEN_ALGORITHM = "grpo"
_PREVIEW_CHARS = 200
_DEFAULT_EPISODES = 3
# characters that give a TOML value structure. text containing none of them is a bare string, so a
# parse failure means the user typed unquoted words; text containing any of them was reaching for
# TOML syntax and a parse failure means they got it wrong.
_TOML_STRUCTURAL_CHARS = frozenset("\"'[]{}=,\n")
# the characters that make a TOML KEY mean something other than its literal spelling: `.` nests,
# quotes make a bare key hold characters it otherwise could not. deliberately narrower than
# _TOML_STRUCTURAL_CHARS, which describes values -- a key is checked before the `=` split, so the
# structural characters of a value are not applicable to it.
_TOML_KEY_STRUCTURAL_CHARS = frozenset(".\"'")
# every TOML scalar that is not a bare `true`/`false`/`inf`/`nan` word starts here: integers,
# floats, and the whole date/time family all begin with a digit, and a signed number with `+`/`-`.
# so a token starting with one of these was reaching for a TOML scalar, and failing to parse means
# it is malformed rather than prose -- the same reasoning _TOML_STRUCTURAL_CHARS applies to
# delimiters, applied to the tokens that carry no delimiter at all.
#
# `.` is here for the leading-dot float. TOML requires a digit before the point, so `.5` is exactly
# as malformed as `+.5` -- which the signs already caught, leaving the same spelling accepted or
# rejected depending on whether it carried a sign. it is not in _TOML_STRUCTURAL_CHARS, so it
# reaches this test rather than being read as a delimiter.
_TOML_SCALAR_LEADING_CHARS = frozenset("0123456789+-.")
# the TOML booleans, which are written as bare words rather than starting with a digit or sign and
# are therefore the blind spot of _TOML_SCALAR_LEADING_CHARS. TOML spells them in lowercase only, so
# a case variant is a malformed literal rather than prose and must not forward as a string.
_TOML_BOOLEAN_WORDS = frozenset({"true", "false"})
# the non-finite floats, the other bare-word family, spelled lowercase only and optionally signed.
# lowercase `nan` parses and _reject_unsubmittable_param then turns it away for not being JSON, but
# a case variant never reaches that check: it fails the TOML parse and falls through the bare-word
# test as the literal STRING "NaN". the offline gate then validates a str where the config would
# hold a float -- or an environment coercing it back gets the non-finite value the lowercase
# spelling was rejected for. so a case variant is malformed, not prose, either way.
#
# `infinity` is the same value written out. it is not a TOML spelling in any case, so it reaches the
# bare-word test rather than the parse, and matching only the abbreviation let it through as the
# string "Infinity" -- which an env normalizing with float() turns straight back into inf, the value
# the abbreviation is rejected for.
_TOML_NON_FINITE_WORDS = frozenset({"inf", "infinity", "nan"})
# TOML has no null. these are the spellings people reach for anyway, borrowed from json, python, and
# yaml -- all bare words, so they land in the same blind spot: the parse fails, the value carries no
# structural character, and it forwards as its own literal STRING. an env testing `if value is None`
# or `if not value` then reads a truthy string, and no [environment.params] assignment could have
# produced it, since the config has no way to spell an absent value either. omitting the parameter
# is what expresses that, so say so rather than forwarding text nothing asked for.
_TOML_NULL_WORDS = frozenset({"null", "none", "nil"})


def _check_messages(messages: object, label: str) -> list[dict]:
    """Validate that `messages` is a well-formed chat message list and return it."""
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{label} is not well-formed: {label} must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{label} is not well-formed: {label} message {index} must be a dict")
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} "
                "must have a non-empty string role"
            )
        if role.strip().lower() not in _ALLOWED_ROLES:
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} has unsupported role {role!r}"
            )
        if "content" not in message:
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} must have a content key"
            )
        # content is a string, a multimodal content-block list, or null for a native
        # assistant tool-call message. reject scalars like ints so a malformed message is
        # not reported as a false pass.
        content = message["content"]
        if content is not None and not isinstance(content, (str, list)):
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} content "
                "must be a string, a content-block list, or null"
            )
    return messages


def _message_text(content: object) -> str:
    # the canonical definition, so replay text and the reward path's own extraction cannot drift.
    from flash.multimodal import message_content_text

    return message_content_text(content)


def _reference_turns(env, example: dict) -> list[str]:
    # the sft_completion gold answer stands in for the missing policy model. validate its
    # envelope like the prompt so a malformed completion (scalar content, missing role)
    # fails the episode instead of silently falling back to echo. text is extracted the
    # same way the real reward path grades a completion, so a gold answer expressed as
    # openai-style text blocks is replayed instead of echoed; text-free turns (null content
    # or image-only blocks) yield an empty replay string that is kept in place.
    messages = _check_messages(env.sft_completion(example), "sft_completion")
    # only assistant turns stand in for the policy model; a gold completion with no assistant
    # message must NOT replay user/system text as the model response -- yield no replay text so
    # _resolve_policy falls back to echo.
    assistant = [m for m in messages if m["role"].strip().lower() == "assistant"]
    # assistant turns only (a gold with no assistant message must echo, not replay user/system);
    # keep text-free turns positionally (empty string) so multi-turn replay stays aligned.
    return [_message_text(m["content"]) for m in assistant]


def _resolve_policy(reference_turns: list[str]) -> str:
    return "replay" if "".join(reference_turns).strip() else "echo"


def _junk_response(reference_turns: list[str]) -> str:
    """A deliberately wrong answer, guaranteed to differ from this row's gold one.

    The control is only evidence while it is wrong. `_ECHO_RESPONSE` is the fixed string `test`,
    and a reference answer that happens to be `test` collides with it, so the probe below fed the
    grader the correct answer and read back the gold reward -- making a scorer that separates
    perfectly look like one that pays junk as much as gold, and failing a working environment
    Lengthening past every reference turn terminates: the turns are finite and each pass
    grows the string. Compared stripped, because a control differing from gold only in whitespace
    is one a grader may well still score as correct."""
    gold = {turn.strip() for turn in reference_turns}
    junk = _ECHO_RESPONSE
    while junk.strip() in gold:
        junk = f"not {junk}"
    return junk


def _carries_thinking_markup(reference_turns: list[str]) -> bool:
    """Whether a gold answer is written in reasoning markup this command cannot reproduce.

    A thinking run grades what `_scored_turn_text` leaves behind, which strips the `<think>` span
    before the reward ever sees it (flash/envs/adapter.py). This command has no run config to read
    `thinking` from -- it builds the environment locally, where it defaults off (adapter.py) -- so
    it replays the tagged reference verbatim. Against a strict answer-only grader every reference
    then scores zero, and the gate below reported a working environment as unable to recognize its
    own gold answers. The reward is still printed and still warned about; only the
    blocking conclusion is withheld, because the evidence for it cannot be produced here."""
    return any("<think>" in turn or "</think>" in turn for turn in reference_turns)


def _preview(value: object) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(f"{item.get('role', '?')}: {item.get('content', '')}")
            else:
                parts.append(str(item))
        text = " | ".join(parts)
    else:
        text = str(value)
    text = " ".join(text.split())
    if not text:
        return "(empty)"
    if len(text) <= _PREVIEW_CHARS:
        return text
    return f"{text[: _PREVIEW_CHARS - 3]}..."


def _new_record() -> dict:
    """Mutable per-episode record so a failing episode still reports real progress."""
    return {
        "policy": "n/a",
        "turns": 0,
        "reward": None,
        "prompt": [],
        "responses": [],
        "thinking_markup": False,
        "replay_incomplete": False,
        # the multi-turn rollout state, kept so the per-turn check below can ask about the episode
        # that was actually scored rather than a fresh one. None for single-turn, which has no state
        # and no per-turn vector either.
        "state": None,
    }


def _drive_single_turn(env, example: dict, record: dict, *, force_echo: bool = False) -> None:
    prompt = _check_messages(env.prompt_messages(example), "prompt")
    record["prompt"] = prompt
    reference_turns = _reference_turns(env, example)
    policy = "echo" if force_echo else _resolve_policy(reference_turns)
    record["policy"] = policy
    record["thinking_markup"] = _carries_thinking_markup(reference_turns)
    response = (
        "\n".join(turn for turn in reference_turns if turn)
        if policy == "replay"
        else _junk_response(reference_turns)
    )
    record["responses"] = [response]
    record["turns"] = 1
    record["reward"] = float(env.reward(response, example))


def _drive_multi_turn(env, example: dict, record: dict, *, force_echo: bool = False) -> None:
    state = env.new_rollout_state(example)
    record["prompt"] = _check_messages(state.get("prompt") or state.get("messages"), "prompt")
    reference_turns = _reference_turns(env, example)
    policy = "echo" if force_echo else _resolve_policy(reference_turns)
    record["policy"] = policy
    record["thinking_markup"] = _carries_thinking_markup(reference_turns)
    # mirror the worker turn loop (flash/engine/worker/grpo_multiturn.py): drive one model
    # turn, then stop at the hard turn ceiling, on the env's own done signal, or when the
    # env yields no reply. the hard cap is fixed at what the trainer passes (env.max_turns)
    # and the turn counter rises every turn until it reaches the cap, so a cooperatively-
    # stepping env terminates here exactly as it would in training; no separate
    # non-termination guard is needed.
    hard_cap = int(env.max_turns)
    turns = 0
    # mirrors the worker's own flag: True while the newest turn has not been through env_reply.
    env_step_pending = False
    while True:
        if policy == "replay" and turns < len(reference_turns):
            content = reference_turns[turns]
        else:
            # a reference shorter than the episode is padded so the rollout still reaches its own
            # termination, but the trajectory graded at the end is then part gold and part junk.
            # scoring it is not evidence about whether the grader recognizes its references, so the
            # flag keeps it out of the blocking gate's totals.
            if policy == "replay":
                record["replay_incomplete"] = True
            content = _junk_response(reference_turns)
        record["responses"].append(content)
        env.record_model_turn(state, content)
        env_step_pending = True
        turns += 1
        record["turns"] = turns
        if turns >= hard_cap or env.rollout_done(state, max_turns=hard_cap):
            break
        env_msgs = env.env_reply(state["messages"], state)
        env_step_pending = False
        if not env_msgs:
            break
        # the env's own reply messages feed the chat template for the next turn in the real
        # rollout, so validate their envelope here too: a malformed reply that would break
        # remotely must fail the episode instead of slipping through on a finite reward.
        _check_messages(env_msgs, "env_reply")
        if env.rollout_done(state, max_turns=hard_cap):
            break

    # the driver-side exits above stop before the inter-turn env_reply, so the last replayed turn
    # is still unapplied. a stateful env would then score a board or transcript missing the last
    # thing the model did, so give it that turn before scoring -- only the inter-turn glue is
    # skipped, since no further model turn is conditioned on the reply. this mirrors the worker
    # loops (opd_train / rl_train), which check termination before requesting an env_reply for the
    # same reason: this command exists to catch a contract break before a paid run does, and it can
    # only do that while it scores the state the run would score.
    #
    # env_step_pending is False once the last generated turn has already been through env_reply (an
    # env that replied with nothing, a natural finish) -- stepping again there would be a spurious
    # extra move. rollout_done covers the env having declared the episode over.
    if env_step_pending and not env.rollout_done(state, hard_cap):
        env.env_reply(state["messages"], state)
    record["reward"] = float(env.reward("", example, state))
    record["state"] = state


def _scores_gold_no_better_than_junk(env, example: dict, gold_reward: float) -> bool:
    """Whether the grader pays a deliberately wrong answer at least as well as its own gold one.

    This is the evidence the flat-reward gate needs. The reward's own value cannot supply it: a
    centered scale that pays 0 for a correct answer and negative for an incorrect one separates
    them perfectly, and reading that zero as failure rejected a trainable environment. What does
    not train is a grader whose gold answer earns no more than junk, whatever number it prints.

    The junk run drives the same env through the same code path as the real episode, with the
    replay suppressed so the echo placeholder stands in for a wrong model answer. A grader that
    raises on it is not evidence of anything -- an env may legitimately require a parseable answer
    -- so treat that as separation and leave the episode to its own contract checks.

    Call this only after every episode has been scored. The probe is an extra pass through the same
    env, and an env that advances state per call (a scripted reward sequence, a cursor into a
    dataset) would hand a later episode the answer meant for this one -- the probe deciding the
    outcome of the run it exists to measure."""
    try:
        probe = _new_record()
        if env.multi_turn:
            _drive_multi_turn(env, example, probe, force_echo=True)
        else:
            _drive_single_turn(env, example, probe, force_echo=True)
        junk_reward = probe["reward"]
    except (Exception, SystemExit):
        return False
    if junk_reward is None or not math.isfinite(junk_reward):
        return False
    return junk_reward >= gold_reward


def _separates_on_turn_rewards(env, example: dict, state: dict | None) -> bool:
    """Whether the env grades this episode on a per-turn vector the episode scalar cannot show.

    `credit_assignment = "per_turn"` trains from `per_turn_rewards` in the score result's metadata,
    which reaches the trainer through `rollout_rewards_many` (flash/envs/adapter.py) and never
    through `env.reward`. So a multi-turn env may legitimately return a flat episode scalar while
    the vector it actually trains on distinguishes the turns, and reading only the scalar reported a
    working environment as unable to recognize its references.

    Unlike the algorithm, this is answerable here: ask the env for the typed reward and look. A
    vector holding more than one distinct value is separation the scalar could not express. Anything
    else -- no such method, single-turn, no vector, a raise -- leaves the scalar as the only evidence
    and the gate proceeds on it."""
    rollout_rewards_many = getattr(env, "rollout_rewards_many", None)
    if not callable(rollout_rewards_many):
        return False
    try:
        rewards = rollout_rewards_many([(example, state or {})])
    except (Exception, SystemExit):
        return False
    if not rewards:
        return False
    turns = getattr(rewards[0], "turns", None)
    if not turns:
        return False
    return len({float(turn) for turn in turns}) > 1


def _load_failure(reason: str) -> int:
    _err(f"env test failed: {reason}")
    print("0/0 episodes passed contract checks")
    return _err("overall: FAIL")


def _evaluation_example(case) -> dict:
    example = dict(case.metadata or {})
    example["input"] = case.input
    example["output"] = case.expected
    if case.id is not None:
        example["id"] = case.id
    return example


def _normalize_prompt_images(env, example: dict, messages: list[dict]) -> None:
    """Resolve the case's images the way the online command and every training worker do.

    Raises whatever `normalize_prompt_images` raises, so an unreadable, oversized, or malformed
    image fails this gate with the message the caller would have hit at generation time. Text-only
    cases -- the vast majority -- return without importing the multimodal machinery at all.
    """
    from flash.multimodal import record_has_images

    if not record_has_images(example, messages):
        return
    from flash.multimodal import normalize_prompt_images

    normalize_prompt_images(example, messages, getattr(env, "package_root", None))


def _evaluation_response(env, case) -> tuple[str, str]:
    example = _evaluation_example(case)
    # build the prompt even though the replayed response does not need it. `flash env eval` sends
    # every case through prompt_messages() (flash/cli/env_eval.py `_case_messages`), so a prompt
    # that raises or returns malformed messages for a held-out case is a suite the online command
    # records a prompt-construction error for. checking only the scorer let this offline gate print
    # `overall: PASS` for exactly that sidecar.
    build = getattr(env, "prompt_messages", None)
    if callable(build):
        messages = _check_messages(build(example), "prompt")
        # `prompt_messages()` is only half of the prompt: `flash env eval` then runs
        # `normalize_prompt_images` (`_remote_prompt_messages`), as every training worker does
        # before tokenization. the envelope check above sees only the message list, so a case
        # carrying a top-level `image`/`images` -- a missing or oversized package-relative file --
        # or a malformed image block inside its prompt passed this gate, and the online command then
        # recorded prompt-construction failures for a suite reported `overall: PASS`. run the same
        # normalization here, against the environment's own package root, so both commands reject
        # the same suites.
        _normalize_prompt_images(env, example, messages)
    reference_turns = _reference_turns(env, example)
    policy = _resolve_policy(reference_turns)
    response = (
        "\n".join(turn for turn in reference_turns if turn)
        if policy == "replay"
        else _ECHO_RESPONSE
    )
    return policy, response


def _check_evaluation_suites(entrypoint: Path, env) -> bool:
    from flash.envs.evaluations import (
        _DEFAULT_EVALUATIONS_PATH,
        EvalSuiteReport,
        has_evaluations,
        load_evaluation_suites,
        normalize_eval_result,
        validate_evaluation_cases,
    )

    if not has_evaluations(entrypoint):
        return True
    source = entrypoint.parent / _DEFAULT_EVALUATIONS_PATH
    try:
        suites = load_evaluation_suites(entrypoint, environment=env)
    except (Exception, SystemExit) as exc:
        reason = str(exc) or exc.__class__.__name__
        _err(f"evaluation checks failed: {reason}")
        return False

    all_valid = True
    for suite in suites:
        results = []
        try:
            cases = validate_evaluation_cases(suite, source=source)
            if not cases:
                all_valid = False
                _err(
                    f"evaluation suite {suite.name} failed contract checks: suite produced no cases"
                )
                continue
            for index, case in enumerate(cases, start=1):
                _policy, response = _evaluation_response(env, case)
                scored = suite.score(case, response)
                result = normalize_eval_result(
                    case,
                    response,
                    scored,
                    case_id=case.id or str(index),
                )
                results.append(result)
            report = EvalSuiteReport(name=suite.name, results=tuple(results))
            # a scorer that reported an error did not grade the case. `flash env eval` counts
            # those as errors and fails the suite, so approving them here would let the offline
            # gate greenlight exactly the sidecar the online command refuses.
            errored = [result for result in results if result.error]
            if errored:
                all_valid = False
                detail = ", ".join(f"{result.case_id}: {result.error}" for result in errored)
                _err(
                    f"evaluation suite {suite.name} failed contract checks: "
                    f"{len(errored)}/{report.total} case(s) reported a scoring error ({detail})"
                )
                continue
            print(
                f"evaluation suite {suite.name}: {report.total}/{report.total} cases "
                f"passed contract checks mean_score={report.mean_score:.6f}"
            )
        except (Exception, SystemExit) as exc:
            all_valid = False
            reason = str(exc) or exc.__class__.__name__
            _err(f"evaluation suite {suite.name} failed contract checks: {reason}")
    return all_valid


def _reject_unsubmittable_param(key: str, value: object) -> None:
    """Reject a parsed TOML value that ``[environment.params]`` could not actually submit.

    TOML has date/time types that JSON does not, so ``--param cutoff=2026-01-01`` parses cleanly
    into a ``datetime.date``. The equivalent config keeps that object in ``EnvironmentSpec.params``
    and the submit fails later at ``json.dumps(body)`` in ``ApiClient._request()``. Approving it
    here would mean the gate passed on a config that cannot be submitted at all.

    ``allow_nan=False`` because the default does NOT raise on ``nan``/``inf``: it emits the
    non-standard tokens ``NaN`` and ``Infinity``, which are not JSON at all. `--param
    threshold=nan` therefore passed the gate and produced a request body a strict parser rejects
    Raised as ValueError, which is a separate exception from the TypeError above.

    ``ensure_ascii=False`` so the encode reaches the text itself rather than escaping it away. The
    default renders every non-ascii character as ``\\uXXXX``, which a lone surrogate survives, and
    the value then forwards to an env that can open the path while no UTF-8 config could carry it
    Encoding what json produced is what makes this cover a surrogate nested inside a
    list or table, not just a bare scalar.
    """
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--param {key} is not JSON-serializable and could not be submitted "
            f"in [environment.params]: {exc}"
        ) from exc
    if not _is_expressible_in_toml(encoded):
        raise ValueError(
            f"--param {key} is not valid UTF-8, so no config file could carry it and the run "
            f"would never receive this value"
        )


def _parse_param_value(key: str, raw: str) -> object:
    """Parse one ``--param`` value the way ``[environment.params]`` would parse it.

    Bare unquoted text is not valid TOML but is what users type most often, so it falls back to a
    string. Everything else must parse, because silently keeping a malformed structured value as a
    literal string would validate the environment against parameters the equivalent config entry
    could never load -- the offline gate would pass on input training rejects.
    """
    value = raw.strip()
    try:
        document = tomllib.loads(f"v = {value}")
    except tomllib.TOMLDecodeError as exc:
        # the fallback is an allowlist, not a blocklist of opening delimiters: `filters=]` opens
        # nothing yet is still malformed TOML, and blocklisting only the openers let it through as
        # the literal string "]". a bare string is text with no TOML structural character in it.
        #
        # ...and no delimiter is needed to be reaching for TOML syntax. `cutoff=2026-13-01` holds
        # none of those characters, so it forwarded as the string "2026-13-01" while the equivalent
        # `[environment.params]` entry fails to load -- the gate passing on a config that cannot be
        # written. same for `1e`, `0x`, `007`, `1_`, `12:99:00`. a leading digit or sign is the
        # tell: every TOML scalar except the bare-word `true`/`false`/`inf`/`nan` spellings starts
        # with one, so such a token is a malformed number or date, not prose.
        if value and not (set(value) & _TOML_STRUCTURAL_CHARS):
            # the booleans are the family of TOML scalars that does NOT start with a digit or sign,
            # so the leading-character test below cannot see them. TOML spells them lowercase only,
            # which makes a python-style `strict=False` parse-fail and fall through here as the
            # STRING "False" -- and a non-empty string is truthy, so an env branching on `if strict`
            # reads it as enabled while the config spelling `false` disables it. the offline gate
            # would pass on the opposite of what the run trains with.
            if value.lower() in _TOML_BOOLEAN_WORDS:
                raise ValueError(
                    f"--param {key} is not a valid TOML value: {exc}. TOML spells "
                    f"{value.lower()} in lowercase; write --param {key}={value.lower()} for the "
                    f"boolean, or --param '{key}=\"{value}\"' to pass it as text"
                ) from exc
            # the non-finite floats are the same blind spot, minus their optional sign -- `-Inf`
            # does start with one of the leading characters, but the message that test raises talks
            # about numbers and dates and would not name what is actually wrong.
            if value.lstrip("+-").lower() in _TOML_NON_FINITE_WORDS:
                raise ValueError(
                    f"--param {key} is not a valid TOML value: {exc}. TOML spells the non-finite "
                    f"floats as lowercase inf and nan, and [environment.params] could not submit "
                    f"one anyway since it is not JSON; pass a finite number, or "
                    f"--param '{key}=\"{value}\"' to pass it as text"
                ) from exc
            # the null spellings, the last bare-word family. unlike the others there is no lowercase
            # form to point at, because TOML cannot express an absent value at all.
            if value.lower() in _TOML_NULL_WORDS:
                raise ValueError(
                    f"--param {key} is not a valid TOML value: {exc}. TOML has no null, so "
                    f"[environment.params] could not carry this either; omit --param {key} to "
                    f"leave it unset, or --param '{key}=\"{value}\"' to pass it as text"
                ) from exc
            if value[0] not in _TOML_SCALAR_LEADING_CHARS:
                # the bare-string path returns before the parsed-value checks below, so the one
                # that applies to text is asked here too. this is the route a surrogate actually
                # takes: it holds no structural character, so it is read as prose.
                _reject_unsubmittable_param(key, value)
                return value
            # quoting is the escape hatch, and it is the same spelling the config needs -- a
            # genuinely textual "3px" has to be written `"3px"` in `[environment.params]` too, so
            # pointing at it keeps the flag and the config in step rather than adding a second
            # spelling that only the flag accepts.
            raise ValueError(
                f"--param {key} is not a valid TOML value: {exc}. it starts like a number or "
                f"date, so [environment.params] would reject it too; quote it "
                f"(--param '{key}=\"{value}\"') to pass it as text"
            ) from exc
        raise ValueError(f"--param {key} is not a valid TOML value: {exc}") from exc
    # a value containing a newline makes `v = <value>` a multi-line document, so tomllib accepts
    # `max_rows=5\nstrict=true` as two assignments and taking only "v" drops the second silently.
    if set(document) != {"v"}:
        extra = ", ".join(sorted(set(document) - {"v"}))
        raise ValueError(
            f"--param {key} contains more than one assignment ({extra}); "
            "pass one KEY=VALUE per --param"
        )
    parsed = document["v"]
    _reject_unsubmittable_param(key, parsed)
    return parsed


def _is_expressible_in_toml(text: str) -> bool:
    """Report whether ``[environment.params]`` can carry ``text``, unchanged.

    The question is not whether the text is a TOML BARE key -- quoted keys and basic strings hold
    spaces, slashes and non-ascii perfectly well -- but whether the config can express THIS text at
    all. A basic string can: every character is either literal or has an escape, so the only text
    left out is what the file cannot physically contain. The config is read as UTF-8
    (``tomllib.load``, flash/schema/__init__.py), so that is exactly the un-encodable text -- a lone
    surrogate, which reaches argv when a command line carries a byte that is not valid UTF-8.

    Asked of both sides of an assignment. A surrogate is no more expressible on the right than on
    the left, and guarding only the name let `--param dataset_path=<surrogate>` forward a path the
    loader can open and the gate can PASS on, while no UTF-8 training TOML could submit the run that
    was validated.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _literal_param_key(key: str) -> str:
    """Resolve one ``--param`` name to the literal name ``[environment.params]`` would produce.

    The left side of a `[environment.params]` entry is a TOML key, not a literal name, so the
    spelling and the name can differ. `difficulty.level = 3` in a config is
    ``{"difficulty": {"level": 3}}``, but taking the source spelling literally forwarded
    ``{"difficulty.level": 3}`` instead -- a different call, which an environment accepting
    ``**kwargs`` swallows without ever exercising the nested parameter the run trains on. The gate
    then passes on input it never actually tested.

    So a structural spelling is resolved through tomllib rather than guessed at, and that also
    supplies the escape for a name that CONTAINS a dot: `"release.channel" = 3` is a quoted key and
    yields the flat ``{"release.channel": 3}``. Classifying dots and quotes as structure outright
    left that valid config with no `--param` spelling at all -- the quoted form is
    both the remedy and the same text the config needs.

    A genuinely nested table is still rejected. Params are splatted as keyword arguments
    (``load_freesolo_environment(..., **params)``, flash/envs/registry.py), so `difficulty` would
    have to arrive as a whole dict; `--param difficulty={level = 3}` already says that exactly, and
    is what the error points at.
    """
    name = key
    if set(key) & _TOML_KEY_STRUCTURAL_CHARS:
        try:
            document = tomllib.loads(f"{key} = 0")
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"--param {key} is not a valid TOML key: {exc}. [environment.params] would reject "
                f"this spelling too; quote the name (--param '\"{key}\"=...') to pass it literally"
            ) from exc
        # one assignment yields one top-level entry, whatever the spelling. it is nested exactly
        # when the dots were read as structure, which is the case this flag cannot forward.
        ((name, resolved),) = document.items()
        if isinstance(resolved, dict):
            raise ValueError(
                f"--param {key} uses TOML key syntax that denotes structure, which this flag "
                f"cannot forward faithfully. pass the containing table as one value instead, for "
                f"example --param {name}='{{ level = 3 }}', or quote the name "
                f"(--param '\"{key}\"=...') if the dot is part of it"
            )
    # whether the config can hold the name at all. almost always yes: a QUOTED key carries `bad
    # key`, `a/b`, `café` and the rest, and the schema loader takes it, so those are configs a run
    # really can receive. an earlier guard here rejected anything outside the BARE-key grammar,
    # which blocked validating a working config while claiming the config could not hold the name.
    # what is left is the names a UTF-8 config file cannot physically contain.
    if not _is_expressible_in_toml(name):
        raise ValueError(
            f"--param {key!r} is not valid UTF-8, so no config file could carry it and the run "
            f"would never receive this parameter"
        )
    return name


def _quoted_key_end(item: str, start: int) -> int | None:
    """Index of the quote closing the one at ``start``, or None when it is never closed."""
    quote = item[start]
    index = start + 1
    while index < len(item):
        # only a basic string takes escapes. in a literal string a backslash is just a character,
        # so consuming the next one there would step over the closing quote.
        if item[index] == "\\" and quote == '"':
            index += 2
            continue
        if item[index] == quote:
            return index
        index += 1
    return None


def _split_param_assignment(item: str) -> tuple[str, str, str]:
    """Split one ``--param`` argument at the ``=`` that separates its key from its value.

    Mirrors ``str.partition("=")`` in shape, but skips over quoted stretches of the key first. A
    quoted TOML key may itself contain an ``=`` -- ``[environment.params]`` accepts ``"a=b" = 1``
    as the flat key ``a=b`` -- and splitting at the first one turned that spelling into the key
    ``"a`` and rejected it, leaving a loadable config with no CLI spelling that could validate it.

    An unterminated quote is not a key this can find the end of, so it falls back to the first
    ``=``; the key check then reports the malformed spelling rather than this returning something
    arbitrary.
    """
    index = 0
    while index < len(item):
        char = item[index]
        if char in "\"'":
            closing = _quoted_key_end(item, index)
            if closing is None:
                break
            index = closing + 1
            continue
        if char == "=":
            return item[:index], "=", item[index + 1 :]
        index += 1
    return item.partition("=")


def _env_params(args) -> dict:
    """Build the ``load_environment()`` kwargs from ``--split`` / ``--param KEY=VALUE``.

    Mirrors ``[environment.params]`` so the local gate can validate the split a run actually
    trains on. Without this the gate always loaded ``dataset/train.jsonl`` and could pass while
    the configured split was never exercised.
    """
    params: dict = {}
    for item in getattr(args, "param", None) or []:
        key, sep, raw = _split_param_assignment(str(item))
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--param must be KEY=VALUE (got {item!r})")
        params[_literal_param_key(key)] = _parse_param_value(key, raw)
    split = getattr(args, "split", None)
    if split is not None:
        # distinguish "not passed" from "passed empty". `--split "$SPLIT"` with an unset variable
        # is an explicit request for a split, and treating it as absent leaves a `--param
        # split=...` in effect -- so the gate silently validates a different split than the one the
        # command asked for, which is the failure this flag exists to prevent.
        split = str(split).strip()
        if not split:
            raise ValueError("--split requires a non-empty split name")
        params["split"] = split
    return params


def cmd_env_test(args) -> int:
    """Load a local environment and drive deterministic offline contract checks.

    a fully non-returning environment hook (one that never yields control back) cannot be
    interrupted in-process, so run this under a ci job timeout to bound that class of
    defect; the per-episode turn cap (env.max_turns) bounds any cooperatively-stepping
    multi-turn loop exactly as the trainer does.
    """
    try:
        _, _, entrypoint, _ = _resolve_local_env_entrypoint(Path(args.path))
    except (Exception, SystemExit) as exc:
        reason = str(exc) or exc.__class__.__name__
        return _load_failure(reason.replace("cannot publish", "cannot test"))

    try:
        params = _env_params(args)
    except ValueError as exc:
        return _load_failure(str(exc))

    try:
        from flash.envs.loader import load_freesolo_environment

        # resolve to an absolute path so the loader takes its local-file branch; a bare
        # relative dir like `my-env` matches the managed-slug pattern and would otherwise
        # resolve remotely, breaking the offline contract.
        env = load_freesolo_environment(str(entrypoint.resolve()), **params)
        dataset = env.dataset()
    except (Exception, SystemExit) as exc:
        reason = str(exc) or exc.__class__.__name__
        return _load_failure(reason)

    if not dataset:
        return _load_failure("dataset is empty")

    episode_count = min(_DEFAULT_EPISODES, len(dataset))
    passed = 0
    # only replay episodes carry a gold answer to score. an echo episode has none, so its reward
    # says nothing about the grader and is counted in neither total.
    replayed = 0
    replayed_zero: list[tuple[dict, dict | None]] = []
    for index, example in enumerate(dataset[:episode_count], start=1):
        record = _new_record()
        failure: str | None = None
        try:
            if env.multi_turn:
                _drive_multi_turn(env, example, record)
            else:
                _drive_single_turn(env, example, record)
            reward = record["reward"]
            if reward is None or not math.isfinite(reward):
                raise ValueError(f"reward is not finite: {reward}")
        except (Exception, SystemExit) as exc:
            failure = str(exc) or exc.__class__.__name__

        reward = record["reward"]
        reward_text = "n/a" if reward is None else f"{reward:.6f}"
        print(
            f"episode {index}: policy={record['policy']} "
            f"turns={record['turns']} reward={reward_text}"
        )
        print(f"  prompt: {_preview(record['prompt'])}")
        print(f"  response: {_preview(record['responses'])}")
        if failure:
            _err(f"episode {index} failed contract checks: {failure}")
            continue

        passed += 1
        if record["policy"] == "replay" and reward is not None:
            # a reference written in reasoning markup cannot be replayed faithfully from here (see
            # `_carries_thinking_markup`), so its score is not evidence about the grader and is kept
            # out of the blocking gate's totals. neither is a multi-turn episode whose gold answer
            # ran out before the rollout did: the graded trajectory is then part reference and part
            # junk, and a zero says nothing about the reference the gate never ran. the advisory
            # warning below still fires for both: a low reward is worth surfacing either way, it
            # just cannot be the reason to fail.
            if not record["thinking_markup"] and not record["replay_incomplete"]:
                replayed += 1
                # an exact zero is what a scorer that recognized nothing returns, so it stays the
                # signature this gate looks for. it is confirmed against a wrong answer once every
                # episode has run, not here: see `_scores_gold_no_better_than_junk`.
                if reward == 0.0:
                    replayed_zero.append((example, record["state"]))
            if reward <= 0.0:
                message = (
                    f"replay gold answer scored low (reward={reward:.6f}); "
                    "check the reward function"
                )
                print(
                    render.warn(message) if render.styled() else f"warning: {message}",
                    file=sys.stderr,
                )

    print(f"{passed}/{episode_count} episodes passed contract checks")
    if passed != episode_count:
        return _err("overall: FAIL")
    # a grader that pays every one of its own reference answers zero cannot recognize them at all --
    # a broken scorer or a missing runtime dependency (LS-005) -- and the run reaches a gpu able to
    # see only flat reward. deliberately narrow: a PARTIAL failure stays the warning above and still
    # passes, because a strict reward function with some hard rows is legitimate.
    #
    # zero alone does not settle it, though. a centered scale paying 0 for a correct answer and
    # negative for an incorrect one separates them perfectly, which is all GRPO's relative advantage
    # needs, and failing on its zero rejected those environments outright. so ask the reward what a
    # deliberately wrong answer is worth, and block only if it is worth as much. one probe settles
    # it: the question is whether this reward can separate at all, and a grader that scored every
    # gold answer identically will answer identically for each of them.
    #
    # and the whole gate presumes the reward IS the training signal, which is true of grpo alone.
    # sft trains on a supervised loss and opd on a teacher token loss; neither reads `env.reward`,
    # so a no-op scorer is legitimate for them and blocking on it failed the recommended pre-push
    # check for environments that train perfectly well. the algorithm lives in the run config
    # (`JobSpec.algorithm`), which an environment directory does not carry, so it is asked for --
    # and defaults to grpo, the case that needs the gate. defaulting to the reward-driven algorithm
    # keeps the gate for a caller that passes no algorithm at all: an absent field must not be the
    # way to switch a blocking check off.
    algorithm = getattr(args, "algorithm", None) or _REWARD_DRIVEN_ALGORITHM
    grader_recognizes_gold = not (
        algorithm == _REWARD_DRIVEN_ALGORITHM
        and replayed
        and len(replayed_zero) == replayed
        and not _separates_on_turn_rewards(env, *replayed_zero[0])
        and _scores_gold_no_better_than_junk(env, replayed_zero[0][0], 0.0)
    )
    if not grader_recognizes_gold:
        _err(
            f"all {replayed} replayed gold answer(s) scored zero, no better than a deliberately "
            "wrong answer; the reward function cannot recognize its own reference answers. check "
            "the grader and that its runtime dependencies are installed in this environment."
        )
    # run the evaluation checks even when the grader gate already failed, so one `flash env test`
    # reports every broken surface at once instead of hiding the sidecar's errors behind the
    # reward function's. both gates block; neither short-circuits the other.
    if not _check_evaluation_suites(entrypoint, env) or not grader_recognizes_gold:
        return _err("overall: FAIL")
    print(render.ok("overall: PASS") if render.styled() else "overall: PASS")
    return 0
