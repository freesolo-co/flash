"""The parent side of the multi-turn GRPO rollout loop.

In multi-turn GRPO the verl child owns generation but not the environment: flash keeps the env
here, in the parent, and the child reaches it over a local reward server. `MultiTurnBridge` holds
that per-session env state, `start_reward_server` serves both the single- and multi-turn scoring
endpoints, and the copy/env helpers stage the child-side modules the verl interpreter imports.

Split out of `flash.engine.worker.train.entry.rl_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler

from flash.content.multimodal import normalize_environment_reply
from flash.engine.worker.train.core.child.glue import (
    dedup_seam_terminator,
    parent_environment_glue,
    parent_image_digests,
    validate_structured_messages,
)
from flash.engine.worker.train.entry.backend_common import BoundedThreadingHTTPServer
from flash.engine.worker.train.entry.score_batcher import ScoreBatcher
from flash.engine.worker.train.rl.rollout.scoring import RolloutScoreRequest, score_rollouts
from flash.engine.worker.verl.parent_work import ParentWorkGauge

# how many concurrently-finished episodes the multi-turn bridge scores in ONE env call. a whole
# generation is prompts_per_step * group_size episodes and they finish at different turn counts,
# so the batch is whatever has landed when the first waiter's grace period expires rather than a
# barrier over the step. the grace period is short because the cost being amortised is a judge
# round-trip: an episode should never wait longer than one is worth.
_MULTI_TURN_SCORE_BATCH_SIZE = 64
_MULTI_TURN_SCORE_FLUSH_WAIT_S = 0.1
_MULTI_TURN_SCORE_SHUTDOWN_WAIT_S = 5.0
# reap sessions abandoned by dead ray actors, which otherwise retain env state forever.
# the timeout must cover a live turn's full generation and env reply; late dead-session cleanup only
# holds memory, while premature cleanup breaks a working rollout.
_MULTI_TURN_SESSION_LEASE_S = 1800.0
_SINGLE_TURN_SCORE_BATCH_SIZE = 64
_SINGLE_TURN_SCORE_FLUSH_WAIT_S = 0.1
_SINGLE_TURN_SCORE_SHUTDOWN_WAIT_S = 5.0

# a body that could not be decoded at all: bad Content-Length, bad utf-8, or bad json.
_UNDECODABLE_BODY_ERRORS = (ValueError, TypeError, UnicodeDecodeError)


class _BadRequest(Exception):
    """A request THIS BRIDGE is deliberately rejecting because the request itself is wrong.

    Raised by the validating code rather than inferred from exception type at the handler, which
    cannot distinguish the bridge rejecting a bad index from a user env raising IndexError inside
    its own scoring -- and reporting the latter as a client error is the bug the 503 split fixes.
    """


class _BadIndex(_BadRequest, IndexError):
    """An index outside the dataset. Also an IndexError: that is what it genuinely is."""


class _BadSession(_BadRequest, KeyError):
    """An unknown or duplicate session id. Also a KeyError, for the same reason.

    ``KeyError.__str__`` reprs its argument, which would double-quote the message; this reports the
    message as written so the client sees the same text every other rejection produces.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


class _BadEnvReply(_BadRequest, ValueError):
    """A reply the environment produced that this transcript cannot carry. Also a ValueError.

    The property the handler splits on is PERMANENT versus TRANSIENT, and this is permanent: the
    environment will produce the same unrepresentable block on every retry, so it belongs with the
    deliberate rejections rather than with the capacity faults. A bare ValueError here would fall
    to the 503 branch and tell the reader the bridge had a resource problem, when the fix is to
    change the environment's reply -- the same misdirection the 400/503 split exists to prevent.
    """


def _request_field(payload: dict, key: str):
    """Read one required payload field, or reject the request.

    A decoded body that is not an object at all (``[]``, ``"x"``, ``3``) is a malformed request, so
    it is rejected here rather than raising TypeError out of the subscript and reading as a fault.
    """
    if not isinstance(payload, dict):
        raise _BadRequest(f"request body must be a json object, got {type(payload).__name__}")
    try:
        return payload[key]
    except KeyError as exc:
        raise _BadRequest(f"missing required field {key!r}") from exc


def request_int(payload: dict, key: str) -> int:
    """Read one required integer payload field, or reject the request."""
    value = _request_field(payload, key)
    try:
        return int(value)
    # OverflowError: json decodes 1e309 to inf, and int(inf) overflows rather than ValueError-ing.
    except (TypeError, ValueError, OverflowError) as exc:
        raise _BadRequest(f"field {key!r} is not an integer: {value!r}") from exc


def request_session_id(payload: dict) -> str:
    """Read the required session id, or reject the request."""
    return str(_request_field(payload, "session_id"))


def _authentication_prompts_equal(expected: list[dict], actual: list[dict]) -> bool:
    """compare prompts after only the text concatenation performed by chat templates."""

    def content_parts(content) -> list[tuple[str, object]]:
        if isinstance(content, str):
            return [("text", content)] if content else []
        parts: list[tuple[str, object]] = []
        text_parts: list[str] = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block["text"])
                continue
            text = "".join(text_parts)
            if text:
                parts.append(("text", text))
            text_parts = []
            parts.append(("block", block))
        text = "".join(text_parts)
        if text:
            parts.append(("text", text))
        return parts

    if len(expected) != len(actual):
        return False
    for expected_message, actual_message in zip(expected, actual, strict=True):
        expected_metadata = {
            key: value for key, value in expected_message.items() if key != "content"
        }
        actual_metadata = {key: value for key, value in actual_message.items() if key != "content"}
        if expected_metadata != actual_metadata:
            return False
        if content_parts(expected_message["content"]) != content_parts(actual_message["content"]):
            return False
    return True


# ONLY the bridge's own deliberate rejections are client errors. classifying by exception TYPE
# instead cannot work: a user env raising IndexError/KeyError deep inside its own scoring would be
# reported as a malformed request -- the same "blame the caller for this side's failure" bug this
# module's 503 split exists to fix, one layer down. so the rejecting code raises _BadRequest itself
# and anything else reaching the handler is this side failing to serve a well-formed request.
_BAD_REQUEST_ERRORS = (_BadRequest,)


# size the listen backlog for the full prompts_per_step * group_size connection burst.
# socketserver's default of 5 resets overflowed clients, and bridge_post intentionally does not retry.
# a fixed backlog only moves the cliff; linux may clamp this value to somaxconn.
_REWARD_BRIDGE_MIN_REQUEST_BACKLOG = 128

# the directory the child-side modules are copied FROM. anchored to `flash/engine/worker` rather
# than to this file: the paths in MULTI_TURN_CHILD_MODULES are relative to the worker package, and
# resolving them against this module's own directory would silently look under train/rl/.
_WORKER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class MultiTurnBridge:
    """hold parent-side env state for the child's multi-turn loop.

    the child owns tokens and generation; this bridge owns the flash env and treats its rollout state
    as authoritative for turn limits, doneness, transcript, and scoring. the callback receives each
    scored episode. multi-turn scoring is scalar, so no component breakdown exists.
    """

    def __init__(
        self,
        env,
        examples: list[dict],
        *,
        env_prompts: list[list[dict]],
        max_turns: int,
        prompt_ids: list[list[int]] | None = None,
        prompt_descriptors: list[list[str] | tuple[str, ...]] | None = None,
        package_root: str | None = None,
        processor=None,
        tokenizer=None,
        thinking: bool = False,
        per_turn_credit: bool = False,
        on_episode_scored: Callable[[object, object, float], None] | None = None,
        parent_work: ParentWorkGauge | None = None,
        identity_ledger=None,
        score_batch_size: int = _MULTI_TURN_SCORE_BATCH_SIZE,
        score_flush_wait_s: float = _MULTI_TURN_SCORE_FLUSH_WAIT_S,
        session_lease_s: float = _MULTI_TURN_SESSION_LEASE_S,
    ) -> None:
        if len(env_prompts) != len(examples):
            raise ValueError("multi-turn env prompts must align one-to-one with examples")
        if prompt_ids is not None and len(prompt_ids) != len(examples):
            raise ValueError("multi-turn prompt ids must align one-to-one with examples")
        prompt_descriptors = prompt_descriptors or [()] * len(examples)
        if len(prompt_descriptors) != len(examples):
            raise ValueError("multi-turn image descriptors must align one-to-one with examples")
        self._env = env
        self._examples = examples
        self._env_prompts = [
            validate_structured_messages(messages, source="frozen environment prompt")
            for messages in env_prompts
        ]
        self._prompt_ids = (
            None
            if prompt_ids is None
            else [tuple(int(token_id) for token_id in ids) for ids in prompt_ids]
        )
        self._prompt_descriptors = [tuple(values) for values in prompt_descriptors]
        self._processor = processor
        self._prompt_digests = [
            tuple(parent_image_digests(processor, values, package_root))
            for values in self._prompt_descriptors
        ]
        self._package_root = package_root
        self._tokenizer = tokenizer
        self._thinking = bool(thinking)
        self._max_turns = int(max_turns)
        self._per_turn_credit = bool(per_turn_credit)
        self._on_episode_scored = on_episode_scored
        self._parent_work = parent_work or ParentWorkGauge()
        self._identity_ledger = identity_ledger
        # the flash env is not required to be thread-safe, and verl runs many rollouts at once.
        # every stateful episode touch below happens under this lock.
        self._lock = threading.Lock()
        self._warned_missing_turn_rewards = False
        # per-run turn totals, published through `turn_accounting`. guarded by the same lock as
        # every other mutable bridge state.
        self._scored_episodes = 0
        # counted from the parent's own `next_turn`, NOT from the child's `turn_count`. these
        # counters exist to prove the child's turn loop really iterated, so deriving them from a
        # number the child reports about itself would make a child that collapsed to one turn
        # report whatever it liked -- the one failure they are here to catch. `step` validates
        # every ordinal against `next_turn` before incrementing it, so the parent's count is exact.
        self._scored_turns = 0
        self._max_observed_turns = 0
        self._sessions: dict[str, dict] = {}
        self._session_lease_s = float(session_lease_s)
        # score many episodes under one lock-held env call so judge work can use the env's own
        # max_score_concurrency. keep the lock: reward_thread_safe permits scorer/scorer concurrency,
        # not racing scoring against env_reply.
        # the defaults are what this path wants: env scoring errors propagate as raised (the reward
        # routes already translate them for the child), and a batch assembled during the flush window
        # is still scored, so the last requests of a run answer instead of failing on shutdown.
        self._scorer = ScoreBatcher(
            self._score_batch,
            max_batch_size=int(score_batch_size),
            flush_wait_s=float(score_flush_wait_s),
            label="multi-turn episode score batcher",
            thread_name="flash-grpo-episode-scorer",
        )

    def routes(self) -> dict:
        def start(payload: dict) -> dict:
            for key in ("raw_prompt", "prompt_ids", "image_count", "image_digests"):
                _request_field(payload, key)
            return self.start(payload)

        def step(payload: dict) -> dict:
            for key in (
                "turn_ordinal",
                "accepted_prefix",
                "response_ids",
                "image_count",
                "image_digests",
            ):
                _request_field(payload, key)
            return self.step(payload)

        return {
            "/multiturn/start": start,
            "/multiturn/step": step,
            "/multiturn/score": self.score,
            "/multiturn/close": self.close,
        }

    def turn_accounting(self) -> dict[str, int | float | None]:
        """Bounded per-run turn totals, for proving the multi-turn loop actually iterated.

        A regression that ended every episode after one turn still produces finite gradients, a
        nonzero adapter delta, and complete artifacts -- every existing gate stays green while the
        environment's multi-turn contract is silently dead. These counters make that visible.
        OPD already publishes the same pair as `episodes_seen` / `mt_turn_records`.
        """
        with self._lock:
            episodes = self._scored_episodes
            turns = self._scored_turns
            maximum = self._max_observed_turns
        return {
            "episodes_scored": episodes,
            "turn_records": turns,
            "max_turns_observed": maximum,
            "mean_turns_per_episode": (turns / episodes if episodes else None),
        }

    def shutdown(self) -> None:
        """stop the scoring thread. distinct from the ``/multiturn/close`` route, which ends one
        episode. any episode still waiting is failed rather than left blocked on its event."""
        self._scorer.close(_MULTI_TURN_SCORE_SHUTDOWN_WAIT_S)

    def _session(self, payload: dict) -> dict:
        session_id = request_session_id(payload)
        session = self._sessions.get(session_id)
        if session is None:
            raise _BadSession(f"unknown multi-turn session {session_id}")
        # every touch renews the lease, so only a session nobody is driving can ever age out. all
        # callers already hold the lock.
        session["touched_at"] = time.monotonic()
        return session

    def _reap_abandoned_sessions(self) -> list[str]:
        """Drop sessions whose owner stopped driving them. Caller holds the lock.

        `monotonic` rather than wall time: a clock step must not expire a live episode, nor keep a
        dead one alive.
        """
        if self._session_lease_s <= 0:
            return []
        cutoff = time.monotonic() - self._session_lease_s
        stale = [sid for sid, s in self._sessions.items() if s.get("touched_at", 0.0) <= cutoff]
        for session_id in stale:
            self._sessions.pop(session_id, None)
        return stale

    def _env_call(self, method: str, *args):
        with self._parent_work.busy():
            return getattr(self._env, method)(*args)

    @staticmethod
    def _payload_media_identity(payload: dict, *, required: bool) -> tuple[int, tuple[str, ...]]:
        if not required and "image_count" not in payload and "image_digests" not in payload:
            return 0, ()
        image_count = request_int(payload, "image_count")
        digests = _request_field(payload, "image_digests")
        if not isinstance(digests, list) or any(not isinstance(value, str) for value in digests):
            raise _BadRequest("field 'image_digests' must be a list of strings")
        if image_count != len(digests):
            raise _BadRequest("image count does not match ordered media digests")
        return image_count, tuple(digests)

    def start(self, payload: dict) -> dict:
        index = request_int(payload, "index")
        if index < 0 or index >= len(self._examples):
            raise _BadIndex(
                f"multi-turn example index {index} is outside [0, {len(self._examples)})"
            )
        session_id = request_session_id(payload)
        example = self._examples[index]
        identity = (
            self._identity_ledger.require_registered(payload.get("identity"), index)
            if self._identity_ledger is not None
            else None
        )
        expected_prompt = self._env_prompts[index]
        expected_digests = self._prompt_digests[index]
        if "raw_prompt" in payload:
            raw_prompt = validate_structured_messages(
                payload["raw_prompt"], source="child initial prompt"
            )
            if not _authentication_prompts_equal(expected_prompt, raw_prompt):
                raise _BadRequest(
                    "multi-turn child prompt does not match the frozen environment prompt"
                )
        image_count, image_digests = self._payload_media_identity(
            payload, required="raw_prompt" in payload
        )
        if image_count != len(expected_digests) or image_digests != expected_digests:
            raise _BadRequest("multi-turn child media does not match the frozen environment prompt")
        prompt_ids = [int(token_id) for token_id in payload.get("prompt_ids", [])]
        if self._prompt_ids is not None and tuple(prompt_ids) != self._prompt_ids[index]:
            raise _BadRequest(
                "multi-turn child prompt ids do not match the frozen processor prompt"
            )
        with self._lock:
            # swept here rather than on a timer thread: a session is only ever abandoned by an
            # actor that stopped calling, and the actors that replace it announce themselves
            # exactly here. no extra thread, and nothing to shut down.
            reaped = self._reap_abandoned_sessions()
            if session_id in self._sessions:
                raise _BadSession(f"duplicate multi-turn session {session_id}")
            state = self._env_call("new_rollout_state", example, expected_prompt)
            self._sessions[session_id] = {
                "example": example,
                "state": state,
                "identity": identity,
                "messages": copy.deepcopy(expected_prompt),
                "descriptors": list(self._prompt_descriptors[index]),
                "image_digests": list(expected_digests),
                "required_prefix": prompt_ids,
                "next_turn": 0,
                "turns": [],
                "touched_at": time.monotonic(),
            }
        if reaped:
            # loud, because it means rollout actors are dying mid-episode. the leak it prevents is
            # silent, so the reap must not be.
            print(
                f"[rl-verl] reaped {len(reaped)} abandoned multi-turn session(s) after "
                f"{self._session_lease_s:.0f}s idle: {', '.join(sorted(reaped))}",
                flush=True,
            )
        # the per-example budget wins over the batch-wide cap, same precedence as rollout_done.
        episode_turns = state.get("max_episode_turns")
        turns = self._max_turns if episode_turns is None else int(episode_turns)
        return {"max_turns": max(1, min(self._max_turns, turns))}

    @staticmethod
    def _step_response(
        session: dict,
        *,
        terminal: bool,
        messages: list[dict],
        image_data_uris: tuple[str, ...] = (),
        authenticated: bool,
    ) -> dict:
        response = {"terminal": terminal, "messages": messages}
        if authenticated:
            response.update(
                {
                    "image_data_uris": list(image_data_uris),
                    "image_count": len(session["image_digests"]),
                    "image_digests": list(session["image_digests"]),
                }
            )
        return response

    def step(self, payload: dict) -> dict:
        with self._lock:
            session = self._session(payload)
            state = session["state"]
            turn_ordinal = int(payload.get("turn_ordinal", session["next_turn"]))
            if turn_ordinal != session["next_turn"]:
                raise _BadRequest(
                    f"multi-turn rollout expected turn {session['next_turn']}, got {turn_ordinal}"
                )
            authenticated = "accepted_prefix" in payload
            accepted_prefix = [
                int(token_id)
                for token_id in payload.get("accepted_prefix", session["required_prefix"])
            ]
            if accepted_prefix != session["required_prefix"]:
                raise _BadRequest(
                    "multi-turn rollout prompt does not exactly match the authenticated environment context"
                )
            image_count, image_digests = self._payload_media_identity(
                payload, required=authenticated
            )
            if authenticated and (
                image_count != len(session["image_digests"])
                or image_digests != tuple(session["image_digests"])
            ):
                raise _BadRequest(
                    "multi-turn rollout media does not match the authenticated context"
                )
            response_ids = [int(token_id) for token_id in payload.get("response_ids", [])]
            completion_text = str(payload.get("completion_text") or "")
            session["next_turn"] += 1
            # an unusable turn is terminal and must not be shown to the env or teacher.
            if bool(payload.get("truncated")) or str(payload.get("skip_reason") or ""):
                session["aborted_turn"] = {
                    "role": "assistant",
                    "content": completion_text,
                }
                return self._step_response(
                    session,
                    terminal=True,
                    messages=[],
                    authenticated=authenticated,
                )
            self._env_call("record_model_turn", state, completion_text)
            session["messages"].append({"role": "assistant", "content": completion_text})
            next_prefix = [*accepted_prefix, *response_ids]
            if self._env_call("rollout_done", state, self._max_turns):
                session["required_prefix"] = next_prefix
                return self._step_response(
                    session,
                    terminal=True,
                    messages=[],
                    authenticated=authenticated,
                )
            replies = self._env_call("env_reply", list(state.get("messages") or ()), state)
            terminal = bool(self._env_call("rollout_done", state, self._max_turns))
            if terminal:
                # terminal replies remain available to environment scoring but are never actor or
                # teacher context, so their media is deliberately neither normalized nor transported.
                session["required_prefix"] = next_prefix
                return self._step_response(
                    session,
                    terminal=True,
                    messages=[],
                    authenticated=authenticated,
                )
            try:
                normalized = normalize_environment_reply(
                    replies,
                    self._package_root,
                    session["descriptors"],
                )
                if authenticated or normalized.descriptors:
                    glue_ids, new_digests = parent_environment_glue(
                        self._processor,
                        self._tokenizer,
                        normalized.messages,
                        normalized.descriptors,
                        self._package_root,
                        thinking=self._thinking,
                    )
                else:
                    glue_ids, new_digests = [], []
            except ValueError as error:
                raise _BadEnvReply(str(error)) from error
            glue_ids = dedup_seam_terminator(response_ids, glue_ids)
            session["messages"].extend(normalized.messages)
            session["descriptors"].extend(normalized.descriptors)
            session["image_digests"].extend(new_digests)
            session["required_prefix"] = [*next_prefix, *glue_ids]
            session["turns"].append(
                {
                    "messages": copy.deepcopy(session["messages"]),
                    "descriptors": tuple(session["descriptors"]),
                    "image_digests": tuple(session["image_digests"]),
                }
            )
            return self._step_response(
                session,
                terminal=False,
                messages=normalized.messages,
                image_data_uris=normalized.data_uris,
                authenticated=authenticated,
            )

    def _score_batch(self, requests: list) -> list:
        """score a whole batch of terminal episodes in ONE env call. runs on the batcher thread.

        takes the same lock every other env touch takes, so scoring never overlaps a concurrent
        episode's ``env_reply``. the win is that one lock acquisition now covers a whole batch.
        """
        with self._lock, self._parent_work.busy():
            return score_rollouts(self._env, requests)

    def score(self, payload: dict) -> dict:
        turn_count = request_int(payload, "turn_count")
        with self._lock:
            session = self._session(payload)
            state = session["state"]
            expected_identity = session.get("identity")
        if self._identity_ledger is not None:
            identity = self._identity_ledger.validate_for_index(
                payload.get("identity"),
                expected_identity.sample_index,
            )
            if identity != expected_identity:
                raise ValueError("multi-turn GRPO score identity does not match its session")
            self._identity_ledger.record(payload.get("identity"), expected_identity.sample_index)
        # queued OUTSIDE the lock so concurrent episodes can coalesce into one env call; the
        # batcher thread reacquires it to do the scoring. safe to read this session's state
        # unlocked because the episode is terminal -- the child sends /score only after its turn
        # loop has ended, so nothing else mutates this session.
        reward = self._scorer.submit(
            RolloutScoreRequest(
                example=session["example"],
                state=state,
                turn_count=turn_count,
            )
        )
        with self._lock:
            # counted here, AFTER identity validation and a scorer reply, so the accounting only
            # ever describes episodes that were really scored. counting on entry let a request the
            # checks below reject -- a mismatched or duplicate identity -- or a scorer failure
            # still inflate the totals, and `turn_accounting()` is published from the runner's
            # `finally` path, so those inflated numbers reach the durable notes of a run that
            # failed. that is the opposite of what the counters exist to prove.
            #
            # one /score call per terminal episode, so this counts episodes exactly once. recorded
            # in `score` rather than in `step` because a truncated turn returns terminal without
            # ever reaching the env, and it is still a turn the model generated and trained on.
            #
            # `next_turn`, not the payload's `turn_count`: they are different quantities on purpose.
            # `turn_count` is the SCOREABLE turn total the child derives from `turn_spans`, which
            # deliberately omits an unusable turn because the env never saw it and returns no reward
            # for it -- that is the scoring contract and it stays as it is. the counters here want
            # the GENERATED total, which is what the sentence above says they mean, and taking it
            # from the parent's own validated ordinal also stops the child from self-reporting the
            # very number that would expose it collapsing to one turn per episode.
            generated_turns = int(session["next_turn"])
            self._scored_episodes += 1
            self._scored_turns += generated_turns
            self._max_observed_turns = max(self._max_observed_turns, generated_turns)
            # snapshot under the same lock that guards the session: `step` mutates this list in
            # place, and a concurrent episode's turn would otherwise be read mid-append.
            prompt = list(state.get("prompt") or ())
            # the episode transcript, NOT the whole message list: new_rollout_state seeds `messages`
            # with a copy of `prompt` and appends turns onto it, so publishing it whole would repeat
            # the prompt inside `completion` when it already rides the sample as `prompt_tail`.
            # slice by length rather than by equality -- an env may legitimately produce a turn that
            # matches a prompt message, and dropping it would silently truncate the episode.
            transcript = list(state.get("messages") or ())[len(prompt) :]
            # the aborted turn never entered `messages` -- the env must not score it -- but the
            # child trained on its tokens, so the sample shows what was actually generated.
            aborted = session.get("aborted_turn")
            if aborted is not None:
                transcript.append(aborted)
        # nan is score_rollouts' unscorable marker; verl has no equivalent, and a nan would
        # propagate into the group baseline and poison every other rollout in the group, so an
        # unscorable episode scores 0.0 the way a failed single-turn grading does.
        episode = float(reward.episode)
        if not math.isfinite(episode):
            print("[rl-verl] multi-turn episode unscorable; scoring 0.0", flush=True)
            episode = 0.0
        if self._on_episode_scored is not None:
            # outside the lock: the callback is the caller's buffer, which has its own lock, and
            # nesting them in this order would invert the single-turn path's acquisition order.
            self._on_episode_scored(prompt, transcript, episode)
        if not self._per_turn_credit:
            return {"score": episode}
        # score_rollouts already validated the vector against turn_count and canonicalised an
        # unusable one to None (multiturn_reward_scoring), so nothing further is checked here.
        # None means this episode falls back to episode credit; the shim widens that fallback to
        # the whole group so a group is never centred on a mix of the two.
        if reward.turns is None:
            with self._lock:
                warn = not self._warned_missing_turn_rewards
                self._warned_missing_turn_rewards = True
            if warn:
                # scoped to the group, not the run: the shim skips only the groups that contain an
                # episode without a usable vector, so other groups keep per-turn credit. saying
                # "this run" would misreport which credit assignment actually trained the model.
                print(
                    "[rl-verl] per-turn credit was requested, but an episode returned no usable "
                    "per_turn_rewards metadata; every rollout group containing such an episode "
                    "falls back to episode-level credit (warned once per run)",
                    flush=True,
                )
        turns = None if reward.turns is None else [float(value) for value in reward.turns]
        return {"score": episode, "turns": turns}

    def close(self, payload: dict) -> dict:
        with self._lock:
            self._sessions.pop(request_session_id(payload), None)
        return {"closed": True}

    def open_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)


# every GRPO child receives one complete flat plugin bundle. copying the multi-turn module even for
# single-turn runs keeps the bundle shape invariant while the plugin decides whether to register it.
GRPO_CHILD_MODULES = (
    (os.path.join("train", "core", "child", "runtime.py"), "flash_verl_runtime.py"),
    (
        os.path.join("..", "..", "content", "reasoning_normalization.py"),
        "flash_reasoning_normalization.py",
    ),
    (os.path.join("train", "core", "child", "glue.py"), "flash_multiturn_glue.py"),
    (os.path.join("train", "rl", "child", "patches.py"), "flash_grpo_patches.py"),
    (os.path.join("train", "rl", "child", "multiturn.py"), "flash_grpo_multiturn.py"),
    (os.path.join("train", "rl", "child", "plugin.py"), "flash_grpo_plugin.py"),
    (os.path.join("train", "rl", "child", "entry.py"), "flash_grpo_entry.py"),
)


def copy_grpo_child_modules(shim_dir: str) -> tuple[str, ...]:
    """copy the complete GRPO plugin bundle and return the paths written."""
    written = []
    for source_name, child_name in GRPO_CHILD_MODULES:
        target = os.path.join(shim_dir, child_name)
        shutil.copy2(os.path.join(_WORKER_DIR, source_name), target)
        written.append(target)
    return tuple(written)


def multi_turn_child_env(inp: dict, *, reward_url: str, thinking: bool) -> dict[str, str]:
    """return only the child env vars consumed by the multi-turn loop.

    the verl interpreter cannot import flash, so the parent resolves generation-config eos values,
    turn limits, halting, and glue rendering into strings.
    """
    return {
        "FLASH_VERL_MULTITURN_URL": reward_url,
        "FLASH_VERL_MAX_TURNS": str(int(inp["max_turns"])),
        "FLASH_VERL_MAX_MODEL_LEN": str(int(inp["engine_len"])),
        # the PER-TURN cap. distinct from the two episode-wide budgets the child already derives:
        # `[train].max_completion_tokens` bounds ONE assistant turn, while max_model_len and the
        # response tensor bound the whole transcript. without it the first turn may consume the
        # entire episode budget, which is both a cost and a behaviour change from the retired
        # driver's `_turn_budget` (per_turn_max_tokens=max_completion).
        "FLASH_VERL_MAX_COMPLETION_TOKENS": str(int(inp["max_completion"])),
        "FLASH_VERL_STOP_SEQUENCES": json.dumps(list(inp["stop_sequences"])),
        # sorted so the child sees a stable set regardless of frozenset iteration order.
        "FLASH_VERL_EOS_TOKEN_IDS": json.dumps(sorted(inp["eos_token_ids"])),
        "FLASH_VERL_THINKING": "1" if thinking else "0",
    }


def start_reward_server(
    score_by_index,
    *,
    example_count: int,
    multi_turn_bridge=None,
    rollout_batch: int = 0,
    score_batch=None,
    identity_ledger=None,
):
    """start the localhost reward server and return ``(server, base_url)``.

    call ``server.shutdown()`` when done. multi-turn adds four episode routes beside ``/score``;
    ``rollout_batch`` sizes the accept queue for the simultaneous episode burst.
    """
    # the scalar compatibility path serializes top-level env calls. training supplies score_batch,
    # which keeps that same one-call-at-a-time boundary while coalescing concurrent verl requests into
    # the env's own batched scorer.
    score_lock = threading.Lock()
    score_batcher = (
        ScoreBatcher(
            score_batch,
            max_batch_size=_SINGLE_TURN_SCORE_BATCH_SIZE,
            flush_wait_s=_SINGLE_TURN_SCORE_FLUSH_WAIT_S,
            label="single-turn reward score batcher",
            thread_name="flash-grpo-reward-scorer",
        )
        if callable(score_batch)
        else None
    )

    def _register_identities(payload: dict) -> dict:
        if identity_ledger is None:
            raise RuntimeError("GRPO identity registration is not configured")
        identities = _request_field(payload, "identities")
        if not isinstance(identities, list):
            raise _BadRequest("field 'identities' must be a list")
        step = identity_ledger.register(identities)
        return {"optimizer_step": step, "registered": len(identities)}

    def _score_route(payload: dict) -> dict:
        index = request_int(payload, "index")
        if index < 0 or index >= example_count:
            raise _BadIndex(f"reward example index {index} is outside [0, {example_count})")
        solution_str = payload.get("solution_str", "")
        if identity_ledger is not None:
            identity_ledger.record(payload.get("identity"), index)
        if score_batcher is not None:
            return {"score": float(score_batcher.submit((index, solution_str)))}
        with score_lock:
            return {"score": float(score_by_index(index, solution_str))}

    routes = {"/identity/register": _register_identities, "/score": _score_route}
    if multi_turn_bridge is not None:
        routes.update(multi_turn_bridge.routes())

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            route = routes.get(self.path)
            if route is None:
                self.send_response(404)
                self.end_headers()
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
            except _UNDECODABLE_BODY_ERRORS as exc:
                status = 400
                detail = f"{type(exc).__name__}: {exc}"
            else:
                try:
                    result = route(payload)
                except _BAD_REQUEST_ERRORS as exc:
                    # the request itself is wrong: a malformed field, an index outside the dataset,
                    # or an unknown/duplicate session. no retry or capacity change makes it succeed.
                    # reported without the internal class name -- the message already says what is
                    # wrong, and "_BadIndex:" only names a private type the caller cannot act on.
                    status = 400
                    detail = str(exc)
                except Exception as exc:
                    # the request was fine and this side could not serve it -- thread exhaustion
                    # ("can't start new thread") under rollout concurrency is the case that drove
                    # this split. 400 blamed the caller's payload and sent every reader to score
                    # code that cannot produce one, so a resource fault reads as a 5xx here.
                    status = 503
                    detail = f"{type(exc).__name__}: {exc}"
                else:
                    status = 200
            if status == 200:
                body = json.dumps(result).encode()
            else:
                print(
                    f"[rl-verl] reward server request failed ({status}) for {self.path}: {detail}",
                    flush=True,
                )
                body = json.dumps({"error": detail}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class _RewardBridgeHTTPServer(BoundedThreadingHTTPServer):
        request_queue_size = max(_REWARD_BRIDGE_MIN_REQUEST_BACKLOG, int(rollout_batch))

        def shutdown(self):
            if score_batcher is not None:
                score_batcher.close(_SINGLE_TURN_SCORE_SHUTDOWN_WAIT_S)
            super().shutdown()

    server = _RewardBridgeHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}"
