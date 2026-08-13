"""The parent side of the multi-turn GRPO rollout loop.

In multi-turn GRPO the verl child owns generation but not the environment: flash keeps the env
here, in the parent, and the child reaches it over a local reward server. `MultiTurnBridge` holds
that per-session env state, `start_reward_server` serves both the single- and multi-turn scoring
endpoints, and the copy/env helpers stage the child-side modules the verl interpreter imports.

Split out of `flash.engine.worker.rl_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler

from flash.engine.worker.backend_common import BoundedThreadingHTTPServer
from flash.engine.worker.score_batcher import ScoreBatcher
from flash.engine.worker.train.core.child.glue import validate_transcript_messages
from flash.engine.worker.train.rl.scoring import RolloutScoreRequest, score_rollouts

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

# size the listen backlog for the full prompts_per_step * group_size connection burst.
# socketserver's default of 5 resets overflowed clients, and bridge_post intentionally does not retry.
# a fixed backlog only moves the cliff; linux may clamp this value to somaxconn.
_REWARD_BRIDGE_MIN_REQUEST_BACKLOG = 128

# the directory the child-side modules are copied FROM. anchored to `flash/engine/worker` rather
# than to this file: the paths in MULTI_TURN_CHILD_MODULES are relative to the worker package, and
# resolving them against this module's own directory would silently look under train/rl/.
_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        per_turn_credit: bool = False,
        on_episode_scored: Callable[[object, object, float], None] | None = None,
        score_batch_size: int = _MULTI_TURN_SCORE_BATCH_SIZE,
        score_flush_wait_s: float = _MULTI_TURN_SCORE_FLUSH_WAIT_S,
        session_lease_s: float = _MULTI_TURN_SESSION_LEASE_S,
    ) -> None:
        if len(env_prompts) != len(examples):
            raise ValueError("multi-turn env prompts must align one-to-one with examples")
        self._env = env
        self._examples = examples
        self._env_prompts = env_prompts
        self._max_turns = int(max_turns)
        self._per_turn_credit = bool(per_turn_credit)
        self._on_episode_scored = on_episode_scored
        # the flash env is not required to be thread-safe, and verl runs many rollouts at once.
        # every stateful episode touch below happens under this lock.
        self._lock = threading.Lock()
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
        return {
            "/multiturn/start": self.start,
            "/multiturn/step": self.step,
            "/multiturn/score": self.score,
            "/multiturn/close": self.close,
        }

    def shutdown(self) -> None:
        """stop the scoring thread. distinct from the ``/multiturn/close`` route, which ends one
        episode. any episode still waiting is failed rather than left blocked on its event."""
        self._scorer.close(_MULTI_TURN_SCORE_SHUTDOWN_WAIT_S)

    def _session(self, payload: dict) -> dict:
        session_id = str(payload["session_id"])
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown multi-turn session {session_id}")
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

    def start(self, payload: dict) -> dict:
        index = int(payload["index"])
        if index < 0 or index >= len(self._examples):
            raise IndexError(
                f"multi-turn example index {index} is outside [0, {len(self._examples)})"
            )
        session_id = str(payload["session_id"])
        example = self._examples[index]
        with self._lock:
            # swept here rather than on a timer thread: a session is only ever abandoned by an
            # actor that stopped calling, and the actors that replace it announce themselves
            # exactly here. no extra thread, and nothing to shut down.
            reaped = self._reap_abandoned_sessions()
            if session_id in self._sessions:
                raise KeyError(f"duplicate multi-turn session {session_id}")
            state = self._env.new_rollout_state(example)
            # new_rollout_state calls start_episode again after dataset preparation. adopt the dataset's
            # prompt so randomized envs do not generate for episode a and score episode b; keep the
            # remaining state created by the env.
            env_prompt = [dict(message) for message in self._env_prompts[index]]
            state["prompt"] = env_prompt
            state["messages"] = [dict(message) for message in env_prompt]
            self._sessions[session_id] = {
                "example": example,
                "state": state,
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

    def step(self, payload: dict) -> dict:
        with self._lock:
            session = self._session(payload)
            state = session["state"]
            # an unusable turn is terminal and must NOT be shown to the env: recording it would
            # append a truncated or empty assistant message to the transcript that gets scored.
            # the child stops on the same condition, so this only decides what the env sees.
            if bool(payload.get("truncated")) or str(payload.get("skip_reason") or ""):
                # it is still the turn the model generated and the child trained on, so it is kept
                # for the DIAGNOSTIC transcript. dropping it entirely would publish an empty
                # completion for a first-turn truncation -- the one sample worth reading, since it
                # is the failure being diagnosed.
                session["aborted_turn"] = {
                    "role": "assistant",
                    "content": str(payload.get("completion_text") or ""),
                }
                return {"terminal": True, "messages": []}
            self._env.record_model_turn(state, str(payload.get("completion_text") or ""))
            if self._env.rollout_done(state, self._max_turns):
                return {"terminal": True, "messages": []}
            replies = self._env.env_reply(list(state.get("messages") or ()), state)
            terminal = bool(self._env.rollout_done(state, self._max_turns))
        # NOT str(content): an env returning image blocks hands back a list of content blocks, and
        # coercing it would put the python repr of that list into the transcript as text -- the
        # model would read "[{'type': 'image_url', ...}]" while no pixels were ever added, because
        # the media on every generate call is fixed from the initial prompt. the shared validator
        # is the same contract the opd bridge applies to its own env replies, and it rejects
        # non-text content loudly instead. rendering a mid-episode image is genuinely unsupported
        # here; being unsupported is survivable, being silent is not.
        return {
            "terminal": terminal,
            "messages": validate_transcript_messages(list(replies), source="environment reply"),
        }

    def _score_batch(self, requests: list) -> list:
        """score a whole batch of terminal episodes in ONE env call. runs on the batcher thread.

        takes the same lock every other env touch takes, so scoring never overlaps a concurrent
        episode's ``env_reply``. the win is that one lock acquisition now covers a whole batch.
        """
        with self._lock:
            return score_rollouts(self._env, requests)

    def score(self, payload: dict) -> dict:
        turn_count = int(payload["turn_count"])
        with self._lock:
            session = self._session(payload)
            state = session["state"]
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
        turns = None if reward.turns is None else [float(value) for value in reward.turns]
        return {"score": episode, "turns": turns}

    def close(self, payload: dict) -> dict:
        with self._lock:
            self._sessions.pop(str(payload["session_id"]), None)
        return {"closed": True}

    def open_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)


# the three stdlib-only modules the child needs beside its shim, mapped to the flat names it
# imports them under. flat and `flash_`-prefixed because the child imports them as top-level
# modules, not as a package: flash itself is NOT importable in the verl interpreter (incompatible
# torch/vllm pins), which is why they are copied rather than imported. same mechanism the opd verl
# path uses for its own loop.
# (path relative to flash/engine/worker/, flat name the child imports it under). the child code
# lives under train/, so these are package-relative paths rather than bare siblings.
MULTI_TURN_CHILD_MODULES = (
    (os.path.join("train", "core", "child", "glue.py"), "flash_multiturn_glue.py"),
    (os.path.join("train", "rl", "child", "multiturn.py"), "flash_grpo_multiturn.py"),
    (os.path.join("train", "rl", "child", "plugin.py"), "flash_grpo_plugin.py"),
)


def copy_multi_turn_child_modules(shim_dir: str) -> tuple[str, ...]:
    """copy the child-side agent loop next to the shim; returns the paths written."""
    written = []
    for source_name, child_name in MULTI_TURN_CHILD_MODULES:
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
        # verl imports this at the end of `verl/__init__` (import_external_libs), which is what
        # registers the loop under the name the agent-loop override selects. without it the
        # override names a loop that was never registered and the child dies at rollout build.
        "VERL_USE_EXTERNAL_MODULES": "flash_grpo_plugin",
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

    def _score_route(payload: dict) -> dict:
        index = int(payload["index"])
        if index < 0 or index >= example_count:
            raise IndexError(f"reward example index {index} is outside [0, {example_count})")
        solution_str = payload.get("solution_str", "")
        if score_batcher is not None:
            return {"score": float(score_batcher.submit((index, solution_str)))}
        with score_lock:
            return {"score": float(score_by_index(index, solution_str))}

    routes = {"/score": _score_route}
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
                result = route(payload)
            except Exception as exc:
                print(f"[rl-verl] reward server request failed: {exc}", flush=True)
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            else:
                body = json.dumps(result).encode()
                self.send_response(200)
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
