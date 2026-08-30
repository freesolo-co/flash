"""Preregistered workload buckets, prompt construction, and metric definitions.

Every definition here is fixed BEFORE any GPU is allocated so a disappointing result cannot be
rescued by redefining the metric afterwards. The report records this module's digest.

Three properties this module is responsible for:

* **Prefix caching must not be measured as capacity.** Hosted serving runs with
  ``ENABLE_PREFIX_CACHING = True``, which is correct for production (shared system prompts) and
  fatal for a capacity benchmark: a shared prefix turns a throughput number into a cache-hit-rate
  number. Every prompt is therefore request-unique from its FIRST token, and any request whose
  engine-reported ``cached_tokens`` is nonzero is an invalid sample, not a fast one.
* **Tokens are counted by the engine, never by the client.** vLLM reports authoritative
  ``prompt_tokens``/``completion_tokens``. Counting SSE deltas instead would silently inflate
  throughput whenever a chunk carries more than one token.
* **Input length is measured after the chat template is applied**, because the template adds tokens
  the caller never wrote.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

# Fixed decoding contract. temperature=0 with n=1 makes each request's work deterministic given its
# prompt, so a concurrency curve varies concurrency and nothing else.
TEMPERATURE = 0.0
TOP_P = 1.0
N_CHOICES = 1
# Thinking is ON for the measured path: reasoning and answer tokens both count toward throughput and
# TTFT is the first reasoning-or-content delta. A base-model serve honors this caller override
# (lora_engine._thinking_default); a trained adapter would not, which is one reason this campaign is
# base-only.
ENABLE_THINKING = True
# How far an assembled prompt may sit from its bucket's target and still count as that bucket. The
# fitter accepts within this band, so the engine-side check must use the SAME number: a tighter one
# would reject prompts the fitter deliberately returned.
PROMPT_TOKEN_TOLERANCE = 16


@dataclass(frozen=True)
class Bucket:
    """One preregistered workload shape."""

    name: str
    target_input_tokens: int
    max_output_tokens: int
    description: str
    # Depth targets, per bucket rather than per call. The error-rate bound only resolves below 1%
    # after ~268 clean attempts (Wilson, z=1.645), which is minutes of wall time for a short turn but
    # hours for a near-32k one. Each bucket therefore carries the deepest floor its shape can afford,
    # and a bucket that cannot reach the depth publishes an unresolved bound instead of a false one.
    min_requests: int = 20
    min_seconds: float = 60.0
    max_seconds: float = 300.0

    @property
    def total_token_ceiling(self) -> int:
        return self.target_input_tokens + self.max_output_tokens


# Near-32k targets 31_744 = 32_768 - 1_024, leaving room for the template plus 512 output tokens
# inside the engine's configured 32_768 max_model_len. Sizing it against the RAW prompt instead
# would overflow the context once the template is applied.
BUCKETS: tuple[Bucket, ...] = (
    Bucket(
        name="short_interactive",
        target_input_tokens=512,
        max_output_tokens=128,
        description="short chat turn; TTFT-dominated",
        # Cheap enough to clear the ~268-attempt bound threshold outright.
        min_requests=300,
        min_seconds=60.0,
        max_seconds=420.0,
    ),
    Bucket(
        name="medium_generation",
        target_input_tokens=8192,
        max_output_tokens=256,
        description="document-sized prompt with a short answer",
        min_requests=120,
        min_seconds=60.0,
        max_seconds=420.0,
    ),
    Bucket(
        name="near_32k",
        target_input_tokens=31744,
        max_output_tokens=512,
        description="near the configured 32768 context limit",
        # 268 attempts at this shape is hours of paid time. Sample what is affordable and let the
        # bound be reported unresolved rather than buying depth the envelope does not need.
        min_requests=20,
        min_seconds=60.0,
        max_seconds=600.0,
    ),
)

BUCKETS_BY_NAME: dict[str, Bucket] = {bucket.name: bucket for bucket in BUCKETS}

# Deterministic filler vocabulary. Ordinary words (not repeated punctuation) keep the tokenizer's
# behavior representative of real traffic rather than of a degenerate byte pattern.
_FILLER_WORDS: tuple[str, ...] = (
    "system",
    "network",
    "capacity",
    "latency",
    "throughput",
    "request",
    "token",
    "engine",
    "kernel",
    "memory",
    "buffer",
    "queue",
    "worker",
    "shard",
    "cache",
    "stream",
    "batch",
    "planner",
    "scheduler",
    "gradient",
    "vector",
    "matrix",
    "cluster",
    "pipeline",
    "adapter",
    "runtime",
    "context",
    "session",
    "channel",
    "segment",
    "profile",
    "counter",
    "sample",
)


def _deterministic_words(seed_material: str, count: int) -> list[str]:
    """``count`` filler words drawn deterministically from ``seed_material``.

    Uses a hash chain rather than ``random`` so the same request id reproduces the same prompt in a
    later replication, on any machine, with no shared RNG state between concurrent workers.
    """
    words: list[str] = []
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    while len(words) < count:
        for byte in digest:
            words.append(_FILLER_WORDS[byte % len(_FILLER_WORDS)])
            if len(words) == count:
                break
        digest = hashlib.sha256(digest).digest()
    return words


def request_uid(bucket: str, concurrency: int, block: int, index: int, invocation: str = "") -> str:
    """Stable unique id for one request. Keys the prompt HEADER, never the filler body.

    ``invocation`` is a per-sweep nonce and it is what makes a RETRY safe. Without it the id depends
    only on the grid coordinates, so re-running a failed sweep at the same block re-sends every
    prompt byte-for-byte; within Modal's 120s scaledown the container and its prefix cache survive,
    and ``_validate`` correctly scores those cache hits ERROR_CACHE_CONTAMINATED -- invalidating a
    paid rerun whose engine was perfectly healthy.

    It deliberately does NOT reach ``corpus_seed``. The filler body stays keyed to the grid
    coordinates alone, so the semantic workload is still reproducible across invocations and a
    curve's points still differ only in offered load; only the per-request header moves.
    """
    suffix = f"-x{invocation}" if invocation else ""
    return f"{bucket}-c{concurrency}-b{block}-i{index}{suffix}"


def corpus_seed(bucket: str, block: int, index: int) -> str:
    """Seed for a prompt's filler BODY. Deliberately carries no concurrency.

    ``request_uid`` must carry the concurrency point, because every request needs an id unique
    across the whole cell. Seeding the body from that id too gave every concurrency point a
    DIFFERENT corpus, so a curve varied the prompt text alongside the one variable it exists to
    isolate. Two points then differed in offered load AND in which words were sent, and the tokenizer
    does not treat every word stream identically.

    Holding the body constant down the grid and letting only the per-request header differ leaves
    concurrency as the sole difference between points, which is what the curve claims to show.
    """
    return f"{bucket}-b{block}-i{index}"


def _prompt_header(uid: str) -> str:
    """The per-request head of a prompt: a digest first, so character ZERO differs per request.

    An earlier form led with ``f"trace {uid}"``, which shares its first ~24 characters across every
    request in a bucket ("trace short_interactive-c4-b0-i" and so on). That shared run is shorter
    than one vLLM cache block today, so it probably would not have been cached, but "probably
    shorter than a block the current build happens to use" is not a property this benchmark should
    rest on. Leading with the digest makes the divergence unconditional.
    """
    nonce = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:16]
    return f"{nonce} trace {uid}"


def build_prompt_text(uid: str, approximate_tokens: int, *, corpus: str | None = None) -> str:
    """A request-unique prompt of roughly ``approximate_tokens`` tokens.

    ``corpus`` seeds the filler body and defaults to ``uid``. Pass a ``corpus_seed`` to hold the body
    fixed across concurrency points while the header stays request-unique.

    The returned length is approximate; ``measure_prompt_tokens`` performs the exact fit against the
    real tokenizer and chat template.
    """
    # ~0.75 words per token for this vocabulary; the exact fit corrects the remainder.
    word_count = max(1, int(approximate_tokens * 0.75))
    words = _deterministic_words(corpus or uid, word_count)
    return _prompt_header(uid) + "\n" + " ".join(words)


def reseed_prompt(messages: list[dict[str, Any]], uid: str) -> list[dict[str, Any]]:
    """Re-key an already-fitted prompt to a new request id WITHOUT re-tokenizing.

    A cell can outrun its prompt pool while it waits on ``min_seconds``. Re-sending a pooled prompt
    is not a cheap reuse: the engine serves it from prefix cache, ``_validate`` correctly marks it
    ERROR_CACHE_CONTAMINATED, and so the FASTER a cell runs the more artificial error rate it
    accumulates -- precisely inverting what the curve is meant to show.

    Refitting on the fly is not the answer either. Fitting is repeated synchronous tokenization, and
    on the event loop it blocks consumption of every other in-flight stream, which is the exact
    distortion ``_build_prompt_pool`` exists to keep out of the measured window.

    So the filler body is reused verbatim and only the header is rewritten. The header carries the
    per-request digest, so the new prompt still diverges from every other at character ZERO and
    cannot share a cache block. Its assembled length moves only by the header's own width, which
    ``PROMPT_TOKEN_TOLERANCE`` accommodates.
    """
    reseeded: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            reseeded.append(message)
            continue
        body = content.split("\n", 1)[1] if "\n" in content else ""
        reseeded.append({**message, "content": _prompt_header(uid) + "\n" + body})
    return reseeded


def messages_for(
    uid: str, approximate_tokens: int, *, corpus: str | None = None
) -> list[dict[str, Any]]:
    """The chat messages for one request: a single user turn, no system prompt.

    No system prompt on purpose. A shared system prompt is the canonical prefix-cache hit, and it
    would be shared across every request in the run.
    """
    return [{"role": "user", "content": build_prompt_text(uid, approximate_tokens, corpus=corpus)}]


def measure_prompt_tokens(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    """Exact assembled prompt length: chat template applied, thinking enabled."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


class PromptFitError(RuntimeError):
    """A prompt could not be fitted inside its bucket's tolerance."""


def fit_prompt_to_tokens(
    tokenizer: Any,
    uid: str,
    target_tokens: int,
    *,
    corpus: str | None = None,
    tolerance: int = PROMPT_TOKEN_TOLERANCE,
    max_iterations: int = 12,
) -> tuple[list[dict[str, Any]], int]:
    """Binary-search a prompt whose ASSEMBLED length is within ``tolerance`` of ``target_tokens``.

    Returns ``(messages, exact_assembled_tokens)``. Every request records its exact measured length,
    so a bucket is reported by what was actually sent rather than by what was requested.

    Raises ``PromptFitError`` when the search cannot land inside ``tolerance``. Returning the best
    near-miss instead would put a materially mis-sized prompt into the advertised bucket: the driver
    checks the engine's reported length against this FITTED count, not against
    ``bucket.target_input_tokens``, so an out-of-band fit is transmitted faithfully and validates
    cleanly while the published bucket label is wrong. A bucket that cannot be fitted is a workload
    defect to surface before the window opens, not a number to publish.
    """
    low, high = 1, max(2, target_tokens * 2)
    best: tuple[list[dict[str, Any]], int] | None = None
    for _ in range(max_iterations):
        guess = (low + high) // 2
        messages = messages_for(uid, guess, corpus=corpus)
        actual = measure_prompt_tokens(tokenizer, messages)
        if best is None or abs(actual - target_tokens) < abs(best[1] - target_tokens):
            best = (messages, actual)
        if abs(actual - target_tokens) <= tolerance:
            return messages, actual
        if actual < target_tokens:
            low = guess + 1
        else:
            high = guess - 1
        if low > high:
            break
    assert best is not None
    if abs(best[1] - target_tokens) > tolerance:
        raise PromptFitError(
            f"could not fit a prompt within {tolerance} tokens of {target_tokens} in "
            f"{max_iterations} iterations; closest was {best[1]}"
        )
    return best


def concurrency_grid(max_num_seqs: int) -> tuple[int, ...]:
    """Preregistered concurrency points for an engine capped at ``max_num_seqs``.

    Powers of two up to the engine cap, then ~1.5x and 2x past it. Measuring past the cap is the
    point: it is where queueing appears, and the saturation criterion needs at least one infeasible
    or clearly-degraded point to be falsifiable.
    """
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be >= 1")
    points: list[int] = []
    value = 1
    while value < max_num_seqs:
        points.append(value)
        value *= 2
    points.append(max_num_seqs)
    points.append(max(max_num_seqs + 1, (max_num_seqs * 3) // 2))
    points.append(max_num_seqs * 2)
    return tuple(sorted(set(points)))


# The grid cap the checksum digests. All three benchmark tiers declare `max_num_seqs=8`, so this
# fixes the digest at the shape actually measured; the ALGORITHM is what is being pinned, and a
# change to it moves the checksum at this cap just as it would at any other.
_CHECKSUM_GRID_CAP = 8

# The functions whose source defines prompt semantics. Named explicitly rather than digesting the
# whole module so unrelated edits (docstrings elsewhere, new helpers, `__all__`) do not invalidate
# comparability between campaigns that ran the same prompt contract.
_CONSTRUCTION_SOURCES: tuple[str, ...] = (
    # `_deterministic_words` decides every filler body and `request_uid` decides every header and
    # cache key, so either can change the workload materially. Neither appeared in any source
    # digested below -- they are CALLED by the digested functions, and `inspect.getsource` reads a
    # function's own text, not its callees. Listed explicitly rather than walked transitively:
    # a transitive walk would sweep in `hashlib` and the tokenizer and make the digest move for
    # reasons that are not the workload.
    "_deterministic_words",
    "request_uid",
    "_prompt_header",
    "build_prompt_text",
    "corpus_seed",
    "reseed_prompt",
    "messages_for",
    "measure_prompt_tokens",
    "fit_prompt_to_tokens",
)


def _construction_digest() -> str:
    """Digest of the prompt-construction implementation itself."""
    import inspect

    joined = chr(31).join(inspect.getsource(globals()[name]) for name in _CONSTRUCTION_SOURCES)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


# The driver and metric code that decides what is actually SENT and how a curve is reduced. The
# prompt contract above fixes what a prompt says; these fix which prompts get issued, which attempts
# become errors, and where the curve is declared saturated. A campaign that changed any of them
# measured different work, so it must not be able to claim this checksum.
#
# Imported lazily inside the digest, not at module scope: `driver` imports `workload`, so a
# module-scope import here would be circular.
_DRIVER_SOURCES: tuple[str, ...] = (
    # What each request asks the engine for, and who issues it when.
    "_payload_for",
    "_prompt_issuer",
    "_build_prompt_pool",
    # What counts as a valid sample. `_validate` and `_absorb_event` decide which attempts become
    # errors, which is the error-rate numerator itself.
    "_absorb_event",
    "_validate",
)
_METRIC_SOURCES: tuple[str, ...] = (
    # The reduction arithmetic and the curve's ceiling/knee/saturation rules. Their THRESHOLDS are
    # keyword defaults in these signatures -- no caller overrides one -- so digesting the source
    # covers the threshold values too.
    "percentile",
    "_distribution",
    "wilson_upper_bound",
    "reduce_cell",
    "summarize_curve",
)
# Module CONSTANTS are digested by value, not by source. A constant's name is all that appears in
# the function text that reads it, so retuning `_POOL_PERIOD_SLACK` from 64 to 128 would leave every
# source digest above byte-identical while changing which prompts the pool wraps to.
_DRIVER_CONSTANTS: tuple[str, ...] = (
    "REQUEST_TIMEOUT_SECONDS",
    "_POOL_PERIOD_SLACK",
    "_PROMPT_FIT_FIXED_SECONDS",
    "_PROMPT_FIT_SECONDS_PER_TOKEN",
    "_PROMPT_FIT_MAX_ITERATIONS",
)


def _execution_digest() -> str:
    """Digest of the driver and metric behaviour that produced a curve.

    Separate from `_construction_digest` on purpose: the two answer different questions. The
    construction digest says "these prompts"; this one says "issued this way, and reduced by these
    rules". Reporting them as one value would make a driver retune indistinguishable from a prompt
    change when a stale artifact is being explained.
    """
    import inspect

    from flash.serving.bench import driver, metrics

    parts = [inspect.getsource(getattr(driver, name)) for name in _DRIVER_SOURCES]
    parts += [inspect.getsource(getattr(metrics, name)) for name in _METRIC_SOURCES]
    parts += [f"{name}={getattr(driver, name)!r}" for name in _DRIVER_CONSTANTS]
    # Every error code is part of the contract: renaming or adding one changes how a failed attempt
    # is reported, and a consumer comparing two campaigns' failure breakdowns needs that to move.
    parts += [
        f"{name}={getattr(metrics, name)!r}"
        for name in sorted(n for n in dir(metrics) if n.startswith("ERROR_"))
    ]
    return hashlib.sha256(chr(31).join(parts).encode()).hexdigest()[:16]


def workload_checksum() -> str:
    """Digest of the preregistered contract, recorded in the report.

    Any later edit to the buckets or decoding contract changes this value, so a report cannot
    silently describe a workload other than the one it measured.

    The depth floors are part of the digest, not decoration: `min_requests` decides whether the
    error-rate bound can resolve at all, and `min_seconds`/`max_seconds` decide how long a cell runs.
    Two campaigns that differ only in those numbers are materially different measurements, so they
    must not share a checksum.
    """
    material = "|".join(
        [
            f"temperature={TEMPERATURE}",
            f"top_p={TOP_P}",
            f"n={N_CHOICES}",
            f"thinking={ENABLE_THINKING}",
            # The prompt CONSTRUCTION, not just the bucket dimensions. Two campaigns can agree on
            # every token count and still measure different work: the filler vocabulary decides what
            # the model actually reads, the fit tolerance decides how far an assembled prompt may sit
            # from its target, and the concurrency grid decides which points the curve contains.
            # Digesting only the dimensions let all three change without moving the checksum, so a
            # published result could claim a preregistered workload it did not run.
            f"tolerance={PROMPT_TOKEN_TOLERANCE}",
            f"filler={hashlib.sha256(chr(31).join(_FILLER_WORDS).encode()).hexdigest()[:16]}",
            # The construction CODE, not an enumeration of its inputs. Listing the filler vocabulary
            # and tolerance still left `_prompt_header`, `build_prompt_text`, `corpus_seed`,
            # `reseed_prompt` and the fitting/template logic outside the digest -- and every one of
            # them changes what the model reads, what the cache sees, or how a prompt is assembled.
            # An enumeration can only ever cover the inputs someone remembered; digesting the source
            # covers the next edit too, so two materially different workload contracts cannot share
            # a checksum.
            f"construction={_construction_digest()}",
            # The EXECUTION contract, not just the prompt contract. `_payload_for`, `_prompt_issuer`,
            # the pool period, the request timeout and the reduction/saturation thresholds all change
            # which prompts are issued, which attempts become errors, or where the curve saturates --
            # and none of them appear in any prompt-construction source. Two materially different
            # campaigns could otherwise publish the same checksum.
            f"execution={_execution_digest()}",
            f"grid={','.join(str(point) for point in concurrency_grid(_CHECKSUM_GRID_CAP))}",
            *(
                f"{b.name}:{b.target_input_tokens}:{b.max_output_tokens}"
                f":{b.min_requests}:{b.min_seconds}:{b.max_seconds}"
                for b in BUCKETS
            ),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


__all__ = [
    "BUCKETS",
    "BUCKETS_BY_NAME",
    "ENABLE_THINKING",
    "N_CHOICES",
    "PROMPT_TOKEN_TOLERANCE",
    "TEMPERATURE",
    "TOP_P",
    "Bucket",
    "PromptFitError",
    "build_prompt_text",
    "concurrency_grid",
    "corpus_seed",
    "fit_prompt_to_tokens",
    "measure_prompt_tokens",
    "messages_for",
    "request_uid",
    "reseed_prompt",
    "workload_checksum",
]
