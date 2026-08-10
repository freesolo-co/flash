"""sample realized rollout length through a hosted inference api, off gpu.

realized completion length is a property of (model, prompt, sampling params) and NOT of the
hardware: a faster card changes how QUICKLY tokens are emitted, never HOW MANY. so the quantity the
grpo/opd quote is missing can be measured against a hosted endpoint on cpu, instead of renting the
gpu whose price this measurement exists to compute.

what this module does NOT do is measure SECONDS. generation seconds are provider-, card- and
moment-specific, so only the token COUNT transfers to the run being quoted.

the endpoint is openai-compatible and named only in configuration, never in identifiers here: the
catalog of hosted models is data, and a second provider must not require a second module.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

# where the samples come from. an openai-compatible chat-completions endpoint is the whole contract:
# the sampler reads `usage.completion_tokens`, `usage.prompt_tokens` and `finish_reason`, which are
# the fields such an api bills on, so a provider has every incentive to report them correctly.
ROLLOUT_SAMPLER_BASE_URL_ENV = "FLASH_ROLLOUT_SAMPLER_BASE_URL"
ROLLOUT_SAMPLER_API_KEY_ENV = "FLASH_ROLLOUT_SAMPLER_API_KEY"
DEFAULT_ROLLOUT_SAMPLER_BASE_URL = "https://openrouter.ai/api/v1"

# per-request ceiling. a hosted generation at a 2048-token cap takes tens of seconds; this bounds a
# hung request without cutting off a legitimately long one, whose length is exactly the tail this
# profile exists to capture.
REQUEST_TIMEOUT_S = 180.0

# overall ceiling on a sampling pass. the draws are serial and a user is waiting on the submit
# behind them, so the per-request timeout alone is not a bound: 32 stalled draws would be ~96
# minutes. an incomplete sample is not a problem -- it fails the trust gate and the quote returns to
# the declared cap, which is what an unmeasured run was priced at anyway.
SAMPLING_DEADLINE_S = 300.0

# a stalling or unauthorized endpoint fails identically on every draw. stopping after a few in a row
# turns a misconfigured sampler into a fast fallback instead of a long wait.
MAX_CONSECUTIVE_FAILURES = 5

# ceiling on one chat-completion response body. this path reads whatever the configured endpoint
# sends, and json.load consumes the stream with no bound -- so a malfunctioning or hostile origin
# could exhaust memory in the user's `flash train` process instead of taking the fail-open path this
# whole module is built on. generous by two orders of magnitude: the largest legitimate reply here is
# one completion at the 2048-token cap plus usage accounting, a few tens of kilobytes.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# ceiling on one REQUEST body, measured on the serialized bytes. the prompt comes from the user's
# own prompt_messages(), so a corpus-sized or malformed row would otherwise be sent in full -- once
# per draw. the same limit as the response for the same reason: a prompt this large is far past any
# run's context budget, so the worker would drop the row anyway and the draw could never have priced
# real work.
MAX_REQUEST_BYTES = 8 * 1024 * 1024

# preflight on the prompt's CHARACTER count, run before anything is serialized. the encoded check
# below is the accurate one, but it cannot fire until json.dumps has already built the whole string
# and its encoded copy: measured, a 200 MB prompt costs ~400 MB peak reaching a check that then
# rejects it. counting characters costs len() on strings that already exist.
#
# the two bounds are complementary, not redundant, and the direction matters. json escaping only ever
# ADDS bytes -- ascii is one byte, and the worst case, a non-bmp character, is twelve (two escaped
# surrogates) -- so a prompt of N characters encodes to AT LEAST N bytes. a prompt over
# MAX_REQUEST_BYTES characters therefore cannot fit the encoded limit whatever it contains, which is
# what makes declining it here provably free of false declines: nothing rejected by this bound could
# have passed the exact one. everything it admits still goes through the exact check, which is what
# accounts for escaping expansion, roles, stop sequences and every other field.
_MAX_PROMPT_CHARS = MAX_REQUEST_BYTES

# how much of a response body is read per socket operation. small enough that a slow-drip origin is
# noticed within one chunk of the deadline, large enough that a legitimate reply -- tens of kilobytes
# -- costs a couple of reads rather than hundreds.
_READ_CHUNK_BYTES = 64 * 1024

# prompt-mix dominates the variance (measured: between-prompt 304846 vs within-prompt 61082, a word
# problem running a median 1642 tokens against ~290 for arithmetic), so spread draws across DISTINCT
# prompts before repeating any one of them.
SAMPLE_POLICY_DISTINCT_PROMPTS = "distinct-prompts-round-robin"


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """refuse to follow redirects, so the api key never reaches an origin the user did not name.

    urllib copies request headers onto the redirected request, Authorization included, so a 30x from
    the configured endpoint hands the user's PAID inference key to whatever host the Location names.
    A 302 also rewrites the POST to a GET, and the redirected response is still parsed as a sample --
    so the same hop both leaks the credential and lets a foreign host dictate the measurement.

    raising here surfaces as urllib.error.HTTPError, which ``_one_completion`` already treats as a
    failed draw: a redirecting endpoint measures nothing and the quote returns to the declared cap.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# module-level: an opener carries no per-request state, and rebuilding it per draw would add a
# handler chain construction to every sample for nothing.
_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirects)


@dataclass(frozen=True)
class RolloutSample:
    """one realized generation: the token counts, and why it stopped."""

    prompt_tokens: int
    completion_tokens: int
    truncated: bool


class _OutOfBudget:
    """sentinel: the draw succeeded, and the row is one the WORKER drops before training.

    distinct from None, which means the draw failed. a failed draw leaves a prompt unmeasured that
    the run will still train on, so it is missing workload; an out-of-budget row is filtered by both
    workers, so its absence is the profile agreeing with training rather than a gap in it.
    """

    __slots__ = ()


OUT_OF_BUDGET = _OutOfBudget()


@dataclass(frozen=True)
class RolloutSampling:
    """what a sampling pass observed, before any trust verdict is applied."""

    samples: tuple[RolloutSample, ...]
    failures: int
    sampled_prompts: int
    offered_prompts: int = 0

    @property
    def completed(self) -> int:
        return len(self.samples)

    @property
    def truncated(self) -> int:
        return sum(1 for s in self.samples if s.truncated)

    def completion_tokens(self) -> list[int]:
        return [s.completion_tokens for s in self.samples]

    def summary(self) -> dict[str, object]:
        """aggregates only. no prompts, completions, token ids or credentials leave this process.

        returns the distribution rather than a mean alone: a mean cannot tell a uniformly short
        workload from a bimodal one whose long tail sets the step time.
        """
        lengths = sorted(self.completion_tokens())
        prompts = [s.prompt_tokens for s in self.samples]
        if not lengths:
            return {
                "sampled_prompts": self.sampled_prompts,
                "offered_prompts": self.offered_prompts,
                "completed_rollouts": 0,
                "failed_rollouts": self.failures,
            }
        return {
            "sampled_prompts": self.sampled_prompts,
            "offered_prompts": self.offered_prompts,
            "completed_rollouts": len(lengths),
            "failed_rollouts": self.failures,
            "completion_tokens_mean": statistics.fmean(lengths),
            "completion_tokens_p50": _percentile(lengths, 0.50),
            "completion_tokens_p90": _percentile(lengths, 0.90),
            "completion_tokens_max": lengths[-1],
            "prompt_tokens_mean": statistics.fmean(prompts),
            "truncated_rollouts": self.truncated,
            "eos_rollouts": len(lengths) - self.truncated,
            "sample_policy": SAMPLE_POLICY_DISTINCT_PROMPTS,
        }


def _percentile(sorted_values: Sequence[int], q: float) -> int:
    """nearest-rank percentile of an already-sorted sequence."""
    if not sorted_values:
        return 0
    index = max(0, min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1))))
    return int(sorted_values[index])


def sampler_credentials() -> tuple[str, str]:
    """(base_url, api_key) for the hosted sampler; api_key is "" when unconfigured."""
    base = (os.environ.get(ROLLOUT_SAMPLER_BASE_URL_ENV) or "").strip()
    key = (os.environ.get(ROLLOUT_SAMPLER_API_KEY_ENV) or "").strip()
    return (base or DEFAULT_ROLLOUT_SAMPLER_BASE_URL), key


def sample_rollouts(
    *,
    model: str,
    prompts: Sequence[Sequence[dict]],
    rollouts: int,
    max_completion_tokens: int,
    temperature: float | None,
    top_p: float | None,
    base_url: str,
    api_key: str,
    stop_sequences: Sequence[str] = (),
    thinking: bool | None = None,
    prompt_budget: int | None = None,
) -> RolloutSampling:
    """draw ``rollouts`` generations, spreading across ``prompts`` before repeating one.

    never raises for a failed draw: a profile is advisory evidence, and a partial sample is reported
    as partial so the trust verdict can reject it. a caller that gets zero samples simply has no
    measurement, which returns the quote to the declared cap.
    """
    if not prompts or rollouts <= 0:
        return RolloutSampling(samples=(), failures=0, sampled_prompts=0, offered_prompts=0)

    collected: list[RolloutSample] = []
    failures = 0
    # prompts that produced a SAMPLE, not prompts attempted. a prompt the endpoint systematically
    # refuses -- a content filter is the usual cause -- contributes no tokens, and counting the
    # attempt let the loop make up the shortfall from the other prompts and still report full
    # coverage. the worker does not reproduce the hosted filter, so that prompt is part of the run
    # being quoted and its (usually long-tail) length would be missing from an accepted profile.
    measured_prompts: set[int] = set()
    # prompts the endpoint reported as over the worker's own prompt budget. these leave the OFFER
    # rather than counting against coverage: the workers drop those rows before training, so the
    # profile is not missing workload by not measuring them.
    filtered_prompts: set[int] = set()
    # this runs in front of a submit the user is waiting on, and the draws are serial. an endpoint
    # that accepts connections but stalls would otherwise burn REQUEST_TIMEOUT_S on every draw
    # before falling back to cap pricing -- 32 draws x 180s is over an hour of dead wait for a
    # measurement that is advisory anyway. so the sampling gives up on whichever comes first: an
    # overall deadline, or enough consecutive failures to show the endpoint is not answering.
    deadline = time.monotonic() + SAMPLING_DEADLINE_S
    consecutive_failures = 0
    # count SUCCESSES, not attempts. `rollouts` defaults to the server's trust floor, so a loop over
    # attempts meant one transient failure left 31 samples and the server rejected the whole
    # measurement as too thin -- the endpoint recovered, and the run was still priced at the cap.
    # the deadline and the consecutive-failure cutoff remain the bounds; only the exit condition
    # changed, so a dead endpoint still gives up after MAX_CONSECUTIVE_FAILURES draws.
    attempts = 0
    while len(collected) < rollouts:
        # the deadline bounds when a draw may START; this bounds how long it may RUN. checking only
        # at the top lets a draw begun one second inside the deadline stall for the full request
        # timeout, so the "bounded" pass takes deadline + REQUEST_TIMEOUT_S. measured against a
        # stalling endpoint: 6.00s wall on a 4.0s deadline, which at production values is ~480s in
        # front of a submit the user is waiting on.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        # round-robin over ATTEMPTS: keeps one prompt from being redrawn just because an earlier
        # draw of it failed, which would concentrate the sample on whichever prompt is flakiest.
        #
        # the rotation is over the prompts still ELIGIBLE. a filtered prompt is settled -- the
        # endpoint already reported its length and the worker will drop the row -- so redrawing it
        # buys nothing and costs a paid completion every time. measured before this: with all 8
        # prompts over budget the loop issued 186,249 paid requests and collected zero samples,
        # because `collected` never grows and neither counter advances.
        eligible = [i for i in range(len(prompts)) if i not in filtered_prompts]
        if not eligible:
            # every prompt is one the worker drops. there is nothing left to measure, and the pass
            # has established that -- so stop rather than spend the remaining deadline re-confirming
            # it. an empty sample fails the trust gate and the quote returns to the declared cap.
            break
        slot = eligible[attempts % len(eligible)]
        attempts += 1
        sample = _one_completion(
            model=model,
            messages=prompts[slot],
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            base_url=base_url,
            api_key=api_key,
            stop_sequences=stop_sequences,
            timeout_s=min(REQUEST_TIMEOUT_S, remaining),
            thinking=thinking,
            prompt_budget=prompt_budget,
        )
        if sample is OUT_OF_BUDGET:
            # not a failure: the endpoint answered, and the row is one both workers filter out
            # before training. it is dropped from the OFFER below rather than counted against
            # coverage, because a prompt the run never trains on cannot be missing from the sample.
            # the consecutive-failure counter is left alone for the same reason -- a dataset with
            # several over-long rows is not a misbehaving endpoint.
            filtered_prompts.add(slot)
        elif sample is None:
            failures += 1
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                break
        else:
            collected.append(sample)
            measured_prompts.add(slot)
            consecutive_failures = 0
    # a prompt that drew a sample is in budget whatever an earlier draw reported: prompt_tokens is a
    # property of the prompt, so a slot in both sets means the filter reading came from a flapping
    # endpoint, not from the row. counting it as offered is the conservative reading.
    filtered_prompts -= measured_prompts
    return RolloutSampling(
        samples=tuple(collected),
        failures=failures,
        sampled_prompts=len(measured_prompts),
        offered_prompts=len(prompts) - len(filtered_prompts),
    )


def _prompt_exceeds_chars(messages: Sequence[dict], limit: int) -> bool:
    """whether a prompt's strings hold more than ``limit`` characters, without building a copy.

    every string is counted, not just ``content``: a message carries whatever prompt_messages() put
    on it, and an oversized row is as easily oversized in a name or a tool-call argument. containers
    are WALKED rather than repr'd -- taking len(repr(value)) would materialize exactly the copy this
    check exists to avoid -- and the walk stops the moment the limit is passed, so a pathological row
    costs a partial traversal instead of a full one.

    a non-string leaf counts as ONE, below the bytes it actually encodes to. undercounting is the
    safe direction: it can only admit a payload, never decline one, which is what keeps the bound
    free of false declines. bulk lives in strings anyway, and the exact encoded check downstream is
    what bounds the body.
    """
    total = 0
    seen: set[int] = set()
    stack: list[object] = [messages]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            total += len(item)
            if total > limit:
                return True
            continue
        if isinstance(item, dict):
            # a self-referential row would otherwise loop here forever. json.dumps raises on one, so
            # skipping the repeat only has to keep THIS walk finite: the draw is declined either way.
            if id(item) in seen:
                continue
            seen.add(id(item))
            stack.extend(item.keys())
            stack.extend(item.values())
            continue
        if isinstance(item, (list, tuple)):
            if id(item) in seen:
                continue
            seen.add(id(item))
            stack.extend(item)
            continue
        total += 1
        if total > limit:
            return True
    return False


def _read_within_deadline(response, limit: int, deadline: float) -> bytes | None:
    """read up to ``limit`` bytes, giving up at ``deadline``. None when the deadline passed.

    ``urlopen(timeout=...)`` bounds each blocking socket operation, NOT the call as a whole, and
    ``read(n)`` loops over as many operations as it takes to fill n bytes. an origin that sends one
    byte before each socket timeout expires therefore resets the clock forever: measured, a 3s
    deadline was still blocked at 30s with no end in sight, which hangs the user's `flash train`
    instead of taking the fail-open path this module is built on.

    ``read1`` is what makes the deadline checkable: it returns as soon as ANY bytes arrive -- one
    byte, measured -- where ``read(n)`` blocks until it has all n. so the loop regains control on
    every drip and can compare the ABSOLUTE deadline, and a drip-feeder buys one socket operation
    rather than an unbounded run of them. None rather than an exception, because a timed-out draw is
    an ordinary failed draw: the quote returns to the declared cap.

    checking between reads is not enough on its own, though. the socket keeps whatever timeout the
    request was opened with, so a byte arriving just inside the deadline passes the check and then
    blocks for that FULL original timeout: measured, a byte at 1.8s held a 2.0s deadline out to 3.8s.
    so the socket's own timeout is shrunk to the remaining budget before each read, which makes the
    absolute deadline the real bound rather than an advisory one.
    """
    # http.client responses implement read1; a test double or an exotic file-like may not, and for
    # those the bounded read is still correct, just without the mid-body deadline check.
    read1 = getattr(response, "read1", None)
    if read1 is None:
        return response.read(limit)
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        left = deadline - time.monotonic()
        if left <= 0:
            return None
        _limit_socket_timeout(response, left)
        chunk = read1(min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _limit_socket_timeout(response, seconds: float) -> None:
    """lower the underlying socket's timeout to ``seconds``, best effort.

    without this the deadline is only checked BETWEEN reads: a byte arriving just inside it starts a
    read that then blocks for the socket's original timeout, so the pass overshoots by up to one
    full REQUEST_TIMEOUT_S. narrowing it per read makes the absolute deadline binding.

    best effort because the socket is reached through http.client internals that a test double or a
    future stdlib layout may not expose. failing to narrow it costs the old overshoot, not
    correctness, so an unexpected shape is left alone rather than raised on.
    """
    sock = getattr(getattr(response, "fp", None), "raw", None)
    sock = getattr(sock, "_sock", None)
    if sock is None:
        return
    try:
        sock.settimeout(max(0.001, seconds))
    except (OSError, ValueError, AttributeError):
        return


def _one_completion(
    *,
    model: str,
    messages: Sequence[dict],
    max_completion_tokens: int,
    temperature: float | None,
    top_p: float | None,
    base_url: str,
    api_key: str,
    stop_sequences: Sequence[str] = (),
    timeout_s: float = REQUEST_TIMEOUT_S,
    thinking: bool | None = None,
    prompt_budget: int | None = None,
) -> RolloutSample | _OutOfBudget | None:
    """one generation's token counts, None when the draw failed, OUT_OF_BUDGET when it was filtered.

    the third outcome exists because "no draw" has two very different meanings for coverage: a row
    the WORKER also drops is not missing workload, while a row the endpoint refuses is.

    ``timeout_s`` is the caller's REMAINING overall budget when that is tighter than the per-request
    ceiling, so no single draw can carry the pass past its deadline.
    """
    # cheap bound first, on strings that already exist, so an absurd row is declined before
    # json.dumps builds a copy of it. see _MAX_PROMPT_CHARS: this can only reject prompts the
    # encoded check below would reject too.
    if _prompt_exceeds_chars(messages, _MAX_PROMPT_CHARS):
        return None
    payload: dict[str, object] = {
        "model": model,
        # the env's own prompt_messages() output, roles intact. flattening a system turn into a
        # user message would change both chat-template tokenization and model behaviour, so the
        # sample would describe a different distribution than the one training runs.
        "messages": [dict(m) for m in messages],
        "max_tokens": int(max_completion_tokens),
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    # nucleus sampling decides which continuations can be selected at all, so it moves the stopping
    # length this profile measures. both workers set it explicitly
    # (`actor_rollout_ref.rollout.top_p`), so leaving it off the request would inherit whatever the
    # endpoint defaults to and measure a decoding the run never uses.
    if top_p is not None:
        payload["top_p"] = float(top_p)
    # the training rollout passes these as `stop`, so a run configured with them ends generation at
    # the first one. sampling without them lets the hosted model run on to the cap and reports a
    # mean no rollout produces -- biased HIGH, the direction that overbills. a stop-string draw ends
    # with finish_reason "stop", which is the natural end this already counts as EOS.
    stops = [str(value) for value in stop_sequences if value]
    if stops:
        payload["stop"] = stops
    if thinking is not None:
        # the worker renders with enable_thinking=thinking, so say so rather than inheriting the
        # endpoint's default. sent both ways: `chat_template_kwargs` is what a vllm-compatible
        # server applies to the template itself, and `reasoning` is what a router-style gateway
        # reads. an endpoint that understands neither ignores both, which is why the RESPONSE is
        # checked below instead of this being trusted.
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        payload["reasoning"] = {"enabled": bool(thinking)}
    # the ceiling is enforced on the ENCODED body, not on the prompt's character count. the content
    # comes from the user's own prompt_messages(), and json escaping is not length-preserving: a
    # non-ascii character can serialize to twelve bytes, so a character count passes a body an order
    # of magnitude over the limit. encoding once and measuring the result is also what makes roles,
    # stop sequences and every other field count toward the same bound.
    #
    # nothing is sent when it is exceeded. declining beats truncating: a shortened prompt would
    # measure a completion distribution belonging to no run at all, and a prompt this large is far
    # past any run's context budget, so the worker would drop the row anyway.
    body = json.dumps(payload).encode()
    if len(body) > MAX_REQUEST_BYTES:
        return None
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=timeout_s) as response:
            # read one byte past the ceiling rather than trusting Content-Length, which an origin
            # can understate or omit entirely. over the limit is a declined draw, not an exception:
            # the caller's fallback is the declared cap either way.
            raw = _read_within_deadline(response, MAX_RESPONSE_BYTES + 1, deadline)
        if raw is None or len(raw) > MAX_RESPONSE_BYTES:
            return None
        body = json.loads(raw)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    return _sample_from_response(
        body,
        max_completion_tokens=int(max_completion_tokens),
        thinking=thinking,
        prompt_budget=prompt_budget,
    )


def _reasoned(usage: dict, choice: object) -> bool:
    """whether this response carries visible reasoning.

    three signals, because providers disagree on where reasoning surfaces: a token accounting
    breakdown, a separate reasoning field on the message, and the literal think tags a chat template
    emits when reasoning is on. any one of them is enough to decide the answer either way.

    the reading is symmetric because BOTH mismatches misprice. reasoning against a thinking=False
    run inflates completion length; a plain answer to a thinking=True run understates it, and that
    is the likelier failure -- `reasoning` and `chat_template_kwargs` are non-standard, so an
    endpoint may accept the request and ignore both, returning a short answer that looks perfectly
    valid. an unreasoned draw cannot stand for a run that renders every prompt with thinking on.
    """
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        try:
            if int(details.get("reasoning_tokens") or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        return False
    # `reasoning` and `reasoning_content` name the same thing at different providers.
    if message.get("reasoning") or message.get("reasoning_content"):
        return True
    content = message.get("content")
    return isinstance(content, str) and "<think>" in content


def _sample_from_response(
    body: object,
    *,
    max_completion_tokens: int,
    thinking: bool | None = None,
    prompt_budget: int | None = None,
) -> RolloutSample | _OutOfBudget | None:
    """token counts for one draw, None when it cannot stand, OUT_OF_BUDGET when the worker drops it."""
    usage = body.get("usage") if isinstance(body, dict) else None
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(usage, dict) or not isinstance(choices, list) or not choices:
        return None
    if thinking is not None and _reasoned(usage, choices[0]) is not thinking:
        # the draw's reasoning does not match the run's, in either direction, so it measures
        # different work. reasoning against a thinking=False run adds tokens billed in
        # completion_tokens that the rollout never generates; a plain answer to a thinking=True run
        # is missing the traces every rollout WILL generate, which underprices by far more. both
        # are dropped, which returns the quote to the cap rather than to a confident wrong number.
        return None
    try:
        completion_tokens = int(usage["completion_tokens"])
        prompt_tokens = int(usage["prompt_tokens"])
    except (KeyError, TypeError, ValueError):
        return None
    if completion_tokens <= 0 or prompt_tokens <= 0:
        return None
    if prompt_budget is not None and prompt_tokens > prompt_budget:
        # both workers drop a row whose rendered prompt exceeds `context - max_completion` before
        # training (rl_train.py and opd_train.py both filter on their own prompt_budget), so this
        # row is never trained on. keeping its draw would put a length in the profile that no step
        # produces -- and the long rows are exactly the ones that would dominate the mean.
        #
        # OUT_OF_BUDGET rather than None, so the caller can tell "the worker drops this row too"
        # from "the endpoint refused a row the worker WILL train on". they are indistinguishable by
        # count -- both leave a prompt with zero draws -- but only the second one means the profile
        # is missing part of the workload.
        return OUT_OF_BUDGET
    first = choices[0]
    finish = first.get("finish_reason") if isinstance(first, dict) else None
    # a truncated draw contributes the CAP rather than its true length, biasing the mean downward --
    # the direction that underbills. counted, so the trust verdict can reject a censored sample
    # rather than report it as a measurement.
    #
    # only "stop" is a natural end. a provider can also return "content_filter", "tool_calls", or
    # an unknown/absent reason, and those completions stopped for a reason the training rollout will
    # not reproduce -- usually short. treating them as EOS would let 32 filtered draws pass the
    # trust gate with zero reported truncations and quote a length no rollout produces, so they are
    # dropped as failed draws instead.
    if finish == "length":
        # a truncated draw is a censored observation: the true length is at least the cap, and the
        # count reported here is only where generation was cut off. when a provider stops BELOW the
        # requested cap -- its own output ceiling, or an exhausted context -- keeping that smaller
        # number would pull the mean down, and up to MAX_TRUSTWORTHY_TRUNCATION_RATE of such draws
        # are accepted, so they would underquote. record the cap the run will actually generate to.
        return RolloutSample(
            prompt_tokens=prompt_tokens,
            completion_tokens=max(completion_tokens, int(max_completion_tokens)),
            truncated=True,
        )
    if finish != "stop":
        return None
    return RolloutSample(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, truncated=False
    )
