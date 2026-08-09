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

# ceiling on one REQUEST body, measured on the rendered prompt before it is serialized. the prompt
# comes from the user's own prompt_messages(), so a corpus-sized or malformed row would otherwise be
# encoded and sent in full -- once per draw. the same limit as the response for the same reason: a
# prompt this large is far past any run's context budget, so the worker would drop the row anyway
# and the draw could never have priced real work.
MAX_REQUEST_BYTES = 8 * 1024 * 1024

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
        slot = attempts % len(prompts)
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
        if sample is None:
            failures += 1
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                break
        else:
            collected.append(sample)
            measured_prompts.add(slot)
            consecutive_failures = 0
    return RolloutSampling(
        samples=tuple(collected),
        failures=failures,
        sampled_prompts=len(measured_prompts),
        offered_prompts=len(prompts),
    )


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
) -> RolloutSample | None:
    """one generation's token counts, or None when the draw failed for any reason.

    ``timeout_s`` is the caller's REMAINING overall budget when that is tighter than the per-request
    ceiling, so no single draw can carry the pass past its deadline.
    """
    # measured on the rendered prompt, before it is encoded into a request body. the content comes
    # from the user's own prompt_messages(), so a corpus-sized row would otherwise be serialized and
    # sent in full on every draw. declining is the right answer rather than truncating: a shortened
    # prompt would measure a completion distribution that belongs to no run at all.
    if sum(len(str(m.get("content", ""))) for m in messages) > MAX_REQUEST_BYTES:
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
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=timeout_s) as response:
            # read one byte past the ceiling rather than trusting Content-Length, which an origin
            # can understate or omit entirely. over the limit is a declined draw, not an exception:
            # the caller's fallback is the declared cap either way.
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
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
) -> RolloutSample | None:
    """token counts for one draw, or None when the response cannot stand as a measurement."""
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
        return None
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
