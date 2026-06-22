"""Serve a trained LoRA adapter via the freesolo platform's multi-LoRA serving app.

Flash no longer runs its own per-run vLLM endpoint. Instead the control plane is a
thin client of the freesolo serving service (a Modal multi-LoRA app that serves every
adapter on a single GPU per base model, scaling to zero when idle — so there is no
flash-side idle billing to track). The same CLI commands and control-plane endpoints
(`deploy`/`undeploy`/`chat`/`deployments`) stay; only what they do under the hood
changed.

The serving service exposes:

- ``POST {FREESOLO_SERVING_URL}/adapters`` — register/deploy an adapter (auth header).
- ``DELETE {FREESOLO_SERVING_URL}/adapters/{adapterId}`` — undeploy (auth header).
- ``POST {FREESOLO_SERVING_URL}/v1/chat/completions`` — OpenAI-style chat (no auth).
- ``GET {FREESOLO_SERVING_URL}/healthz`` / ``GET .../adapters`` — health / list.

The registration/teardown calls carry the shared ``X-Freesolo-Internal-Key`` header
(the same internal credential flash already holds, ``FREESOLO_INTERNAL_KEY``); chat is
unauthenticated.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

import httpx

from flash._logging import get_logger
from flash.providers.base import canonical_gpu, gpu_short

logger = get_logger(__name__)

# Default freesolo serving base URL (the Modal multi-LoRA app). Overridable per-env.
DEFAULT_FREESOLO_SERVING_URL = "https://clado-ai--freesolo-lora-serving.modal.run"

MODES = ("dev", "always-on")
DEFAULT_IDLE_TIMEOUT_S = 300

# A chat must complete (warm or cold start) within this many seconds of wall-clock time, or
# we raise instead of hanging. Modal scale-from-zero cold starts are minutes, not 30 min, so
# this is generous yet bounded — the old code used a 30-minute httpx timeout applied *per
# redirect hop*, which (combined with Modal's long-polling async-result redirects) let a single
# chat crawl for many minutes / appear to hang forever. Env-overridable for slow base models.
CHAT_DEADLINE_S = 8 * 60.0
# Per-HTTP-request socket timeout (one POST or one async-result poll). Bounded so a single
# stalled hop can't wedge us; the overall budget is enforced by CHAT_DEADLINE_S across hops.
CHAT_HTTP_TIMEOUT_S = 90.0
# How long to wait between async-result polls (Modal returns the result via a redirect to a
# ?__modal_function_call_id= poll URL; we poll it ourselves with a small backoff rather than
# letting httpx chase the redirect chain, which has no global deadline).
_CHAT_POLL_INTERVAL_S = 2.0


class ServingError(RuntimeError):
    """The freesolo serving backend (Modal LoRA app) rejected a request or was unreachable.

    Carries the upstream status (when there was an HTTP response) so the API layer can
    surface a clean ``502 Bad Gateway`` with the real reason instead of letting an
    ``httpx`` exception escape as an unhandled ``500`` + traceback.
    """

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _post_adapter_or_raise(url: str, body: dict) -> httpx.Response:
    """POST an adapter registration to the serving backend, translating any transport- or
    status-level failure into a ``ServingError`` that carries the upstream detail."""
    try:
        # follow_redirects: Modal answers a slow request with a 303 to an async-result poll URL
        # (?__modal_function_call_id=...); without following it httpx raises on the 303 (see chat).
        resp = httpx.post(
            url,
            json=body,
            headers=_internal_key_header(),
            timeout=60.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp
    except httpx.HTTPStatusError as exc:
        raise _serving_status_error(url, exc) from exc
    except httpx.RequestError as exc:
        raise ServingError(f"could not reach the serving backend at {url}: {exc}") from exc


def _serving_status_error(url: str, exc: httpx.HTTPStatusError) -> ServingError:
    """Build a ``ServingError`` from an upstream HTTP failure, carrying the status and a
    4xx-vs-5xx-tailored hint (shared by the deploy POST and the undeploy DELETE)."""
    # raise_for_status() always carries a response, but a hand-built HTTPStatusError may
    # not — guard so error translation can never itself raise.
    resp = exc.response
    status = resp.status_code if resp is not None else None
    detail = ((resp.text if resp is not None else "") or "").strip()[:500]
    msg = f"serving backend error for {url}"
    if status is not None:
        msg += f" (HTTP {status})"
    if detail:
        msg += f": {detail}"
    # Tailor the hint to the upstream status: a 4xx is a client/auth problem with THIS request
    # (e.g. a missing/invalid FREESOLO_INTERNAL_KEY), not a serving outage; a 5xx (or unknown)
    # means the backend itself failed / has no engine for the base model.
    if status is not None and status < 500:
        msg += (
            " — the serving backend rejected the request (4xx); check FREESOLO_INTERNAL_KEY "
            "and the request payload (this is a client/auth error, not a serving outage)"
        )
    else:
        msg += (
            " — the serving backend is unavailable or has no engine for this base model; "
            "an operator must check the freesolo serving deployment"
        )
    return ServingError(msg, status_code=status)


def serving_base_url() -> str:
    """The freesolo serving base URL (env-overridable, trailing slash stripped)."""
    return (os.environ.get("FREESOLO_SERVING_URL") or DEFAULT_FREESOLO_SERVING_URL).rstrip("/")


def _internal_key_header() -> dict[str, str]:
    key = os.environ.get("FREESOLO_INTERNAL_KEY") or ""
    return {"X-Freesolo-Internal-Key": key} if key else {}


@dataclass
class Deployment:
    run_id: str
    model: str
    adapter_hf_prefix: str
    gpu: str
    openai_model: str
    endpoint_name: str
    mode: str = "dev"
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S
    # freesolo serving scales to zero per base model, so flash never bills for idle
    # serving — there is no flash-side per-run endpoint to keep warm.
    est_idle_cost_usd_per_day: float = 0.0
    state: str = "ready"

    def to_dict(self) -> dict:
        return asdict(self)


def serve_endpoint_name(friendly_gpu: str, run_id: str) -> str:
    """Cosmetic endpoint label (the freesolo app serves all adapters on one endpoint)."""
    tail = (run_id or "").split("-")[-1][:24]
    base = f"flash-serve-{gpu_short(canonical_gpu(friendly_gpu))}"
    return f"{base}-{tail}" if tail else base


def servable_gpu(gpu_name: str) -> str:
    """Resolve a friendly GPU class for the deployment record.

    Serving is delegated to freesolo (one GPU per base model, chosen there), so this is
    now informational. We still canonicalize the name and fall back to the cheapest RunPod
    class big enough when the trained class isn't a RunPod class, so the recorded ``gpu`` is
    a sensible, valid class (and junk GPU names still raise)."""
    from flash.providers.base import GPU_INFO, cheapest_gpu

    friendly = canonical_gpu(gpu_name)
    info = GPU_INFO[friendly]
    if info.enum_member:  # a RunPod class — serve it directly
        return friendly
    return cheapest_gpu(info.vram_gb)  # else the cheapest RunPod class that fits


def deploy_adapter(
    run_id: str,
    model: str,
    hf_repo: str,
    adapter_prefix: str,
    gpu_name: str = "RTX 5090",
    mode: str = "dev",
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S,
    dry_run: bool = False,
    thinking: bool = False,
) -> Deployment:
    """Register the trained adapter with the freesolo serving app.

    The adapter artifacts already live in the run's HF dataset repo (the trainer
    streamed them there); freesolo serving pulls them from
    ``{hf_repo}:{adapter_prefix}/adapter``. ``dry_run`` validates/shapes the deployment
    without making the network call.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    friendly = servable_gpu(gpu_name)
    subfolder = f"{adapter_prefix}/adapter"
    dep = Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=subfolder,
        gpu=friendly,
        openai_model=run_id,
        endpoint_name=serving_base_url(),
        mode=mode,
        idle_timeout_s=idle_timeout_s,
        est_idle_cost_usd_per_day=0.0,
        state="dry_run" if dry_run else "ready",
    )
    if dry_run:
        return dep
    base = serving_base_url()
    body = {
        "adapterId": run_id,
        "repoId": hf_repo,
        "baseModel": model,
        "subfolder": subfolder,
        # The trainer always streams the adapter into a *dataset* repo (the worker's
        # hf_upload_folder uses repo_type="dataset"), so serving must pull from the dataset
        # namespace. Without this the serving app defaults repoType to "model" and
        # snapshot_download 404s on the model namespace — deploy returns 200 but the engine
        # warmup fails, the adapter is silently disabled, and the first chat 404s.
        "repoType": "dataset",
        "status": "ready",
    }
    _post_adapter_or_raise(f"{base}/adapters", body)
    logger.info("registered adapter %s with freesolo serving (%s)", run_id, base)
    return dep


def undeploy_adapter(run_id: str) -> list[str]:
    """Deregister the run's adapter from the freesolo serving app.

    Returns ``[run_id]`` when the adapter was removed (200), ``[]`` when it was already
    gone (404). Any other failure — a non-404 HTTP status or a transport error — is
    translated into a ``ServingError`` (carrying the upstream status), exactly like
    ``deploy_adapter``, so callers see a stable error surface (the API maps it to a clean
    502) instead of a raw ``httpx`` exception escaping as an unhandled 500.
    """
    base = serving_base_url()
    url = f"{base}/adapters/{run_id}"
    try:
        resp = httpx.delete(
            url,
            headers=_internal_key_header(),
            timeout=60.0,
            # Modal answers a slow request with a 303 to an async-result poll URL; follow it (see chat).
            follow_redirects=True,
        )
        # Undeploy is idempotent: an already-absent adapter (404) is a no-op success, not an
        # error — handle it before raise_for_status() so it never becomes a ServingError.
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _serving_status_error(url, exc) from exc
    except httpx.RequestError as exc:
        raise ServingError(f"could not reach the serving backend at {url}: {exc}") from exc
    logger.info("deregistered adapter %s from freesolo serving (%s)", run_id, base)
    return [run_id]


def _chat_deadline_s() -> float:
    """Overall wall-clock budget for one chat (env-overridable, never unbounded)."""
    raw = os.environ.get("FREESOLO_CHAT_TIMEOUT_S")
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return CHAT_DEADLINE_S


def _is_modal_async_redirect(resp: httpx.Response) -> bool:
    """True when Modal answered with a redirect to its async-result poll URL.

    Modal serves a request that outruns its synchronous window by returning a 303/302/307 whose
    ``Location`` is the same path with a ``?__modal_function_call_id=`` query — polling that URL
    eventually yields the completion. (A redirect *without* that marker is something else and is
    surfaced as an error rather than silently chased.)"""
    if resp.status_code not in (302, 303, 307, 308):
        return False
    loc = resp.headers.get("location") or ""
    return "__modal_function_call_id=" in loc


def chat(
    run_id: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
) -> dict:
    """Send an OpenAI-style chat request for the run's adapter to freesolo serving.

    The adapter is addressed by ``model=run_id`` (its registered ``adapterId``); the
    response is the parsed OpenAI chat-completion dict, so
    ``resp["choices"][0]["message"]["content"]`` keeps working downstream.

    Cold starts (scale-from-zero per base model) can take a couple of minutes. The warm path
    returns ``200`` directly (exactly like a raw POST). On a cold start Modal answers with a
    redirect to an async-result poll URL (``?__modal_function_call_id=...``); we poll that URL
    ourselves with a small backoff against a hard wall-clock deadline and raise on timeout.

    Earlier this delegated redirect-following to httpx (``follow_redirects=True``,
    ``max_redirects=100``, ``timeout=30*60``). That hung: Modal's async-result URL *long-polls*
    (each poll blocks a few seconds, then redirects again until the result is ready), httpx
    applies its timeout *per hop* (no global budget), and chasing up to 100 such hops let a
    single chat crawl for many minutes / appear to hang forever — while a plain POST returned in
    seconds. Polling explicitly with one bounded deadline fixes that.
    """
    base = serving_base_url()
    body = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        # Per-run thinking parity: a run trained with thinking must serve with thinking, so
        # forward the flag to the chat template (enable_thinking is the kwarg the renderer and
        # rollout path use, e.g. multiturn_rollout.build_rollout_func). Without this the served
        # completions diverge from training behavior even though the caller passes thinking=.
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    _deadline_s = _chat_deadline_s()
    deadline = time.monotonic() + _deadline_s
    url = f"{base}/v1/chat/completions"
    _base_url = httpx.URL(url)  # origin reference for the same-origin guard on the poll redirect
    # follow_redirects=False: we never let httpx chase Modal's async-result redirect chain (it has
    # no global deadline and the poll URL long-polls). We handle the one redirect ourselves below.
    with httpx.Client(follow_redirects=False, timeout=CHAT_HTTP_TIMEOUT_S) as client:
        try:
            resp = client.post(url, json=body, headers=_internal_key_header())
            # Cold start: chase the async-result poll URL ourselves, bounded by `deadline`.
            while _is_modal_async_redirect(resp):
                if time.monotonic() >= deadline:
                    raise ServingError(
                        f"chat for {run_id} timed out after {_deadline_s:.0f}s waiting on the "
                        "serving backend (cold start did not finish); retry once the engine is warm"
                    )
                poll_url = str(resp.url.join(resp.headers["location"]))
                # The poll GET carries the internal key — NEVER send it off-origin. Modal's
                # async-result redirect is same-origin; refuse a redirect to a different scheme/host/
                # port (an absolute cross-origin Location would otherwise leak FREESOLO_INTERNAL_KEY).
                _pu = httpx.URL(poll_url)
                if (_pu.scheme, _pu.host, _pu.port) != (_base_url.scheme, _base_url.host, _base_url.port):
                    raise ServingError(
                        f"chat for {run_id} refused a cross-origin async-result redirect to "
                        f"{_pu.host!r} (expected {_base_url.host!r}); not sending the internal key off-origin"
                    )
                time.sleep(_CHAT_POLL_INTERVAL_S)
                # Modal's async-result poll is a GET; it returns 200 when ready, or redirects again.
                resp = client.get(poll_url, headers=_internal_key_header())
        except httpx.TimeoutException as exc:
            raise ServingError(
                f"chat for {run_id} timed out talking to the serving backend at {base}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise ServingError(
                f"could not reach the serving backend at {base}: {exc}"
            ) from exc
    # If we exited the poll loop still holding a 3xx, it is NOT Modal's async-result redirect (those
    # are what the loop chases). httpx only raises_for_status on 4xx/5xx, so such a redirect would
    # slip through and blow up confusingly in resp.json(); surface it as a clear serving error.
    if 300 <= resp.status_code < 400:
        raise ServingError(
            f"chat for {run_id} got an unexpected {resp.status_code} redirect to "
            f"{resp.headers.get('location')!r} from the serving backend at {base} (not a Modal "
            "async-result redirect); the deployment may be misconfigured"
        )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _serving_status_error(url, exc) from exc
    return resp.json()
